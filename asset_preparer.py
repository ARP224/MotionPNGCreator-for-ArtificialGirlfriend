#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
asset_preparer.py

MotionPNGCreator for ArtificialGirlfriend 素材準備 GUI。
video_generator.py での連続生成前の事前準備をタブ順に行う。

タブ構成（左から順に進める）:
  ⓪ インストラクション: ワークフォルダ選択・APIキー・事前準備チェックリスト
  ① 画像リスケール:     ワークフォルダ直下の画像を 720x1280 に統一
  ② 背景色置換:         画像の背景色を単色に置換（1枚ずつ）
  ③ 目・口検出 (SAM3):  face_info.json 生成 + キャラクターフォルダへ移動
  ④ Vidu動画生成:       キャラクターフォルダごとに口パク用ループ動画を生成
  ⑤-1 抽出 / ⑤-2 出力:  動画から口スプライト (open/closed/half) を抽出
  ⑥ 口消し (ブラシ):    立ち絵画像の口をブラシで塗りつぶして消す
  ⑦ 状態確認:           video_generator が期待するフォルダ構造かを検証・自動修正

License: AGPL-3.0-only (Ultralyticsライセンスに準拠)
"""

from __future__ import annotations

import json
import math
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import traceback
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from vidu_api import (
    IMAGE_EXTS,
    VIDU_MODELS,
    ViduAPIError,
    image_to_base64_uri,
    vidu_download,
    vidu_get,
    vidu_post,
)
from batch_engine import (scan_characters, EXCLUDED_DIRS, calc_credits, path_ansi_safe,
                          char_name_error, normalize_char_name)
from i18n import tr, current_language, catalog_load_errors

# pythonw（コンソール非表示）起動では stdout/stderr が None になり print() が
# 例外を出すため、ダミー出力へ差し替える。ログを見たい場合はコンソールから
# `uv run python asset_preparer.py` で起動する
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")

# リダイレクト時（> log.txt 等）のUnicodeEncodeError防止。
# コンソール表示のエンコーディング自体は変更しない
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(errors="replace")
        except Exception:
            pass

# Optional: drag & drop support
_HAS_TK_DND = False
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    _HAS_TK_DND = True
except Exception:
    pass

try:
    from PIL import Image, ImageTk
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False
    print("[warn] PIL not installed. Preview will be limited.")

# SAM3の利用可否。torch/ultralyticsの初回importは数十秒かかるため、
# ウィンドウ表示後にバックグラウンドスレッドで判定する（App._start_ai_loading）。
# None = 判定中。判定完了までSAM3を使う機能（③実行・⑤解析）は無効化される。
from sam3_mouth_detector import SAM3MouthDetector, is_sam3_available

_SAM3_OK: Optional[bool] = None
_SAM3_ERR = ""

from mouth_sprite_extractor import (
    MouthSpriteExtractor,
    MouthFrameInfo,
    warp_frame_to_norm,
    ensure_even_ge2,
    extract_mouth_sprite,
    adjust_quad,
    classify_mouth_frames,
    center_to_quad,
    imwrite_jp,
)


# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
SETTINGS_PATH = os.path.join(HERE, ".batch_settings.json")      # video_generator と共有
LEGACY_CONFIG_PATH = os.path.join(HERE, ".vidu_batch_config.json")  # 旧utility.pyの設定（初回移行のみ）
SAM3_MODEL_PATH = os.path.join(HERE, "Sam3", "sam3.pt")

TARGET_W = 720
TARGET_H = 1280

# ② 背景色置換の置換先。Phase2 の背景透過（erase_mouth_offline.create_alpha_hybrid）が
# #00FF00 固定で判定するため、ここも固定であって変更してはいけない。
FIXED_BG_HEX = "#00FF00"

POLL_INTERVAL = 10
MAX_POLL_FAILURES = 5  # ポーリング連続失敗の上限（超えたらタスクを失敗扱い）
MAX_POLL_SECONDS = 30 * 60  # ポーリングの最大経過時間（超えたらタスクを失敗扱い）
ASPECT_WARN_RATIO = 0.03  # リスケール時、9:16からのずれがこれを超えたら警告

# ③ 口の検出結果の妥当性チェック（目間距離に対する比率）。
# 口の無い画像で検出すると SAM3 が小さなゴミを「口」と返すことがあり、
# 量産時の口消し・口合成の位置がずれる。閉じた口の bbox は高さ≒0.05 なので
# 高さの下限は極端に小さい値だけを弾く。
MOUTH_BBOX_W_NORM_MIN = 0.12
MOUTH_BBOX_H_NORM_MIN = 0.02
MOUTH_NORM_Y_ABS_RANGE = (0.3, 0.9)   # 目の中点から口までの距離 / 目間距離


def _mouth_detection_warning(result: dict) -> Optional[str]:
    """③の検出結果が不自然なら理由の文字列を返す（正常なら None）。"""
    mouth = result.get("mouth") or {}
    bbox = mouth.get("bbox")
    eye_dist = float(result.get("eye_distance") or 0.0)
    if not bbox or len(bbox) != 4 or eye_dist <= 0:
        return None
    bw = float(bbox[2] - bbox[0])
    bh = float(bbox[3] - bbox[1])
    problems = []
    if bw / eye_dist < MOUTH_BBOX_W_NORM_MIN or bh / eye_dist < MOUTH_BBOX_H_NORM_MIN:
        problems.append(f"mouth bbox {bw:.0f}x{bh:.0f}px vs eye distance {eye_dist:.0f}px")
    norm_y = result.get("mouth_norm_y")
    if norm_y is not None:
        lo, hi = MOUTH_NORM_Y_ABS_RANGE
        if not (lo <= abs(float(norm_y)) <= hi):
            problems.append(f"mouth offset {abs(float(norm_y)):.2f} x eye distance")
    return ", ".join(problems) if problems else None

DEFAULT_PREP_PROMPT = "The character naturally opens and closes their mouth while speaking"

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _split_alpha(img: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """画像を (BGR, alpha) に分離する。

    編集・表示・検出はBGR 3chで行い、アルファは保存時に _merge_alpha で
    再合成して透過を維持する（グレースケール/3ch入力は alpha=None）。
    """
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), None
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR), img[:, :, 3].copy()
    return img, None


def _merge_alpha(bgr: np.ndarray, alpha: Optional[np.ndarray]) -> np.ndarray:
    """_split_alpha で分離したアルファを再合成する（alpha=Noneならそのまま）。"""
    if alpha is None:
        return bgr
    return np.dstack([bgr, alpha])

# --- ⑤ 口スプライト抽出 ---
INITIAL_CANDIDATES_PER_CATEGORY = 5
MAX_CANDIDATES_PER_CATEGORY = 20
THUMB_HEIGHT = 40  # サムネイルの高さ（アスペクト比維持）
PREVIEW_MAX_SIZE = 350  # 右側プレビューの最大サイズ

DEFAULT_FEATHER = 0
MAX_FEATHER = 50
DEFAULT_PADDING = 0.5

DEFAULT_OFFSET_X = 0
DEFAULT_OFFSET_Y = 0
DEFAULT_SCALE = 1.0
MAX_OFFSET = 100
MIN_SCALE = 0.5
# 出力キャンバスは「全候補の最大quad × 1.1」なので、これ以上拡大しても口の端が
# 切れるだけで実用にならない
MAX_SCALE = 1.5

DEFAULT_MASK_DILATE = 3
MAX_MASK_DILATE = 30

MOUTH_CATEGORIES = ["open", "closed", "half"]
CATEGORY_LABELS_SHORT = {
    "open": "Open",
    "closed": "Closed",
    "half": "Half",
}


# ---------------------------------------------------------------------------
# 設定（.batch_settings.json を video_generator と共有・マージ保存）
# ---------------------------------------------------------------------------

def _load_settings() -> dict:
    if os.path.isfile(SETTINGS_PATH):
        try:
            # utf-8-sig: 外部エディタで編集されBOMが付いたファイルも読めるように
            with open(SETTINGS_PATH, "r", encoding="utf-8-sig") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_settings(updates: dict):
    """既存の設定とマージして保存（他ツールのキーを消さない）。"""
    merged = _load_settings()
    merged.update(updates)
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)


def _migrate_legacy_config():
    """旧 .vidu_batch_config.json から初回のみ設定を取り込む。"""
    settings = _load_settings()
    if settings.get("_prep_migrated"):
        return
    legacy = {}
    if os.path.isfile(LEGACY_CONFIG_PATH):
        try:
            # 旧設定ファイルはBOM付きで保存されていたため utf-8-sig で読む
            with open(LEGACY_CONFIG_PATH, "r", encoding="utf-8-sig") as f:
                legacy = json.load(f)
        except Exception:
            legacy = {}
    updates: dict = {"_prep_migrated": True}
    if not settings.get("api_key") and legacy.get("api_key"):
        updates["api_key"] = legacy["api_key"]
    for src, dst in [("model", "prep_model"), ("duration", "prep_duration"),
                     ("resolution", "prep_resolution"), ("prompt", "prep_prompt")]:
        if dst not in settings and src in legacy:
            updates[dst] = legacy[src]
    if "work_folder" not in settings:
        folder = legacy.get("folder", "")
        if folder and os.path.isdir(folder):
            updates["work_folder"] = folder
    _save_settings(updates)


# ---------------------------------------------------------------------------
# 共通ユーティリティ
# ---------------------------------------------------------------------------

def _imread_jp(path: str):
    buf = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)


def _find_images(folder: str) -> list[str]:
    """フォルダ直下の画像ファイル一覧（非再帰）。"""
    result = []
    try:
        entries = sorted(os.listdir(folder))
    except OSError:
        return result
    for f in entries:
        full = os.path.join(folder, f)
        if os.path.isfile(full) and os.path.splitext(f)[1].lower() in IMAGE_EXTS:
            result.append(full)
    return result


def _find_videos(folder: str) -> list[str]:
    """フォルダ直下の動画ファイル一覧（非再帰）。"""
    result = []
    try:
        entries = sorted(os.listdir(folder))
    except OSError:
        return result
    for f in entries:
        full = os.path.join(folder, f)
        if os.path.isfile(full) and os.path.splitext(f)[1].lower() in VIDEO_EXTS:
            result.append(full)
    return result


def _list_char_folders(work_folder: str) -> list[str]:
    """ワークフォルダ直下のキャラクターフォルダ候補（除外セット適用）。"""
    result = []
    try:
        entries = sorted(os.listdir(work_folder))
    except OSError:
        return result
    for f in entries:
        full = os.path.join(work_folder, f)
        if not os.path.isdir(full):
            continue
        if f.startswith(".") or f.lower() in EXCLUDED_DIRS:
            continue
        result.append(full)
    return result


def _is_valid_char_name(name: str) -> bool:
    """キャラクター名（=画像ファイル名）が使えるか。判定本体は batch_engine.char_name_problem。"""
    return char_name_error(name) == ""


def _bgr_to_pil(bgr: np.ndarray, max_w: int = 500, max_h: int = 400) -> "Image.Image":
    """BGR画像をPIL Imageに変換（表示用にリサイズ）"""
    h, w = bgr.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def numpy_to_photoimage(
    img: np.ndarray,
    max_size: Optional[int] = None,
    target_height: Optional[int] = None,
) -> Optional["ImageTk.PhotoImage"]:
    """numpy配列をPhotoImageに変換"""
    if not _HAS_PIL:
        return None
    try:
        if len(img.shape) == 2:
            pil_img = Image.fromarray(img)
        elif img.shape[2] == 4:
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA))
        else:
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        w, h = pil_img.size

        if target_height is not None:
            new_h = target_height
            new_w = max(1, int(w * target_height / h))
            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        elif max_size is not None:
            scale = min(max_size / w, max_size / h, 1.0)
            if scale < 1.0:
                new_w, new_h = int(w * scale), int(h * scale)
                pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        return ImageTk.PhotoImage(pil_img)
    except Exception:
        return None


def composite_on_checkerboard(bgra: np.ndarray, checker_size: int = 8) -> np.ndarray:
    """チェッカーボード背景に合成"""
    h, w = bgra.shape[:2]
    checker = np.zeros((h, w, 3), dtype=np.uint8)

    for y in range(0, h, checker_size):
        for x in range(0, w, checker_size):
            if ((x // checker_size) + (y // checker_size)) % 2 == 0:
                checker[y:y+checker_size, x:x+checker_size] = [200, 200, 200]
            else:
                checker[y:y+checker_size, x:x+checker_size] = [255, 255, 255]

    alpha = bgra[:, :, 3:4].astype(np.float32) / 255.0
    rgb = bgra[:, :, :3].astype(np.float32)
    result = (rgb * alpha + checker.astype(np.float32) * (1 - alpha)).astype(np.uint8)

    return result


# ---------------------------------------------------------------------------
# 背景色置換
# ---------------------------------------------------------------------------

def replace_background(img: np.ndarray, colors: list[dict], target_hex: str) -> np.ndarray:
    tgt_h = target_hex.lstrip("#")
    if len(tgt_h) != 6:
        return img
    tgt_bgr = np.array(
        [int(tgt_h[4:6], 16), int(tgt_h[2:4], 16), int(tgt_h[0:2], 16)],
        dtype=np.uint8,
    )
    h, w = img.shape[:2]
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    for c in colors:
        s = c["color"].lstrip("#")
        if len(s) != 6:
            continue
        try:
            src_bgr = np.array(
                [int(s[4:6], 16), int(s[2:4], 16), int(s[0:2], 16)],
                dtype=np.float32,
            )
        except ValueError:
            continue
        tol = max(1, c.get("tolerance", 30))
        threshold = tol * 3.0
        diff = np.sum(np.abs(img.astype(np.float32) - src_bgr), axis=2)
        combined_mask = np.maximum(combined_mask, (diff < threshold).astype(np.uint8))
    result = img.copy()
    result[combined_mask == 1] = tgt_bgr
    return result


def _auto_detect_bg_color(img_bgr: np.ndarray) -> str:
    h, w = img_bgr.shape[:2]
    corners = [img_bgr[0:3, 0:3], img_bgr[0:3, w - 3:w]]
    avg = np.mean([c.mean(axis=(0, 1)) for c in corners], axis=0).astype(np.uint8)
    return f"#{avg[2]:02X}{avg[1]:02X}{avg[0]:02X}"


# ---------------------------------------------------------------------------
# ⓪ インストラクション本文
# ---------------------------------------------------------------------------

# 赤字にするフレーズ（INSTRUCTION_TEXT内に含めること）
INSTRUCTION_RED_PHRASE = tr("prep.instruction.red_phrase")

INSTRUCTION_TEXT = tr("prep.instruction.text")

# 言語コード → 表示名（各言語のネイティブ表記のため翻訳しない）
LANG_DISPLAY = {"ja": "日本語", "en": "English"}


# ---------------------------------------------------------------------------
# メインGUI
# ---------------------------------------------------------------------------

class BatchPreparerApp(TkinterDnD.Tk if _HAS_TK_DND else tk.Tk):
    """素材準備GUIアプリケーション"""

    def __init__(self):
        super().__init__()
        self.title("MotionPNGCreator for ArtificialGirlfriend - Asset Preparer / Step1 ("
                   + tr("prep.window.subtitle") + ")")
        self.geometry("1200x800")
        self.minsize(1000, 650)

        _migrate_legacy_config()
        self._settings = _load_settings()

        # 共有状態
        self._work_folder: str = ""
        wf = self._settings.get("work_folder", "")
        if wf and os.path.isdir(wf):
            self._work_folder = wf

        # GPU排他（③検出と⑤解析の同時実行を防ぐ）
        self._gpu_lock = threading.Lock()
        self._gpu_task: str = ""

        # SAM3検出器（③⑤で共有・遅延ロード）
        self._detector: Optional["SAM3MouthDetector"] = None

        # CUDA利用可否（import torch が重いため初回チェック結果をキャッシュ）
        self._cuda_ok: Optional[bool] = None

        # --- ④ Vidu用 ---
        self._t4_stop_event = threading.Event()
        self._t4_count_lock = threading.Lock()

        # --- ⑤ 抽出用状態 ---
        self.video_path: str = ""
        self.extractor: Optional[MouthSpriteExtractor] = None
        self.classified_frames: Dict[str, List[MouthFrameInfo]] = {}
        self.all_candidates: List[MouthFrameInfo] = []
        self.selected_frames: Dict[str, Optional[MouthFrameInfo]] = {
            cat: None for cat in MOUTH_CATEGORIES
        }
        self.preview_sprites: Dict[str, np.ndarray] = {}
        self.preview_images: Dict[str, "ImageTk.PhotoImage"] = {}
        self.unified_size: Optional[Tuple[int, int]] = None
        self.is_analyzing = False
        self.current_preview_mf: Optional[MouthFrameInfo] = None
        self._cached_cap: Optional[cv2.VideoCapture] = None
        self.log_queue: queue.Queue[str] = queue.Queue()
        # tkinterはメインスレッド以外からの操作が安全でないため、
        # ワーカーからの進捗/UI更新はキュー経由でメインスレッドが反映する
        self.progress_queue: queue.Queue[float] = queue.Queue()
        self.ui_queue: queue.Queue[Callable[[], None]] = queue.Queue()
        self._thumb_images: List["ImageTk.PhotoImage"] = []
        self._preview_photo: Optional["ImageTk.PhotoImage"] = None

        self._build_ui()
        self._t0_refresh_checklist()
        self._poll_logs()
        # torch/ultralyticsのロードは重いので、ウィンドウ表示後にバックグラウンドで行う
        self._start_ai_loading()

    # ================================================================
    # UI構築
    # ================================================================

    def _build_ui(self):
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)

        self.tab0_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.tab0_frame, text=tr("prep.tab.t0"))
        self._build_tab0(self.tab0_frame)

        self.tab1_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.tab1_frame, text=tr("prep.tab.t1"))
        self._build_tab1(self.tab1_frame)

        self.tab2_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.tab2_frame, text=tr("prep.tab.t2"))
        self._build_tab2(self.tab2_frame)

        self.tab3_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.tab3_frame, text=tr("prep.tab.t3"))
        self._build_tab3(self.tab3_frame)

        self.tab4_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.tab4_frame, text=tr("prep.tab.t4"))
        self._build_tab4(self.tab4_frame)

        self.tab51_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.tab51_frame, text=tr("prep.tab.t51"))
        self._build_tab51(self.tab51_frame)

        self.tab52_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.tab52_frame, text=tr("prep.tab.t52"))
        self._build_tab52(self.tab52_frame)

        self.tab6_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.tab6_frame, text=tr("prep.tab.t6"))
        self._build_tab6(self.tab6_frame)

        self.tab7_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.tab7_frame, text=tr("prep.tab.t7"))
        self._build_tab7(self.tab7_frame)

        # タブ切替時に実行するハンドラ（タブの追加・並べ替えの影響を受けないようフレーム基準）
        self._tab_change_handlers: Dict[tk.Widget, tuple] = {
            self.tab0_frame: (self._t0_refresh_checklist,),
            self.tab2_frame: (self._t2_refresh_images, self._t2_auto_show),
            self.tab3_frame: (self._t3_refresh_list,),
            self.tab4_frame: (self._t4_refresh_targets,),
            self.tab51_frame: (self._t5_refresh_chars,),
            self.tab52_frame: (self._on_update_preview,),
            self.tab6_frame: (self._t6_refresh_chars,),
            self.tab7_frame: (self._t7_rescan,),
        }

        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

    def _on_tab_changed(self, event=None):
        try:
            selected = self.nametowidget(self.notebook.select())
        except (tk.TclError, KeyError):
            return
        for handler in self._tab_change_handlers.get(selected, ()):
            handler()

    # ================================================================
    # 共有: ワークフォルダ / APIキー / GPU排他 / SAM3共有
    # ================================================================

    def _require_work_folder(self) -> Optional[str]:
        if self._work_folder and os.path.isdir(self._work_folder):
            return self._work_folder
        messagebox.showwarning(
            tr("prep.msgbox.title_warn"), tr("prep.msgbox.no_work_folder"))
        return None

    def _gpu_acquire(self, task_name: str) -> bool:
        with self._gpu_lock:
            if self._gpu_task:
                messagebox.showwarning(
                    tr("prep.msgbox.title_warn"),
                    tr("prep.msgbox.gpu_busy", task=self._gpu_task))
                return False
            self._gpu_task = task_name
            return True

    def _gpu_release(self):
        with self._gpu_lock:
            self._gpu_task = ""

    def _shared_detector(self) -> "SAM3MouthDetector":
        """③⑤で共有するSAM3検出器（遅延ロード）。"""
        if self._detector is None:
            self._detector = SAM3MouthDetector()
        return self._detector

    def _start_ai_loading(self):
        """torch/ultralyticsをバックグラウンドで読み込み、SAM3/CUDAの利用可否を判定する。

        完了するまで _SAM3_OK は None（判定中）のままで、③実行・⑤解析ボタンは
        無効。完了時にメインスレッドで _on_ai_loaded() が呼ばれUIへ反映される。
        """
        def _worker():
            print("Loading AI libraries (torch / ultralytics)... "
                  "this may take up to 30 seconds.")
            ok, err = is_sam3_available()
            cuda_ok = False
            if ok:
                try:
                    import torch
                    cuda_ok = torch.cuda.is_available()
                except Exception:
                    cuda_ok = False
            self.after(0, lambda: self._on_ai_loaded(ok, err, cuda_ok))

        threading.Thread(target=_worker, daemon=True, name="AI-Loader").start()

    def _on_ai_loaded(self, ok: bool, err: str, cuda_ok: bool):
        """AIライブラリ読み込み完了後のUI反映（メインスレッド）。"""
        global _SAM3_OK, _SAM3_ERR
        _SAM3_OK, _SAM3_ERR = ok, err
        self._cuda_ok = cuda_ok

        if ok:
            self.log(tr("prep.log.sam3_ok"))
            self._t3_btn.config(state="normal")
            self._t3_progress.config(text="", foreground="black")
            # 読み込み中に動画が選択されていた場合は⑤の解析ボタンを有効化
            if self.video_path and not self.is_analyzing:
                self.analyze_btn.configure(state=tk.NORMAL)
        else:
            self.log(tr("prep.log.sam3_unavailable", error=err))
            self._t3_progress.config(text="", foreground="black")
            self._t3_sam3_err_label.config(text=f"SAM3: {err}")

        self._t0_refresh_checklist()

    # ================================================================
    # ⓪ インストラクション
    # ================================================================

    def _build_tab0(self, parent):
        # ワークフォルダ
        wf_frame = ttk.LabelFrame(parent, text=tr("prep.tab0.work_folder_frame"), padding=5)
        wf_frame.pack(fill="x", padx=10, pady=(10, 5))
        row = ttk.Frame(wf_frame)
        row.pack(fill="x")
        ttk.Button(row, text=tr("prep.tab0.select_work_folder"),
                   command=self._t0_select_work_folder).pack(
            side="left", padx=2)
        self._var_work_folder = tk.StringVar(value=self._work_folder)
        ttk.Entry(row, textvariable=self._var_work_folder, state="readonly").pack(
            side="left", fill="x", expand=True, padx=2)

        # APIキー
        api_frame = ttk.LabelFrame(parent, text=tr("prep.tab0.api_frame"), padding=5)
        api_frame.pack(fill="x", padx=10, pady=5)
        row2 = ttk.Frame(api_frame)
        row2.pack(fill="x")
        ttk.Label(row2, text=tr("prep.tab0.api_key_label")).pack(side="left")
        self._var_api_key = tk.StringVar(value=self._settings.get("api_key", ""))
        ttk.Entry(row2, textvariable=self._var_api_key, width=55, show="*").pack(
            side="left", fill="x", expand=True, padx=2)
        ttk.Button(row2, text=tr("prep.tab0.save"), command=self._t0_save_api_key).pack(
            side="left", padx=2)

        # 言語設定
        lang_frame = ttk.LabelFrame(parent, text=tr("common.lang.frame"), padding=8)
        lang_frame.pack(fill="x", padx=10, pady=5)
        row_lang = ttk.Frame(lang_frame)
        row_lang.pack(fill="x")
        ttk.Label(row_lang, text=tr("common.lang.label")).pack(side="left")
        self._var_language = tk.StringVar(
            value=LANG_DISPLAY.get(current_language(), LANG_DISPLAY["en"]))
        lang_cb = ttk.Combobox(row_lang, textvariable=self._var_language,
                               values=list(LANG_DISPLAY.values()),
                               state="readonly", width=12)
        lang_cb.pack(side="left", padx=5)
        lang_cb.bind("<<ComboboxSelected>>", self._on_language_change)
        ttk.Label(row_lang, text=tr("common.lang.restart_hint")).pack(side="left", padx=8)

        # 下段: 左=チェックリスト / 右=使い方
        body = ttk.Frame(parent)
        body.pack(fill="both", expand=True, padx=10, pady=5)

        check_frame = ttk.LabelFrame(body, text=tr("prep.tab0.checklist_frame"), padding=8)
        check_frame.pack(side="left", fill="y", padx=(0, 5))

        # インストラクション「■ 事前準備」の説明順に対応させる
        self._t0_check_items = [
            ("uv", tr("prep.check.uv")),
            ("git", tr("prep.check.git")),
            ("ffmpeg", tr("prep.check.ffmpeg")),
            ("nodejs", tr("prep.check.nodejs")),
            ("sam3_model", tr("prep.check.sam3_model")),
            ("sam3_import", tr("prep.check.sam3_import")),
            ("cuda", tr("prep.check.cuda")),
            ("api_key", tr("prep.check.api_key")),
            ("work_folder", tr("prep.check.work_folder")),
            ("names", tr("prep.check.names")),
        ]
        self._t0_check_labels: Dict[str, ttk.Label] = {}
        for key, label in self._t0_check_items:
            row = ttk.Frame(check_frame)
            row.pack(fill="x", pady=2)
            mark = ttk.Label(row, text="－", width=3, anchor="center")
            mark.pack(side="left")
            ttk.Label(row, text=label).pack(side="left")
            self._t0_check_labels[key] = mark
        ttk.Button(check_frame, text=tr("prep.tab0.recheck"), command=self._t0_refresh_checklist).pack(
            fill="x", pady=(10, 0))
        self._t0_check_detail = ttk.Label(check_frame, text="", foreground="red",
                                          wraplength=220, justify="left")
        self._t0_check_detail.pack(fill="x", pady=(5, 0))

        text_frame = ttk.LabelFrame(body, text=tr("prep.tab0.usage_frame"), padding=5)
        text_frame.pack(side="left", fill="both", expand=True)
        txt = tk.Text(text_frame, wrap="word")
        sb = ttk.Scrollbar(text_frame, command=txt.yview)
        txt.config(yscrollcommand=sb.set)
        txt.insert("1.0", INSTRUCTION_TEXT)
        txt.tag_configure("red", foreground="red")
        idx = txt.search(INSTRUCTION_RED_PHRASE, "1.0")
        if idx:
            txt.tag_add("red", idx, f"{idx}+{len(INSTRUCTION_RED_PHRASE)}c")
        txt.config(state="disabled")
        txt.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

    def _t0_select_work_folder(self):
        folder = filedialog.askdirectory(title=tr("prep.tab0.select_work_folder_dialog"))
        if not folder:
            return
        if not path_ansi_safe(folder):
            messagebox.showwarning(tr("prep.msgbox.title_warn"),
                                   tr("common.warn.path_not_ansi"))
        self._work_folder = folder
        self._var_work_folder.set(folder)
        _save_settings({"work_folder": folder})
        self._t0_refresh_checklist()

    def _on_language_change(self, event=None):
        """言語変更時: 設定に保存し、再起動後に反映される旨を案内。"""
        display = self._var_language.get()
        code = next((c for c, d in LANG_DISPLAY.items() if d == display), "en")
        if code == current_language():
            return
        _save_settings({"language": code})
        messagebox.showinfo(tr("common.lang.changed_title"), tr("common.lang.restart_note"))

    def _t0_save_api_key(self):
        key = self._var_api_key.get().strip()
        _save_settings({"api_key": key})
        self._settings["api_key"] = key
        messagebox.showinfo(tr("prep.msgbox.title_save"), tr("prep.msgbox.api_saved"))
        self._t0_refresh_checklist()

    def _t0_refresh_checklist(self):
        detail_msgs = []

        def set_mark(key: str, ok: Optional[bool], warn_only: bool = False):
            lbl = self._t0_check_labels[key]
            if ok is None:  # 判定中（AIライブラリ読み込み待ち）
                lbl.config(text="…", foreground="gray")
            elif ok:
                lbl.config(text="✓", foreground="green")
            elif warn_only:
                lbl.config(text="△", foreground="#B8860B")
            else:
                lbl.config(text="✗", foreground="red")

        # ソフトウェア（インストーラーでも手動でも導入可）
        set_mark("uv", bool(shutil.which("uv"))
                 or os.path.isfile(os.path.join(
                     os.path.expanduser("~"), ".local", "bin", "uv.exe")))
        # git はvenv構築済みなら当面不要、Node.jsはプレイヤー起動時のみ必要 → 警告扱い
        set_mark("git", bool(shutil.which("git")), warn_only=True)
        set_mark("nodejs", bool(shutil.which("npx")), warn_only=True)

        set_mark("sam3_model", os.path.isfile(SAM3_MODEL_PATH))
        # SAM3/CUDAの判定はAIライブラリ読み込み完了後（_on_ai_loaded）に確定する。
        # それまでは「…」（判定中）を表示。torchのimportをここで行うと起動時に
        # メインスレッドが数十秒固まるため、判定はローダースレッドに任せる
        set_mark("sam3_import", _SAM3_OK)
        if _SAM3_OK is False:
            detail_msgs.append(f"SAM3: {_SAM3_ERR}")

        # CUDA必須（CPU動作は未確認・サポート対象外）
        set_mark("cuda", self._cuda_ok)

        set_mark("ffmpeg", bool(shutil.which("ffmpeg")))

        set_mark("api_key", bool(self._var_api_key.get().strip()))
        set_mark("work_folder", bool(self._work_folder and os.path.isdir(self._work_folder)))

        # 画像ファイル名チェック（ルート直下 + キャラフォルダ名）
        names_ok = True
        if self._work_folder and os.path.isdir(self._work_folder):
            bad = []
            for p in _find_images(self._work_folder):
                stem = os.path.splitext(os.path.basename(p))[0]
                err = char_name_error(stem)
                if err:
                    bad.append(f"{os.path.basename(p)}（{err}）")
            for d in _list_char_folders(self._work_folder):
                err = char_name_error(os.path.basename(d))
                if err:
                    bad.append(f"{os.path.basename(d)}/（{err}）")
            if bad:
                names_ok = False
                names_str = ", ".join(bad[:5]) + ("..." if len(bad) > 5 else "")
                detail_msgs.append(tr("prep.check.bad_names", names=names_str))
            set_mark("names", names_ok)
        else:
            self._t0_check_labels["names"].config(text="－", foreground="gray")

        notes = []
        if _SAM3_OK is None:
            notes.append(tr("prep.check.ai_loading"))
        self._t0_check_detail.config(
            text="\n".join(notes + detail_msgs),
            foreground="red" if detail_msgs else "gray")

    # ================================================================
    # ① 画像リスケール
    # ================================================================

    def _build_tab1(self, parent):
        note = ttk.Label(
            parent, foreground="#555",
            text=tr("prep.tab1.note", w=TARGET_W, h=TARGET_H))
        note.pack(anchor="w", padx=10, pady=(10, 0))

        row2 = ttk.Frame(parent, padding=5)
        row2.pack(fill="x", padx=5)
        self._t1_btn = ttk.Button(row2, text=tr("prep.tab1.execute"), command=self._t1_execute)
        self._t1_btn.pack(side="left", padx=2)
        self._t1_status = ttk.Label(row2, text="")
        self._t1_status.pack(side="left", padx=10)

        lf = ttk.LabelFrame(parent, text=tr("prep.common.log"), padding=5)
        lf.pack(fill="both", expand=True, padx=10, pady=5)
        self._t1_log = tk.Text(lf, height=10, state="disabled", wrap="word")
        sb = ttk.Scrollbar(lf, command=self._t1_log.yview)
        self._t1_log.config(yscrollcommand=sb.set)
        self._t1_log.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

    def _t1_logmsg(self, msg: str):
        def _do():
            self._t1_log.config(state="normal")
            self._t1_log.insert("end", msg + "\n")
            self._t1_log.see("end")
            self._t1_log.config(state="disabled")
        self.after(0, _do)

    def _t1_execute(self):
        folder = self._require_work_folder()
        if not folder:
            return
        images = _find_images(folder)
        if not images:
            messagebox.showwarning(
                tr("prep.msgbox.title_warn"), tr("prep.msgbox.t1_no_images"))
            return
        # 元画像を上書きするため、実行前に確認（事前コピーの保管を案内）
        if not messagebox.askyesno(
                tr("prep.msgbox.title_confirm"),
                tr("prep.msgbox.t1_confirm", n=len(images), w=TARGET_W, h=TARGET_H)):
            return
        self._t1_btn.config(state="disabled")
        threading.Thread(target=self._t1_run, args=(images,), daemon=True).start()

    def _t1_run(self, images: list[str]):
        self._t1_logmsg(tr("prep.log.t1_start", n=len(images)))
        done = 0
        for path in images:
            name = os.path.basename(path)
            try:
                img = _imread_jp(path)
                if img is None:
                    self._t1_logmsg(tr("prep.log.t1_skip_unreadable", name=name))
                    continue
                h, w = img.shape[:2]
                if w == TARGET_W and h == TARGET_H:
                    self._t1_logmsg(tr("prep.log.t1_skip_already",
                                       w=TARGET_W, h=TARGET_H, name=name))
                    continue
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                # 9:16 から大きく外れた画像は強制リサイズで歪むため警告
                target_aspect = TARGET_W / TARGET_H
                deviation = abs(w / h - target_aspect) / target_aspect
                if deviation > ASPECT_WARN_RATIO:
                    self._t1_logmsg(tr("prep.log.aspect_warn", name=name,
                                       w=w, h=h, pct=round(deviation * 100, 1)))
                scale_min = min(TARGET_W / w, TARGET_H / h)
                interp = cv2.INTER_AREA if scale_min < 1 else cv2.INTER_LANCZOS4
                resized = cv2.resize(img, (TARGET_W, TARGET_H), interpolation=interp)
                if imwrite_jp(path, resized):
                    self._t1_logmsg(tr("prep.log.t1_done_one", name=name,
                                       w=w, h=h, tw=TARGET_W, th=TARGET_H))
                    done += 1
                else:
                    self._t1_logmsg(tr("prep.log.t1_save_failed", name=name))
            except Exception as e:
                self._t1_logmsg(tr("prep.log.t1_error", name=name, error=e))
        self._t1_logmsg(tr("prep.log.t1_finish", done=done, total=len(images)))
        self.after(0, lambda: self._t1_status.config(
            text=tr("prep.tab1.status_done", done=done, total=len(images)),
            foreground="green"))
        self.after(0, lambda: self._t1_btn.config(state="normal"))

    # ================================================================
    # ② 背景色置換
    # ================================================================

    def _build_tab2(self, parent):
        note = ttk.Label(
            parent, foreground="#555",
            text=tr("prep.tab2.note"))
        note.pack(anchor="w", padx=10, pady=(10, 0))

        top = ttk.Frame(parent, padding=5)
        top.pack(fill="x")
        ttk.Label(top, text=tr("prep.tab2.image_label")).pack(side="left")
        self._t2_combo = ttk.Combobox(top, state="readonly", width=50)
        self._t2_combo.pack(side="left", fill="x", expand=True, padx=2)
        self._t2_combo.bind("<<ComboboxSelected>>", self._t2_on_image_selected)
        ttk.Button(top, text=tr("prep.common.refresh"),
                   command=self._t2_refresh_images).pack(side="left", padx=2)
        self._t2_image_map: Dict[str, str] = {}

        body = ttk.Frame(parent, padding=5)
        body.pack(fill="both", expand=True)

        # 左: プレビュー
        left = ttk.LabelFrame(body, text=tr("prep.tab2.preview_frame"), padding=5)
        left.pack(side="left", fill="both", expand=True)
        self._t2_canvas = tk.Canvas(left, bg="#2b2b2b", highlightthickness=0)
        self._t2_canvas.pack(fill="both", expand=True)
        self._t2_canvas.bind("<Button-1>", self._t2_on_preview_click)
        self._t2_canvas_image_id = None
        self._t2_preview_img = None
        self._t2_img_bgr = None
        self._t2_img_alpha = None  # 元画像のアルファ（保存時に再合成して透過を維持）
        self._t2_path = ""
        self._t2_preview_scale = 1.0
        self._t2_preview_offset = (0, 0)  # 画像描画オフセット (ox, oy)
        self._t2_preview_active = False  # プレビュー更新が一度でも押されたか

        # 右: 設定
        right = ttk.Frame(body, padding=5)
        right.pack(side="right", fill="y")

        cf = ttk.LabelFrame(right, text=tr("prep.tab2.detect_color_frame"), padding=5)
        cf.pack(fill="x", pady=5)
        ttk.Label(cf, text=tr("prep.tab2.color_label")).pack(side="left")
        self._t2_src_color = tk.StringVar(value="")
        self._t2_src_canvas = tk.Canvas(cf, width=20, height=20, bg="#808080", highlightthickness=1)
        self._t2_src_canvas.pack(side="left", padx=3)
        ttk.Entry(cf, textvariable=self._t2_src_color, width=9).pack(side="left", padx=3)
        self._t2_src_color.trace_add("write", lambda *_: self._t2_update_color_preview())
        ttk.Button(cf, text=tr("prep.tab2.auto_detect"),
                   command=self._t2_auto_detect).pack(side="left", padx=5)

        tf = ttk.Frame(right, padding=5)
        tf.pack(fill="x")
        ttk.Label(tf, text=tr("prep.tab2.tolerance")).pack(side="left")
        self._t2_tolerance = tk.IntVar(value=30)
        ttk.Scale(tf, from_=5, to=100, variable=self._t2_tolerance,
                  orient="horizontal", length=120).pack(side="left", padx=2)
        self._t2_tol_label = ttk.Label(tf, text="30", width=3)
        self._t2_tol_label.pack(side="left")
        self._t2_tolerance.trace_add("write", lambda *_: self._t2_on_tolerance_change())

        # 置換先は #00FF00 固定（Phase2 の背景透過がグリーンバック固定のため変更不可）
        rf = ttk.LabelFrame(right, text=tr("prep.tab2.target_color_frame"), padding=5)
        rf.pack(fill="x", pady=5)
        tk.Canvas(rf, width=20, height=20, bg=FIXED_BG_HEX,
                  highlightthickness=1).pack(side="left", padx=3)
        ttk.Label(rf, text=FIXED_BG_HEX).pack(side="left", padx=3)

        bf = ttk.Frame(right, padding=5)
        bf.pack(fill="x", pady=10)
        ttk.Button(bf, text=tr("prep.tab2.save"), command=self._t2_save).pack(fill="x", pady=2)

        self._t2_status = ttk.Label(right, text="")
        self._t2_status.pack(fill="x", pady=5)

    def _t2_refresh_images(self):
        """ワークフォルダ直下＋キャラフォルダ内の画像をドロップダウンに列挙。"""
        self._t2_image_map = {}
        values = []
        work = self._work_folder
        if work and os.path.isdir(work):
            for p in _find_images(work):
                disp = os.path.basename(p)
                self._t2_image_map[disp] = p
                values.append(disp)
            for d in _list_char_folders(work):
                for p in _find_images(d):
                    disp = f"{os.path.basename(d)}/{os.path.basename(p)}"
                    self._t2_image_map[disp] = p
                    values.append(disp)
        self._t2_combo.configure(values=values)
        current = self._t2_combo.get()
        if current not in values:
            self._t2_combo.set("")

    def _t2_auto_show(self):
        """タブ表示時: 画像未選択なら先頭を自動選択し、プレビュー更新を実行。"""
        values = list(self._t2_combo["values"])
        if not values:
            return
        if not self._t2_combo.get():
            self._t2_combo.set(values[0])
            self._t2_on_image_selected()
        if self._t2_img_bgr is not None:
            self._t2_update_preview()

    def _t2_on_image_selected(self, event=None):
        disp = self._t2_combo.get()
        p = self._t2_image_map.get(disp)
        if not p:
            return
        img = _imread_jp(p)
        if img is None:
            messagebox.showerror(tr("prep.msgbox.title_error"),
                                 tr("prep.msgbox.image_load_failed", path=disp))
            return
        img, alpha = _split_alpha(img)
        self._t2_path = p
        self._t2_img_bgr = img
        self._t2_img_alpha = alpha
        self._t2_preview_active = False
        self._t2_show_preview(img)
        self._t2_auto_detect()
        self._t2_status.config(text="")
        # 選択と同時に置換範囲プレビューまで表示
        self._t2_update_preview()

    def _t2_auto_detect(self):
        if self._t2_img_bgr is not None:
            hex_c = _auto_detect_bg_color(self._t2_img_bgr)
            self._t2_src_color.set(hex_c)

    def _t2_on_tolerance_change(self):
        self._t2_tol_label.config(text=str(self._t2_tolerance.get()))
        if self._t2_preview_active:
            self._t2_update_preview()

    def _t2_update_color_preview(self):
        hex_c = self._t2_src_color.get().strip()
        if len(hex_c) == 7 and hex_c.startswith("#"):
            try:
                self._t2_src_canvas.config(bg=hex_c)
            except tk.TclError:
                pass
            if self._t2_preview_active:
                self._t2_update_preview()

    def _t2_show_preview(self, bgr: np.ndarray):
        if not _HAS_PIL:
            return
        h, w = bgr.shape[:2]
        self._t2_canvas.update_idletasks()
        cw = max(self._t2_canvas.winfo_width(), 100)
        ch = max(self._t2_canvas.winfo_height(), 100)
        self._t2_preview_scale = min(cw / w, ch / h, 1.0)
        s = self._t2_preview_scale
        dw, dh = int(w * s), int(h * s)
        pil_img = _bgr_to_pil(bgr, cw, ch)
        self._t2_preview_img = ImageTk.PhotoImage(pil_img)
        ox = cw // 2
        oy = ch // 2
        self._t2_preview_offset = (ox - dw // 2, oy - dh // 2)
        if self._t2_canvas_image_id is not None:
            self._t2_canvas.delete(self._t2_canvas_image_id)
        self._t2_canvas_image_id = self._t2_canvas.create_image(
            ox, oy, image=self._t2_preview_img, anchor="center")

    def _t2_on_preview_click(self, event):
        if self._t2_img_bgr is None:
            return
        h, w = self._t2_img_bgr.shape[:2]
        s = self._t2_preview_scale
        ox, oy = self._t2_preview_offset
        x = int((event.x - ox) / s)
        y = int((event.y - oy) / s)
        if 0 <= x < w and 0 <= y < h:
            bgr = self._t2_img_bgr[y, x]
            self._t2_src_color.set(f"#{bgr[2]:02X}{bgr[1]:02X}{bgr[0]:02X}")

    def _t2_update_preview(self):
        if self._t2_img_bgr is None:
            return
        self._t2_preview_active = True
        src = self._t2_src_color.get().strip()
        tol = self._t2_tolerance.get()
        if not src or len(src) != 7:
            self._t2_show_preview(self._t2_img_bgr)
            return
        # 検出範囲（許容値内の画素）をマゼンタでハイライト表示。
        # 「置換で変化した画素」基準だと、既に置換済み（検出色=置換先色）の画像で
        # 何も表示されなくなるため、検出範囲そのものを描画する。
        s = src.lstrip("#")
        try:
            src_bgr = np.array(
                [int(s[4:6], 16), int(s[2:4], 16), int(s[0:2], 16)],
                dtype=np.float32,
            )
        except ValueError:
            self._t2_show_preview(self._t2_img_bgr)
            return
        threshold = max(1, tol) * 3.0
        diff = np.sum(np.abs(self._t2_img_bgr.astype(np.float32) - src_bgr), axis=2)
        detect_mask = diff < threshold
        display = self._t2_img_bgr.copy()
        display[detect_mask] = [255, 0, 255]  # マゼンタ (BGR)
        self._t2_show_preview(display)

    def _t2_save(self):
        if self._t2_img_bgr is None:
            messagebox.showwarning(tr("prep.msgbox.title_warn"),
                                   tr("prep.msgbox.t2_select_image"))
            return
        src = self._t2_src_color.get().strip()
        tol = self._t2_tolerance.get()
        if not src or len(src) != 7:
            messagebox.showwarning(tr("prep.msgbox.title_warn"),
                                   tr("prep.msgbox.t2_set_color"))
            return
        colors = [{"color": src, "tolerance": tol}]
        result = replace_background(self._t2_img_bgr, colors, FIXED_BG_HEX)
        out = _merge_alpha(result, self._t2_img_alpha)
        if imwrite_jp(self._t2_path, out):
            self._t2_img_bgr = result
            # 置換後は背景がグリーンになっているので、検出色を取り直して
            # ハイライト（マゼンタ）が消えないようにする
            self._t2_auto_detect()
            self._t2_update_preview()
            self._t2_status.config(
                text=tr("prep.tab2.saved_status", name=os.path.basename(self._t2_path)))
        else:
            messagebox.showerror(tr("prep.msgbox.title_error"), tr("prep.msgbox.save_failed"))

    # ================================================================
    # ③ 目・口検出 (SAM3) + キャラクターフォルダ化
    # ================================================================

    def _build_tab3(self, parent):
        note = ttk.Label(
            parent, foreground="#555",
            text=tr("prep.tab3.note"))
        note.pack(anchor="w", padx=10, pady=(10, 0))

        r2 = ttk.Frame(parent, padding=5)
        r2.pack(fill="x", padx=5)
        # AIライブラリ読み込み完了（_on_ai_loaded）まで実行不可
        self._t3_btn = ttk.Button(r2, text=tr("prep.tab3.execute"),
                                  command=self._t3_execute, state="disabled")
        self._t3_btn.pack(side="left", padx=2)
        ttk.Button(r2, text=tr("prep.tab3.refresh_list"),
                   command=self._t3_refresh_list).pack(side="left", padx=2)
        self._t3_bar = ttk.Progressbar(r2, length=160, mode="determinate")
        self._t3_bar.pack(side="left", padx=(10, 0))
        self._t3_progress = ttk.Label(r2, text=tr("prep.tab3.ai_loading"),
                                      foreground="gray")
        self._t3_progress.pack(side="left", padx=10)
        self._t3_sam3_err_label = ttk.Label(r2, text="", foreground="red")
        self._t3_sam3_err_label.pack(side="right")

        body = ttk.Frame(parent, padding=5)
        body.pack(fill="both", expand=True)

        # 左: 画像リスト
        left = ttk.Frame(body, width=240)
        left.pack(side="left", fill="y", padx=(0, 5))
        left.pack_propagate(False)
        ttk.Label(left, text=tr("prep.tab3.target_list")).pack()
        self._t3_listbox = tk.Listbox(left, exportselection=False)
        self._t3_listbox.pack(fill="both", expand=True)
        self._t3_listbox.bind("<<ListboxSelect>>", self._t3_on_select)
        # entries: {path, name, in_root, char_dir, invalid_reason, overwrite}
        self._t3_entries: list[dict] = []
        self._t3_results: dict[str, dict] = {}

        # 右: プレビュー
        right = ttk.LabelFrame(body, text=tr("prep.tab3.preview_frame"), padding=5)
        right.pack(side="left", fill="both", expand=True)
        self._t3_preview_label = ttk.Label(right, text=tr("prep.tab3.run_detection_hint"))
        self._t3_preview_label.pack(fill="both", expand=True)
        self._t3_preview_img = None

        # 下: ログ（他タブと同形式。検出の実行経過を表示）
        lf = ttk.LabelFrame(parent, text=tr("prep.common.log"), padding=5)
        lf.pack(fill="x", padx=5, pady=5)
        self._t3_log = tk.Text(lf, height=6, state="disabled", wrap="word")
        sb = ttk.Scrollbar(lf, command=self._t3_log.yview)
        self._t3_log.config(yscrollcommand=sb.set)
        self._t3_log.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

    def _t3_collect_entries(self) -> list[dict]:
        """検出対象を列挙: ルート直下の画像 + キャラフォルダ内の画像（再検出用）。"""
        entries: list[dict] = []
        work = self._work_folder
        if not work or not os.path.isdir(work):
            return entries
        for p in _find_images(work):
            name = os.path.splitext(os.path.basename(p))[0]
            reason = char_name_error(name)
            entries.append({"path": p, "name": name, "in_root": True,
                            "char_dir": os.path.join(work, name),
                            "invalid_reason": reason, "overwrite": False})
        for d in _list_char_folders(work):
            imgs = _find_images(d)
            name = os.path.basename(d)
            if len(imgs) == 0:
                continue
            # face_info.json が既にあるキャラは処理済みとしてスキップ。
            # ⑥で口消し済みの画像を再検出すると口が見つからず、正常な
            # face_info を劣化した結果で上書きしてしまうため。
            # 再検出したい場合は face_info.json を削除してから実行する。
            if (os.path.isfile(os.path.join(d, f"{name}_face_info.json"))
                    or os.path.isfile(os.path.join(d, "face_info.json"))):
                entries.append({"path": imgs[0], "name": name, "in_root": False,
                                "char_dir": d, "done": True,
                                "invalid_reason": tr("prep.tab3.reason_done"),
                                "overwrite": False})
                continue
            if len(imgs) > 1:
                entries.append({"path": imgs[0], "name": name, "in_root": False,
                                "char_dir": d,
                                "invalid_reason": tr("prep.tab3.reason_multi_images",
                                                     n=len(imgs)),
                                "overwrite": False})
                continue
            reason = char_name_error(name)
            entries.append({"path": imgs[0], "name": name, "in_root": False,
                            "char_dir": d, "invalid_reason": reason, "overwrite": False})
        return entries

    def _t3_entry_display(self, entry: dict) -> str:
        if entry["in_root"]:
            rel = os.path.basename(entry["path"])
        else:
            rel = f"{entry['name']}/{os.path.basename(entry['path'])}"
        if entry.get("done"):
            mark = "OK"  # 処理済み（face_info.jsonあり）: 実行対象から除外される
        elif entry["invalid_reason"]:
            mark = "NG"
        elif entry["path"] in self._t3_results:
            r = self._t3_results[entry["path"]]
            has_eyes = "left_eye" in r and "right_eye" in r
            has_mouth = "mouth" in r
            mark = "OK" if (has_eyes and has_mouth) else "!"
        else:
            mark = " "
        return f"[{mark}] {rel}"

    def _t3_refresh_list(self):
        self._t3_entries = self._t3_collect_entries()
        self._t3_listbox.delete(0, "end")
        for entry in self._t3_entries:
            self._t3_listbox.insert("end", self._t3_entry_display(entry))

    def _t3_execute(self):
        if not _SAM3_OK:
            # None（読み込み中）の間はボタン自体が無効なので通常到達しない
            if _SAM3_OK is not None:
                messagebox.showerror(tr("prep.msgbox.title_error"),
                                     tr("prep.msgbox.sam3_unavailable", error=_SAM3_ERR))
            return
        folder = self._require_work_folder()
        if not folder:
            return
        self._t3_refresh_list()
        entries = self._t3_entries
        valid = [e for e in entries if not e["invalid_reason"]]
        if not valid:
            messagebox.showwarning(tr("prep.msgbox.title_warn"),
                                   tr("prep.msgbox.t3_no_targets"))
            return

        # 移動先フォルダに既存画像がある場合は上書き確認（メインスレッドで一括）
        conflicts = []
        for e in valid:
            if e["in_root"] and os.path.isdir(e["char_dir"]) and _find_images(e["char_dir"]):
                conflicts.append(e)
        if conflicts:
            names = ", ".join(e["name"] for e in conflicts)
            if messagebox.askyesno(
                    tr("prep.msgbox.title_confirm"),
                    tr("prep.msgbox.t3_overwrite_confirm", names=names)):
                for e in conflicts:
                    e["overwrite"] = True
            else:
                for e in conflicts:
                    e["invalid_reason"] = tr("prep.tab3.reason_skip_existing")

        if not self._gpu_acquire(tr("prep.gpu.task_detect")):
            return
        self._t3_btn.config(state="disabled")
        threading.Thread(target=self._t3_run, args=(entries,), daemon=True).start()

    def _t3_run(self, entries: list[dict]):
        try:
            targets = [e for e in entries if not e["invalid_reason"]]
            total = len(targets)
            self._t3_logmsg(tr("prep.tab3.log_start", total=total))
            # SAM3モデルロード中は所要時間が読めないためマーキー表示
            self._t3_bar_marquee(True)
            self._t3_set_progress(tr("prep.log.t3_loading"))
            self._t3_logmsg(tr("prep.log.t3_loading"))
            try:
                detector = self._shared_detector()
            except Exception as e:
                self._t3_logmsg(tr("prep.msgbox.sam3_init_failed", error=e))
                self.after(0, lambda e=e: messagebox.showerror(
                    tr("prep.msgbox.title_error"),
                    tr("prep.msgbox.sam3_init_failed", error=e)))
                return
            finally:
                self._t3_bar_marquee(False)

            done = 0
            for i, entry in enumerate(targets):
                path = entry["path"]
                name = entry["name"]
                self._t3_bar_set(i, total)
                self._t3_set_progress(f"{i+1}/{total}: {name}")

                def on_part(part_key: str, i=i, name=name):
                    self._t3_set_progress(tr(
                        "prep.tab3.detecting", i=i + 1, total=total,
                        name=name, part=tr(part_key)))

                try:
                    result = self._t3_detect_one(detector, path, on_part)
                    if result is None:
                        self._t3_logmsg(tr("prep.tab3.log_unreadable", name=name))
                        continue
                    final_path = self._t3_save_result(entry, result)
                    self._t3_results[final_path] = result
                    done += 1
                    # 口の検出結果の妥当性（口の無い画像で検出した疑い等）
                    suspicious = _mouth_detection_warning(result)
                    if suspicious:
                        self._t3_logmsg(tr("prep.tab3.log_mouth_suspicious",
                                           name=name, detail=suspicious))
                    parts = " / ".join(
                        f"{tr(part_key)} {'✓' if result_key in result else '✗'}"
                        for part_key, result_key in (
                            ("prep.tab3.part_left_eye", "left_eye"),
                            ("prep.tab3.part_right_eye", "right_eye"),
                            ("prep.tab3.part_mouth", "mouth"),
                        ))
                    self._t3_logmsg(tr("prep.tab3.log_result", i=i + 1, total=total,
                                       name=name, parts=parts))
                except Exception as e:
                    self._t3_set_progress(tr("prep.log.t3_item_error", name=name, error=e))
                    self._t3_logmsg(tr("prep.log.t3_item_error", name=name, error=e))
                    traceback.print_exc()

            self._t3_bar_set(total, total)
            if total > 0 and done == total:
                self._t3_set_done(tr("prep.tab3.status_done", done=done, total=total))
                self._t3_logmsg(tr("prep.tab3.status_done", done=done, total=total))
            else:
                self._t3_set_progress(tr("prep.tab3.status_progress", done=done, total=total))
                self._t3_logmsg(tr("prep.tab3.status_progress", done=done, total=total))
        finally:
            self._gpu_release()
            self.after(0, lambda: self._t3_btn.config(state="normal"))
            self.after(0, self._t3_after_run_refresh)

    def _t3_detect_one(self, detector, path: str,
                       on_part: Optional[Callable[[str], None]] = None) -> Optional[dict]:
        """1枚の画像に対して目・口を検出し、face_info辞書を返す。

        on_part: 検出部位が変わるたびに呼ばれる進捗コールバック
                 （部位名のロケールキーを渡す）
        """
        img = _imread_jp(path)
        if img is None:
            return None
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        h, w = img.shape[:2]

        if on_part:
            on_part("prep.tab3.part_left_eye")
        left_eye = detector.detect_mouth(img, prompt="left eye")
        if on_part:
            on_part("prep.tab3.part_right_eye")
        right_eye = detector.detect_mouth(img, prompt="right eye")
        if on_part:
            on_part("prep.tab3.part_mouth")
        mouth = detector.detect_mouth(img, prompt="mouth")

        result = {
            "image": os.path.basename(path),
            "image_width": w,
            "image_height": h,
        }

        def _mask_centroid(mask):
            """マスク重心を計算（bbox中心より安定）"""
            coords = np.where(mask > 0)
            if len(coords[0]) == 0:
                return None
            return float(np.mean(coords[1])), float(np.mean(coords[0]))

        if left_eye:
            mask, bbox, center = left_eye
            centroid = _mask_centroid(mask)
            cx, cy = centroid if centroid else center
            result["left_eye"] = {"cx": cx, "cy": cy}
        if right_eye:
            mask, bbox, center = right_eye
            centroid = _mask_centroid(mask)
            cx, cy = centroid if centroid else center
            result["right_eye"] = {"cx": cx, "cy": cy}
        if mouth:
            mask, bbox, center = mouth
            centroid = _mask_centroid(mask)
            cx, cy = centroid if centroid else center
            result["mouth"] = {
                "cx": cx, "cy": cy,
                "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
            }

        # 目座標系での位置関係を計算
        if "left_eye" in result and "right_eye" in result:
            le = result["left_eye"]
            re_ = result["right_eye"]
            mid_x = (le["cx"] + re_["cx"]) / 2
            mid_y = (le["cy"] + re_["cy"]) / 2
            eye_dx = re_["cx"] - le["cx"]
            eye_dy = re_["cy"] - le["cy"]
            eye_dist = math.sqrt(eye_dx**2 + eye_dy**2)
            eye_angle = math.degrees(math.atan2(eye_dy, eye_dx))

            result["eye_midpoint"] = {"cx": mid_x, "cy": mid_y}
            result["eye_distance"] = eye_dist
            result["eye_angle_deg"] = eye_angle

            if "mouth" in result and eye_dist > 0:
                m = result["mouth"]
                dx = m["cx"] - mid_x
                dy = m["cy"] - mid_y
                # 目座標系の単位ベクトル
                # X軸: 左目→右目方向, Y軸: Xに直交(下向き)
                ux_x, ux_y = eye_dx / eye_dist, eye_dy / eye_dist
                uy_x, uy_y = -eye_dy / eye_dist, eye_dx / eye_dist
                # 正規化オフセット（目間距離を単位とする）
                result["mouth_norm_x"] = (dx * ux_x + dy * ux_y) / eye_dist
                result["mouth_norm_y"] = (dx * uy_x + dy * uy_y) / eye_dist
                # 口サイズの正規化
                b = m["bbox"]
                result["mouth_w_norm"] = (b[2] - b[0]) / eye_dist
                result["mouth_h_norm"] = (b[3] - b[1]) / eye_dist

        return result

    def _t3_save_result(self, entry: dict, result: dict) -> str:
        """face_info.json 保存 + ルート画像はキャラフォルダへ移動。最終画像パスを返す。"""
        name = entry["name"]
        if entry["in_root"]:
            src_path = entry["path"]
            dest_dir = entry["char_dir"]
            nfc = normalize_char_name(name)
            if nfc != name:
                # 分解形（NFD）のファイル名は合成形に揃えてからフォルダを作る
                ext = os.path.splitext(src_path)[1]
                renamed = os.path.join(os.path.dirname(src_path), nfc + ext)
                os.replace(src_path, renamed)
                src_path = renamed
                name = nfc
                dest_dir = os.path.join(os.path.dirname(dest_dir), nfc)
            os.makedirs(dest_dir, exist_ok=True)
            if entry["overwrite"]:
                for old in _find_images(dest_dir):
                    os.remove(old)
            new_path = os.path.join(dest_dir, os.path.basename(src_path))
            if os.path.abspath(new_path) != os.path.abspath(src_path):
                if os.path.isfile(new_path):
                    os.remove(new_path)
                shutil.move(src_path, new_path)
            json_path = os.path.join(dest_dir, f"{name}_face_info.json")
            final_path = new_path
        else:
            json_path = os.path.join(entry["char_dir"], f"{name}_face_info.json")
            final_path = entry["path"]

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return final_path

    def _t3_logmsg(self, msg: str):
        def _do():
            self._t3_log.config(state="normal")
            self._t3_log.insert("end", msg + "\n")
            self._t3_log.see("end")
            self._t3_log.config(state="disabled")
        self.after(0, _do)

    def _t3_set_progress(self, text: str):
        self.after(0, lambda: self._t3_progress.config(text=text, foreground="black"))

    def _t3_set_done(self, text: str):
        self.after(0, lambda: self._t3_progress.config(text=text, foreground="green"))

    def _t3_bar_marquee(self, on: bool):
        """進捗バーのマーキー（不定長）表示を切り替える（ワーカーから呼べる）。"""
        def _apply():
            if on:
                self._t3_bar.config(mode="indeterminate")
                self._t3_bar.start(12)
            else:
                self._t3_bar.stop()
                self._t3_bar.config(mode="determinate", value=0)
        self.after(0, _apply)

    def _t3_bar_set(self, value: float, maximum: float):
        """進捗バーを件数ベースで更新する（ワーカーから呼べる）。"""
        def _apply():
            self._t3_bar.stop()
            self._t3_bar.config(mode="determinate",
                                maximum=max(1, maximum), value=value)
        self.after(0, _apply)

    def _t3_after_run_refresh(self):
        self._t3_refresh_list()
        # 最初の検出済みアイテムを自動選択してプレビュー表示
        for i, entry in enumerate(self._t3_entries):
            if entry["path"] in self._t3_results:
                self._t3_listbox.selection_set(i)
                self._t3_show_result(entry["path"])
                break

    def _t3_on_select(self, event):
        sel = self._t3_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        if idx >= len(self._t3_entries):
            return
        self._t3_show_result(self._t3_entries[idx]["path"])

    def _t3_show_result(self, path: str):
        if not _HAS_PIL:
            return
        try:
            self._t3_show_result_impl(path)
        except Exception as e:
            err = traceback.format_exc()
            self._t3_logmsg(tr("prep.tab3.display_error", error=e, trace=err))

    def _t3_show_result_impl(self, path: str):
        img = _imread_jp(path)
        if img is None:
            return
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        result = self._t3_results.get(path, {})
        display = img.copy()

        def draw_cross(cx, cy, color, label):
            cx, cy = int(cx), int(cy)
            s = 15
            cv2.line(display, (cx - s, cy), (cx + s, cy), color, 2)
            cv2.line(display, (cx, cy - s), (cx, cy + s), color, 2)
            cv2.putText(display, label, (cx + s + 5, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        # 赤い線で目と口を結ぶ（三角形）
        pts = {}
        if "left_eye" in result:
            pts["le"] = (int(result["left_eye"]["cx"]), int(result["left_eye"]["cy"]))
        if "right_eye" in result:
            pts["re"] = (int(result["right_eye"]["cx"]), int(result["right_eye"]["cy"]))
        if "mouth" in result:
            pts["m"] = (int(result["mouth"]["cx"]), int(result["mouth"]["cy"]))

        red = (0, 0, 255)  # BGR
        if "le" in pts and "re" in pts:
            cv2.line(display, pts["le"], pts["re"], red, 2, cv2.LINE_AA)
        if "le" in pts and "m" in pts:
            cv2.line(display, pts["le"], pts["m"], red, 2, cv2.LINE_AA)
        if "re" in pts and "m" in pts:
            cv2.line(display, pts["re"], pts["m"], red, 2, cv2.LINE_AA)

        if "le" in pts:
            draw_cross(result["left_eye"]["cx"], result["left_eye"]["cy"], (255, 0, 0), "L-Eye")
        if "re" in pts:
            draw_cross(result["right_eye"]["cx"], result["right_eye"]["cy"], (255, 0, 0), "R-Eye")
        if "m" in pts:
            m = result["mouth"]
            draw_cross(m["cx"], m["cy"], (0, 255, 0), "Mouth")
            if "bbox" in m:
                b = m["bbox"]
                cv2.rectangle(display, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
        if "eye_midpoint" in result:
            em = result["eye_midpoint"]
            cv2.circle(display, (int(em["cx"]), int(em["cy"])), 5, (0, 255, 255), -1)

        pil_img = _bgr_to_pil(display, 550, 550)
        self._t3_preview_img = ImageTk.PhotoImage(pil_img)
        self._t3_preview_label.config(image=self._t3_preview_img, text="")

    # ================================================================
    # ④ Vidu動画生成（キャラクターフォルダ単位）
    # ================================================================

    def _build_tab4(self, parent):
        note = ttk.Label(
            parent, foreground="#555",
            text=tr("prep.tab4.note"))
        note.pack(anchor="w", padx=10, pady=(10, 0))

        r2 = ttk.Frame(parent, padding=5)
        r2.pack(fill="x", padx=5)
        ttk.Label(r2, text=tr("prep.tab4.model")).pack(side="left")
        self._t4_model = tk.StringVar(value=self._settings.get("prep_model", "viduq2-pro-fast"))
        cb_model = ttk.Combobox(r2, textvariable=self._t4_model, values=VIDU_MODELS,
                                state="readonly", width=18)
        cb_model.pack(side="left", padx=2)
        ttk.Label(r2, text=tr("prep.tab4.duration")).pack(side="left", padx=(15, 0))
        self._t4_duration = tk.IntVar(value=self._settings.get("prep_duration", 4))
        cb_duration = ttk.Combobox(r2, textvariable=self._t4_duration, values=list(range(1, 9)),
                                   state="readonly", width=4)
        cb_duration.pack(side="left", padx=2)
        ttk.Label(r2, text=tr("prep.tab4.resolution")).pack(side="left", padx=(15, 0))
        # 解像度は720p固定: ①のリスケール(720×1280)と量産動画(720p固定)に合わせる。
        # プレイヤーは口素材を実寸で描き、量産時の口消し範囲も口素材の実寸を
        # 基準にするため、準備動画と量産動画の縮尺が一致していなければならない
        self._t4_resolution = tk.StringVar(value="720p")
        cb_resolution = ttk.Combobox(r2, textvariable=self._t4_resolution,
                                     values=["720p"],
                                     state="readonly", width=6)
        cb_resolution.pack(side="left", padx=2)
        self._t4_skip_existing = tk.BooleanVar(value=True)
        ttk.Checkbutton(r2, text=tr("prep.tab4.skip_existing"),
                        variable=self._t4_skip_existing,
                        command=self._t4_update_estimate).pack(side="left", padx=(15, 0))
        # 設定変更のたびに画面上の見積を更新
        for cb in (cb_model, cb_duration, cb_resolution):
            cb.bind("<<ComboboxSelected>>", lambda e: self._t4_update_estimate())

        pf = ttk.LabelFrame(parent, text=tr("prep.tab4.prompt_frame"), padding=5)
        pf.pack(fill="x", padx=10, pady=2)
        self._t4_prompt = tk.Text(pf, height=3, wrap="word")
        self._t4_prompt.pack(fill="x")
        self._t4_prompt.insert("1.0", self._settings.get("prep_prompt", DEFAULT_PREP_PROMPT))

        r5 = ttk.Frame(parent, padding=5)
        r5.pack(fill="x", padx=5)
        self._t4_btn_start = ttk.Button(r5, text=tr("prep.tab4.start"), command=self._t4_start)
        self._t4_btn_start.pack(side="left", padx=2)
        self._t4_btn_stop = ttk.Button(r5, text=tr("prep.tab4.stop"),
                                       command=self._t4_stop, state="disabled")
        self._t4_btn_stop.pack(side="left", padx=2)
        self._t4_progress = ttk.Label(r5, text=tr("prep.tab4.waiting"))
        self._t4_progress.pack(side="left", padx=10)
        self._t4_target_label = ttk.Label(r5, text="", foreground="#555")
        self._t4_target_label.pack(side="right", padx=5)
        # 開始前でも費用感が分かるよう、見積を常時表示する
        self._t4_estimate_label = ttk.Label(r5, text="", foreground="#555")
        self._t4_estimate_label.pack(side="right", padx=5)

        lf = ttk.LabelFrame(parent, text=tr("prep.common.log"), padding=5)
        lf.pack(fill="both", expand=True, padx=10, pady=5)
        self._t4_log = tk.Text(lf, height=15, state="disabled", wrap="word")
        sb = ttk.Scrollbar(lf, command=self._t4_log.yview)
        self._t4_log.config(yscrollcommand=sb.set)
        self._t4_log.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

    def _t4_collect_targets(self) -> list[dict]:
        """画像1枚を持つキャラクターフォルダを列挙。"""
        targets = []
        work = self._work_folder
        if not work or not os.path.isdir(work):
            return targets
        for d in _list_char_folders(work):
            imgs = _find_images(d)
            if len(imgs) != 1:
                continue
            name = os.path.basename(d)
            video = os.path.join(d, f"{name}.mp4")
            targets.append({
                "name": name, "folder": d, "image": imgs[0],
                "video": video, "has_video": os.path.isfile(video),
            })
        return targets

    def _t4_refresh_targets(self):
        targets = self._t4_collect_targets()
        n_done = sum(1 for t in targets if t["has_video"])
        self._t4_target_label.config(
            text=tr("prep.tab4.targets", n=len(targets), done=n_done))
        self._t4_update_estimate(targets)

    def _t4_update_estimate(self, targets: Optional[list] = None):
        """画面上の見積表示を更新する（開始ボタンとは独立・課金なし）。"""
        if targets is None:
            targets = self._t4_collect_targets()
        if self._t4_skip_existing.get():
            n = sum(1 for t in targets if not t["has_video"])
        else:
            n = len(targets)
        try:
            per_job = calc_credits(self._t4_model.get(), self._t4_resolution.get(),
                                   self._t4_duration.get())
        except Exception:
            per_job = 0
        if per_job <= 0:
            self._t4_estimate_label.config(
                text=tr("prep.tab4.estimate_unknown"), foreground="#B00000")
            return
        total = per_job * n
        self._t4_estimate_label.config(
            text=tr("prep.tab4.estimate", n=n, per=per_job, credits=total,
                    cost=f"{total * 0.005:.2f}"),
            foreground="#555")

    def _t4_logmsg(self, msg: str):
        def _do():
            self._t4_log.config(state="normal")
            self._t4_log.insert("end", msg + "\n")
            self._t4_log.see("end")
            self._t4_log.config(state="disabled")
        self.after(0, _do)

    def _t4_set_progress(self, text: str):
        self.after(0, lambda: self._t4_progress.config(text=text, foreground="black"))

    def _t4_set_done(self, text: str):
        self.after(0, lambda: self._t4_progress.config(text=text, foreground="green"))

    def _t4_save_settings(self):
        _save_settings({
            "api_key": self._var_api_key.get().strip(),
            "prep_model": self._t4_model.get(),
            "prep_duration": self._t4_duration.get(),
            "prep_resolution": self._t4_resolution.get(),
            "prep_prompt": self._t4_prompt.get("1.0", "end-1c").strip(),
        })

    def _t4_start(self):
        api_key = self._var_api_key.get().strip()
        if not api_key:
            messagebox.showwarning(
                tr("prep.msgbox.title_warn"), tr("prep.msgbox.t4_no_api_key"))
            return
        folder = self._require_work_folder()
        if not folder:
            return
        targets = self._t4_collect_targets()
        if not targets:
            messagebox.showwarning(
                tr("prep.msgbox.title_warn"), tr("prep.msgbox.t4_no_targets"))
            return
        if self._t4_skip_existing.get():
            skipped = [t for t in targets if t["has_video"]]
            targets = [t for t in targets if not t["has_video"]]
            if skipped:
                self._t4_logmsg(tr("prep.log.t4_skip_existing",
                                   names=", ".join(t['name'] for t in skipped)))
            if not targets:
                messagebox.showinfo(tr("prep.msgbox.title_info"),
                                    tr("prep.msgbox.t4_all_done"))
                self._t4_set_done(tr("prep.tab4.status_all_done"))
                return
        else:
            existing = [t for t in targets if t["has_video"]]
            if existing:
                if not messagebox.askyesno(
                        tr("prep.msgbox.title_confirm"),
                        tr("prep.msgbox.t4_overwrite_confirm", n=len(existing))):
                    return

        model = self._t4_model.get()
        prompt = self._t4_prompt.get("1.0", "end-1c").strip()
        duration = self._t4_duration.get()
        resolution = self._t4_resolution.get()

        # 課金前の見積・確認。料金を算出できない組み合わせは開始しない
        per_job = calc_credits(model, resolution, duration)
        if per_job <= 0:
            messagebox.showerror(tr("prep.msgbox.title_error"),
                                 tr("prep.msgbox.price_unknown"))
            return
        total_credits = per_job * len(targets)
        if not messagebox.askyesno(
                tr("prep.msgbox.title_confirm"),
                tr("prep.msgbox.t4_cost_confirm",
                   n=len(targets), model=model, duration=duration,
                   resolution=resolution, credits=total_credits,
                   cost=f"{total_credits * 0.005:.2f}")):
            return

        self._t4_save_settings()
        self._t4_stop_event.clear()
        self._t4_btn_start.config(state="disabled")
        self._t4_btn_stop.config(state="normal")
        threading.Thread(target=self._t4_run_batch,
                         args=(api_key, targets, model, prompt, duration, resolution),
                         daemon=True).start()

    def _t4_stop(self):
        self._t4_stop_event.set()
        self._t4_logmsg(tr("prep.log.t4_stop_req"))
        self._t4_btn_stop.config(state="disabled")

    def _t4_on_done(self):
        self.after(0, lambda: self._t4_btn_start.config(state="normal"))
        self.after(0, lambda: self._t4_btn_stop.config(state="disabled"))
        self.after(0, self._t4_refresh_targets)

    def _t4_run_batch(self, api_key, targets, model, prompt, duration, resolution):
        total = len(targets)
        self._t4_logmsg(tr("prep.log.t4_batch_start", n=total, model=model))
        max_concurrent = 5
        try:
            info = vidu_get(api_key, "credits")
            r = info.get("remains", [{}])[0]
            credit = r.get("credit_remain", "?")
            max_concurrent = max(1, int(r.get("concurrency_limit", 5) or 1))
            self._t4_logmsg(tr("prep.log.t4_credits", credit=credit, limit=max_concurrent))
        except Exception as e:
            self._t4_logmsg(tr("prep.log.t4_credit_error", error=e))

        completed = 0
        failed = 0

        def process_one(target: dict):
            nonlocal completed, failed
            if self._t4_stop_event.is_set():
                return
            name = target["name"]
            try:
                b64 = image_to_base64_uri(target["image"])
                payload = {"model": model, "images": [b64, b64],
                           "prompt": prompt, "duration": duration, "resolution": resolution}
                task_id = None
                for attempt in range(4):
                    try:
                        result = vidu_post(api_key, "start-end2video", payload)
                        task_id = result.get("task_id")
                        break
                    except ViduAPIError as ve:
                        is_rate_limited = (ve.status_code == 429
                                           or ve.err_code == "TooManyRequests")
                        if is_rate_limited and attempt < 3:
                            time.sleep(10 * (attempt + 1))
                        else:
                            raise
                if not task_id:
                    with self._t4_count_lock:
                        failed += 1
                    self._t4_logmsg(tr("prep.log.t4_fail_no_taskid", name=name))
                    return
                self._t4_logmsg(tr("prep.log.t4_submitted", name=name, task=task_id[:20]))
                poll_failures = 0
                poll_start = time.monotonic()
                while not self._t4_stop_event.is_set():
                    time.sleep(POLL_INTERVAL)
                    # success/failed 以外のまま固まるタスクへの保険（無限ポーリング防止）
                    if time.monotonic() - poll_start > MAX_POLL_SECONDS:
                        with self._t4_count_lock:
                            failed += 1
                        self._t4_logmsg(tr("prep.log.t4_fail_timeout", name=name,
                                           minutes=MAX_POLL_SECONDS // 60))
                        self._t4_set_progress(tr("prep.tab4.status_progress",
                                                 done=completed, total=total, failed=failed))
                        return
                    try:
                        status = vidu_get(api_key, f"tasks/{task_id}/creations")
                        poll_failures = 0
                    except Exception as pe:
                        poll_failures += 1
                        self._t4_logmsg(tr("prep.log.t4_poll_fail", n=poll_failures,
                                           max=MAX_POLL_FAILURES, name=name, error=pe))
                        if poll_failures >= MAX_POLL_FAILURES:
                            with self._t4_count_lock:
                                failed += 1
                            self._t4_logmsg(tr("prep.log.t4_fail_poll_abort", name=name))
                            self._t4_set_progress(tr("prep.tab4.status_progress",
                                                     done=completed, total=total, failed=failed))
                            return
                        continue
                    state = status.get("state", "")
                    if state == "success":
                        creations = status.get("creations", [])
                        if creations:
                            vidu_download(creations[0].get("url", ""), target["video"])
                            with self._t4_count_lock:
                                completed += 1
                            self._t4_logmsg(tr("prep.log.t4_done_one", name=name,
                                               file=os.path.basename(target['video'])))
                        else:
                            with self._t4_count_lock:
                                failed += 1
                            self._t4_logmsg(tr("prep.log.t4_fail_empty", name=name))
                        self._t4_set_progress(tr("prep.tab4.status_progress",
                                                 done=completed, total=total, failed=failed))
                        return
                    elif state == "failed":
                        with self._t4_count_lock:
                            failed += 1
                        self._t4_logmsg(tr("prep.log.t4_fail_state", name=name,
                                           code=status.get('err_code')))
                        self._t4_set_progress(tr("prep.tab4.status_progress",
                                                 done=completed, total=total, failed=failed))
                        return
            except Exception as e:
                with self._t4_count_lock:
                    failed += 1
                self._t4_logmsg(tr("prep.log.t4_error", name=name, error=e))
            self._t4_set_progress(tr("prep.tab4.status_progress",
                                     done=completed, total=total, failed=failed))

        try:
            with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
                futures = [executor.submit(process_one, t) for t in targets
                           if not self._t4_stop_event.is_set()]
                for fut in futures:
                    try:
                        fut.result()
                    except Exception:
                        pass
        except Exception as e:
            self._t4_logmsg(tr("prep.log.t4_batch_error", error=e))
        self._t4_logmsg(tr("prep.log.t4_batch_finish",
                           done=completed, total=total, failed=failed))
        if failed == 0 and completed == total and total > 0:
            self._t4_set_done(tr("prep.tab4.status_done", done=completed, total=total))
        else:
            self._t4_set_progress(tr("prep.tab4.status_progress",
                                     done=completed, total=total, failed=failed))
        self._t4_on_done()

    # ================================================================
    # ⑤-1 抽出
    # ================================================================

    def _build_tab51(self, parent):
        note = ttk.Label(
            parent, foreground="#555",
            text=tr("prep.tab51.note"))
        note.pack(anchor="w", padx=10, pady=(10, 0))

        frame = ttk.Frame(parent, padding=5)
        frame.pack(fill="both", expand=True)

        # 上部: キャラクター選択 & 解析
        top_frame = ttk.Frame(frame)
        top_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(top_frame, text=tr("prep.common.char_label")).pack(side=tk.LEFT)
        self._t5_char_combo = ttk.Combobox(top_frame, state="readonly", width=16)
        self._t5_char_combo.pack(side=tk.LEFT, padx=2)
        self._t5_char_combo.bind("<<ComboboxSelected>>", self._t5_on_char_selected)

        ttk.Button(top_frame, text=tr("prep.common.refresh"),
                   command=self._t5_refresh_chars).pack(side=tk.LEFT, padx=2)

        # 口周りの余白（検出した口の周囲をどれだけ広く切り出すか）
        ttk.Label(top_frame, text=tr("prep.tab51.padding")).pack(side=tk.LEFT)
        self.padding_var = tk.DoubleVar(value=DEFAULT_PADDING)
        self.padding_slider = ttk.Scale(
            top_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL,
            variable=self.padding_var, length=100
        )
        self.padding_slider.pack(side=tk.LEFT, padx=2)
        self.padding_label = ttk.Label(top_frame, text="30%", width=4)
        self.padding_label.pack(side=tk.LEFT)

        # 解析ボタン
        self.analyze_btn = ttk.Button(
            top_frame, text=tr("prep.tab51.analyze"), command=self._on_analyze, state=tk.DISABLED
        )
        self.analyze_btn.pack(side=tk.LEFT, padx=10)

        # プログレスバー
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            top_frame, variable=self.progress_var, maximum=100, length=150
        )
        self.progress_bar.pack(side=tk.LEFT, padx=5)

        # ⑤-2へ進むボタン（進捗バーの右横）
        self.goto_step3_btn = ttk.Button(
            top_frame, text=tr("prep.tab51.goto52"),
            command=self._goto_step52, state=tk.DISABLED,
        )
        self.goto_step3_btn.pack(side=tk.LEFT, padx=5)

        # Drag & drop（⑤のタブにのみ登録）
        if _HAS_TK_DND:
            frame.drop_target_register(DND_FILES)
            frame.dnd_bind("<<Drop>>", self._on_drop)

        # 左右分割メインエリア
        main_paned = ttk.PanedWindow(frame, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True)

        # 左側: 候補一覧 + ログ（上下分割）
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=1)

        left_paned = ttk.PanedWindow(left_frame, orient=tk.VERTICAL)
        left_paned.pack(fill=tk.BOTH, expand=True)

        candidates_frame = ttk.LabelFrame(left_paned, text=tr("prep.tab51.candidates_frame"),
                                          padding=5)
        left_paned.add(candidates_frame, weight=1)

        self.candidates_canvas = tk.Canvas(candidates_frame, width=450)
        candidates_scrollbar = ttk.Scrollbar(
            candidates_frame, orient=tk.VERTICAL, command=self.candidates_canvas.yview
        )
        self.candidates_inner = ttk.Frame(self.candidates_canvas)

        self.candidates_canvas.configure(yscrollcommand=candidates_scrollbar.set)
        candidates_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.candidates_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.candidates_canvas.create_window(
            (0, 0), window=self.candidates_inner, anchor=tk.NW
        )
        self.candidates_inner.bind("<Configure>", lambda e: self.candidates_canvas.configure(
            scrollregion=self.candidates_canvas.bbox("all")
        ))

        # 左下: ログ
        log_frame = ttk.LabelFrame(left_paned, text=tr("prep.common.log"), padding=3)
        left_paned.add(log_frame, weight=1)

        self.log_text = tk.Text(log_frame, height=6, state=tk.DISABLED, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # カテゴリごとのUI
        self.category_frames: Dict[str, ttk.Frame] = {}
        self.candidate_buttons: Dict[str, List[ttk.Button]] = {}
        self.more_buttons: Dict[str, ttk.Button] = {}

        for cat in MOUTH_CATEGORIES:
            self._build_category_row(cat)

        # 右側: プレビュー & 選択状況
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=1)

        preview_panel = ttk.LabelFrame(right_frame, text=tr("prep.tab51.preview_frame"),
                                       padding=5)
        preview_panel.pack(fill=tk.BOTH, expand=True)

        self.preview_label = ttk.Label(preview_panel, text=tr("prep.tab51.click_candidate"))
        self.preview_label.pack(pady=10)

        self.preview_info_label = ttk.Label(preview_panel, text="", foreground="gray")
        self.preview_info_label.pack()

        # カテゴリ選択ボタン
        self.cat_btn_frame = ttk.Frame(preview_panel)
        self.cat_btn_frame.pack(pady=10)

        ttk.Label(self.cat_btn_frame, text=tr("prep.tab51.assign_to")).pack(side=tk.LEFT, padx=(0, 5))

        self.assign_buttons: Dict[str, ttk.Button] = {}
        for cat in MOUTH_CATEGORIES:
            btn = ttk.Button(
                self.cat_btn_frame, text=cat.upper(), width=8,
                command=lambda c=cat: self._assign_current_to_category(c),
                state=tk.DISABLED
            )
            btn.pack(side=tk.LEFT, padx=2)
            self.assign_buttons[cat] = btn

        # 選択状況
        status_frame = ttk.LabelFrame(right_frame, text=tr("prep.tab51.selection_frame"),
                                      padding=5)
        status_frame.pack(fill=tk.X, pady=(5, 0))

        self.status_labels: Dict[str, ttk.Label] = {}
        for cat in MOUTH_CATEGORIES:
            row = ttk.Frame(status_frame)
            row.pack(fill=tk.X, pady=1)

            ttk.Label(row, text=tr("prep.tab51.required", cat=cat.upper()),
                      width=15).pack(side=tk.LEFT)

            status_lbl = ttk.Label(row, text=tr("prep.tab51.not_selected"), foreground="gray")
            status_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.status_labels[cat] = status_lbl

    def _build_category_row(self, cat: str):
        """カテゴリ行を構築"""
        cat_frame = ttk.Frame(self.candidates_inner)
        cat_frame.pack(fill=tk.X, pady=3)
        self.category_frames[cat] = cat_frame

        label_text = f"▼ {CATEGORY_LABELS_SHORT[cat]}"
        ttk.Label(cat_frame, text=label_text, font=("", 9, "bold")).pack(anchor=tk.W)

        thumb_frame = ttk.Frame(cat_frame)
        thumb_frame.pack(fill=tk.X)

        self.candidate_buttons[cat] = []

        more_btn = ttk.Button(
            thumb_frame, text="+", width=3,
            command=lambda c=cat: self._load_more_candidates(c),
            state=tk.DISABLED
        )
        more_btn.pack(side=tk.RIGHT, padx=2)
        self.more_buttons[cat] = more_btn

    # ---- キャラ/動画選択 ----

    def _t5_refresh_chars(self):
        """動画を持つキャラクターフォルダをドロップダウンに列挙。"""
        values = []
        work = self._work_folder
        if work and os.path.isdir(work):
            for d in _list_char_folders(work):
                if _find_videos(d):
                    values.append(os.path.basename(d))
        self._t5_char_combo.configure(values=values)
        if self._t5_char_combo.get() not in values:
            self._t5_char_combo.set("")

    def _t5_on_char_selected(self, event=None):
        """キャラ選択で動画を自動決定（{キャラ名}.mp4 を優先）。"""
        char = self._t5_char_combo.get()
        if not char or not self._work_folder:
            return
        char_dir = os.path.join(self._work_folder, char)
        videos = _find_videos(char_dir)
        if not videos:
            return
        default = os.path.join(char_dir, f"{char}.mp4")
        pick = default if os.path.isfile(default) else videos[0]
        self._set_video(pick)

    def _on_drop(self, event):
        """ドラッグ&ドロップ（ワークフォルダ外の動画も直接指定可能）"""
        path = event.data.strip()
        if path.startswith("{") and path.endswith("}"):
            path = path[1:-1]
        if os.path.isfile(path):
            self._set_video(path)

    def _set_video(self, path: str):
        """動画を設定。動画が変わった場合は前の動画の解析状態を全リセットする。"""
        # 解析中の差し替えは旧動画の解析結果と新動画のフレームが混ざるため拒否
        if self.is_analyzing:
            messagebox.showwarning(tr("prep.msgbox.title_warn"),
                                   tr("prep.msgbox.t5_analyzing"))
            return
        # 同じ動画の再選択では解析済みの候補・割り当てを保持する
        if path == self.video_path:
            return

        self.video_path = path
        # 前の動画（別キャラクター）の候補一覧・割り当てが残ったまま⑤-2へ
        # 進めると誤ったスプライトを出力できてしまうため、ここで必ずリセット
        self._reset_extract_state()
        if _SAM3_OK:
            self.analyze_btn.configure(state=tk.NORMAL)
        self.log(tr("prep.log.t5_video_selected", path=path))

    # ---- 解析 ----

    def _reset_extract_state(self):
        """⑤の状態をリセット"""
        self.extractor = None
        self.classified_frames = {}
        self.all_candidates = []
        self.selected_frames = {cat: None for cat in MOUTH_CATEGORIES}
        self.preview_sprites = {}
        self.preview_images = {}
        self.unified_size = None
        self._thumb_images = []
        self.current_preview_mf = None

        if self._cached_cap is not None:
            self._cached_cap.release()
            self._cached_cap = None

        self._clear_candidates_ui()
        self._clear_preview_panel()
        self._update_selection_status()
        self.goto_step3_btn.configure(state=tk.DISABLED)
        # ⑤-2の出力プレビューも無効になるので、PNG出力を押せない状態に戻す
        self.output_btn.configure(state=tk.DISABLED)

    def _on_analyze(self):
        """解析開始"""
        if not self.video_path or self.is_analyzing:
            return
        if not self._gpu_acquire(tr("prep.gpu.task_analyze")):
            return

        self._reset_extract_state()
        self.is_analyzing = True
        self.analyze_btn.configure(state=tk.DISABLED)
        self.progress_var.set(0)

        # tk変数はメインスレッドで読み取り、ワーカーへは引数で渡す
        padding_ratio = self.padding_var.get()
        thread = threading.Thread(target=self._analyze_worker,
                                  args=(padding_ratio,), daemon=True)
        thread.start()

    def _analyze_worker(self, padding_ratio: float):
        """解析ワーカースレッド"""
        try:
            self.log(tr("prep.log.t5_analyze_start"))

            self.extractor = MouthSpriteExtractor(
                self.video_path,
                padding_ratio=padding_ratio,
                model_path="",  # 空の場合はモジュール基準のデフォルト (Sam3/sam3.pt)
            )
            # ③と共有するSAM3検出器を注入（二重ロード防止）
            try:
                self.extractor._detector = self._shared_detector()
            except Exception:
                pass  # 失敗時はextractor側の遅延ロードに任せる

            def on_progress(current: int, total: int):
                if total > 0:
                    self.progress_queue.put(current / total * 100.0)

            self.extractor.analyze(callback=self.log, progress=on_progress)

            if len(self.extractor.mouth_frames) == 0:
                self.log(tr("prep.log.t5_no_mouth"))
                return

            # 多めに分類
            self.classified_frames = classify_mouth_frames(
                self.extractor.mouth_frames,
                self.extractor.cluster_mask,
                candidates_per_category=MAX_CANDIDATES_PER_CATEGORY,
            )

            # 全候補リスト
            self.all_candidates = []
            seen_frames = set()
            for frames in self.classified_frames.values():
                for mf in frames:
                    if mf.frame_idx not in seen_frames:
                        self.all_candidates.append(mf)
                        seen_frames.add(mf.frame_idx)

            # 候補に選ばれなかったフレームのSAM3マスクを解放（メモリ節約）
            for mf in self.extractor.mouth_frames:
                if mf.frame_idx not in seen_frames:
                    mf.mask = None

            self.log(tr("prep.log.t5_classified", n=len(self.all_candidates)))

            # 統一サイズ計算（quadサイズ = padding適用済みを使用）
            if self.all_candidates:
                max_qw = 0.0
                max_qh = 0.0
                for mf in self.all_candidates:
                    q = mf.quad
                    qw = float(np.linalg.norm(q[1] - q[0]))
                    qh = float(np.linalg.norm(q[3] - q[0]))
                    max_qw = max(max_qw, qw)
                    max_qh = max(max_qh, qh)
                self.unified_size = (
                    ensure_even_ge2(max(32, int(max_qw * 1.1))),
                    ensure_even_ge2(max(32, int(max_qh * 1.1))),
                )
                self.log(tr("prep.log.t5_unified_size", size=self.unified_size))

            # UIを更新（メインスレッドで実行）
            self.ui_queue.put(self._populate_candidates)

            self.progress_queue.put(100.0)
            self.log(tr("prep.log.t5_analyze_done"))

        except Exception as e:
            self.log(tr("prep.log.t5_error", error=e))
            traceback.print_exc()
        finally:
            self.is_analyzing = False
            self._gpu_release()
            self.ui_queue.put(
                lambda: self.analyze_btn.configure(state=tk.NORMAL)
            )

    def _clear_candidates_ui(self):
        """候補UIをクリア"""
        for cat in MOUTH_CATEGORIES:
            for btn in self.candidate_buttons[cat]:
                btn.destroy()
            self.candidate_buttons[cat] = []
            self.more_buttons[cat].configure(state=tk.DISABLED)

    def _populate_candidates(self):
        """候補をUIに表示"""
        if not self.classified_frames or not self.unified_size:
            return

        cap = self._get_video_capture()
        unified_w, unified_h = self.unified_size

        self._thumb_images = []

        for cat in MOUTH_CATEGORIES:
            self._populate_category_candidates(cat, cap, unified_w, unified_h)

    def _populate_category_candidates(
        self, cat: str, cap: cv2.VideoCapture, unified_w: int, unified_h: int
    ):
        """カテゴリの候補を表示"""
        frames = self.classified_frames.get(cat, [])
        shown_count = INITIAL_CANDIDATES_PER_CATEGORY

        for btn in self.candidate_buttons[cat]:
            btn.destroy()
        self.candidate_buttons[cat] = []

        cat_frame = self.category_frames[cat]
        thumb_frame = None
        for child in cat_frame.winfo_children():
            if isinstance(child, ttk.Frame):
                thumb_frame = child
                break

        if thumb_frame is None:
            return

        for i in range(min(shown_count, len(frames))):
            mf = frames[i]

            unified_quad = center_to_quad(mf.center, unified_w, unified_h)

            cap.set(cv2.CAP_PROP_POS_FRAMES, float(mf.frame_idx))
            ok, frame = cap.read()

            btn = ttk.Button(thumb_frame, text="", width=8)
            btn.pack(side=tk.LEFT, padx=1, before=self.more_buttons[cat])

            if ok and frame is not None:
                patch = warp_frame_to_norm(frame, unified_quad, unified_w, unified_h)
                photo = numpy_to_photoimage(patch, target_height=THUMB_HEIGHT)
                if photo:
                    self._thumb_images.append(photo)
                    btn.configure(image=photo, compound=tk.TOP)

            btn.configure(command=lambda c=cat, idx=i: self._on_candidate_click(c, idx))
            self.candidate_buttons[cat].append(btn)

        if len(frames) > shown_count:
            self.more_buttons[cat].configure(state=tk.NORMAL)
        else:
            self.more_buttons[cat].configure(state=tk.DISABLED)

        self._update_candidate_buttons_appearance()

    def _load_more_candidates(self, cat: str):
        """候補を追加ロード - 別ウィンドウで表示"""
        frames = self.classified_frames.get(cat, [])
        if not frames:
            return
        self._open_candidate_window(cat, frames)

    def _open_candidate_window(self, cat: str, frames: List[MouthFrameInfo]):
        """カテゴリの全候補を別ウィンドウで表示"""
        win = tk.Toplevel(self)
        win.title(tr("prep.tab51.more_window_title", cat=cat.upper()))
        win.geometry("600x500")
        win.transient(self)

        canvas = tk.Canvas(win)
        scrollbar_y = ttk.Scrollbar(win, orient=tk.VERTICAL, command=canvas.yview)
        scrollbar_x = ttk.Scrollbar(win, orient=tk.HORIZONTAL, command=canvas.xview)
        inner_frame = ttk.Frame(canvas)

        canvas.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        canvas.create_window((0, 0), window=inner_frame, anchor=tk.NW)
        inner_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        win._thumb_images = []

        cap = self._get_video_capture()
        unified_w, unified_h = self.unified_size

        cols = 4
        thumb_height = 60

        for i, mf in enumerate(frames):
            row = i // cols
            col = i % cols

            unified_quad = center_to_quad(mf.center, unified_w, unified_h)
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(mf.frame_idx))
            ok, frame = cap.read()

            btn_frame = ttk.Frame(inner_frame)
            btn_frame.grid(row=row, column=col, padx=5, pady=5)

            btn = ttk.Button(btn_frame, text=f"#{mf.frame_idx}")

            if ok and frame is not None:
                patch = warp_frame_to_norm(frame, unified_quad, unified_w, unified_h)
                photo = numpy_to_photoimage(patch, target_height=thumb_height)
                if photo:
                    win._thumb_images.append(photo)
                    btn.configure(image=photo, compound=tk.TOP)

            def on_select(mf=mf, win=win):
                self.current_preview_mf = mf
                self._update_preview_panel(mf)
                for btn in self.assign_buttons.values():
                    btn.configure(state=tk.NORMAL)
                win.destroy()

            btn.configure(command=on_select)
            btn.pack()

            ttk.Label(btn_frame, text=f"F{mf.frame_idx}", font=("", 8)).pack()

        self.log(tr("prep.log.t5_more_shown", cat=cat, n=len(frames)))

    def _on_candidate_click(self, cat: str, index: int):
        """候補クリック - プレビュー表示 + 自動割り当て"""
        frames = self.classified_frames.get(cat, [])
        if index >= len(frames):
            return

        mf = frames[index]
        self.current_preview_mf = mf
        self._update_preview_panel(mf)

        for btn in self.assign_buttons.values():
            btn.configure(state=tk.NORMAL)

        # 自動割り当て: そのカテゴリに既に同じフレームが割当済みなら何もしない
        current = self.selected_frames.get(cat)
        if current is not None and current.frame_idx == mf.frame_idx:
            return

        self._assign_current_to_category(cat)

    def _update_preview_panel(self, mf: MouthFrameInfo):
        """プレビューパネルを更新"""
        cap = self._get_video_capture()
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(mf.frame_idx))
        ok, frame = cap.read()

        if not ok or frame is None:
            return

        unified_w, unified_h = self.unified_size
        unified_quad = center_to_quad(mf.center, unified_w, unified_h)

        frame_draw = frame.copy()
        quad_int = unified_quad.astype(np.int32)
        cv2.polylines(frame_draw, [quad_int], isClosed=True, color=(0, 255, 0), thickness=2)

        photo = numpy_to_photoimage(frame_draw, max_size=PREVIEW_MAX_SIZE)
        if photo:
            self._preview_photo = photo
            self.preview_label.configure(image=photo, text="")

        self.preview_info_label.configure(
            text=tr("prep.tab51.preview_info", frame=mf.frame_idx, cat=mf.category,
                    w=unified_w, h=unified_h)
        )

    def _clear_preview_panel(self):
        """プレビューパネルをクリア"""
        self.preview_label.configure(image="", text=tr("prep.tab51.click_candidate"))
        self.preview_info_label.configure(text="")
        self.current_preview_mf = None
        self._preview_photo = None

        for btn in self.assign_buttons.values():
            btn.configure(state=tk.DISABLED)

    def _assign_current_to_category(self, category: str):
        """現在プレビュー中のフレームをカテゴリに割り当て"""
        if self.current_preview_mf is None:
            return

        mf = self.current_preview_mf
        self.selected_frames[category] = mf

        source_cat = mf.category if mf.category else "?"
        self.log(tr("prep.log.t5_assigned", cat=category, frame=mf.frame_idx,
                    source=source_cat))

        self._update_candidate_buttons_appearance()
        self._update_selection_status()

    def _update_candidate_buttons_appearance(self):
        """候補ボタンの見た目を更新"""
        frame_to_assigned_cat: Dict[int, str] = {}
        for cat, mf in self.selected_frames.items():
            if mf is not None:
                frame_to_assigned_cat[mf.frame_idx] = cat

        for cat in MOUTH_CATEGORIES:
            frames = self.classified_frames.get(cat, [])
            buttons = self.candidate_buttons[cat]

            for i, btn in enumerate(buttons):
                if i < len(frames):
                    mf = frames[i]
                    assigned_cat = frame_to_assigned_cat.get(mf.frame_idx)

                    if assigned_cat:
                        btn.configure(text=f"→{assigned_cat.upper()}", compound=tk.BOTTOM)
                    else:
                        btn.configure(text="", compound=tk.TOP)

    def _update_selection_status(self):
        """選択状況を更新"""
        for cat in MOUTH_CATEGORIES:
            mf = self.selected_frames.get(cat)
            if mf:
                source_cat = mf.category if mf.category else "?"
                self.status_labels[cat].configure(
                    text=tr("prep.tab51.selected_status", frame=mf.frame_idx,
                            source=source_cat),
                    foreground="green"
                )
            else:
                self.status_labels[cat].configure(text=tr("prep.tab51.not_selected"),
                                                  foreground="gray")

        required_complete = all(
            self.selected_frames[cat] is not None
            for cat in MOUTH_CATEGORIES
        )

        if required_complete:
            self.goto_step3_btn.configure(state=tk.NORMAL)
        else:
            self.goto_step3_btn.configure(state=tk.DISABLED)

    def _goto_step52(self):
        """⑤-2へ進む"""
        required_complete = all(
            self.selected_frames[cat] is not None
            for cat in MOUTH_CATEGORIES
        )

        if not required_complete:
            missing = [cat for cat in MOUTH_CATEGORIES if self.selected_frames[cat] is None]
            messagebox.showwarning(tr("prep.msgbox.title_warn"),
                                   tr("prep.msgbox.t5_missing_required",
                                      names=", ".join(missing)))
            return

        self.notebook.select(self.tab52_frame)
        self._on_update_preview()

    def _get_video_capture(self) -> cv2.VideoCapture:
        """キャッシュされたVideoCaptureを取得"""
        if self._cached_cap is None or not self._cached_cap.isOpened():
            self._cached_cap = cv2.VideoCapture(self.video_path)
        return self._cached_cap

    # ================================================================
    # ⑤-2 出力
    # ================================================================

    def _build_tab52(self, parent):
        note = ttk.Label(
            parent, foreground="#555",
            text=tr("prep.tab52.note"))
        note.pack(anchor="w", padx=10, pady=(10, 0))

        frame = ttk.Frame(parent, padding=5)
        frame.pack(fill="both", expand=True)

        top_frame = ttk.Frame(frame)
        top_frame.pack(fill=tk.X, pady=(0, 10))

        # マスク設定
        settings_frame = ttk.LabelFrame(top_frame, text=tr("prep.tab52.mask_frame"), padding=5)
        settings_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # Mask type (SAM3 fixed)
        self.mask_type_var = tk.StringVar(value="sam3")

        # Feather
        feather_row = ttk.Frame(settings_frame)
        feather_row.pack(fill=tk.X, pady=2)
        ttk.Label(feather_row, text=tr("prep.tab52.feather")).pack(side=tk.LEFT)
        self.feather_var = tk.IntVar(value=DEFAULT_FEATHER)
        ttk.Scale(
            feather_row, from_=0, to=MAX_FEATHER, orient=tk.HORIZONTAL,
            variable=self.feather_var, length=100
        ).pack(side=tk.LEFT, padx=5)
        self.feather_label = ttk.Label(feather_row, text=f"{DEFAULT_FEATHER}px", width=5)
        self.feather_label.pack(side=tk.LEFT)

        # Dilate
        dilate_row = ttk.Frame(settings_frame)
        dilate_row.pack(fill=tk.X, pady=2)
        ttk.Label(dilate_row, text=tr("prep.tab52.dilate")).pack(side=tk.LEFT)
        self.dilate_var = tk.IntVar(value=DEFAULT_MASK_DILATE)
        self.dilate_slider = ttk.Scale(
            dilate_row, from_=-MAX_MASK_DILATE, to=MAX_MASK_DILATE, orient=tk.HORIZONTAL,
            variable=self.dilate_var, length=100
        )
        self.dilate_slider.pack(side=tk.LEFT, padx=5)
        self.dilate_label = ttk.Label(dilate_row, text=f"{DEFAULT_MASK_DILATE}px", width=5)
        self.dilate_label.pack(side=tk.LEFT)

        # 口の位置・大きさ（値は「再生時の見え方」基準。切り抜き枠への変換は
        # _on_update_preview 側で符号反転して行う）
        tuning_frame = ttk.LabelFrame(top_frame, text=tr("prep.tab52.tuning_frame"), padding=5)
        tuning_frame.pack(side=tk.LEFT, fill=tk.Y)

        ox_row = ttk.Frame(tuning_frame)
        ox_row.pack(fill=tk.X, pady=1)
        ttk.Label(ox_row, text="X:").pack(side=tk.LEFT)
        self.offset_x_var = tk.IntVar(value=DEFAULT_OFFSET_X)
        ttk.Scale(
            ox_row, from_=-MAX_OFFSET, to=MAX_OFFSET, orient=tk.HORIZONTAL,
            variable=self.offset_x_var, length=100
        ).pack(side=tk.LEFT, padx=5)
        self.offset_x_label = ttk.Label(ox_row, text="0px", width=5)
        self.offset_x_label.pack(side=tk.LEFT)

        oy_row = ttk.Frame(tuning_frame)
        oy_row.pack(fill=tk.X, pady=1)
        ttk.Label(oy_row, text="Y:").pack(side=tk.LEFT)
        self.offset_y_var = tk.IntVar(value=DEFAULT_OFFSET_Y)
        ttk.Scale(
            oy_row, from_=-MAX_OFFSET, to=MAX_OFFSET, orient=tk.HORIZONTAL,
            variable=self.offset_y_var, length=100
        ).pack(side=tk.LEFT, padx=5)
        self.offset_y_label = ttk.Label(oy_row, text="0px", width=5)
        self.offset_y_label.pack(side=tk.LEFT)

        scale_row = ttk.Frame(tuning_frame)
        scale_row.pack(fill=tk.X, pady=1)
        ttk.Label(scale_row, text=tr("prep.tab52.scale")).pack(side=tk.LEFT)
        self.scale_var = tk.DoubleVar(value=DEFAULT_SCALE)
        ttk.Scale(
            scale_row, from_=MIN_SCALE, to=MAX_SCALE, orient=tk.HORIZONTAL,
            variable=self.scale_var, length=100
        ).pack(side=tk.LEFT, padx=5)
        self.scale_label = ttk.Label(scale_row, text="100%", width=5)
        self.scale_label.pack(side=tk.LEFT)

        # ボタン
        btn_frame = ttk.Frame(top_frame)
        btn_frame.pack(side=tk.LEFT, padx=10)

        ttk.Button(btn_frame, text=tr("prep.tab52.reset"),
                   command=self._reset_settings).pack(fill=tk.X, pady=2)

        # 出力ボタン（プレビュー更新/リセットの右横）
        out_frame = ttk.Frame(top_frame)
        out_frame.pack(side=tk.LEFT, padx=10, fill=tk.Y)
        self.output_btn = ttk.Button(
            out_frame, text=tr("prep.tab52.output"), command=self._on_output,
            state=tk.DISABLED
        )
        self.output_btn.pack(fill=tk.BOTH, expand=True, pady=2)

        # スライダー変更時に自動でプレビュー更新（デバウンス付き）
        self._preview_update_job: Optional[str] = None
        for var in (self.feather_var, self.dilate_var,
                    self.offset_x_var, self.offset_y_var, self.scale_var):
            var.trace_add("write", lambda *_: self._schedule_preview_update())

        # プレビューエリア
        preview_frame = ttk.LabelFrame(frame, text=tr("prep.tab52.preview_frame"), padding=10)
        preview_frame.pack(fill=tk.BOTH, expand=True)

        preview_inner = ttk.Frame(preview_frame)
        preview_inner.pack()

        self.output_preview_labels: Dict[str, ttk.Label] = {}
        self.output_preview_name_labels: Dict[str, ttk.Label] = {}

        for cat in MOUTH_CATEGORIES:
            col = ttk.Frame(preview_inner)
            col.pack(side=tk.LEFT, padx=15, pady=5)

            name_lbl = ttk.Label(col, text=f"[{cat}]", font=("", 9, "bold"))
            name_lbl.pack()
            self.output_preview_name_labels[cat] = name_lbl

            img_lbl = ttk.Label(col, text="---")
            img_lbl.pack(pady=5)
            self.output_preview_labels[cat] = img_lbl

    def _schedule_preview_update(self):
        """スライダー変更から300ms後にプレビュー更新（連続変更をまとめる）。"""
        if self._preview_update_job is not None:
            try:
                self.after_cancel(self._preview_update_job)
            except Exception:
                pass
        self._preview_update_job = self.after(300, self._run_scheduled_preview)

    def _run_scheduled_preview(self):
        self._preview_update_job = None
        # 解析済み・全カテゴリ選択済みの場合のみ（_on_update_preview内でもガードされる）
        self._on_update_preview()

    def _reset_settings(self):
        """設定をリセット"""
        self.feather_var.set(DEFAULT_FEATHER)
        self.dilate_var.set(DEFAULT_MASK_DILATE)
        self.offset_x_var.set(DEFAULT_OFFSET_X)
        self.offset_y_var.set(DEFAULT_OFFSET_Y)
        self.scale_var.set(DEFAULT_SCALE)
        self.log(tr("prep.log.t52_reset"))

    def _on_update_preview(self):
        """出力プレビュー更新"""
        if not self.unified_size:
            return

        required_complete = all(
            self.selected_frames[cat] is not None
            for cat in MOUTH_CATEGORIES
        )
        if not required_complete:
            return

        cap = self._get_video_capture()
        unified_w, unified_h = self.unified_size

        feather_px = self.feather_var.get()
        offset_x = self.offset_x_var.get()
        offset_y = self.offset_y_var.get()
        scale = max(self.scale_var.get(), MIN_SCALE)  # 0除算ガード
        use_sam_mask = self.mask_type_var.get() == "sam3"
        mask_dilate = self.dilate_var.get()

        self.preview_sprites = {}

        for cat in MOUTH_CATEGORIES:
            mf = self.selected_frames.get(cat)
            if mf is None:
                self.output_preview_labels[cat].configure(image="", text="---")
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, float(mf.frame_idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            unified_quad = center_to_quad(mf.center, unified_w, unified_h)
            # UIの値は「出力（＝再生時）での口の位置・大きさ」。実際に動かすのは
            # 切り抜き枠なので、向き・拡縮とも反転して渡す。
            # 枠を1/scale倍にすると元画像→出力の拡大率が scale 倍になるため、
            # オフセットも scale で割って出力ピクセル基準に揃える。
            adjusted_quad = adjust_quad(
                unified_quad, -offset_x / scale, -offset_y / scale, 1.0 / scale)

            bgra = extract_mouth_sprite(
                frame, adjusted_quad, unified_w, unified_h,
                feather_px=feather_px,
                sam_mask=mf.mask,
                use_sam_mask=use_sam_mask,
                mask_dilate=mask_dilate,
            )

            self.preview_sprites[cat] = bgra

            composited = composite_on_checkerboard(bgra)
            photo = numpy_to_photoimage(composited, target_height=80)
            if photo:
                self.preview_images[cat] = photo
                self.output_preview_labels[cat].configure(image=photo, text="")

        self.output_btn.configure(state=tk.NORMAL)
        self.log(tr("prep.log.t52_preview_updated", n=len(self.preview_sprites)))

    def _on_output(self):
        """PNG出力: {キャラフォルダ}/mouth/ へ上書き保存"""
        if not self.preview_sprites:
            messagebox.showwarning(tr("prep.msgbox.title_warn"),
                                   tr("prep.msgbox.t52_update_first"))
            return

        video_dir = os.path.dirname(self.video_path)
        output_dir = os.path.join(video_dir, "mouth")

        existing = [c for c in ("open.png", "closed.png", "half.png")
                    if os.path.isfile(os.path.join(output_dir, c))]
        if existing:
            if not messagebox.askyesno(
                    tr("prep.msgbox.title_confirm"),
                    tr("prep.msgbox.t52_overwrite_confirm", path=output_dir)):
                return

        os.makedirs(output_dir, exist_ok=True)

        # 正方形サイズを計算（最大辺に合わせる）
        max_side = max(
            max(bgra.shape[0], bgra.shape[1])
            for bgra in self.preview_sprites.values()
        )
        square_size = max_side
        self.log(tr("prep.log.t52_output_size", size=square_size))

        failed = []
        for cat, bgra in self.preview_sprites.items():
            # 正方形キャンバスに中央配置
            h, w = bgra.shape[:2]
            square_bgra = np.zeros((square_size, square_size, 4), dtype=np.uint8)
            x_offset = (square_size - w) // 2
            y_offset = (square_size - h) // 2
            square_bgra[y_offset:y_offset+h, x_offset:x_offset+w] = bgra

            out_path = os.path.join(output_dir, f"{cat}.png")
            if imwrite_jp(out_path, square_bgra):
                self.log(tr("prep.log.t52_output", path=out_path))
            else:
                failed.append(out_path)

        if failed:
            messagebox.showerror(tr("prep.msgbox.title_error"),
                                 tr("prep.msgbox.t52_output_failed",
                                    files="\n".join(failed)))
            return

        self.log(tr("prep.log.t52_done", path=output_dir))
        messagebox.showinfo(tr("prep.msgbox.title_done"),
                            tr("prep.msgbox.t52_done", path=output_dir))

    # ---- ⑤共通: ログ/進捗ポーリング ----

    def _poll_logs(self):
        """ログキューをポーリング"""
        while not self.log_queue.empty():
            try:
                msg = self.log_queue.get_nowait()
                self._append_log(msg)
            except queue.Empty:
                break

        # ワーカースレッドからの進捗を反映（最後の値のみ使用）
        latest_progress: Optional[float] = None
        while not self.progress_queue.empty():
            try:
                latest_progress = self.progress_queue.get_nowait()
            except queue.Empty:
                break
        if latest_progress is not None:
            self.progress_var.set(latest_progress)

        # ワーカースレッドから依頼されたUI更新を実行
        while not self.ui_queue.empty():
            try:
                func = self.ui_queue.get_nowait()
            except queue.Empty:
                break
            func()

        # ラベル更新
        padding_pct = int(self.padding_var.get() * 100)
        self.padding_label.configure(text=f"{padding_pct}%")
        self.feather_label.configure(text=f"{self.feather_var.get()}px")
        self.dilate_label.configure(text=f"{self.dilate_var.get()}px")
        self.offset_x_label.configure(text=f"{self.offset_x_var.get()}px")
        self.offset_y_label.configure(text=f"{self.offset_y_var.get()}px")
        scale_pct = int(self.scale_var.get() * 100)
        self.scale_label.configure(text=f"{scale_pct}%")

        self.after(100, self._poll_logs)

    def _append_log(self, msg: str):
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def log(self, msg: str):
        """スレッドセーフなログ（⑤のログ欄に出力）"""
        self.log_queue.put(msg)

    # ================================================================
    # ⑥ 口消し（ブラシ）
    # ================================================================

    def _build_tab6(self, parent):
        note = ttk.Label(
            parent, foreground="#555",
            text=tr("prep.tab6.note"))
        note.pack(anchor="w", padx=10, pady=(10, 0))

        top = ttk.Frame(parent, padding=5)
        top.pack(fill="x", padx=5)
        ttk.Label(top, text=tr("prep.common.char_label")).pack(side="left")
        self._t6_combo = ttk.Combobox(top, state="readonly", width=14)
        self._t6_combo.pack(side="left", padx=2)
        self._t6_combo.bind("<<ComboboxSelected>>", self._t6_on_char_selected)
        ttk.Button(top, text=tr("prep.common.refresh"),
                   command=self._t6_refresh_chars).pack(side="left", padx=2)

        ttk.Label(top, text=tr("prep.tab6.brush")).pack(side="left")
        self._t6_brush = tk.IntVar(value=12)
        ttk.Scale(top, from_=2, to=60, variable=self._t6_brush,
                  orient="horizontal", length=80).pack(side="left", padx=2)
        self._t6_brush_label = ttk.Label(top, text="12px", width=5)
        self._t6_brush_label.pack(side="left")
        self._t6_brush.trace_add(
            "write", lambda *_: self._t6_brush_label.config(text=f"{self._t6_brush.get()}px"))

        ttk.Label(top, text=tr("prep.tab6.soften")).pack(side="left", padx=(8, 0))
        self._t6_soft = tk.IntVar(value=40)
        ttk.Scale(top, from_=0, to=90, variable=self._t6_soft,
                  orient="horizontal", length=70).pack(side="left", padx=2)
        self._t6_soft_label = ttk.Label(top, text="40%", width=4)
        self._t6_soft_label.pack(side="left")
        self._t6_soft.trace_add(
            "write", lambda *_: self._t6_soft_label.config(text=f"{self._t6_soft.get()}%"))

        ttk.Label(top, text=tr("prep.tab6.sampled_color")).pack(side="left")
        self._t6_color_swatch = tk.Canvas(top, width=20, height=20, bg="#808080",
                                          highlightthickness=1)
        self._t6_color_swatch.pack(side="left", padx=3)

        row2 = ttk.Frame(parent, padding=(5, 0))
        row2.pack(fill="x", padx=5)
        ttk.Button(row2, text=tr("prep.tab6.discard"), command=self._t6_reload).pack(
            side="left", padx=2)
        ttk.Button(row2, text=tr("prep.tab6.save"), command=self._t6_save).pack(
            side="left", padx=8)
        self._t6_status = ttk.Label(row2, text="")
        self._t6_status.pack(side="left", padx=10)

        body = ttk.Frame(parent, padding=5)
        body.pack(fill="both", expand=True)

        left = ttk.LabelFrame(body, text=tr("prep.tab6.preview_frame"), padding=5)
        left.pack(side="left", fill="both", expand=True)
        self._t6_prev_canvas = tk.Canvas(left, bg="#2b2b2b", highlightthickness=0)
        self._t6_prev_canvas.pack(fill="both", expand=True)

        right = ttk.LabelFrame(body, text=tr("prep.tab6.edit_frame"), padding=5)
        right.pack(side="left", fill="both", expand=True, padx=(5, 0))
        self._t6_edit_canvas = tk.Canvas(right, bg="#2b2b2b", highlightthickness=0)
        self._t6_edit_canvas.pack(fill="both", expand=True)
        self._t6_edit_canvas.bind("<ButtonPress-1>", self._t6_on_paint_start)
        self._t6_edit_canvas.bind("<B1-Motion>", self._t6_on_paint_move)
        self._t6_edit_canvas.bind("<ButtonRelease-1>", self._t6_on_paint_end)
        self._t6_edit_canvas.bind("<Button-3>", self._t6_on_sample)
        self._t6_edit_canvas.bind("<Motion>", self._t6_on_cursor_move)
        self._t6_edit_canvas.bind("<Leave>", lambda e: self._t6_edit_canvas.delete("cursor"))

        # 編集状態
        self._t6_img: Optional[np.ndarray] = None       # 編集中のBGR画像（フル解像度）
        self._t6_alpha: Optional[np.ndarray] = None     # 元画像のアルファ（保存時に再合成）
        self._t6_path: str = ""
        self._t6_char_dir: str = ""
        self._t6_region: Tuple[int, int, int, int] = (0, 0, 0, 0)  # 拡大表示範囲
        self._t6_zoom: float = 1.0
        self._t6_edit_offset: Tuple[int, int] = (0, 0)
        self._t6_color: Optional[np.ndarray] = None      # 採取色 (BGR float32)
        self._t6_last_pt: Optional[Tuple[int, int]] = None
        self._t6_prev_photo = None
        self._t6_edit_photo = None
        self._t6_prev_image_id = None
        self._t6_edit_image_id = None

    # ---- キャラ選択・画像ロード ----

    def _t6_refresh_chars(self):
        values = []
        work = self._work_folder
        if work and os.path.isdir(work):
            for d in _list_char_folders(work):
                if len(_find_images(d)) == 1:
                    values.append(os.path.basename(d))
        self._t6_combo.configure(values=values)
        if self._t6_combo.get() not in values:
            self._t6_combo.set("")

    def _t6_on_char_selected(self, event=None):
        char = self._t6_combo.get()
        if not char or not self._work_folder:
            return
        char_dir = os.path.join(self._work_folder, char)
        imgs = _find_images(char_dir)
        if len(imgs) != 1:
            return
        self._t6_char_dir = char_dir
        self._t6_path = imgs[0]
        self._t6_load_image()

    def _t6_load_image(self):
        img = _imread_jp(self._t6_path)
        if img is None:
            messagebox.showerror(tr("prep.msgbox.title_error"),
                                 tr("prep.msgbox.image_load_failed", path=self._t6_path))
            return
        img, alpha = _split_alpha(img)
        self._t6_img = img
        self._t6_alpha = alpha
        self._t6_color = None
        self._t6_color_swatch.config(bg="#808080")
        self._t6_region = self._t6_compute_region()
        self._t6_set_status(
            tr("prep.tab6.loaded", name=os.path.basename(self._t6_path)), "black")
        self._t6_redraw_all()

    def _t6_compute_region(self) -> Tuple[int, int, int, int]:
        """face_info の口bboxを基準に拡大表示範囲を決める（無ければ全体）。"""
        h, w = self._t6_img.shape[:2]
        name = os.path.basename(self._t6_char_dir) if self._t6_char_dir else ""
        info = None
        for fname in (f"{name}_face_info.json", "face_info.json"):
            p = os.path.join(self._t6_char_dir, fname)
            if os.path.isfile(p):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        info = json.load(f)
                except Exception:
                    info = None
                break
        if not info or "mouth" not in info or "bbox" not in info["mouth"]:
            return (0, 0, w, h)
        b = info["mouth"]["bbox"]
        cx = (b[0] + b[2]) / 2.0
        cy = (b[1] + b[3]) / 2.0
        half = max(b[2] - b[0], b[3] - b[1]) * 2.5
        half = max(half, 80)
        x0 = max(0, int(cx - half))
        y0 = max(0, int(cy - half))
        x1 = min(w, int(cx + half))
        y1 = min(h, int(cy + half))
        return (x0, y0, x1, y1)

    def _t6_set_status(self, text: str, color: str = "black"):
        self._t6_status.config(text=text, foreground=color)

    # ---- 描画 ----

    def _t6_redraw_all(self):
        self._t6_redraw_edit()
        self._t6_redraw_preview()

    def _t6_redraw_edit(self):
        if self._t6_img is None or not _HAS_PIL:
            return
        x0, y0, x1, y1 = self._t6_region
        crop = self._t6_img[y0:y1, x0:x1]
        if crop.size == 0:
            return
        ch_img, cw_img = crop.shape[:2]
        self._t6_edit_canvas.update_idletasks()
        cw = max(self._t6_edit_canvas.winfo_width(), 100)
        ch = max(self._t6_edit_canvas.winfo_height(), 100)
        zoom = min(cw / cw_img, ch / ch_img)
        zoom = min(zoom, 8.0)
        self._t6_zoom = zoom
        dw, dh = max(1, int(cw_img * zoom)), max(1, int(ch_img * zoom))
        interp = cv2.INTER_NEAREST if zoom >= 2.0 else cv2.INTER_AREA
        disp = cv2.resize(crop, (dw, dh), interpolation=interp)
        pil_img = Image.fromarray(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB))
        self._t6_edit_photo = ImageTk.PhotoImage(pil_img)
        ox = (cw - dw) // 2
        oy = (ch - dh) // 2
        self._t6_edit_offset = (ox, oy)
        if self._t6_edit_image_id is not None:
            self._t6_edit_canvas.delete(self._t6_edit_image_id)
        self._t6_edit_image_id = self._t6_edit_canvas.create_image(
            ox, oy, image=self._t6_edit_photo, anchor="nw")
        self._t6_edit_canvas.tag_lower(self._t6_edit_image_id)

    def _t6_redraw_preview(self):
        if self._t6_img is None or not _HAS_PIL:
            return
        display = self._t6_img.copy()
        x0, y0, x1, y1 = self._t6_region
        h, w = display.shape[:2]
        if (x1 - x0) < w or (y1 - y0) < h:
            cv2.rectangle(display, (x0, y0), (x1, y1), (0, 255, 0), 2)
        self._t6_prev_canvas.update_idletasks()
        cw = max(self._t6_prev_canvas.winfo_width(), 100)
        ch = max(self._t6_prev_canvas.winfo_height(), 100)
        pil_img = _bgr_to_pil(display, cw, ch)
        self._t6_prev_photo = ImageTk.PhotoImage(pil_img)
        if self._t6_prev_image_id is not None:
            self._t6_prev_canvas.delete(self._t6_prev_image_id)
        self._t6_prev_image_id = self._t6_prev_canvas.create_image(
            cw // 2, ch // 2, image=self._t6_prev_photo, anchor="center")

    def _t6_canvas_to_img(self, event) -> Optional[Tuple[int, int]]:
        """編集キャンバス座標 → 画像座標。範囲外は None。"""
        if self._t6_img is None or self._t6_zoom <= 0:
            return None
        ox, oy = self._t6_edit_offset
        x0, y0, x1, y1 = self._t6_region
        ix = x0 + int((event.x - ox) / self._t6_zoom)
        iy = y0 + int((event.y - oy) / self._t6_zoom)
        if x0 <= ix < x1 and y0 <= iy < y1:
            return (ix, iy)
        return None

    # ---- ブラシ ----

    def _t6_dab(self, ix: int, iy: int):
        """1点のブラシスタンプ（ぼかし付きの円）を画像に適用。"""
        img = self._t6_img
        h, w = img.shape[:2]
        r = max(1, int(self._t6_brush.get()))
        soft = self._t6_soft.get() / 100.0
        x0 = max(0, ix - r)
        y0 = max(0, iy - r)
        x1 = min(w, ix + r + 1)
        y1 = min(h, iy + r + 1)
        if x0 >= x1 or y0 >= y1:
            return
        yy, xx = np.ogrid[y0:y1, x0:x1]
        dist = np.sqrt((xx - ix) ** 2 + (yy - iy) ** 2)
        if soft <= 0:
            alpha = (dist <= r).astype(np.float32)
        else:
            inner = r * (1.0 - soft)
            alpha = np.clip((r - dist) / max(r - inner, 1e-3), 0.0, 1.0).astype(np.float32)
            alpha[dist > r] = 0.0
        roi = img[y0:y1, x0:x1].astype(np.float32)
        img[y0:y1, x0:x1] = (
            self._t6_color[None, None, :] * alpha[..., None]
            + roi * (1.0 - alpha[..., None])
        ).astype(np.uint8)

    def _t6_stroke_to(self, ix: int, iy: int):
        """前回位置から補間しながらスタンプを打つ。"""
        if self._t6_last_pt is None:
            self._t6_dab(ix, iy)
        else:
            lx, ly = self._t6_last_pt
            dist = math.hypot(ix - lx, iy - ly)
            step = max(1, int(self._t6_brush.get()) // 3)
            n = max(1, int(dist / step))
            for i in range(1, n + 1):
                px = int(lx + (ix - lx) * i / n)
                py = int(ly + (iy - ly) * i / n)
                self._t6_dab(px, py)
        self._t6_last_pt = (ix, iy)

    def _t6_on_paint_start(self, event):
        if self._t6_img is None:
            return
        if self._t6_color is None:
            self._t6_set_status(tr("prep.tab6.sample_first"), "red")
            return
        pt = self._t6_canvas_to_img(event)
        if pt is None:
            return
        self._t6_last_pt = None
        self._t6_stroke_to(*pt)
        self._t6_redraw_edit()

    def _t6_on_paint_move(self, event):
        if self._t6_img is None or self._t6_color is None or self._t6_last_pt is None:
            return
        pt = self._t6_canvas_to_img(event)
        if pt is None:
            return
        self._t6_stroke_to(*pt)
        self._t6_redraw_edit()

    def _t6_on_paint_end(self, event):
        if self._t6_last_pt is not None:
            self._t6_last_pt = None
            self._t6_redraw_preview()

    def _t6_on_sample(self, event):
        """右クリック: ブラシ範囲の平均色を採取。"""
        if self._t6_img is None:
            return
        pt = self._t6_canvas_to_img(event)
        if pt is None:
            return
        ix, iy = pt
        img = self._t6_img
        h, w = img.shape[:2]
        r = max(1, int(self._t6_brush.get()))
        x0 = max(0, ix - r)
        y0 = max(0, iy - r)
        x1 = min(w, ix + r + 1)
        y1 = min(h, iy + r + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask = (xx - ix) ** 2 + (yy - iy) ** 2 <= r * r
        pixels = img[y0:y1, x0:x1][mask]
        if len(pixels) == 0:
            return
        self._t6_color = pixels.mean(axis=0).astype(np.float32)
        b, g, rr = int(self._t6_color[0]), int(self._t6_color[1]), int(self._t6_color[2])
        self._t6_color_swatch.config(bg=f"#{rr:02X}{g:02X}{b:02X}")
        self._t6_set_status(tr("prep.tab6.sampled", color=f"#{rr:02X}{g:02X}{b:02X}"))

    def _t6_on_cursor_move(self, event):
        """ブラシ範囲をカーソル位置に円で表示。"""
        self._t6_edit_canvas.delete("cursor")
        if self._t6_img is None:
            return
        r = max(1, int(self._t6_brush.get())) * self._t6_zoom
        self._t6_edit_canvas.create_oval(
            event.x - r, event.y - r, event.x + r, event.y + r,
            outline="#FF00FF", tags="cursor")

    # ---- 操作 ----

    def _t6_reload(self):
        if not self._t6_path:
            return
        self._t6_load_image()
        self._t6_set_status(tr("prep.tab6.discarded"))

    def _t6_save(self):
        if self._t6_img is None or not self._t6_path:
            messagebox.showwarning(tr("prep.msgbox.title_warn"),
                                   tr("prep.msgbox.t6_select_char"))
            return
        out = _merge_alpha(self._t6_img, self._t6_alpha)
        if imwrite_jp(self._t6_path, out):
            self._t6_set_status(
                tr("prep.tab6.saved", name=os.path.basename(self._t6_path)),
                "green")
        else:
            messagebox.showerror(tr("prep.msgbox.title_error"), tr("prep.msgbox.save_failed"))

    # ================================================================
    # ⑦ 状態確認
    # ================================================================

    def _build_tab7(self, parent):
        note = ttk.Label(
            parent, foreground="#555",
            text=tr("prep.tab7.note"))
        note.pack(anchor="w", padx=10, pady=(10, 0))

        btn_row = ttk.Frame(parent, padding=5)
        btn_row.pack(fill="x", padx=5)
        ttk.Button(btn_row, text=tr("prep.tab7.rescan"),
                   command=self._t7_rescan).pack(side="left", padx=2)
        self._t7_launch_btn = ttk.Button(
            btn_row, text=tr("prep.tab7.launch"),
            command=self._t7_launch, state="disabled")
        self._t7_launch_btn.pack(side="right", padx=2)
        self._t7_summary = ttk.Label(btn_row, text="")
        self._t7_summary.pack(side="left", padx=10)

        columns = ("name", "nameok", "image", "faceinfo", "mouth", "video", "errors")
        tree_frame = ttk.Frame(parent)
        tree_frame.pack(fill="both", expand=True, padx=10, pady=5)
        self._t7_tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=12)
        headings = {
            "name": (tr("prep.tab7.col_name"), 140),
            "nameok": (tr("prep.tab7.col_nameok"), 50),
            "image": (tr("prep.tab7.col_image"), 50),
            "faceinfo": ("face_info", 70),
            "mouth": ("mouth", 60),
            "video": (tr("prep.tab7.col_video"), 50),
            "errors": (tr("prep.tab7.col_errors"), 400),
        }
        for col, (label, width) in headings.items():
            self._t7_tree.heading(col, text=label)
            self._t7_tree.column(col, width=width, anchor="w" if col in ("name", "errors") else "center")
        tsb = ttk.Scrollbar(tree_frame, command=self._t7_tree.yview)
        self._t7_tree.config(yscrollcommand=tsb.set)
        self._t7_tree.pack(side="left", fill="both", expand=True)
        tsb.pack(side="right", fill="y")

        lf = ttk.LabelFrame(parent, text=tr("prep.tab7.leftover_frame"), padding=5)
        lf.pack(fill="x", padx=10, pady=(0, 10))
        self._t7_leftover = tk.Text(lf, height=5, state="disabled", wrap="word")
        self._t7_leftover.pack(fill="x")

    def _t7_rescan(self):
        work = self._work_folder
        self._t7_tree.delete(*self._t7_tree.get_children())
        self._t7_leftover.config(state="normal")
        self._t7_leftover.delete("1.0", "end")

        if not work or not os.path.isdir(work):
            self._t7_summary.config(text=tr("prep.tab7.no_work_folder"), foreground="red")
            self._t7_launch_btn.config(state="disabled")
            self._t7_leftover.config(state="disabled")
            return

        chars = scan_characters(work)
        all_ok = bool(chars)
        mark_ok = tr("prep.tab7.mark_ok")
        mark_ng = tr("prep.tab7.mark_ng")
        mark_none = tr("prep.tab7.mark_none")
        for c in chars:
            name_ok = _is_valid_char_name(c.name)
            errors = list(c.errors)  # 名前の問題は scan_characters が先頭に入れている
            has_video = os.path.isfile(os.path.join(c.folder, f"{c.name}.mp4"))
            row_ok = c.is_valid
            if not row_ok:
                all_ok = False
            self._t7_tree.insert("", "end", values=(
                c.name,
                mark_ok if name_ok else mark_ng,
                mark_ok if c.image_path else mark_ng,
                mark_ok if c.face_info_path else mark_ng,
                mark_ok if c.mouth_dir else mark_ng,
                mark_ok if has_video else mark_none,
                " / ".join(errors) if errors else tr("prep.tab7.status_ok"),
            ))

        # ルート直下の未処理ファイル
        leftover_lines = []
        root_images = _find_images(work)
        if root_images:
            leftover_lines.append(tr(
                "prep.tab7.leftover_images",
                names=", ".join(os.path.basename(p) for p in root_images)))
        stray_jsons = [f for f in sorted(os.listdir(work))
                       if f.endswith("_face_info.json")
                       and os.path.isfile(os.path.join(work, f))]
        if stray_jsons:
            leftover_lines.append(tr("prep.tab7.leftover_faceinfo",
                                     names=", ".join(stray_jsons)))

        self._t7_leftover.insert(
            "1.0", "\n".join(leftover_lines) if leftover_lines else tr("prep.tab7.leftover_none"))
        self._t7_leftover.config(state="disabled")

        if root_images:
            all_ok = False

        if not chars:
            self._t7_summary.config(text=tr("prep.tab7.no_chars"), foreground="red")
        elif all_ok:
            self._t7_summary.config(
                text=tr("prep.tab7.all_ok", n=len(chars)), foreground="green")
        else:
            ng = sum(1 for c in chars if not c.is_valid)
            self._t7_summary.config(
                text=tr("prep.tab7.some_ng", n=len(chars), ng=ng), foreground="red")

        self._t7_launch_btn.config(state="normal" if all_ok else "disabled")

    def _t7_launch(self):
        script = os.path.join(HERE, "video_generator.py")
        try:
            subprocess.Popen([sys.executable, script], cwd=HERE)
        except Exception as e:
            messagebox.showerror(tr("prep.msgbox.title_error"),
                                 tr("prep.msgbox.t7_launch_failed", error=e))
            return
        # 準備は完了したので asset_preparer は閉じる
        self.destroy()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app = BatchPreparerApp()
    _warn_if_catalog_broken()
    app.mainloop()
    return 0


def _warn_if_catalog_broken():
    """翻訳カタログが読めていない場合に警告する。

    この状態では tr() が生のキー名を返すため、メッセージは日英ハードコード。
    """
    errors = catalog_load_errors()
    if errors:
        messagebox.showwarning(
            "Translation files / 翻訳ファイル",
            "翻訳ファイルの読み込みに失敗しました。UIがキー名のまま表示されます。\n"
            "locales フォルダを含めてリポジトリを再取得してください。\n\n"
            "Failed to load translation files. UI text will appear as raw keys.\n"
            "Please re-download the repository including the locales folder.\n\n"
            + "\n".join(errors))


if __name__ == "__main__":
    raise SystemExit(main())
