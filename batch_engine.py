#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_engine.py

MotionPNGCreator for ArtificialGirlfriend バッチ処理エンジン（非GUI）。

Phase1: Vidu API で動画生成 → ダウンロード
Phase2: SAM3 トラッキング＋口消し WebM 生成

GUIから BatchEngine をインスタンス化して start() で実行。
コールバックで進捗・エラーを通知する。
"""

from __future__ import annotations

import gc
import json
import os
import queue
import shutil
import sys
import tempfile
import threading
import time
import unicodedata
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
from i18n import tr
from vidu_api import (
    IMAGE_EXTS,
    ViduAPIError,
    image_to_base64_uri,
    vidu_download,
    vidu_get,
    vidu_post,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

JOB_STATE_PENDING = "pending"
JOB_STATE_PHASE1_SUBMITTED = "phase1_submitted"
JOB_STATE_PHASE1_DONE = "phase1_done"
JOB_STATE_PHASE2_RUNNING = "phase2_running"
JOB_STATE_COMPLETED = "completed"
JOB_STATE_FAILED = "failed"

BATCH_STATUS_IDLE = "idle"
BATCH_STATUS_RUNNING = "running"
BATCH_STATUS_PAUSED = "paused"
BATCH_STATUS_STOPPED = "stopped"
BATCH_STATUS_COMPLETED = "completed"

# scan_characters が無視するフォルダ名（asset_preparer 側と共通の定義）
# "backup" は旧バージョンが作成していた元画像バックアップ先（現在は作成しないが、
# 既存ワークフォルダやユーザーが手動で置いたコピーをキャラクター扱いしないよう除外を維持）
EXCLUDED_DIRS = {"prompts", "miss_data", ".git", "__pycache__", "backup"}

# Phase2 品質ゲート: トラッキング平均信頼度がこの値未満なら失敗扱い
MIN_CONF_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Pricing tables
# ---------------------------------------------------------------------------

# (base_credit_for_1st_sec, per_sec_credit_for_2nd_onward)
# For Q3 models: base=0, per_sec=linear_rate (every second same)
# For Q1 / v2.0: fixed price (use special key)

PRICING: dict[str, dict[str, dict]] = {
    "viduq3-pro": {
        "1080p": {"type": "linear", "per_sec": 30},
        "720p":  {"type": "linear", "per_sec": 25},
        "540p":  {"type": "linear", "per_sec": 10},
    },
    "viduq3-turbo": {
        "1080p": {"type": "linear", "per_sec": 14},
        "720p":  {"type": "linear", "per_sec": 12},
        "540p":  {"type": "linear", "per_sec": 8},
    },
    "viduq2-pro": {
        "1080p": {"type": "base_plus", "base": 55, "per_sec": 15},
        "720p":  {"type": "base_plus", "base": 15, "per_sec": 10},
        "540p":  {"type": "base_plus_sec2", "base": 8, "sec2": 10, "per_sec": 5},
    },
    "viduq2-pro-fast": {
        "1080p": {"type": "base_plus", "base": 16, "per_sec": 4},
        "720p":  {"type": "base_plus", "base": 8,  "per_sec": 2},
        # 540p not supported
    },
    "viduq2-turbo": {
        "1080p": {"type": "base_plus", "base": 35, "per_sec": 10},
        "720p":  {"type": "base_plus_sec2", "base": 8, "sec2": 10, "per_sec": 10},
        "540p":  {"type": "base_plus", "base": 6,  "per_sec": 2},
    },
    "viduq1": {
        "1080p": {"type": "fixed", "credits": 80},
    },
    "viduq1-classic": {
        "1080p": {"type": "fixed", "credits": 80},
    },
    "vidu2.0": {
        "360p_4s":  {"type": "fixed", "credits": 20},
        "720p_4s":  {"type": "fixed", "credits": 40},
        "1080p_4s": {"type": "fixed", "credits": 100},
        "720p_8s":  {"type": "fixed", "credits": 100},
    },
}


def calc_credits(model: str, resolution: str, duration: int) -> int:
    """1本あたりの推定クレジットを算出。"""
    if model not in PRICING:
        return 0
    model_pricing = PRICING[model]

    # vidu2.0 special handling
    if model == "vidu2.0":
        key = f"{resolution}_{duration}s"
        if key not in model_pricing:
            return 0
        return model_pricing[key]["credits"]

    if resolution not in model_pricing:
        return 0

    info = model_pricing[resolution]
    ptype = info["type"]

    if ptype == "fixed":
        credits = info["credits"]
    elif ptype == "linear":
        credits = info["per_sec"] * duration
    elif ptype == "base_plus":
        credits = info["base"] + info["per_sec"] * max(0, duration - 1)
    elif ptype == "base_plus_sec2":
        # base + 2秒目はsec2 + 3秒目以降はper_sec (Q2-pro 540p / Q2-turbo 720p)
        if duration == 1:
            credits = info["base"]
        elif duration == 2:
            credits = info["base"] + info["sec2"]
        else:
            credits = info["base"] + info["sec2"] + info["per_sec"] * (duration - 2)
    else:
        credits = 0

    return credits


def get_duration_range(model: str) -> tuple[int, int]:
    """モデルのduration範囲を返す。"""
    if model.startswith("viduq3"):
        return (1, 16)
    if model.startswith("viduq2") or model == "vidu2.0":
        return (1, 8)
    if model.startswith("viduq1"):
        return (5, 5)
    return (1, 8)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class JobInfo:
    """1本の動画生成ジョブ。"""
    id: str
    character: str
    prompt_list: str
    prompt_index: int
    prompt_text: str
    state: str = JOB_STATE_PENDING
    vidu_task_id: Optional[str] = None
    phase1_retries: int = 0
    phase2_retries: int = 0
    original_path: Optional[str] = None
    output_webm: Optional[str] = None
    # 旧バージョンの batch_state.json 互換のため残す（新規エラーは error_key に記録）
    error_message: Optional[str] = None
    error_key: Optional[str] = None
    error_params: dict = field(default_factory=dict)

    def set_error(self, key: str, **params):
        """エラーを言語非依存のキー+パラメータで記録する（batch_state.json に永続化される）。"""
        self.error_key = key
        self.error_params = {k: str(v) for k, v in params.items()}
        self.error_message = None

    def clear_error(self):
        """エラー記録を消去する（リトライ再開時など）。"""
        self.error_key = None
        self.error_params = {}
        self.error_message = None

    def error_text(self) -> str:
        """表示用のエラー文字列（現在のUI言語で翻訳）。"""
        if self.error_key:
            return tr(self.error_key, **self.error_params)
        return self.error_message or ""

    @property
    def has_error(self) -> bool:
        return bool(self.error_key or self.error_message)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "JobInfo":
        return JobInfo(**{k: v for k, v in d.items() if k in JobInfo.__dataclass_fields__})


@dataclass
class BatchConfig:
    """バッチ実行設定。"""
    api_key: str = ""
    model: str = "viduq2-pro-fast"
    duration: int = 5
    resolution: str = "720p"
    bg_tolerance: float = 65.0
    smooth_cutoff: float = 3.0
    gpu_rest_count: int = 10
    gpu_rest_seconds: int = 60
    poll_interval: int = 10

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "BatchConfig":
        return BatchConfig(**{k: v for k, v in d.items() if k in BatchConfig.__dataclass_fields__})


@dataclass
class BatchState:
    """バッチ全体の永続化状態。"""
    version: int = 1
    config_snapshot: Optional[dict] = None
    jobs: list[dict] = field(default_factory=list)
    status: str = BATCH_STATUS_IDLE

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "BatchState":
        return BatchState(
            version=d.get("version", 1),
            config_snapshot=d.get("config_snapshot"),
            jobs=d.get("jobs", []),
            status=d.get("status", BATCH_STATUS_IDLE),
        )


# ---------------------------------------------------------------------------
# Material validation
# ---------------------------------------------------------------------------

@dataclass
class CharacterInfo:
    """キャラクター素材情報。"""
    name: str
    folder: str
    image_path: Optional[str] = None
    face_info_path: Optional[str] = None
    mouth_dir: Optional[str] = None
    sprite_size: Optional[tuple[int, int]] = None  # 口PNGのキャンバス (幅, 高さ)
    errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0


def path_ansi_safe(path: str) -> bool:
    """パスがANSIコードページで表現可能かを検査する。

    OpenCVの動画I/O（VideoCapture/VideoWriter）はC++層でANSIコードページ変換を
    行うため、表現できない文字を含むパスでは開けない。GUIはワークフォルダ選択時に
    これを確認してユーザーへ警告する。
    """
    if sys.platform != "win32":
        return True
    try:
        path.encode("mbcs")
        return True
    except UnicodeEncodeError:
        return False


# ---------------------------------------------------------------------------
# Character name validation
# ---------------------------------------------------------------------------

# キャラクター名（= 画像ファイル名の拡張子なし）は、生成物すべてのパス成分になり、
# AG側では Motion フォルダ名としてそのまま使われる。日本語を含む非ASCIIは可。
# 弾くのは「ファイル名に使えない文字」「AGプレイヤーのURL経路で壊れる文字（# %）」
# 「Windowsの予約名」「先頭末尾のスペース・末尾のドット」「長すぎる名前」
# 「このPCの標準文字コードで表せない文字（OpenCVの動画I/Oが開けない）」。
CHAR_NAME_FORBIDDEN_CHARS = '\\/:*?"<>|#%'
CHAR_NAME_MAX_LEN = 60
_WINDOWS_RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL"} | {
    f"{base}{i}" for base in ("COM", "LPT") for i in range(1, 10)}


def normalize_char_name(name: str) -> str:
    """フォルダ・ファイル名に使う前のNFC正規化（HFS+由来の分解形が混ざっても揃える）。"""
    return unicodedata.normalize("NFC", name)


def char_name_problem(name: str) -> str:
    """キャラクター名の問題を locale キーで返す。問題なければ ""。"""
    if not name or not name.strip():
        return "engine.name.empty"
    name = normalize_char_name(name)  # 分解形は③で合成形に揃えるので、判定も合成形で行う
    if any(ch in CHAR_NAME_FORBIDDEN_CHARS or ord(ch) < 32 for ch in name):
        return "engine.name.forbidden_char"
    if name[0] == " " or name[-1] in " .":
        return "engine.name.edge"
    if name.split(".")[0].upper() in _WINDOWS_RESERVED_NAMES:
        return "engine.name.reserved"
    if len(name) > CHAR_NAME_MAX_LEN:
        return "engine.name.too_long"
    if not path_ansi_safe(name):
        return "engine.name.not_ansi"
    return ""


def char_name_error(name: str) -> str:
    """キャラクター名の問題を表示用の文言で返す。問題なければ ""。"""
    key = char_name_problem(name)
    if not key:
        return ""
    return tr(key, max=CHAR_NAME_MAX_LEN) if key == "engine.name.too_long" else tr(key)


def scan_characters(work_folder: str) -> list[CharacterInfo]:
    """WorkFolder直下のキャラクターフォルダをスキャンし、素材チェック。"""
    work = Path(work_folder)
    if not work.is_dir():
        return []

    characters: list[CharacterInfo] = []

    for entry in sorted(work.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name.startswith(".") or entry.name.lower() in EXCLUDED_DIRS:
            continue

        info = CharacterInfo(name=entry.name, folder=str(entry))

        # --- キャラクター名（= フォルダ名）の検査 ---
        name_err = char_name_error(entry.name)
        if name_err:
            info.errors.append(name_err)

        # --- ソース画像の検出 ---
        images = [
            f for f in entry.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS
        ]
        if len(images) == 0:
            info.errors.append(tr("engine.err.source_image_missing"))
        elif len(images) > 1:
            info.errors.append(tr("engine.scan.source_image_multiple", n=len(images)))
        else:
            info.image_path = str(images[0])

        # --- face_info.json の検出 ---
        # {CharacterName}_face_info.json を優先、なければ face_info.json
        fi_named = entry / f"{entry.name}_face_info.json"
        fi_plain = entry / "face_info.json"
        if fi_named.is_file():
            info.face_info_path = str(fi_named)
        elif fi_plain.is_file():
            info.face_info_path = str(fi_plain)
        else:
            info.errors.append(tr("engine.err.face_info_missing"))

        # --- mouth/ フォルダ ---
        mouth_dir = entry / "mouth"
        if mouth_dir.is_dir():
            # open.png / closed.png / half.png (または mouth_open.png 等)
            sprite_patterns = [
                (["open.png", "mouth_open.png"], "open"),
                (["closed.png", "mouth_closed.png"], "closed"),
                (["half.png", "mouth_half.png"], "half"),
            ]
            missing = []
            for candidates, label in sprite_patterns:
                if not any((mouth_dir / c).is_file() for c in candidates):
                    missing.append(label)
            if missing:
                info.errors.append(tr("engine.scan.mouth_missing_files", files=", ".join(missing)))
            else:
                info.mouth_dir = str(mouth_dir)
                # 3枚のキャンバスサイズは同一が必須（位置データの quad はこのサイズを
                # 基準に書く。準備ツールの出力は必ず同一で、手で差し替えた場合だけ
                # 不一致になりうる）
                from track_utils import sprite_canvas_size
                try:
                    info.sprite_size = sprite_canvas_size(str(mouth_dir))
                except ValueError as e:
                    info.errors.append(tr("engine.scan.mouth_size_mismatch", detail=str(e)))
        else:
            info.errors.append(tr("engine.scan.mouth_dir_missing"))

        characters.append(info)

    return characters


def load_prompt_list(path: str) -> list[str]:
    """プロンプトリストJSONを読み込み。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [s for s in data if isinstance(s, str) and s.strip()]
        return []
    except Exception as e:
        # 構文エラー等を無言で0件扱いにしない（挙動は維持しつつ警告を出す）
        print(tr("engine.log.prompt_list_load_failed", path=path, error=e))
        return []


def save_prompt_list(path: str, prompts: list[str]):
    """プロンプトリストJSONを保存。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    clean = [s for s in prompts if s.strip()]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Atomic JSON write
# ---------------------------------------------------------------------------

def _atomic_json_write(path: str, data: dict):
    """アトミックにJSONを書き込む（tempfile→rename）。"""
    dir_path = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        # Windows: 上書き rename は os.replace で可
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# BatchEngine
# ---------------------------------------------------------------------------

class BatchEngine:
    """バッチ処理エンジン。Phase1(API)とPhase2(GPU)をパイプラインで実行。"""

    def __init__(self):
        self._jobs: list[JobInfo] = []
        self._config = BatchConfig()
        self._status = BATCH_STATUS_IDLE
        self._work_folder = ""
        self._state_path = ""

        # キャラクター情報キャッシュ (start時に構築)
        self._char_cache: dict[str, CharacterInfo] = {}
        # base64画像キャッシュ: {char_name: data_uri}
        self._image_b64_cache: dict[str, str] = {}
        # 口消し塗り色キャッシュ: {char_name: BGR ndarray | None(=フォールバック)}
        # Phase2スレッドからのみアクセスするためロック不要
        self._fill_color_cache: dict[str, Optional[np.ndarray]] = {}
        # 口消し楕円の幾何キャッシュ: {char_name: EraseGeometry | None}
        # （口素材 mouth/ の実寸基準。Phase2スレッドからのみアクセス）
        self._erase_geom_cache: dict[str, object] = {}

        # キュー: Phase1完了 → Phase2投入
        self._phase2_queue: queue.Queue[Optional[JobInfo]] = queue.Queue()

        # スレッド
        self._phase1_thread: Optional[threading.Thread] = None
        self._phase2_thread: Optional[threading.Thread] = None

        # 制御
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()  # set=running, clear=paused
        self._pause_event.set()
        self._lock = threading.Lock()
        # 世代カウンタ: 再start毎にインクリメントし、旧世代スレッドを無効化
        self._generation = 0
        # 料金不明警告を一度だけ出すためのフラグ
        self._price_warned = False

        # Phase1の現在の活動（キューモニター表示用の翻訳済みテキスト）。
        # タスク投入はAPI残高確認＋数MBのアップロードを直列で行うため、
        # 開始直後はスロットが埋まるまで十数秒かかる。その間の
        # 「何も起きていないように見える」状態を防ぐためGUIがこれを表示する
        self.phase1_activity: str = ""

        # コールバック
        self.on_progress: Optional[Callable[[str, dict], None]] = None
        self.on_job_update: Optional[Callable[[JobInfo], None]] = None
        self.on_error: Optional[Callable[[str, str], None]] = None
        self.on_auto_pause: Optional[Callable[[str], None]] = None
        self.on_batch_complete: Optional[Callable[[], None]] = None

    # ================================================================
    # Properties
    # ================================================================

    @property
    def status(self) -> str:
        return self._status

    @property
    def jobs(self) -> list[JobInfo]:
        return list(self._jobs)

    @property
    def config(self) -> BatchConfig:
        return self._config

    # ================================================================
    # Job building
    # ================================================================

    def build_jobs(
        self,
        characters: list[CharacterInfo],
        prompt_assignments: dict[str, list[str]],
        prompts_folder: str,
    ) -> list[JobInfo]:
        """キャラクター × プロンプト から全ジョブリストを構築。

        Args:
            characters: 有効なキャラクターリスト
            prompt_assignments: {character_name: [list_name, ...]}
            prompts_folder: prompts/ フォルダのパス
        Returns:
            構築されたジョブリスト
        """
        jobs: list[JobInfo] = []
        for char in characters:
            if not char.is_valid:
                continue
            lists = prompt_assignments.get(char.name, [])
            for list_name in lists:
                list_path = os.path.join(prompts_folder, f"{list_name}.json")
                prompts = load_prompt_list(list_path)
                for idx, prompt_text in enumerate(prompts):
                    job_id = f"{char.name}_{list_name}_{idx+1:02d}"
                    jobs.append(JobInfo(
                        id=job_id,
                        character=char.name,
                        prompt_list=list_name,
                        prompt_index=idx,
                        prompt_text=prompt_text,
                    ))
        return jobs

    # ================================================================
    # State persistence
    # ================================================================

    def save_state(self):
        """現在のバッチ状態をJSONに保存。

        APIキーは平文で残さないよう snapshot から除外する
        （resume時に resume_from_state(api_key=...) で再注入する）。
        """
        if not self._state_path:
            return
        with self._lock:
            config_snapshot = self._config.to_dict()
            config_snapshot.pop("api_key", None)
            state = BatchState(
                version=1,
                config_snapshot=config_snapshot,
                jobs=[j.to_dict() for j in self._jobs],
                status=self._status,
            )
        _atomic_json_write(self._state_path, state.to_dict())

    def load_state(self, path: str) -> bool:
        """保存済みバッチ状態を読み込み。"""
        if not os.path.isfile(path):
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            bs = BatchState.from_dict(data)
            if bs.config_snapshot:
                self._config = BatchConfig.from_dict(bs.config_snapshot)
            self._jobs = [JobInfo.from_dict(j) for j in bs.jobs]
            self._status = bs.status
            return True
        except Exception:
            return False

    # ================================================================
    # Preflight check
    # ================================================================

    def check_api_credits(self) -> dict:
        """API残高を確認。 {credit_remain, concurrency_limit, current_concurrency}"""
        try:
            result = vidu_get(self._config.api_key, "credits")
            remains = result.get("remains", [])
            if remains:
                r = remains[0]
                return {
                    "credit_remain": r.get("credit_remain", 0),
                    "concurrency_limit": r.get("concurrency_limit", 5),
                    "current_concurrency": r.get("current_concurrency", 0),
                }
        except Exception as e:
            return {"error": str(e)}
        return {"credit_remain": 0, "concurrency_limit": 5, "current_concurrency": 0}

    # ================================================================
    # Start / Pause / Stop
    # ================================================================

    def _prepare_restart(self) -> tuple:
        """再start前の後始末。旧スレッドへの停止要求と世代更新のみ行う。

        stop()→再start でキューに残った None や旧ジョブを新スレッドが
        消費してハング/誤動作するのを防ぐため、Phase2キューは世代ごとに
        新しい Queue を作り直す（旧スレッドは自分が捕捉した旧キューにしか
        触れないため、排水は不要）。旧スレッドの join はGUIスレッドを
        ブロックしないよう、新スレッド側（_phase*_thread_main）で行う。

        Returns:
            (旧Phase1スレッド, 旧Phase2スレッド)。生きていないものは None。
        """
        # 旧スレッドに停止を要求（pause中でも進めるように pause も解除）
        self._stop_event.set()
        self._pause_event.set()
        self._generation += 1

        prev_p1 = self._phase1_thread if (
            self._phase1_thread is not None and self._phase1_thread.is_alive()) else None
        prev_p2 = self._phase2_thread if (
            self._phase2_thread is not None and self._phase2_thread.is_alive()) else None
        self._phase1_thread = None
        self._phase2_thread = None

        # Phase2キューを世代ごとに作り直す（旧世代の残骸と完全に分離）
        self._phase2_queue = queue.Queue()

        # base64画像キャッシュをクリア（素材差し替えに追従）
        self._image_b64_cache.clear()
        # 塗り色キャッシュもクリア（立ち絵差し替えに追従）
        self._fill_color_cache.clear()
        # 消去幾何キャッシュもクリア（口素材差し替えに追従）
        self._erase_geom_cache.clear()

        self._stop_event.clear()
        self._pause_event.set()
        return prev_p1, prev_p2

    def _join_prev_thread(self, prev: Optional[threading.Thread]):
        """旧世代スレッドの終了を待つ（新ワーカースレッド内で実行される）。

        旧スレッドがGPUジョブ処理中の場合は完了まで数分かかりうるが、
        GUIスレッドはブロックしない。二重SAM3ロードによるVRAM枯渇や
        同一出力ファイルへの二重書き込みを防ぐため、完了を待ってから
        新世代の処理を開始する。
        """
        if prev is None or not prev.is_alive():
            return
        if self.on_error:
            self.on_error("engine", tr("engine.log.waiting_prev_gen"))
        prev.join()

    def _phase1_thread_main(self, gen: int, prev: Optional[threading.Thread]):
        self._join_prev_thread(prev)
        if gen != self._generation:
            return
        self._phase1_loop(gen)

    def _phase2_thread_main(self, gen: int, q: "queue.Queue",
                            prev: Optional[threading.Thread]):
        self._join_prev_thread(prev)
        if gen != self._generation:
            return
        self._phase2_loop(gen, q)

    def start(
        self,
        work_folder: str,
        jobs: list[JobInfo],
        config: BatchConfig,
    ):
        """バッチ処理を開始。"""
        prev_p1, prev_p2 = self._prepare_restart()

        self._work_folder = work_folder
        self._state_path = os.path.join(work_folder, "batch_state.json")
        with self._lock:
            self._config = config
            self._jobs = jobs
            self._status = BATCH_STATUS_RUNNING

        # キャラクター情報をキャッシュ
        chars = scan_characters(work_folder)
        self._char_cache = {c.name: c for c in chars}

        # prompts/ フォルダ作成
        os.makedirs(os.path.join(work_folder, "prompts"), exist_ok=True)

        # キャラクターの original/ フォルダ作成
        for job in self._jobs:
            char_folder = os.path.join(work_folder, job.character)
            os.makedirs(os.path.join(char_folder, "original"), exist_ok=True)

        self.save_state()

        # スレッド起動
        self.phase1_activity = tr("engine.activity.checking")
        self._phase1_thread = threading.Thread(
            target=self._phase1_thread_main, args=(self._generation, prev_p1),
            daemon=True, name="Phase1-API")
        self._phase2_thread = threading.Thread(
            target=self._phase2_thread_main,
            args=(self._generation, self._phase2_queue, prev_p2),
            daemon=True, name="Phase2-GPU")
        self._phase1_thread.start()
        self._phase2_thread.start()

    def resume_from_state(self, work_folder: str, api_key: str = ""):
        """batch_state.json から復帰して再開。

        Args:
            api_key: 再注入するAPIキー（stateファイルには保存されないため必須）
        """
        prev_p1, prev_p2 = self._prepare_restart()

        self._work_folder = work_folder
        self._state_path = os.path.join(work_folder, "batch_state.json")
        if not self.load_state(self._state_path):
            raise RuntimeError(tr("engine.log.state_load_failed"))
        if api_key:
            self._config.api_key = api_key

        # キャラクター情報をキャッシュ
        chars = scan_characters(work_folder)
        self._char_cache = {c.name: c for c in chars}

        # phase1_submitted のジョブを pending に戻す（ポーリング再開で回収試行）
        # ただし vidu_task_id がある場合はポーリングで回収できる可能性
        # → phase1_submitted のまま残す

        # Phase1完了だがPhase2未処理のジョブをキューに投入。
        # クラッシュ時に phase2_running のまま残ったジョブも再投入しないと、
        # 誰にも処理されないまま「残作業」として数えられ続けてハングする。
        with self._lock:
            self._status = BATCH_STATUS_RUNNING
            for job in self._jobs:
                if job.state == JOB_STATE_PHASE2_RUNNING:
                    job.state = JOB_STATE_PHASE1_DONE
                if job.state == JOB_STATE_PHASE1_DONE:
                    self._phase2_queue.put(job)

        self.save_state()

        self.phase1_activity = tr("engine.activity.checking")
        self._phase1_thread = threading.Thread(
            target=self._phase1_thread_main, args=(self._generation, prev_p1),
            daemon=True, name="Phase1-API")
        self._phase2_thread = threading.Thread(
            target=self._phase2_thread_main,
            args=(self._generation, self._phase2_queue, prev_p2),
            daemon=True, name="Phase2-GPU")
        self._phase1_thread.start()
        self._phase2_thread.start()

    def start_phase2_only(
        self,
        work_folder: str,
        jobs: list[JobInfo],
        config: BatchConfig,
    ):
        """Phase2のみ再実行。Phase1(API)はスキップし、既存original動画を処理。"""
        _prev_p1, prev_p2 = self._prepare_restart()

        self._work_folder = work_folder
        self._state_path = os.path.join(work_folder, "batch_state.json")

        # キャラクター情報をキャッシュ
        chars = scan_characters(work_folder)
        self._char_cache = {c.name: c for c in chars}

        # 全ジョブをPHASE1_DONEとしてPhase2キューに投入
        with self._lock:
            self._config = config
            self._jobs = jobs
            self._status = BATCH_STATUS_RUNNING
            for job in self._jobs:
                job.state = JOB_STATE_PHASE1_DONE
                self._phase2_queue.put(job)

        self.save_state()

        # Phase2スレッドのみ起動（Phase1はスキップ）
        self._phase1_thread = None
        self._phase2_thread = threading.Thread(
            target=self._phase2_thread_main,
            args=(self._generation, self._phase2_queue, prev_p2),
            daemon=True, name="Phase2-GPU")
        self._phase2_thread.start()

    def pause(self):
        """一時停止。"""
        self._pause_event.clear()
        with self._lock:
            self._status = BATCH_STATUS_PAUSED
        self.save_state()

    def resume(self):
        """再開。"""
        with self._lock:
            self._status = BATCH_STATUS_RUNNING
        self._pause_event.set()
        self.save_state()

    def stop(self):
        """中止。"""
        self._stop_event.set()
        self._pause_event.set()  # pause解除してスレッドが終了できるように
        with self._lock:
            self._status = BATCH_STATUS_STOPPED
        self.save_state()
        # Phase2キューに終了シグナル
        self._phase2_queue.put(None)

    # ================================================================
    # Phase 1: Vidu API
    # ================================================================

    def _phase1_loop(self, gen: int):
        """Phase1: API呼び出し→ポーリング→ダウンロードのメインループ。

        gen: 起動時の世代番号。再startで世代が進んだら旧スレッドは終了する。
        """
        consecutive_failures = 0

        while not self._stop_event.is_set() and gen == self._generation:
            # pause チェック
            self._pause_event.wait()
            if self._stop_event.is_set() or gen != self._generation:
                break

            try:
                # --- 1. アクティブタスクのポーリング ---
                with self._lock:
                    active = [j for j in self._jobs
                              if j.state == JOB_STATE_PHASE1_SUBMITTED]

                for job in active:
                    if self._stop_event.is_set() or gen != self._generation:
                        break
                    self._poll_phase1_task(job)

                    # ダウンロード成功ならPhase2キューへ
                    # （旧世代スレッドは新世代のキューに投入しない）
                    if job.state == JOB_STATE_PHASE1_DONE:
                        if gen == self._generation:
                            self._phase2_queue.put(job)
                        consecutive_failures = 0
                    elif job.state == JOB_STATE_FAILED:
                        consecutive_failures += 1

                # 旧世代スレッドは自動pause・投入・保存で新世代を汚染しない
                if self._stop_event.is_set() or gen != self._generation:
                    break

                # --- 2. 連続失敗チェック ---
                if consecutive_failures >= 3:
                    with self._lock:
                        self._status = BATCH_STATUS_PAUSED
                    self._pause_event.clear()
                    self.save_state()
                    if self.on_auto_pause:
                        self.on_auto_pause(tr("engine.log.api_consecutive_failures"))
                    consecutive_failures = 0
                    self._pause_event.wait()
                    if self._stop_event.is_set() or gen != self._generation:
                        break
                    continue

                # --- 3. 空きスロットに新規投入 ---
                with self._lock:
                    pending = [j for j in self._jobs if j.state == JOB_STATE_PENDING]

                if pending:
                    try:
                        self.phase1_activity = tr("engine.activity.checking")
                        credits_info = vidu_get(self._config.api_key, "credits")
                        remains = credits_info.get("remains", [{}])
                        r = remains[0] if remains else {}
                        concurrency_limit = r.get("concurrency_limit", 5)
                        current_concurrency = r.get("current_concurrency", 0)

                        # クレジット残高チェック
                        credit_remain = r.get("credit_remain", 0)
                        per_job = calc_credits(
                            self._config.model, self._config.resolution,
                            self._config.duration)
                        if per_job <= 0 and not self._price_warned:
                            # 料金不明の設定では残高チェックが機能しない旨を一度だけ通知
                            self._price_warned = True
                            if self.on_error:
                                self.on_error("phase1", tr("engine.warn.price_unknown"))
                        if per_job > 0 and credit_remain < per_job:
                            with self._lock:
                                self._status = BATCH_STATUS_PAUSED
                            self._pause_event.clear()
                            self.phase1_activity = ""
                            self.save_state()
                            if self.on_auto_pause:
                                self.on_auto_pause(
                                    tr("engine.log.credit_low",
                                       remain=credit_remain, per_job=per_job))
                            self._pause_event.wait()
                            if self._stop_event.is_set() or gen != self._generation:
                                break
                            continue

                        free_slots = max(0, concurrency_limit - current_concurrency)

                        submitted_count = 0
                        for job in pending[:free_slots]:
                            if self._stop_event.is_set() or gen != self._generation:
                                break
                            self.phase1_activity = tr("engine.activity.submitting",
                                                      id=job.id)
                            success = self._submit_phase1_task(job)
                            if success:
                                submitted_count += 1
                            else:
                                # TooManyRequests の場合はループを抜けて待機
                                break

                    except Exception as e:
                        if self.on_error:
                            self.on_error("phase1", tr("engine.log.api_conn_error", error=e))
                        # credits取得失敗でも、ポーリングは継続する（次のループへ）
                    finally:
                        # 投入ラウンド終了。以降の待機中は活動表示を消す
                        self.phase1_activity = ""

                # --- 4. 全Phase1完了チェック ---
                with self._lock:
                    phase1_remaining = sum(
                        1 for j in self._jobs
                        if j.state in (JOB_STATE_PENDING, JOB_STATE_PHASE1_SUBMITTED)
                    )

                if phase1_remaining == 0:
                    # Phase1全完了 → 最終状態を保存してからPhase2の終了待ちへ
                    self.save_state()
                    break

                # --- 5. 状態保存＆待機 ---
                self.save_state()
                self._notify_progress()

                self._sleep_interruptible(self._config.poll_interval)

            except Exception as e:
                if self.on_error:
                    self.on_error("phase1", tr("engine.log.phase1_loop_exception", error=e))
                self._sleep_interruptible(10)

        # Phase1終了後、Phase2の完了を待つ（旧世代スレッドは何もせず終了）
        if gen == self._generation:
            self._wait_phase2_completion(gen)

    def _submit_phase1_task(self, job: JobInfo) -> bool:
        """Vidu APIにタスクを投入。"""
        char_info = self._char_cache.get(job.character)
        if not char_info or not char_info.image_path:
            with self._lock:
                job.set_error("engine.err.source_image_missing")
                job.state = JOB_STATE_FAILED
            if self.on_job_update:
                self.on_job_update(job)
            return False

        try:
            # base64キャッシュ（同じキャラクターの画像は1回だけエンコード）
            if job.character not in self._image_b64_cache:
                self._image_b64_cache[job.character] = image_to_base64_uri(
                    char_info.image_path)
            image_b64 = self._image_b64_cache[job.character]
            data = {
                "model": self._config.model,
                "images": [image_b64, image_b64],  # start=end (ループ動画)
                "prompt": job.prompt_text,
                "duration": self._config.duration,
                "resolution": self._config.resolution,
            }
            result = vidu_post(self._config.api_key, "start-end2video", data)
            with self._lock:
                job.vidu_task_id = result.get("task_id")
                job.state = JOB_STATE_PHASE1_SUBMITTED
            if self.on_job_update:
                self.on_job_update(job)
            return True

        except ViduAPIError as e:
            # TooManyRequests (429) → リトライカウントに含めない
            if e.err_code == "TooManyRequests" or e.status_code == 429:
                self._sleep_interruptible(5)
                return False
            # QuotaExceeded → スロット空き待ち
            if e.err_code == "QuotaExceeded":
                return False
            # CreditInsufficient → 即一時停止
            if e.err_code == "CreditInsufficient":
                with self._lock:
                    self._status = BATCH_STATUS_PAUSED
                self._pause_event.clear()
                self.save_state()
                if self.on_auto_pause:
                    self.on_auto_pause(tr("engine.log.credit_insufficient", error=e))
                return False

            max_retries = 3
            with self._lock:
                job.phase1_retries += 1
                failed = job.phase1_retries > max_retries
                if failed:
                    job.set_error("engine.err.phase1_failed_retries", retries=max_retries, error=e)
                    job.state = JOB_STATE_FAILED
            if failed:
                if self.on_job_update:
                    self.on_job_update(job)
            else:
                if self.on_error:
                    self.on_error("phase1",
                                  tr("engine.log.retry", id=job.id,
                                     n=job.phase1_retries, max=max_retries, error=e))
            return False

        except Exception as e:
            max_retries = 3
            with self._lock:
                job.phase1_retries += 1
                failed = job.phase1_retries > max_retries
                if failed:
                    job.set_error("engine.err.phase1_network", error=e)
                    job.state = JOB_STATE_FAILED
            if failed:
                if self.on_job_update:
                    self.on_job_update(job)
            else:
                if self.on_error:
                    self.on_error("phase1",
                                  tr("engine.log.network_retry", id=job.id,
                                     n=job.phase1_retries, max=max_retries, error=e))
            return False

    def _poll_phase1_task(self, job: JobInfo):
        """投入済みタスクの状態を確認し、成功ならダウンロード。"""
        if not job.vidu_task_id:
            with self._lock:
                job.state = JOB_STATE_PENDING
            return

        try:
            result = vidu_get(
                self._config.api_key,
                f"tasks/{job.vidu_task_id}/creations",
            )
            state = result.get("state", "")

            if state == "success":
                creations = result.get("creations", [])
                if creations:
                    video_url = creations[0].get("url", "")
                    if video_url:
                        self._download_phase1(job, video_url)
                        return
                # creationsが空 → 失敗扱い
                with self._lock:
                    job.set_error("engine.err.no_creations")
                    job.state = JOB_STATE_FAILED

            elif state == "failed":
                err_code = result.get("err_code", "unknown")
                max_retries = 3
                with self._lock:
                    job.phase1_retries += 1
                    if job.phase1_retries <= max_retries:
                        job.state = JOB_STATE_PENDING
                        job.vidu_task_id = None
                    else:
                        job.set_error("engine.err.vidu_api_failed", code=err_code)
                        job.state = JOB_STATE_FAILED

            # created / queueing / processing → 待機

        except Exception as e:
            if self.on_error:
                self.on_error("phase1", tr("engine.log.poll_failed", id=job.id, error=e))

        if self.on_job_update:
            self.on_job_update(job)

    def _download_phase1(self, job: JobInfo, video_url: str):
        """Phase1完了動画をダウンロード。"""
        char_folder = os.path.join(self._work_folder, job.character)
        original_dir = os.path.join(char_folder, "original")
        os.makedirs(original_dir, exist_ok=True)

        filename = f"{job.id}_original.mp4"
        save_path = os.path.join(original_dir, filename)

        try:
            vidu_download(video_url, save_path)
            with self._lock:
                job.original_path = save_path
                job.state = JOB_STATE_PHASE1_DONE
        except Exception as e:
            # 例外文字列には署名付き動画URLが含まれうるため、クラス名のみ記録
            err_name = type(e).__name__
            # URLが失効している可能性 → Phase1からやり直し
            max_retries = 3
            with self._lock:
                job.phase1_retries += 1
                failed = job.phase1_retries > max_retries
                if failed:
                    job.set_error("engine.err.download_failed_retries",
                                  retries=max_retries, error=err_name)
                    job.state = JOB_STATE_FAILED
                else:
                    # リトライ中はエラー表示を残さない
                    job.clear_error()
                    job.state = JOB_STATE_PENDING
                    job.vidu_task_id = None
            if not failed and self.on_error:
                self.on_error("phase1",
                              tr("engine.log.retry", id=job.id,
                                 n=job.phase1_retries, max=max_retries, error=err_name))

        if self.on_job_update:
            self.on_job_update(job)

    # ================================================================
    # Phase 2: GPU processing
    # ================================================================

    def _phase2_loop(self, gen: int, q: "queue.Queue"):
        """Phase2: SAM3トラッキング＋口消しのメインループ。

        gen: 起動時の世代番号。再startで世代が進んだら旧スレッドは終了する。
        q: この世代のPhase2キュー。self._phase2_queue ではなく必ず q を使う
           （再startで self._phase2_queue は新しいオブジェクトに差し替わるため、
           旧世代スレッドが新世代のキューを汚染しないように）。
        """
        # SAM3 遅延ロード
        detector = None
        processed_count = 0

        while not self._stop_event.is_set() and gen == self._generation:
            # pause チェック
            self._pause_event.wait()
            if self._stop_event.is_set() or gen != self._generation:
                break

            # キューからジョブ取得
            try:
                job = q.get(timeout=5)
            except queue.Empty:
                # Phase1全完了かつキュー空かつ未処理なし → 終了
                with self._lock:
                    phase1_remaining = sum(
                        1 for j in self._jobs
                        if j.state in (JOB_STATE_PENDING, JOB_STATE_PHASE1_SUBMITTED)
                    )
                    phase2_remaining = sum(
                        1 for j in self._jobs
                        if j.state in (JOB_STATE_PHASE1_DONE, JOB_STATE_PHASE2_RUNNING)
                    )
                if phase1_remaining == 0 and phase2_remaining == 0:
                    break
                continue

            if gen != self._generation:
                # 旧世代スレッド: 取得したものを自世代のキューに戻して終了
                if job is not None:
                    q.put(job)
                break

            if job is None:  # 終了シグナル
                break

            # 二重実行防止: 既に処理済み or 処理中ならスキップ
            if job.state != JOB_STATE_PHASE1_DONE:
                continue

            # SAM3ロード（初回のみ）
            if detector is None:
                try:
                    from sam3_mouth_detector import SAM3MouthDetector
                    detector = SAM3MouthDetector(device="auto")
                except Exception as e:
                    # SAM3が使えない場合はリトライしても回復しないため、
                    # Phase1側も含めてバッチ全体を停止する（Phase1だけ走り続けて
                    # 完了待ちが永久にブロックするのを防ぐ）
                    with self._lock:
                        job.set_error("engine.err.sam3_load_failed", error=e)
                        job.state = JOB_STATE_FAILED
                        self._status = BATCH_STATUS_STOPPED
                    if self.on_job_update:
                        self.on_job_update(job)
                    self._stop_event.set()
                    self._pause_event.set()
                    self.save_state()
                    if self.on_auto_pause:
                        self.on_auto_pause(tr("engine.log.sam3_stop", error=e))
                    elif self.on_error:
                        self.on_error("phase2", tr("engine.err.sam3_load_failed", error=e))
                    break

            # ジョブ処理
            with self._lock:
                job.state = JOB_STATE_PHASE2_RUNNING
            if self.on_job_update:
                self.on_job_update(job)
            self.save_state()

            success = self._process_phase2_job(job, detector)

            if gen != self._generation:
                # 処理中に再startされた旧世代スレッド: リトライ再キュー・
                # miss_data移動・state保存で新世代の状態を汚染せず即終了
                break

            if not success:
                # detector破棄 → VRAMを確実に解放してから再初期化
                if detector is not None:
                    del detector
                    detector = None
                gc.collect()
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass

                with self._lock:
                    retry = job.phase2_retries < 1
                    if retry:
                        job.phase2_retries += 1
                        job.state = JOB_STATE_PHASE1_DONE
                    else:
                        if not job.has_error:
                            job.set_error("engine.err.phase2_failed_after_retry")
                        job.state = JOB_STATE_FAILED

                if retry:
                    q.put(job)
                    # detector再初期化（次のループ冒頭で自動ロードされる）
                    # ここでは明示的にNoneのままにして、ループ冒頭のロード処理に委ねる
                else:
                    # 元動画はoriginal/に残す（再生成ページのfailed_onlyフィルタが
                    # metrics無し/低confで検出して再試行できる）
                    if self.on_job_update:
                        self.on_job_update(job)

            self.save_state()
            self._notify_progress()

            # GPU休憩
            processed_count += 1
            if processed_count % self._config.gpu_rest_count == 0:
                # SAM3 detector 再初期化（長時間稼働での状態劣化・VRAMリーク防止）
                if detector is not None:
                    del detector
                    detector = None
                gc.collect()
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                self._sleep_interruptible(self._config.gpu_rest_seconds)

        # 全完了チェック（旧世代スレッドは何もせず終了）
        if gen == self._generation:
            self._check_batch_complete()

    def _process_phase2_job(self, job: JobInfo, detector) -> bool:
        """1本のPhase2処理: トラッキング→口消し→NPZ/JSON保存。"""
        from eye_pipeline import (
            phase1_eye_tracking,
            phase2_mouth_erasure,
            extract_fill_color_from_image,
            compute_erase_geometry,
        )
        from convert_npz_to_json import convert_npz_to_json
        from track_utils import (
            calc_track_metrics,
            save_metrics_json,
            sprite_canvas_size,
            TRACK_FORMAT_VERSION,
            TRACK_SMOOTH_VERSION,
        )

        char_folder = os.path.join(self._work_folder, job.character)
        video_path = job.original_path

        if not video_path or not os.path.isfile(video_path):
            with self._lock:
                job.set_error("engine.err.video_missing", path=video_path)
            return False

        # face_info.json
        char_info = self._char_cache.get(job.character)
        if not char_info or not char_info.face_info_path:
            with self._lock:
                job.set_error("engine.err.face_info_missing")
            return False

        try:
            with open(char_info.face_info_path, "r", encoding="utf-8") as f:
                face_info = json.load(f)
        except Exception as e:
            with self._lock:
                job.set_error("engine.err.face_info_read_failed", error=e)
            return False

        # 口 PNG のキャンバスサイズ（quad の大きさの基準）。通常は走査時に取得済み
        sprite_size = char_info.sprite_size
        if sprite_size is None:
            try:
                sprite_size = sprite_canvas_size(
                    char_info.mouth_dir or os.path.join(char_folder, "mouth"))
            except ValueError as e:
                with self._lock:
                    job.set_error("engine.err.sprite_size_unavailable", error=e)
                return False

        # 口消し塗り色: 口消し済み立ち絵から取得（キャラ単位でキャッシュ）
        # 取得できない場合は None のままリングサンプリングにフォールバック
        if job.character in self._fill_color_cache:
            fill_color = self._fill_color_cache[job.character]
        else:
            fill_color, fc_reason = extract_fill_color_from_image(
                char_info.image_path, face_info)
            self._fill_color_cache[job.character] = fill_color
            if fill_color is not None:
                print(f"[phase2] {job.character}: fill color "
                      f"BGR={fill_color.astype(int).tolist()}")
            else:
                print(f"[phase2] {job.character}: fill color fallback ({fc_reason})")
                if self.on_error:
                    self.on_error("phase2", tr(
                        "engine.warn.fill_color_fallback",
                        char=job.character, reason=fc_reason))

        # 出力パス
        out_webm = os.path.join(char_folder, f"{job.id}.webm")
        out_track = os.path.join(char_folder, f"{job.id}.npz")
        with self._lock:
            job.output_webm = out_webm

        try:
            # --- Phase1: Eye Tracking ---
            track_stats: dict = {}
            quads, valid, confidence, video_w, video_h, fps, raw_valid = phase1_eye_tracking(
                video_path, face_info, detector, self._config.smooth_cutoff,
                sprite_size=sprite_size, stats=track_stats)

            # --- フォールバック検知 ---
            # SAM3が全フレーム検出失敗→静的座標フォールバック(conf=0.1)の場合、
            # 出力品質が保証できないため失敗扱いにしてdetector再初期化→リトライ
            mean_conf = float(np.mean(confidence))
            if mean_conf < MIN_CONF_THRESHOLD:
                with self._lock:
                    job.set_error("engine.err.sam3_eye_detect_failed", conf=f"{mean_conf:.3f}")
                if self.on_error:
                    self.on_error("phase2", f"{job.id}: {job.error_text()}")
                return False

            # --- NPZ保存 ---
            # quad は口 PNG 全体を写す枠（track_format = 2）。refSpriteSize = PNG のキャンバス
            np.savez_compressed(
                out_track,
                quad=quads,
                valid=valid,
                confidence=confidence,
                w=np.int64(video_w),
                h=np.int64(video_h),
                fps=np.float64(fps),
                ref_sprite_w=np.int64(sprite_size[0]),
                ref_sprite_h=np.int64(sprite_size[1]),
                calib_offset=np.array([0.0, 0.0], dtype=np.float64),
                calib_scale=np.float64(1.0),
                calib_rotation=np.float64(0.0),
                track_format=np.int64(TRACK_FORMAT_VERSION),
                track_smooth=np.int64(TRACK_SMOOTH_VERSION),
            )

            # calibrated copy
            calib_path = out_track.replace(".npz", "_calibrated.npz")
            if out_track != calib_path:
                shutil.copy2(out_track, calib_path)

            # metrics（valid_rate は補間前の生 valid で算出）
            metrics = calc_track_metrics(quads, raw_valid, confidence,
                                         min_conf=MIN_CONF_THRESHOLD)
            metrics.update(track_stats)  # 無効化・補間の内訳（稜が数字で確認するため）
            metrics_path = out_track.replace(".npz", "_metrics.json")
            save_metrics_json(metrics_path, metrics)

            # --- 消去楕円の幾何: 口素材 mouth/ の実寸基準（キャラ単位でキャッシュ）---
            # face_info の口bbox高さは使わない（口を閉じた立ち絵では線状になり、
            # 生成口の下半分が残る）。口素材が読めない場合は既定比率で警告。
            if job.character in self._erase_geom_cache:
                erase_geom = self._erase_geom_cache[job.character]
            else:
                mouth_dir = char_info.mouth_dir or os.path.join(char_folder, "mouth")
                erase_geom = compute_erase_geometry(
                    mouth_dir, face_info, video_w, video_h)
                self._erase_geom_cache[job.character] = erase_geom
                if erase_geom is None:
                    print(f"[phase2] {job.character}: erase geometry legacy "
                          f"(face_info has no eye_distance)")
                else:
                    print(f"[phase2] {job.character}: erase geometry "
                          f"{erase_geom.describe()}")
                    if erase_geom.source != "sprites" and self.on_error:
                        self.on_error("phase2", tr(
                            "engine.warn.erase_geom_fallback",
                            char=job.character, reason=erase_geom.reason))

            # --- Phase2: Mouth Erasure (WebM alpha) ---
            phase2_mouth_erasure(
                video_path, quads, valid,
                out_webm, video_w, video_h, fps,
                keep_audio=True,
                use_alpha=True,
                bg_tolerance=self._config.bg_tolerance,
                fill_color=fill_color,
                erase_geom=erase_geom,
            )

            # --- JSON変換 (Electronプレイヤー用) ---
            convert_npz_to_json(Path(calib_path), Path(char_folder))
            # rename mouth_track.json → {job.id}.json
            generic_json = os.path.join(char_folder, "mouth_track.json")
            target_json = os.path.join(char_folder, f"{job.id}.json")
            if os.path.isfile(generic_json):
                os.replace(generic_json, target_json)

            # torch cache clear
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

            with self._lock:
                job.state = JOB_STATE_COMPLETED
            if self.on_job_update:
                self.on_job_update(job)
            return True

        except Exception as e:
            with self._lock:
                job.set_error("engine.err.phase2_failed", error=e)
            if self.on_error:
                self.on_error("phase2", f"{job.id}: {e}")
            return False

    # ================================================================
    # Helpers
    # ================================================================

    def _wait_phase2_completion(self, gen: int):
        """Phase1完了後、Phase2の処理完了を待つ。"""
        # Phase2キューが空になり、全ジョブが完了/失敗するのを待つ
        # (ジョブはポーリング成功時に既にキューに投入済み。ここでは再投入しない)
        while not self._stop_event.is_set() and gen == self._generation:
            with self._lock:
                remaining = sum(
                    1 for j in self._jobs
                    if j.state in (JOB_STATE_PHASE1_DONE, JOB_STATE_PHASE2_RUNNING)
                )
            if remaining == 0:
                break
            time.sleep(2)

        # Phase2スレッドに終了シグナル（旧世代なら新世代のキューを汚さない）
        if gen == self._generation:
            self._phase2_queue.put(None)

    def _check_batch_complete(self):
        """全ジョブ完了チェック。"""
        with self._lock:
            all_done = all(
                j.state in (JOB_STATE_COMPLETED, JOB_STATE_FAILED)
                for j in self._jobs
            )
            completed = all_done and self._status == BATCH_STATUS_RUNNING
            if completed:
                self._status = BATCH_STATUS_COMPLETED
        if completed:
            self.save_state()
            if self.on_batch_complete:
                self.on_batch_complete()

    def _notify_progress(self):
        """進捗コールバックを発火。"""
        if not self.on_progress:
            return
        with self._lock:
            stats = {
                "total": len(self._jobs),
                "pending": sum(1 for j in self._jobs if j.state == JOB_STATE_PENDING),
                "phase1_submitted": sum(1 for j in self._jobs
                                        if j.state == JOB_STATE_PHASE1_SUBMITTED),
                "phase1_done": sum(1 for j in self._jobs
                                   if j.state == JOB_STATE_PHASE1_DONE),
                "phase2_running": sum(1 for j in self._jobs
                                      if j.state == JOB_STATE_PHASE2_RUNNING),
                "completed": sum(1 for j in self._jobs
                                 if j.state == JOB_STATE_COMPLETED),
                "failed": sum(1 for j in self._jobs if j.state == JOB_STATE_FAILED),
            }
        self.on_progress("update", stats)

    def _sleep_interruptible(self, seconds: float):
        """中断可能なsleep。"""
        self._stop_event.wait(timeout=seconds)

    # ================================================================
    # Statistics
    # ================================================================

    def get_stats(self) -> dict:
        """現在のジョブ統計を返す。"""
        with self._lock:
            return {
                "total": len(self._jobs),
                "pending": sum(1 for j in self._jobs if j.state == JOB_STATE_PENDING),
                "phase1_submitted": sum(1 for j in self._jobs
                                        if j.state == JOB_STATE_PHASE1_SUBMITTED),
                "phase1_done": sum(1 for j in self._jobs
                                   if j.state == JOB_STATE_PHASE1_DONE),
                "phase2_running": sum(1 for j in self._jobs
                                      if j.state == JOB_STATE_PHASE2_RUNNING),
                "completed": sum(1 for j in self._jobs
                                 if j.state == JOB_STATE_COMPLETED),
                "failed": sum(1 for j in self._jobs if j.state == JOB_STATE_FAILED),
                "status": self._status,
            }

    def get_phase1_active_jobs(self) -> list[JobInfo]:
        """Phase1でアクティブ（submitted）なジョブ一覧。"""
        with self._lock:
            return [j for j in self._jobs if j.state == JOB_STATE_PHASE1_SUBMITTED]
