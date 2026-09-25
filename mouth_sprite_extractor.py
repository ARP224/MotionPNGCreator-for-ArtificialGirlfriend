#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Origin / 出自（本ファイルは2系統の上流に由来します）:
#   - rotejin/MotionPNGTuber (MIT License, Copyright (c) 2025 rotejin)
#     https://github.com/rotejin/MotionPNGTuber
#   - kazuya-bros/MouthSpriteExtractor-SAM3 (AGPL-3.0, Copyright (c) 2026 kazuya-bros)
#     https://github.com/kazuya-bros/MouthSpriteExtractor-SAM3
#     ※上記は rotejin 氏の口スプライト抽出機能を SAM3 で再実装した派生版です。
# Modified by Ryo (ARP224) in 2026 (5口形状→3口形状化、モデルパス指定、GPU指定、
#                          バッチ処理連携、日英i18n対応ほか).
# License: AGPL-3.0-only. 原著作権表示とMITの許諾表示はリポジトリ直下の
#          NOTICE.md に保持されています。ライセンス全文は LICENSE を参照。
#
"""
mouth_sprite_extractor.py

SAM3を使用して動画から口スプライト（3種類のPNG）を自動抽出するコアモジュール。

機能:
1. SAM3のテキストプロンプト"mouth"で口を自動検出
2. 3種類の口形状を自動選別（open, closed, half）
3. 楕円マスク＋フェザーで透過PNG出力

License: AGPL-3.0-only (Ultralyticsライセンスに準拠)
Note: SAM3モデルの使用にはMeta SAM Licenseが適用されます。
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from sam3_mouth_detector import (
    SAM3MouthDetector,
    bbox_to_quad,
    feather_mask,
    is_sam3_available,
)
from i18n import tr


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MouthFrameInfo:
    """1フレームの口情報"""
    frame_idx: int
    quad: np.ndarray          # (4, 2) float32
    center: np.ndarray        # (2,) float32 - 口の中心座標
    width: float              # 口quadの幅
    height: float             # 口quadの高さ
    confidence: float         # 検出信頼度
    valid: bool               # 検出が有効か
    mask: Optional[np.ndarray] = None  # SAM3マスク (H, W) uint8
    bbox: Optional[Tuple[int, int, int, int]] = None  # 元のbbox
    category: str = ""        # 自動分類カテゴリ (open/closed/half)


@dataclass
class MouthTypeSelection:
    """3種類の口の選択結果"""
    open_idx: int             # 口を大きく開けたフレーム
    closed_idx: int           # 口を閉じたフレーム
    half_idx: int             # 半開きフレーム

    def as_dict(self) -> Dict[str, int]:
        return {
            "open": self.open_idx,
            "closed": self.closed_idx,
            "half": self.half_idx,
        }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def imwrite_jp(path: str, img: np.ndarray) -> bool:
    """cv2.imwrite の日本語パス対応版（imencode + tofile）。"""
    ext = os.path.splitext(path)[1]
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)
        return True
    return False


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def center_to_quad(center: np.ndarray, width: float, height: float) -> np.ndarray:
    """
    中心座標と幅・高さからquadを生成する。

    すべてのフレームで統一サイズのquadを使用するために、
    各フレームの口中心から統一サイズのquadを作成する。

    Args:
        center: (2,) float32 - 口の中心座標 (x, y)
        width: quadの幅
        height: quadの高さ

    Returns:
        quad: (4, 2) float32 - 左上、右上、右下、左下の順
    """
    cx, cy = float(center[0]), float(center[1])
    half_w = width / 2.0
    half_h = height / 2.0

    quad = np.array([
        [cx - half_w, cy - half_h],  # 左上
        [cx + half_w, cy - half_h],  # 右上
        [cx + half_w, cy + half_h],  # 右下
        [cx - half_w, cy + half_h],  # 左下
    ], dtype=np.float32)

    return quad


def ensure_even_ge2(n: int) -> int:
    """偶数に丸める（最小2）"""
    n = int(n)
    if n < 2:
        return 2
    return n if (n % 2 == 0) else (n - 1)


def adjust_quad(
    quad: np.ndarray,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    scale: float = 1.0,
) -> np.ndarray:
    """
    quadに微調整を適用する。

    Args:
        quad: (4, 2) float32 配列
        offset_x: X方向のオフセット（ピクセル）
        offset_y: Y方向のオフセット（ピクセル）
        scale: スケール係数（1.0 = 元のサイズ）

    Returns:
        調整後のquad
    """
    quad = quad.copy()

    # オフセットを適用
    quad[:, 0] += offset_x
    quad[:, 1] += offset_y

    # スケールを適用（中心を基準に拡大/縮小）
    if scale != 1.0:
        center = quad.mean(axis=0)
        quad = center + (quad - center) * scale

    return quad


# ---------------------------------------------------------------------------
# Position-aware clustering
# ---------------------------------------------------------------------------

def find_stable_position_cluster(
    centers: np.ndarray,
    valid: np.ndarray,
    distance_threshold: float = 50.0,
) -> np.ndarray:
    """
    口の中心座標が安定しているフレーム群を特定する。
    """
    N = len(centers)
    valid_indices = np.where(valid)[0]

    if len(valid_indices) < 5:
        return valid.copy()

    valid_centers = centers[valid_indices]

    counts = np.zeros(len(valid_indices), dtype=np.int32)
    for i, c in enumerate(valid_centers):
        dists = np.linalg.norm(valid_centers - c, axis=1)
        counts[i] = np.sum(dists <= distance_threshold)

    best_idx = np.argmax(counts)
    best_center = valid_centers[best_idx]

    dists_from_best = np.linalg.norm(valid_centers - best_center, axis=1)
    cluster_valid_mask = dists_from_best <= distance_threshold

    cluster_mask = np.zeros(N, dtype=bool)
    for i, orig_idx in enumerate(valid_indices):
        if cluster_valid_mask[i]:
            cluster_mask[orig_idx] = True

    return cluster_mask


# ---------------------------------------------------------------------------
# Mouth type selection
# ---------------------------------------------------------------------------

def classify_mouth_frames(
    mouth_frames: List[MouthFrameInfo],
    cluster_mask: np.ndarray,
    candidates_per_category: int = 5,
) -> Dict[str, List[MouthFrameInfo]]:
    """
    口フレームを3カテゴリ（open, closed, half）に自動分類する。

    各カテゴリでなるべく異なるフレームを選ぶようにする。

    Args:
        mouth_frames: 全フレームの口情報
        cluster_mask: 位置クラスタに属するフレームのマスク
        candidates_per_category: 各カテゴリの候補数

    Returns:
        {"open": [...], "closed": [...], "half": [...]}
    """
    # クラスタ内の有効フレームのみを対象
    candidates = [
        mf for mf in mouth_frames
        if mf.valid and cluster_mask[mf.frame_idx]
    ]

    if len(candidates) < 3:
        candidates = [mf for mf in mouth_frames if mf.valid]

    if len(candidates) == 0:
        return {"open": [], "closed": [], "half": []}

    # 各種メトリクス
    heights = np.array([mf.height for mf in candidates])
    widths = np.array([mf.width for mf in candidates])

    median_height = np.median(heights)

    # 各カテゴリのスコア計算
    # open: 高さが大きい（大きく開いた口）
    open_scores = heights

    # closed: 高さが小さく、幅も小さめ（閉じた口）
    closed_scores = -heights - 0.3 * widths

    # half: 高さが中央付近（半開き）
    half_scores = -np.abs(heights - median_height)

    all_scores = {
        "open": open_scores,
        "closed": closed_scores,
        "half": half_scores,
    }

    # カテゴリの処理順序（特徴が明確なものから）
    category_order = ["open", "closed", "half"]

    result: Dict[str, List[MouthFrameInfo]] = {}
    used_count: Dict[int, int] = {}  # frame_idx -> 何回使われたか

    for cat in category_order:
        scores = all_scores[cat].copy()

        # 既に多く使われているフレームにはペナルティ
        for i, mf in enumerate(candidates):
            count = used_count.get(mf.frame_idx, 0)
            if count > 0:
                # 使われた回数に応じてスコアを下げる
                scores[i] -= count * 0.5 * np.std(scores)

        sorted_indices = np.argsort(scores)[::-1]
        selected = []

        for idx in sorted_indices:
            if len(selected) >= candidates_per_category:
                break
            mf = candidates[idx]
            # このカテゴリに追加
            mf_copy = MouthFrameInfo(
                frame_idx=mf.frame_idx,
                quad=mf.quad,
                center=mf.center,
                width=mf.width,
                height=mf.height,
                confidence=mf.confidence,
                valid=mf.valid,
                mask=mf.mask,
                bbox=mf.bbox,
                category=cat,
            )
            selected.append(mf_copy)
            used_count[mf.frame_idx] = used_count.get(mf.frame_idx, 0) + 1

        result[cat] = selected

    return {cat: result[cat] for cat in ["open", "closed", "half"]}


def select_mouth_types(
    mouth_frames: List[MouthFrameInfo],
    cluster_mask: np.ndarray,
) -> MouthTypeSelection:
    """
    3種類の口タイプ（open, closed, half）を自動選別する。
    """
    candidates = [
        mf for mf in mouth_frames
        if mf.valid and cluster_mask[mf.frame_idx]
    ]

    if len(candidates) < 3:
        candidates = [mf for mf in mouth_frames if mf.valid]

    if len(candidates) == 0:
        raise ValueError("No valid mouth frames found")

    heights = np.array([mf.height for mf in candidates])
    widths = np.array([mf.width for mf in candidates])

    used_indices = set()

    def pick_best(scores: np.ndarray, maximize: bool = True) -> int:
        sorted_indices = np.argsort(scores)
        if maximize:
            sorted_indices = sorted_indices[::-1]

        for idx in sorted_indices:
            frame_idx = candidates[idx].frame_idx
            if frame_idx not in used_indices:
                used_indices.add(frame_idx)
                return frame_idx

        return candidates[sorted_indices[0]].frame_idx

    open_idx = pick_best(heights, maximize=True)
    closed_idx = pick_best(heights, maximize=False)

    median_height = np.median(heights)
    half_scores = -np.abs(heights - median_height)
    half_idx = pick_best(half_scores, maximize=True)

    return MouthTypeSelection(
        open_idx=open_idx,
        closed_idx=closed_idx,
        half_idx=half_idx,
    )


# ---------------------------------------------------------------------------
# Mask generation
# ---------------------------------------------------------------------------

def make_ellipse_mask(w: int, h: int, rx: int, ry: int) -> np.ndarray:
    """楕円マスクを生成（0/255）"""
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    rx = int(max(1, min(rx, w // 2 - 1)))
    ry = int(max(1, min(ry, h // 2 - 1)))
    cv2.ellipse(mask, (cx, cy), (rx, ry), 0.0, 0.0, 360.0, 255, -1)
    return mask


# ---------------------------------------------------------------------------
# Sprite extraction
# ---------------------------------------------------------------------------

def warp_frame_to_norm(
    frame_bgr: np.ndarray,
    quad: np.ndarray,
    norm_w: int,
    norm_h: int,
) -> np.ndarray:
    """フレームから口パッチを正規化空間に変換"""
    src = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    dst = np.array([
        [0, 0],
        [norm_w - 1, 0],
        [norm_w - 1, norm_h - 1],
        [0, norm_h - 1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    patch = cv2.warpPerspective(
        frame_bgr,
        M,
        (int(norm_w), int(norm_h)),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return patch


def extract_mouth_sprite(
    frame_bgr: np.ndarray,
    quad: np.ndarray,
    unified_w: int,
    unified_h: int,
    feather_px: int = 15,
    mask_scale: float = 0.85,
    sam_mask: Optional[np.ndarray] = None,
    use_sam_mask: bool = False,
    mask_dilate: int = 0,
) -> np.ndarray:
    """
    フレームから口スプライトを抽出する。

    Args:
        frame_bgr: 入力フレーム（BGR）
        quad: 口のquad (4, 2)
        unified_w: 出力幅
        unified_h: 出力高さ
        feather_px: フェザー幅
        mask_scale: 楕円マスクのスケール
        sam_mask: SAM3が生成したマスク（元画像サイズ）
        use_sam_mask: SAM3マスクを使用するか
        mask_dilate: マスクの膨張/収縮ピクセル数（正:膨張、負:収縮）

    Returns:
        bgra: (H, W, 4) uint8 - 透過PNG用
    """
    patch = warp_frame_to_norm(frame_bgr, quad, unified_w, unified_h)

    if use_sam_mask and sam_mask is not None:
        # SAM3マスクを使用
        # マスクをquadの領域に合わせて変換
        mask_u8 = warp_mask_to_norm(sam_mask, quad, unified_w, unified_h)

        # 膨張/収縮を適用
        if mask_dilate != 0:
            kernel_size = abs(mask_dilate) * 2 + 1
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
            )
            if mask_dilate > 0:
                mask_u8 = cv2.dilate(mask_u8, kernel)
            else:
                mask_u8 = cv2.erode(mask_u8, kernel)
    else:
        # 楕円マスクを使用
        rx = int((unified_w * mask_scale) * 0.5)
        ry = int((unified_h * mask_scale) * 0.5)
        mask_u8 = make_ellipse_mask(unified_w, unified_h, rx, ry)

    mask_f = feather_mask(mask_u8, feather_px)

    bgra = np.zeros((unified_h, unified_w, 4), dtype=np.uint8)
    bgra[:, :, :3] = patch
    bgra[:, :, 3] = (mask_f * 255).astype(np.uint8)

    return bgra


def warp_mask_to_norm(
    mask: np.ndarray,
    quad: np.ndarray,
    norm_w: int,
    norm_h: int,
) -> np.ndarray:
    """SAM3マスクをquadの領域に合わせて正規化空間に変換"""
    src = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    dst = np.array([
        [0, 0],
        [norm_w - 1, 0],
        [norm_w - 1, norm_h - 1],
        [0, norm_h - 1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        mask,
        M,
        (int(norm_w), int(norm_h)),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # 二値化
    return (warped > 127).astype(np.uint8) * 255


def get_unique_output_dir(base_name: str = "mouth") -> str:
    """ユニークな出力ディレクトリ名を生成"""
    if not os.path.exists(base_name):
        return base_name

    i = 1
    while os.path.exists(f"{base_name}_{i:02d}"):
        i += 1
    return f"{base_name}_{i:02d}"


# ---------------------------------------------------------------------------
# Main extractor class
# ---------------------------------------------------------------------------

class MouthSpriteExtractor:
    """SAM3を使用した口スプライト抽出器"""

    def __init__(
        self,
        video_path: str,
        device: str = "auto",
        padding_ratio: float = 0.3,
        model_path: str = "",
    ):
        """
        Args:
            video_path: 入力動画のパス
            device: SAM3のデバイス ("cuda", "cpu", "auto")
            padding_ratio: 口検出時のパディング比率
            model_path: SAM3モデルパス (空の場合はデフォルト Sam3/sam3.pt)
        """
        self.video_path = video_path
        self.device = device
        self.padding_ratio = padding_ratio
        self.model_path = model_path

        # 動画情報
        self.vid_w = 0
        self.vid_h = 0
        self.fps = 0.0
        self.n_frames = 0

        # 解析結果
        self.mouth_frames: List[MouthFrameInfo] = []
        self.cluster_mask: Optional[np.ndarray] = None
        self.selection: Optional[MouthTypeSelection] = None
        self.unified_size: Optional[Tuple[int, int]] = None

        # SAM3検出器（遅延初期化）
        self._detector: Optional[SAM3MouthDetector] = None

        self._load_video_info()

    def _load_video_info(self):
        """動画の情報を取得"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")

        self.vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()

    def _ensure_detector(self):
        """SAM3検出器を初期化"""
        if self._detector is None:
            self._detector = SAM3MouthDetector(
                model_path=self.model_path,
                device=self.device,
            )

    def analyze(
        self,
        max_frames: int = 100,
        callback: Optional[Callable[[str], None]] = None,
        progress: Optional[Callable[[int, int], None]] = None,
    ):
        """
        動画を解析して口情報を取得。

        Args:
            max_frames: 処理する最大フレーム数
            callback: ログコールバック
            progress: 進捗コールバック (current_frame, total_frames)
        """
        def log(msg: str):
            if callback:
                callback(msg)
            else:
                print(msg)

        log(tr("ext.log.video_info", w=self.vid_w, h=self.vid_h,
               frames=self.n_frames, fps=self.fps))

        if self.n_frames <= 0:
            raise RuntimeError(
                tr("ext.err.no_frame_count", n=self.n_frames, path=self.video_path)
            )

        # SAM3検出器を初期化
        log(tr("ext.log.sam3_init"))
        self._ensure_detector()
        log(tr("ext.log.sam3_ready"))

        # フレーム間隔を計算（max_frames が処理数の上限になるよう切り上げ）
        stride = max(1, math.ceil(self.n_frames / max_frames))
        log(tr("ext.log.stride", stride=stride))

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")

        self.mouth_frames = []

        try:
            for frame_idx in range(0, self.n_frames, stride):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue

                # SAM3で口を検出（padding=0で生のbboxを取得）
                result = self._detector.detect_mouth_with_confidence(
                    frame,
                    prompt="mouth",
                    padding_ratio=0.0,
                )

                if result is not None:
                    mask, bbox_raw, center, conf = result
                    # 生のbboxサイズ（分類用）
                    raw_w = float(bbox_raw[2] - bbox_raw[0])
                    raw_h = float(bbox_raw[3] - bbox_raw[1])

                    # padding適用したbboxでquadを生成（切り出し範囲用）
                    h_img, w_img = frame.shape[:2]
                    pad_w = raw_w * self.padding_ratio
                    pad_h = raw_h * self.padding_ratio
                    bbox_padded = (
                        max(0, int(bbox_raw[0] - pad_w)),
                        max(0, int(bbox_raw[1] - pad_h)),
                        min(w_img, int(bbox_raw[2] + pad_w)),
                        min(h_img, int(bbox_raw[3] + pad_h)),
                    )
                    quad = bbox_to_quad(bbox_padded)

                    mf = MouthFrameInfo(
                        frame_idx=frame_idx,
                        quad=quad,
                        center=np.array([center[0], center[1]], dtype=np.float32),
                        width=raw_w,
                        height=raw_h,
                        confidence=conf,
                        valid=True,
                        mask=mask,  # SAM3マスクを保存
                        bbox=bbox_raw,  # 元のbboxを保存
                    )
                    self.mouth_frames.append(mf)

                if progress:
                    progress(frame_idx, self.n_frames)

                if (frame_idx // stride) % 10 == 0:
                    log(tr("ext.log.progress", current=frame_idx, total=self.n_frames,
                           found=len(self.mouth_frames)))
        finally:
            cap.release()

        log(tr("ext.log.detect_done", n=len(self.mouth_frames)))

        if len(self.mouth_frames) == 0:
            log(tr("ext.log.no_mouth"))
            return

        # クラスタリング
        centers = np.array([mf.center for mf in self.mouth_frames])
        valid = np.array([mf.valid for mf in self.mouth_frames])

        # フレームインデックスでマッピング
        max_idx = max(mf.frame_idx for mf in self.mouth_frames) + 1
        centers_full = np.zeros((max_idx, 2), dtype=np.float32)
        valid_full = np.zeros(max_idx, dtype=bool)

        for mf in self.mouth_frames:
            centers_full[mf.frame_idx] = mf.center
            valid_full[mf.frame_idx] = mf.valid

        self.cluster_mask = find_stable_position_cluster(centers_full, valid_full)

        # 3種類の口を選択
        self.selection = select_mouth_types(self.mouth_frames, self.cluster_mask)

        # 統一サイズを計算（quadのサイズから。quadにはpadding適用済み）
        selected_indices = list(self.selection.as_dict().values())
        idx_to_mf = {mf.frame_idx: mf for mf in self.mouth_frames}
        max_qw = 0.0
        max_qh = 0.0
        for idx in selected_indices:
            if idx in idx_to_mf:
                mf = idx_to_mf[idx]
                q = mf.quad
                qw = float(np.linalg.norm(q[1] - q[0]))
                qh = float(np.linalg.norm(q[3] - q[0]))
                max_qw = max(max_qw, qw)
                max_qh = max(max_qh, qh)
        self.unified_size = (
            ensure_even_ge2(max(32, int(max_qw * 1.1))),
            ensure_even_ge2(max(32, int(max_qh * 1.1))),
        )

        log(tr("ext.log.selection", selection=self.selection.as_dict()))
        log(tr("ext.log.unified_size", size=self.unified_size))

    def extract_sprites(
        self,
        output_dir: str,
        feather_px: int = 15,
        mask_scale: float = 0.85,
        callback: Optional[Callable[[str], None]] = None,
    ):
        """
        選択された3種類の口スプライトを出力。

        Args:
            output_dir: 出力ディレクトリ
            feather_px: フェザー幅
            mask_scale: マスクスケール
            callback: ログコールバック
        """
        def log(msg: str):
            if callback:
                callback(msg)
            else:
                print(msg)

        if self.selection is None or self.unified_size is None:
            raise RuntimeError(tr("ext.err.analyze_first"))

        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")

        idx_to_mf = {mf.frame_idx: mf for mf in self.mouth_frames}
        unified_w, unified_h = self.unified_size

        mouth_names = {
            "open": "mouth_open",
            "closed": "mouth_closed",
            "half": "mouth_half",
        }

        try:
            for mouth_type, frame_idx in self.selection.as_dict().items():
                if frame_idx not in idx_to_mf:
                    log(tr("ext.log.missing_frame", type=mouth_type, frame=frame_idx))
                    continue

                mf = idx_to_mf[frame_idx]

                cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
                ok, frame = cap.read()
                if not ok or frame is None:
                    log(tr("ext.log.frame_read_failed", frame=frame_idx))
                    continue

                bgra = extract_mouth_sprite(
                    frame, mf.quad, unified_w, unified_h,
                    feather_px=feather_px, mask_scale=mask_scale
                )

                out_path = os.path.join(output_dir, f"{mouth_names[mouth_type]}.png")
                if imwrite_jp(out_path, bgra):
                    log(tr("ext.log.output", path=out_path))
                else:
                    log(tr("ext.log.write_failed", path=out_path))
        finally:
            cap.release()
        log(tr("ext.log.done", path=output_dir))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract mouth sprites (open/closed/half PNGs) from a video using SAM3"
    )
    parser.add_argument("--video", "-v", required=True, help="input video file")
    parser.add_argument("--out", "-o", default="", help="output directory")
    parser.add_argument("--feather", type=int, default=15, help="feather width (px)")
    parser.add_argument("--padding", type=float, default=0.3,
                        help="mouth detection padding ratio")
    parser.add_argument("--device", default="auto", help="device (cuda/cpu/auto)")
    parser.add_argument("--model", default="", help="SAM3 model path (default: Sam3/sam3.pt)")

    args = parser.parse_args()

    # SAM3の利用可否をチェック
    available, error = is_sam3_available()
    if not available:
        print(f"Error: SAM3 is not available: {error}")
        return 1

    output_dir = args.out or get_unique_output_dir("mouth")

    extractor = MouthSpriteExtractor(
        args.video,
        device=args.device,
        padding_ratio=args.padding,
        model_path=args.model,
    )
    extractor.analyze()
    extractor.extract_sprites(output_dir, feather_px=args.feather)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
