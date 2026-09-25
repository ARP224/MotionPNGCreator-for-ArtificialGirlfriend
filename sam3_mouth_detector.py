#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Origin / 出自:
#   - kazuya-bros/MouthSpriteExtractor-SAM3 の sam3_detector.py
#     (AGPL-3.0, Copyright (c) 2026 kazuya-bros)
#     https://github.com/kazuya-bros/MouthSpriteExtractor-SAM3
# Modified by Ryo (ARP224) in 2026 (モデルパス指定、GPU指定、torch/ultralyticsの遅延import、
#                          日英i18n対応ほか).
# License: AGPL-3.0-only. 原著作権表示はリポジトリ直下の NOTICE.md に
#          保持されています。ライセンス全文は LICENSE を参照。
#
"""
sam3_mouth_detector.py

SAM3 (Segment Anything Model 3) を使用した口検出モジュール。
Ultralytics経由でSAM3を使用。

MotionPNGCreator for ArtificialGirlfriend用に適応:
- モデルパスを設定可能（デフォルト: Sam3/sam3.pt）
- GPU指定サポート（cuda:0, cuda:1, auto）

License: AGPL-3.0-only (Ultralyticsライセンスに準拠)
Note: SAM3モデルの使用にはHuggingFaceでのアクセス承認が必要です。
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from i18n import tr

# ---------------------------------------------------------------------------
# SAM3 availability check (lazy)
# ---------------------------------------------------------------------------
# torch / ultralytics の初回importは数十秒かかるため、モジュールimport時ではなく
# is_sam3_available() の初回呼び出しで行う（GUIはウィンドウ表示後に
# バックグラウンドスレッドから呼ぶ）。結果はキャッシュされる。

_SAM3_AVAILABLE: Optional[bool] = None
_SAM3_ERROR = ""
_availability_lock = threading.Lock()

# is_sam3_available() 成功時にセットされる（遅延import）
torch = None
SAM3SemanticPredictor = None


# Default model path relative to this file's directory
_DEFAULT_MODEL_PATH = str(Path(__file__).resolve().parent / "Sam3" / "sam3.pt")


def is_sam3_available() -> Tuple[bool, str]:
    """SAM3が利用可能かチェック

    初回呼び出しで torch / ultralytics をimportするため数十秒かかることがある。
    2回目以降はキャッシュ済みの結果を即座に返す。

    Returns:
        (available, error_message)
    """
    global _SAM3_AVAILABLE, _SAM3_ERROR, torch, SAM3SemanticPredictor
    with _availability_lock:
        if _SAM3_AVAILABLE is not None:
            return _SAM3_AVAILABLE, _SAM3_ERROR

        try:
            import torch as _torch
        except ImportError as e:
            _SAM3_AVAILABLE = False
            _SAM3_ERROR = f"PyTorch not available: {e}"
            return _SAM3_AVAILABLE, _SAM3_ERROR
        torch = _torch

        try:
            from ultralytics.models.sam import SAM3SemanticPredictor as _predictor
            SAM3SemanticPredictor = _predictor
            _SAM3_AVAILABLE = True
        except ImportError as e:
            _SAM3_AVAILABLE = False
            _SAM3_ERROR = f"ultralytics SAM3 not available: {e}"
        except Exception as e:
            _SAM3_AVAILABLE = False
            _SAM3_ERROR = str(e)
        return _SAM3_AVAILABLE, _SAM3_ERROR


# ---------------------------------------------------------------------------
# SAM3 Mouth Detector
# ---------------------------------------------------------------------------

class SAM3MouthDetector:
    """SAM3を使用した口検出クラス（Ultralytics版）"""

    def __init__(
        self,
        model_path: str = "",
        device: str = "auto",
        confidence_threshold: float = 0.25,
    ):
        """
        Args:
            model_path: SAM3モデルファイルのパス。空の場合はデフォルト (Sam3/sam3.pt)
            device: "cuda", "cuda:0", "cuda:1", "cpu", or "auto"
            confidence_threshold: 検出の信頼度閾値
        """
        available, error = is_sam3_available()
        if not available:
            raise RuntimeError(f"SAM3 is not available: {error}")

        # Resolve model path
        if not model_path:
            model_path = _DEFAULT_MODEL_PATH
        model_path = str(Path(model_path).resolve())

        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                tr("errors.sam3_model_missing", path=model_path))

        # Resolve device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold

        print(f"[SAM3] Loading model from {model_path} on {device}...")

        overrides = dict(
            conf=confidence_threshold,
            task="segment",
            mode="predict",
            model=model_path,
            device=device,
            verbose=False,
            save=False,
            imgsz=644,  # SAM3 stride=14の倍数に合わせて警告を抑制
        )
        self.predictor = SAM3SemanticPredictor(overrides=overrides)

        print("[SAM3] Model loaded successfully")

    def detect_mouth(
        self,
        image: np.ndarray,
        prompt: str = "mouth",
        padding_ratio: float = 0.0,
        padding_px: int = 0,
    ) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int], Tuple[float, float]]]:
        """
        テキストプロンプトで口を検出する。

        Args:
            image: BGR画像（OpenCV形式）
            prompt: テキストプロンプト（デフォルト: "mouth"）
            padding_ratio: bboxを広げる比率（0.3 = 30%広げる）
            padding_px: bboxを広げるピクセル数

        Returns:
            (mask, bbox, center) または検出失敗時はNone
            - mask: (H, W) uint8 マスク（0/255）
            - bbox: (x_min, y_min, x_max, y_max) パディング適用済み
            - center: (cx, cy) 口の中心座標（パディング前）
        """
        result = self.detect_mouth_with_confidence(
            image, prompt=prompt, padding_ratio=padding_ratio, padding_px=padding_px
        )
        if result is None:
            return None
        mask, bbox, center, _confidence = result
        return mask, bbox, center

    def detect_mouth_with_confidence(
        self,
        image: np.ndarray,
        prompt: str = "mouth",
        padding_ratio: float = 0.0,
        padding_px: int = 0,
    ) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int], Tuple[float, float], float]]:
        """
        detect_mouth() と同じだが、信頼度スコアも返す。

        Returns:
            (mask, bbox, center, confidence) または None
        """
        try:
            h, w = image.shape[:2]
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.predictor.set_image(image_rgb)

            results = self.predictor(text=[prompt])
            if results is None or len(results) == 0:
                return None

            result = results[0]
            if result.masks is None or len(result.masks) == 0:
                return None
            if result.boxes is None or len(result.boxes) == 0:
                return None

            masks = result.masks.data
            if hasattr(masks, "cpu"):
                masks = masks.cpu().numpy()
            boxes = result.boxes.xyxy
            if hasattr(boxes, "cpu"):
                boxes = boxes.cpu().numpy()
            scores = result.boxes.conf
            if hasattr(scores, "cpu"):
                scores = scores.cpu().numpy()

            best_idx = int(np.argmax(scores)) if len(scores) > 0 else 0
            confidence = float(scores[best_idx]) if len(scores) > 0 else 1.0

            mask = masks[best_idx]
            if mask.ndim == 3:
                mask = mask.squeeze()
            if mask.shape != (h, w):
                mask = cv2.resize(
                    mask.astype(np.float32), (w, h),
                    interpolation=cv2.INTER_LINEAR,
                )
            mask_u8 = (mask > 0.5).astype(np.uint8) * 255

            bbox = boxes[best_idx]
            x_min, y_min, x_max, y_max = map(int, bbox[:4])
            cx = (x_min + x_max) / 2.0
            cy = (y_min + y_max) / 2.0

            bbox_w = x_max - x_min
            bbox_h = y_max - y_min
            pad_w = bbox_w * padding_ratio + padding_px
            pad_h = bbox_h * padding_ratio + padding_px
            x_min = max(0, int(x_min - pad_w))
            y_min = max(0, int(y_min - pad_h))
            x_max = min(w, int(x_max + pad_w))
            y_max = min(h, int(y_max + pad_h))

            return mask_u8, (x_min, y_min, x_max, y_max), (cx, cy), confidence

        except Exception as e:
            print(f"[SAM3] Detection error: {e}")
            import traceback
            traceback.print_exc()
            return None


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def bbox_to_quad(
    bbox: Tuple[int, int, int, int],
    sprite_aspect: float = 0.0,
) -> np.ndarray:
    """
    bboxからquad（4点）を生成する。

    Args:
        bbox: (x_min, y_min, x_max, y_max)
        sprite_aspect: スプライトのアスペクト比 (w/h)。0以下の場合は元のアスペクト比を維持。

    Returns:
        quad: (4, 2) float32 配列 [TL, TR, BR, BL]
    """
    x_min, y_min, x_max, y_max = bbox
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    w = x_max - x_min
    h = y_max - y_min

    if sprite_aspect > 0:
        current_aspect = w / max(1, h)
        if current_aspect > sprite_aspect:
            h = w / sprite_aspect
        else:
            w = h * sprite_aspect

    hw, hh = w / 2.0, h / 2.0

    quad = np.array([
        [cx - hw, cy - hh],  # TL
        [cx + hw, cy - hh],  # TR
        [cx + hw, cy + hh],  # BR
        [cx - hw, cy + hh],  # BL
    ], dtype=np.float32)

    return quad


def feather_mask(mask_u8: np.ndarray, feather_px: int) -> np.ndarray:
    """マスクにフェザー（グラデーション）を適用"""
    if feather_px <= 0:
        return (mask_u8.astype(np.float32) / 255.0).clip(0.0, 1.0)

    k = 2 * int(feather_px) + 1
    m = cv2.GaussianBlur(mask_u8, (k, k), sigmaX=0)
    return (m.astype(np.float32) / 255.0).clip(0.0, 1.0)
