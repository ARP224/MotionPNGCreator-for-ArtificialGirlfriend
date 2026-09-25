#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
track_utils.py

検出器非依存のトラッキングユーティリティ。
quad配列のスムージング、補間、分解/合成、品質指標計算などを提供する。

eye_pipeline.py / batch_engine.py などから利用される純粋関数群。
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def one_pole_beta(cutoff_hz: float, fps: float) -> float:
    """EMAのbeta係数を計算 (1-pole lowpass)"""
    return float(1.0 - np.exp(-2.0 * np.pi * float(cutoff_hz) / float(max(1e-6, fps))))


# ---------------------------------------------------------------------------
# Quad decomposition / composition
# ---------------------------------------------------------------------------

def decompose_quads_vectorized(
    quads: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized version of decompose_quad for N quads.

    Args:
        quads: (N, 4, 2) array

    Returns:
        (centers (N,2), widths (N,), heights (N,), angles_deg (N,))
    """
    centers = quads.mean(axis=1)  # (N, 2)

    v_top = quads[:, 1, :] - quads[:, 0, :]  # (N, 2)
    widths = np.linalg.norm(v_top, axis=1)  # (N,)
    angles = np.degrees(np.arctan2(v_top[:, 1], v_top[:, 0]))  # (N,)

    v_left = quads[:, 3, :] - quads[:, 0, :]  # (N, 2)
    heights = np.linalg.norm(v_left, axis=1)  # (N,)

    return (
        centers.astype(np.float32),
        widths.astype(np.float32),
        heights.astype(np.float32),
        angles.astype(np.float32),
    )


def compose_quad(
    center: np.ndarray, width: float, height: float, angle_deg: float,
) -> np.ndarray:
    """
    中心、幅、高さ、角度からquadを再構成。

    Returns:
        quad: (4, 2) float32 - [TL, TR, BR, BL]
    """
    hw, hh = width / 2, height / 2

    local = np.array([
        [-hw, -hh],
        [+hw, -hh],
        [+hw, +hh],
        [-hw, +hh],
    ], dtype=np.float32)

    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)

    rotated = local @ R.T
    return (rotated + center).astype(np.float32)


def compose_quads_vectorized(
    centers: np.ndarray,
    widths: np.ndarray,
    heights: np.ndarray,
    angles_deg: np.ndarray,
) -> np.ndarray:
    """
    Vectorized version of compose_quad for N quads.

    Args:
        centers: (N, 2), widths: (N,), heights: (N,), angles_deg: (N,)

    Returns:
        quads: (N, 4, 2) float32
    """
    N = len(centers)
    hw = widths / 2
    hh = heights / 2

    local = np.zeros((N, 4, 2), dtype=np.float32)
    local[:, 0, 0] = -hw;  local[:, 0, 1] = -hh
    local[:, 1, 0] = +hw;  local[:, 1, 1] = -hh
    local[:, 2, 0] = +hw;  local[:, 2, 1] = +hh
    local[:, 3, 0] = -hw;  local[:, 3, 1] = +hh

    angles_rad = np.radians(angles_deg)
    cos_a = np.cos(angles_rad)
    sin_a = np.sin(angles_rad)

    rotated = np.zeros_like(local)
    rotated[:, :, 0] = cos_a[:, None] * local[:, :, 0] - sin_a[:, None] * local[:, :, 1]
    rotated[:, :, 1] = sin_a[:, None] * local[:, :, 0] + cos_a[:, None] * local[:, :, 1]

    result = rotated + centers[:, None, :]
    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# Angle limiting
# ---------------------------------------------------------------------------

def limit_angle_change(angles: np.ndarray, max_change_deg: float) -> np.ndarray:
    """連続するフレーム間の角度変化を制限。"""
    if max_change_deg <= 0:
        return angles.copy()

    result = angles.copy()
    for i in range(1, len(result)):
        diff = result[i] - result[i - 1]
        if abs(diff) > max_change_deg:
            result[i] = result[i - 1] + np.sign(diff) * max_change_deg
    return result


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

_scipy_median_cache: Any = None


def _get_scipy_median():
    """Get cached scipy median filter function, or None if unavailable."""
    global _scipy_median_cache
    if _scipy_median_cache is None:
        try:
            from scipy.ndimage import median_filter
            _scipy_median_cache = median_filter
        except ImportError:
            _scipy_median_cache = False
    return _scipy_median_cache if _scipy_median_cache else None


def median_filter_1d(arr: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """1Dメディアンフィルタ（外れ値除去用）"""
    scipy_median = _get_scipy_median()
    if scipy_median is not None:
        return scipy_median(arr, size=kernel_size, mode="nearest")

    N = len(arr)
    result = arr.copy()
    half = kernel_size // 2
    for i in range(N):
        start = max(0, i - half)
        end = min(N, i + half + 1)
        result[i] = np.median(arr[start:end])
    return result


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------

def interpolate_invalid_quads(quads: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """valid==0 の区間を前後フレームから補間して埋める。

    (center, width, height, angle) に分解して補間 → 再構成する方式。
    """
    N = len(quads)
    if N == 0:
        return quads

    res = quads.copy()
    v = valid.astype(np.uint8).reshape(-1)

    def angle_shortest_diff(a0: float, a1: float) -> float:
        return float(((a1 - a0 + 180.0) % 360.0) - 180.0)

    i = 0
    while i < N:
        if v[i] == 1:
            i += 1
            continue

        start = i
        while i < N and v[i] == 0:
            i += 1
        end = i  # [start, end) is invalid

        prev_ok = start - 1 if start > 0 and v[start - 1] == 1 else -1
        next_ok = end if end < N and v[end] == 1 else -1

        if prev_ok >= 0 and next_ok >= 0:
            c0, w0, h0, a0 = decompose_quads_vectorized(quads[prev_ok:prev_ok + 1])
            c1, w1, h1, a1 = decompose_quads_vectorized(quads[next_ok:next_ok + 1])
            c0 = c0[0]; c1 = c1[0]
            w0 = float(w0[0]); w1 = float(w1[0])
            h0 = float(h0[0]); h1 = float(h1[0])
            a0 = float(a0[0]); a1 = float(a1[0])

            da = angle_shortest_diff(a0, a1)
            gap = float(next_ok - prev_ok)
            for j in range(start, end):
                t = float(j - prev_ok) / gap
                c = (1.0 - t) * c0 + t * c1
                w = (1.0 - t) * w0 + t * w1
                h = (1.0 - t) * h0 + t * h1
                a = a0 + t * da
                res[j] = compose_quad(c.astype(np.float32), float(w), float(h), float(a))
        elif prev_ok >= 0:
            for j in range(start, end):
                res[j] = quads[prev_ok]
        elif next_ok >= 0:
            for j in range(start, end):
                res[j] = quads[next_ok]

    return res


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

def smooth_quads_zero_phase(
    quads: np.ndarray,
    valid: np.ndarray,
    beta: float,
    max_angle_change: float = 15.0,
) -> np.ndarray:
    """
    valid=1のフレームのみを使って、zero-phase EMAで平滑化。
    角度は分離してスムージング + メディアンフィルタで外れ値除去。

    Args:
        quads: (N, 4, 2)
        valid: (N,) bool or uint8
        beta: EMA係数 (one_pole_beta()で計算)
        max_angle_change: 1フレームあたりの最大角度変化(度)。0で無制限。
    """
    N = len(quads)
    if N == 0:
        return quads.copy()

    out = quads.copy()

    # invalidを線形補間で埋める
    out = interpolate_invalid_quads(out, valid)

    # quad → (center, w, h, angle)
    centers, widths, heights, angles = decompose_quads_vectorized(out)

    # 角度のunwrap
    angles_unwrapped = np.degrees(np.unwrap(np.radians(angles)))

    # メディアンフィルタで外れ値除去
    angles_filtered = median_filter_1d(angles_unwrapped, kernel_size=3)

    def smooth_1d(arr: np.ndarray, b: float) -> np.ndarray:
        out = arr.copy()
        for i in range(1, len(out)):
            out[i] = out[i - 1] + b * (out[i] - out[i - 1])
        for i in range(len(out) - 2, -1, -1):
            out[i] = out[i + 1] + b * (out[i] - out[i + 1])
        return out

    cx_smooth = smooth_1d(centers[:, 0], beta)
    cy_smooth = smooth_1d(centers[:, 1], beta)
    w_smooth = smooth_1d(widths, beta)
    h_smooth = smooth_1d(heights, beta)
    angle_smooth = smooth_1d(angles_filtered, beta)

    if max_angle_change > 0:
        angle_smooth = limit_angle_change(angle_smooth, max_angle_change)

    centers_smooth = np.column_stack([cx_smooth, cy_smooth]).astype(np.float32)
    return compose_quads_vectorized(centers_smooth, w_smooth, h_smooth, angle_smooth)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _quad_width_px(quads: np.ndarray) -> np.ndarray:
    """Approx width (px) of each quad frame."""
    if quads.ndim != 3 or quads.shape[1:] != (4, 2):
        return np.zeros((int(quads.shape[0]),), dtype=np.float32)
    p0, p1, p2, p3 = quads[:, 0], quads[:, 1], quads[:, 2], quads[:, 3]
    w_top = np.linalg.norm(p1 - p0, axis=1)
    w_bot = np.linalg.norm(p2 - p3, axis=1)
    return (0.5 * (w_top + w_bot)).astype(np.float32)


def _center_jitter_px(quads: np.ndarray) -> np.ndarray:
    """Per-frame jitter = L2 distance of quad center vs previous frame."""
    if quads.ndim != 3 or quads.shape[1:] != (4, 2):
        return np.zeros((int(quads.shape[0]),), dtype=np.float32)
    centers = quads.mean(axis=1)
    d = np.linalg.norm(np.diff(centers, axis=0), axis=1).astype(np.float32)
    return np.concatenate([np.zeros((1,), np.float32), d])


def calc_track_metrics(
    quads: np.ndarray,
    valid: np.ndarray,
    conf: Optional[np.ndarray],
    *,
    min_conf: float,
) -> dict:
    """トラッキング品質指標を計算。"""
    valid_b = valid.astype(bool)
    n = int(valid_b.size)
    vr = float(valid_b.mean()) if n > 0 else 0.0
    conf_v = conf[valid_b] if (conf is not None and np.any(valid_b)) else np.array([], np.float32)
    jitter = _center_jitter_px(quads)
    w = _quad_width_px(quads)
    w_med = float(np.median(w[valid_b])) if np.any(valid_b) else 0.0
    jitter_p95 = (
        float(np.quantile(jitter[1:][valid_b[1:]], 0.95))
        if (jitter.size > 1 and np.any(valid_b[1:]))
        else 0.0
    )
    jitter_ratio_p95 = float(jitter_p95 / max(1.0, w_med)) if w_med > 0 else float(jitter_p95)
    return {
        "n_frames": int(n),
        "n_valid": int(valid_b.sum()),
        "valid_rate": vr,
        "mean_conf": float(conf_v.mean()) if conf_v.size else None,
        "p10_conf": float(np.quantile(conf_v, 0.10)) if conf_v.size else None,
        "min_conf_threshold": float(min_conf),
        "quad_width_median_px": w_med,
        "jitter_p95_px": jitter_p95,
        "jitter_p95_over_width": jitter_ratio_p95,
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_metrics_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)



# ---------------------------------------------------------------------------
# Mouth sprite / track format helpers
# ---------------------------------------------------------------------------

# 位置データ（NPZ の track_format / JSON の trackFormat）の版。
#   未指定（旧形式）: quad の大きさが立ち絵の口 bbox。AG のプレイヤーは口 PNG を
#                    この枠に押し込んで描くため、口が小さく薄くなる
#   2: quad は口 PNG（mouth/ の open/half/closed、同一キャンバス）全体を写す枠。
#      大きさ = PNG サイズ × 目間距離比、角度は立ち絵の目の角度基準の相対角、
#      refSpriteSize = PNG のキャンバスサイズ。AG のプレイヤーと共通の契約（2026-09-13）
#      旧形式は tools/convert_track_format.py で変換する
TRACK_FORMAT_VERSION = 2

SPRITE_FILE_CANDIDATES = (
    ("open.png", "mouth_open.png"),
    ("half.png", "mouth_half.png"),
    ("closed.png", "mouth_closed.png"),
)


def read_sprite_canvas_sizes(mouth_dir: Optional[str]) -> dict:
    """mouth/ の口素材 PNG（open/half/closed）のキャンバスサイズを返す。

    Returns:
        {ファイル名: (幅, 高さ)}。読めないファイルは含めない。
        画像ヘッダだけを読む（Pillow の遅延読み込み）ので軽い。
    """
    from PIL import Image

    sizes: dict = {}
    if not mouth_dir or not os.path.isdir(mouth_dir):
        return sizes
    for candidates in SPRITE_FILE_CANDIDATES:
        for name in candidates:
            path = os.path.join(mouth_dir, name)
            if not os.path.isfile(path):
                continue
            try:
                with Image.open(path) as im:
                    sizes[name] = (int(im.width), int(im.height))
            except Exception:
                pass
            break
    return sizes


def sprite_canvas_size(mouth_dir: Optional[str]) -> Tuple[int, int]:
    """口素材 3 枚に共通のキャンバスサイズ (幅, 高さ) を返す。

    位置データの quad はこのサイズを基準に書く（quad = PNG 全体を写す枠）ため、
    3 枚は同一サイズでなければならない。読めない・不一致は ValueError。
    """
    sizes = read_sprite_canvas_sizes(mouth_dir)
    if not sizes:
        raise ValueError(f"no readable mouth sprite in {mouth_dir}")
    uniq = set(sizes.values())
    if len(uniq) != 1:
        detail = ", ".join(f"{k}={w}x{h}" for k, (w, h) in sizes.items())
        raise ValueError(f"mouth sprite sizes differ: {detail}")
    return next(iter(uniq))


# ---------------------------------------------------------------------------
# Moving-median smoothing (blink jitter)
# ---------------------------------------------------------------------------

# 位置データの平滑化の版（NPZ の track_smooth / JSON の trackSmooth）。
#   未指定（0）: 平滑化なし
#   1: 中心 x/y・幅・高さ・角度の各系列に SMOOTH_MEDIAN_SIZE コマの移動中央値
#      （端は端の値で延長）。まばたきの半開きで目マスクの重心が下にずれ、口が
#      縦に 1〜2 px 跳ねるのを除く（2026-09-13 稜裁定）。単調な動き（傾ける・
#      寄る・戻す）は値そのまま通し、4 コマ以内に行って戻る動きだけを削る。
#      プレイヤーの契約（trackFormat）は変えない。既存素材は
#      tools/smooth_track_median.py で同じ処理を後からかける
TRACK_SMOOTH_VERSION = 1
SMOOTH_MEDIAN_SIZE = 9


def moving_median_1d(arr: np.ndarray, size: int = SMOOTH_MEDIAN_SIZE) -> np.ndarray:
    """1D 移動中央値（窓 size、端は端の値で延長）。scipy 不要。"""
    a = np.asarray(arr)
    n = len(a)
    if n == 0 or size <= 1:
        return a.copy()
    half = size // 2
    padded = np.concatenate([np.full(half, a[0]), a, np.full(half, a[-1])])
    windows = np.lib.stride_tricks.sliding_window_view(padded, size)
    return np.median(windows, axis=1).astype(a.dtype)


def smooth_quads_moving_median(
    quads: np.ndarray, size: int = SMOOTH_MEDIAN_SIZE,
) -> np.ndarray:
    """quad 列を (中心, 幅, 高さ, 角度) に分解して各系列に移動中央値をかけ、再構成する。

    角度は unwrap してから中央値を取る（±180 度の折り返し対策）。valid は触らない。
    パイプライン（EMA 平滑化の前段）と後処理ツールの両方がこれを使う。
    """
    quads = np.asarray(quads)
    if len(quads) == 0:
        return quads.copy()
    centers, widths, heights, angles = decompose_quads_vectorized(quads.astype(np.float32))
    angles_unwrapped = np.degrees(np.unwrap(np.radians(angles.astype(np.float64))))
    cx = moving_median_1d(centers[:, 0], size)
    cy = moving_median_1d(centers[:, 1], size)
    w = moving_median_1d(widths, size)
    h = moving_median_1d(heights, size)
    a = moving_median_1d(angles_unwrapped, size).astype(np.float32)
    out = compose_quads_vectorized(np.column_stack([cx, cy]).astype(np.float32), w, h, a)
    return out.astype(quads.dtype)
