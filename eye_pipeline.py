#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eye_pipeline.py

SAM3で目を検出し、face_info.jsonの幾何学的比率から口位置を算出。
トラッキング（Phase 1）と口消し（Phase 2）を1プロセスで実行する。

Usage:
    python eye_pipeline.py \
        --video input.mp4 \
        --face-info character_face_info.json \
        --out-track mouth_track.npz \
        --out-video output_mouthless.mp4 \
        --model Sam3/sam3.pt \
        --device auto
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
from sam3_mouth_detector import SAM3MouthDetector
from i18n import tr

# pythonw（コンソール非表示）起動時に子プロセスが黒いコンソール窓を
# 開かないようにするフラグ（Windows以外では0）
CREATE_NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

from track_utils import (
    compose_quads_vectorized,
    decompose_quads_vectorized,
    median_filter_1d,
    one_pole_beta,
    smooth_quads_zero_phase,
    smooth_quads_moving_median,
    SMOOTH_MEDIAN_SIZE,
    calc_track_metrics,
    save_metrics_json,
)
from erase_mouth_offline import (
    FFmpegAlphaWriter,
    abort_ffmpeg_writer,
    mux_audio_ffmpeg,
    mux_audio_ffmpeg_webm,
    create_alpha_mask_multi_color,
    create_alpha_hybrid,
    despill_green,
    hex_to_bgr,
    parse_color_spec,
)


# ---------------------------------------------------------------------------
# H.264 Writer (ffmpeg, browser-compatible MP4)
# ---------------------------------------------------------------------------

class _FFmpegH264Writer:
    """ffmpegでH.264 MP4を出力するライター（ブラウザ互換）"""

    def __init__(self, output_path: str, fps: float, width: int, height: int):
        import tempfile as _tf
        self.output_path = output_path
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            raise RuntimeError(tr("errors.ffmpeg_missing"))
        self._log = _tf.NamedTemporaryFile(mode='w', suffix='.log', delete=False)
        cmd = [
            ffmpeg_path, '-y',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{width}x{height}',
            '-r', str(fps),
            '-i', '-',
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
            '-pix_fmt', 'yuv420p',
            '-an', output_path,
        ]
        self.process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=self._log,
            creationflags=CREATE_NO_WINDOW,
        )

    def write(self, frame_bgr: np.ndarray):
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("FFmpeg process is not running")
        try:
            self.process.stdin.write(frame_bgr.tobytes())
        except BrokenPipeError:
            # FFmpegがエラーで終了した場合
            raise RuntimeError(f"FFmpeg pipe broken. Check log: {self._log.name}")

    def release(self) -> bool:
        if self.process:
            if self.process.stdin:
                try:
                    self.process.stdin.close()
                except Exception:
                    pass
            rc = self.process.wait()
            self.process = None
            log_path = self._log.name
            self._log.close()
            if rc != 0:
                print(f"[warn] FFmpeg H.264 exited with code {rc}")
                if os.path.exists(log_path):
                    try:
                        # ffmpegのログはUTF-8。ロケール既定で読むと非日本語環境で
                        # UnicodeDecodeErrorになり診断情報が失われる
                        with open(log_path, encoding="utf-8", errors="replace") as f:
                            for line in f.readlines()[-5:]:
                                print(f"[ffmpeg] {line.rstrip()}")
                    except Exception:
                        pass
                return False
            if os.path.exists(log_path):
                try:
                    os.unlink(log_path)
                except Exception:
                    pass
            return True
        return True

    def isOpened(self) -> bool:
        return self.process is not None and self.process.poll() is None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OUTLIER_DEVIATION_RATIO = 0.2   # eye_distの20%以上ずれたらoutlier
# 目間距離の基準比較（2026-09-13）。顔が横を向いた瞬間に両目の検出が同じ点に崩れて
# 20 フレーム前後続くと、5 フレームのメディアン比較（_reject_outliers）は崩れた値に
# 追随して弾けない。立ち絵の目間距離（× 動画/画像の解像度比）に対する比率で判定し、
# 範囲外のフレームを無効化して前後から補間する。
# 実測（19 体 568 本）: 正常な動画の最小比率は 0.87、追跡破綻 8 本は 0.01〜0.41。
# 上限は顔がカメラに寄る演出（Rin_list_b_08 で 1.44 倍）を弾かないよう 2.0。
EYE_DIST_RATIO_MIN = 0.6
EYE_DIST_RATIO_MAX = 2.0
MOUTH_RING_PX = 15              # フォールバック塗り色のリングサンプリング幅

# --- 消去楕円（口素材の実寸基準、compute_erase_geometry）---
# 大きさは mouth/ の口素材（open/half/closed のアルファ実寸の最大値）から決める。
# face_info の口bboxの高さは使わない: 口を閉じた立ち絵では bbox が線状
# （高さ≒目間距離の5%）になり、Viduが生成する開いた口の下半分が
# 消去範囲の外に残るため（2026-09-12）。
ERASE_SPRITE_W_SCALE = 1.4      # 口素材幅に対する楕円の倍率（横）。1.3 では大笑いの口角が残る
ERASE_SPRITE_H_SCALE = 1.1      # 同（縦）。1.2 以上は小顔キャラで顎の陰影・頬の赤みまで塗る
# 楕円半径（膨張込み）の上限（目間距離に対する比率）。小顔キャラで口素材が
# 顔に対して大きい場合に、楕円が顎や頬に届くのを防ぐ。横は 0.30 で大笑いの
# 口角が残り 0.34 で小顔キャラの頬を塗るため 0.32、縦は 0.22 で小顔キャラの
# 口が残るため 0.24
ERASE_SEMI_W_MAX_NORM = 0.32
ERASE_SEMI_H_MAX_NORM = 0.24
ERASE_DILATE_RATIO = 0.06       # 膨張px = clip(目間距離×比率, MIN, MAX)
ERASE_DILATE_MIN_PX = 3
ERASE_DILATE_MAX_PX = 10
ERASE_FEATHER_RATIO = 0.04      # フェザー帯の幅px = clip(目間距離×比率, MIN, MAX)。境界を中心に内外へ半分ずつ
ERASE_FEATHER_MIN_PX = 4
ERASE_FEATHER_MAX_PX = 8
# 口素材サイズ / 目間距離 の許容範囲（半開きを open に選んだ場合や、
# 別倍率の手描き素材に差し替えた場合の保険）と、素材が読めない時の既定値
ERASE_SPRITE_W_NORM_MIN = 0.20
ERASE_SPRITE_W_NORM_MAX = 0.50
ERASE_SPRITE_H_NORM_MIN = 0.15
ERASE_SPRITE_H_NORM_MAX = 0.35
ERASE_SPRITE_W_NORM_DEFAULT = 0.37
ERASE_SPRITE_H_NORM_DEFAULT = 0.25
SPRITE_ALPHA_THRESHOLD = 128    # 口素材の「不透明」判定

# --- 旧方式（face_info の口bbox基準）。erase_geom 未指定時の互換用 ---
MOUTH_DILATE_PX = 10
MOUTH_FEATHER_PX = 10
ERASE_H_SCALE = 1.5  # 消去楕円の高さ倍率（生成口が下唇付近に出るため拡大）
ERASE_W_SCALE = 1.2  # 消去楕円の幅倍率（左右も少し余裕を持たせる）


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _progress(phase: int, frame: int = 0, total: int = 0, msg: str = ""):
    """進捗をstdoutに出力（GUIが解析）"""
    parts = [f"[PROGRESS] phase={phase}"]
    if total > 0:
        parts.append(f"frame={frame}/{total}")
    if msg:
        parts.append(f"msg={msg}")
    print(" ".join(parts), flush=True)


def _mask_centroid(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    """マスク重心を計算"""
    coords = np.where(mask > 0)
    if len(coords[0]) == 0:
        return None
    return float(np.mean(coords[1])), float(np.mean(coords[0]))


def _interpolate_points(points: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """2Dポイント列の無効フレームを線形補間で埋める。

    Args:
        points: (N, 2) float32
        valid: (N,) bool/uint8
    Returns:
        filled: (N, 2) float32
    """
    N = len(points)
    filled = points.copy()
    v = valid.astype(np.uint8).reshape(-1)

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
            gap = float(next_ok - prev_ok)
            for j in range(start, end):
                t = float(j - prev_ok) / gap
                filled[j] = (1.0 - t) * points[prev_ok] + t * points[next_ok]
        elif prev_ok >= 0:
            for j in range(start, end):
                filled[j] = points[prev_ok]
        elif next_ok >= 0:
            for j in range(start, end):
                filled[j] = points[next_ok]

    return filled


def _compute_mouth_positions(
    left_eyes: np.ndarray,
    right_eyes: np.ndarray,
    mouth_norm_x: float,
    mouth_norm_y: float,
    mouth_w_norm: float,
    mouth_h_norm: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """目位置からface_info比率を使って口の中心・サイズ・角度を計算。

    Returns:
        centers (N,2), widths (N,), heights (N,), angles_deg (N,)
    """
    N = len(left_eyes)
    eye_mids = (left_eyes + right_eyes) / 2.0
    eye_vecs = right_eyes - left_eyes
    eye_dists = np.linalg.norm(eye_vecs, axis=1)
    # ゼロ除算防止
    eye_dists_safe = np.maximum(eye_dists, 1e-6)

    # 単位ベクトル
    ux = eye_vecs / eye_dists_safe[:, None]  # (N, 2) X軸: 左目→右目
    # Y軸: Xに直交、下向き (90° CCW in image coords)
    uy = np.column_stack([-ux[:, 1], ux[:, 0]])  # (-uy_x, ux_x) → down

    # 口中心
    centers = (eye_mids
               + mouth_norm_x * eye_dists_safe[:, None] * ux
               + mouth_norm_y * eye_dists_safe[:, None] * uy)

    # 口サイズ
    widths = mouth_w_norm * eye_dists_safe
    heights = mouth_h_norm * eye_dists_safe

    # 顔の角度（quadの回転用）
    # eye_vecは解剖学的left→rightで画像上は右→左(≈180°)なので、
    # 180°足して顔の向きに変換し、[-180,180]に正規化
    raw_angles = np.degrees(np.arctan2(eye_vecs[:, 1], eye_vecs[:, 0]))
    angles_deg = ((raw_angles + 180.0 + 180.0) % 360.0) - 180.0

    return centers.astype(np.float32), widths.astype(np.float32), \
        heights.astype(np.float32), angles_deg.astype(np.float32)


def _reject_outliers(
    positions: np.ndarray,
    valid: np.ndarray,
    ref_eye_dist: float,
    kernel_size: int = 5,
) -> np.ndarray:
    """メディアンフィルタで外れ値を除去。

    eye_distの OUTLIER_DEVIATION_RATIO 以上ずれたフレームを invalid にする。
    """
    if np.sum(valid) < 3:
        return valid.copy()

    new_valid = valid.copy()
    threshold = ref_eye_dist * OUTLIER_DEVIATION_RATIO

    for dim in range(2):  # x, y
        vals = positions[:, dim].copy()
        # invalid部分を前後で埋めてからmedian filter
        filled = _interpolate_points(
            positions, valid
        )[:, dim]
        med = median_filter_1d(filled, kernel_size=kernel_size)
        deviation = np.abs(vals - med)
        # valid かつ deviation が大きいフレームを reject
        outlier = (valid.astype(bool)) & (deviation > threshold)
        new_valid[outlier] = 0

    rejected = int(np.sum(valid.astype(int) - new_valid.astype(int)))
    if rejected > 0:
        print(f"[eye_tracker] Rejected {rejected} outlier frames")
    return new_valid


def _wrap_angle_deg(angle):
    """角度を [-180, 180) に正規化（float / ndarray）。"""
    return ((angle + 180.0) % 360.0) - 180.0


def _reject_eye_distance(
    left_eyes: np.ndarray,
    right_eyes: np.ndarray,
    valid: np.ndarray,
    ref_eye_dist: float,
) -> np.ndarray:
    """目間距離が基準の EYE_DIST_RATIO_MIN〜MAX 倍を外れたフレームを無効にする。

    両目の検出が同じ点に崩れる追跡破綻（目間距離 → ほぼ 0）を弾くのが目的。
    顔が寄る演出で目間距離が伸びるのは正常なので上限は緩い。
    """
    if ref_eye_dist <= 0 or np.sum(valid) == 0:
        return valid.copy()
    v = valid.astype(bool)
    dist = np.linalg.norm(right_eyes - left_eyes, axis=1)
    ratio = dist / float(ref_eye_dist)
    bad = v & ((ratio < EYE_DIST_RATIO_MIN) | (ratio > EYE_DIST_RATIO_MAX))
    new_valid = valid.copy()
    new_valid[bad] = 0
    n = int(np.sum(bad))
    if n > 0:
        print(f"[eye_tracker] Rejected {n} frames by eye distance "
              f"(ratio {ratio[v].min():.2f}-{ratio[v].max():.2f} vs allowed "
              f"{EYE_DIST_RATIO_MIN}-{EYE_DIST_RATIO_MAX})")
    return new_valid


# ===================================================================
# Phase 1: Eye Tracking
# ===================================================================

def phase1_eye_tracking(
    video_path: str,
    face_info: dict,
    detector: SAM3MouthDetector,
    smooth_cutoff: float = 3.0,
    *,
    sprite_size: Tuple[int, int],
    stats: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, float, np.ndarray]:
    """目を検出し、口位置のquadを算出する。

    quad の契約（AG の MotionPNGPlayer と共通、2026-09-13、track_format = 2）:
      - quad は口 PNG（mouth/ の open/half/closed、同一キャンバス）全体を写す枠。
        大きさ = sprite_size × (そのフレームの目間距離 ÷ 立ち絵の目間距離)。
        PNG も立ち絵の目間距離も元画像 px なので、動画/画像の解像度比は約分される。
      - 角度は「口 PNG を切り出した向きからの回転量」。立ち絵の目の角度を基準にした
        相対角で、静止時 0 度（口 PNG は立ち絵と同じ向きで切り出されているため）。
      - 中心は face_info の口の相対位置（目の中点からの比率）から。

    Args:
        sprite_size: 口 PNG のキャンバスサイズ (幅, 高さ)。track_utils.sprite_canvas_size()
        stats: dict を渡すと無効化・補間の内訳（フレーム数）を書き込む（メトリクス用）

    Returns:
        quads (N,4,2), valid (N,), confidence (N,),
        video_w, video_h, fps, raw_valid (N,)
        valid は補間で全1化した出力用フラグ（NPZ→プレイヤー互換）、
        raw_valid は補間前の生の有効フラグ（メトリクスの valid_rate 用）。
    """
    # face_info から比率を取得
    mouth_norm_x = face_info["mouth_norm_x"]
    mouth_norm_y = face_info["mouth_norm_y"]
    ref_eye_dist = float(face_info["eye_distance"])

    # quad の大きさは face_info の口 bbox ではなく口 PNG のキャンバスから決める
    # （口を閉じた立ち絵では bbox が線状になり、PNG を枠に押し込む AG のプレイヤーで
    # 口が潰れる）。_compute_mouth_positions は「目間距離 × 比率」で幅・高さを出す
    # ので、比率に PNG サイズ ÷ 立ち絵の目間距離 を渡す。
    sprite_w, sprite_h = int(sprite_size[0]), int(sprite_size[1])
    if sprite_w <= 0 or sprite_h <= 0 or ref_eye_dist <= 0:
        raise ValueError(f"invalid sprite_size={sprite_size} / eye_distance={ref_eye_dist}")
    sprite_w_norm = sprite_w / ref_eye_dist
    sprite_h_norm = sprite_h / ref_eye_dist

    # フォールバック用: face_info の目位置
    ref_left = np.array([face_info["left_eye"]["cx"], face_info["left_eye"]["cy"]])
    ref_right = np.array([face_info["right_eye"]["cx"], face_info["right_eye"]["cy"]])

    # 角度の基準: 立ち絵の目の並び（face_info.eye_angle_deg と同じ量を目位置から算出）。
    # _compute_mouth_positions と同じ変換（right−left の向き + 180 度）で揃える
    ref_vec = ref_right - ref_left
    ref_angle = float(_wrap_angle_deg(
        np.degrees(np.arctan2(ref_vec[1], ref_vec[0])) + 180.0))
    ref_img_w = face_info["image_width"]
    ref_img_h = face_info["image_height"]

    # 動画を開く
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    # CAP_PROP_FRAME_COUNT はWebM/VFRで0や過大値を返すことがあるため参考値扱い。
    # 実際に読めたフレームをリストに蓄積し、最後に配列化してフレーム数を確定する。
    reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[phase1] Video: {video_w}x{video_h}, {reported_frames} frames (reported), {fps:.1f} fps")

    # 解像度スケール (face_info画像 → 動画)
    scale_x = video_w / ref_img_w
    scale_y = video_h / ref_img_h

    left_list: list = []
    right_list: list = []
    valid_list: list = []
    conf_list: list = []
    le_area_list: list = []
    re_area_list: list = []

    try:
        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            _progress(1, i + 1, max(reported_frames, i + 1), "Eye tracking")

            # 左目検出
            le_result = detector.detect_mouth(frame, prompt="left eye")
            # 右目検出
            re_result = detector.detect_mouth(frame, prompt="right eye")

            if le_result is not None and re_result is not None:
                le_mask, le_bbox, le_center = le_result
                re_mask, re_bbox, re_center = re_result

                # マスク重心を使用（bbox中心より安定）
                le_centroid = _mask_centroid(le_mask)
                re_centroid = _mask_centroid(re_mask)

                left_list.append(le_centroid if le_centroid else le_center)
                right_list.append(re_centroid if re_centroid else re_center)
                valid_list.append(1)
                conf_list.append(1.0)

                # マスク面積を記録（瞬き検出用）
                le_area_list.append(float(np.sum(le_mask > 0)))
                re_area_list.append(float(np.sum(re_mask > 0)))
            else:
                left_list.append((0.0, 0.0))
                right_list.append((0.0, 0.0))
                valid_list.append(0)
                conf_list.append(0.0)
                le_area_list.append(0.0)
                re_area_list.append(0.0)

            i += 1
    finally:
        cap.release()

    # 実読取数を正とする
    total_frames = len(valid_list)
    if total_frames == 0:
        raise RuntimeError(f"No frames could be read from video: {video_path}")
    if reported_frames != total_frames:
        print(f"[phase1] Frame count adjusted: reported={reported_frames} "
              f"-> actual={total_frames}")

    left_eyes = np.asarray(left_list, dtype=np.float32).reshape(total_frames, 2)
    right_eyes = np.asarray(right_list, dtype=np.float32).reshape(total_frames, 2)
    valid = np.asarray(valid_list, dtype=np.uint8)
    confidence = np.asarray(conf_list, dtype=np.float32)
    le_areas = np.asarray(le_area_list, dtype=np.float32)
    re_areas = np.asarray(re_area_list, dtype=np.float32)

    n_valid = int(np.sum(valid))
    print(f"[phase1] Eye detection: {n_valid}/{total_frames} frames valid "
          f"({100*n_valid/max(1,total_frames):.1f}%)")
    n_undetected = total_frames - n_valid

    # フォールバック: 全フレーム検出失敗
    if n_valid == 0:
        print("[phase1] WARNING: No eyes detected in any frame. Using face_info fallback.")
        for i in range(total_frames):
            left_eyes[i] = ref_left * np.array([scale_x, scale_y])
            right_eyes[i] = ref_right * np.array([scale_x, scale_y])
            valid[i] = 1
            confidence[i] = 0.1

    # 瞬き検出: マスク面積が中央値の70%以下のフレームをinvalidにする
    # + 前後2フレームのパディングで半目フレームも補間対象にする
    BLINK_AREA_THRESHOLD = 0.70  # 中央値の70%以下で瞬きと判定
    # パディングは ±2 のまま。±4 も試したが（2026-09-13）、顔を横に振る区間で瞬き判定が
    # 面積低下を拾って無効化が伸び、実際の動きが直線補間で潰れて口が最大 10 px ずれた
    # （Hina_list_a_24 f43〜59・f77〜94）。半開きの跳ねは後段の移動中央値で消える
    BLINK_PAD_FRAMES = 2         # 瞬き区間の前後に追加するパディング

    n_blink = 0
    valid_mask = valid.astype(bool)
    if np.sum(valid_mask) > 0:
        le_median_area = float(np.median(le_areas[valid_mask]))
        re_median_area = float(np.median(re_areas[valid_mask]))

        blink_raw = np.zeros(total_frames, dtype=bool)
        for i in range(total_frames):
            if not valid_mask[i]:
                continue
            le_ratio = le_areas[i] / max(1.0, le_median_area)
            re_ratio = re_areas[i] / max(1.0, re_median_area)
            if le_ratio < BLINK_AREA_THRESHOLD or re_ratio < BLINK_AREA_THRESHOLD:
                blink_raw[i] = True

        # パディング: 瞬きフレームの前後にも invalid を広げる
        blink_padded = blink_raw.copy()
        for i in range(total_frames):
            if blink_raw[i]:
                for j in range(max(0, i - BLINK_PAD_FRAMES),
                               min(total_frames, i + BLINK_PAD_FRAMES + 1)):
                    blink_padded[j] = True

        blink_count = int(np.sum(blink_padded & valid_mask))
        valid[blink_padded] = 0
        n_blink = blink_count

        if blink_count > 0:
            print(f"[phase1] Blink detection: {blink_count} frames invalidated "
                  f"(threshold={BLINK_AREA_THRESHOLD:.0%}, pad=±{BLINK_PAD_FRAMES})")

    # 外れ値除去
    scaled_ref_eye_dist = ref_eye_dist * scale_x
    n_before = int(np.sum(valid))
    valid = _reject_outliers(left_eyes, valid, scaled_ref_eye_dist)
    valid = _reject_outliers(right_eyes, valid, scaled_ref_eye_dist)
    n_outlier_pos = n_before - int(np.sum(valid))

    # 目間距離の基準比較（両目が同じ点に崩れる追跡破綻を弾く）
    n_before = int(np.sum(valid))
    valid = _reject_eye_distance(left_eyes, right_eyes, valid, scaled_ref_eye_dist)
    n_outlier_eye_dist = n_before - int(np.sum(valid))

    if stats is not None:
        stats.update({
            "n_undetected": int(n_undetected),
            "n_blink": int(n_blink),
            "n_outlier_pos": int(n_outlier_pos),
            "n_outlier_eye_dist": int(n_outlier_eye_dist),
            "eye_dist_ratio_min": EYE_DIST_RATIO_MIN,
            "eye_dist_ratio_max": EYE_DIST_RATIO_MAX,
            "n_interpolated": int(total_frames - int(np.sum(valid))),
            "blink_pad_frames": int(BLINK_PAD_FRAMES),
            "smooth_median_size": int(SMOOTH_MEDIAN_SIZE),
        })

    # 補間（瞬き・外れ値・目間距離異常のフレームを前後から線形補間）
    left_eyes = _interpolate_points(left_eyes, valid)
    right_eyes = _interpolate_points(right_eyes, valid)

    # 口位置を計算
    centers, widths, heights, angles = _compute_mouth_positions(
        left_eyes, right_eyes,
        mouth_norm_x, mouth_norm_y,
        sprite_w_norm, sprite_h_norm,
    )
    # 立ち絵の目の角度を基準にした相対角（静止時 0 度）
    angles = _wrap_angle_deg(angles - ref_angle).astype(np.float32)

    # quad構築
    quads = compose_quads_vectorized(centers, widths, heights, angles)

    # 移動中央値（まばたきの半開きで目マスクの重心が下にずれ、口が縦に 1〜2 px
    # 跳ねるのを除く。単調な動きは通し、4 コマ以内に行って戻る動きだけ削る。
    # EMA の前段に置く。既存素材は tools/smooth_track_median.py で同じ処理）
    quads = smooth_quads_moving_median(quads, SMOOTH_MEDIAN_SIZE)

    # スムージング
    beta = one_pole_beta(smooth_cutoff, fps)
    quads = smooth_quads_zero_phase(quads, valid, beta)

    # 補間・スムージング後は全フレーム有効（プレイヤーが口スプライトを表示するために必要）
    # メトリクス計算用に補間前の生validは raw_valid として保持する
    raw_valid = valid.astype(np.uint8).copy()
    valid_out = np.ones(total_frames, dtype=np.uint8)

    _progress(1, total_frames, total_frames, "Eye tracking done")
    return quads, valid_out, confidence, video_w, video_h, fps, raw_valid


# ===================================================================
# Fill color extraction (from mouth-erased base image)
# ===================================================================
# 口消しの塗り色は、タブ⑥で人間がブラシ塗りした「口消し済み立ち絵」の
# mouth bbox領域から読み取る。動画からのリングサンプリングと違い推定を
# 含まないため、チーク・輪郭線・影の混入が原理的に起きない。
# 読み取れない場合は None を返し、従来のリングサンプリングへフォールバック。

FILL_COLOR_MAX_STD = 12.0     # これ以上ばらつく領域は「単色でない」とみなす（実測は全キャラ<5）
FILL_COLOR_DARK_V = 120       # max(B,G,R)がこの値未満の画素を「暗画素」とする
                              # 注: 褐色肌キャラでは誤フォールバックし得るが、
                              #     フォールバック先が従来動作のため劣化はしない
FILL_COLOR_DARK_RATIO = 0.05  # 暗画素がこの割合以上なら口消し忘れ・服などの混入を疑う
FILL_COLOR_MIN_PIXELS = 50    # bbox領域がこれ未満なら信頼しない
FILL_COLOR_MIN_ALPHA = 200.0  # bbox領域の平均アルファがこれ未満なら透明領域とみなす


def extract_fill_color_from_image(
    image_path: Optional[str],
    face_info: dict,
) -> Tuple[Optional[np.ndarray], str]:
    """口消し済み立ち絵のmouth bbox領域から塗り色(BGR)を抽出する。

    face_infoのbbox座標は face_info生成時の画像サイズ基準なので、
    現在のPNG実サイズとの比でスケール補正する（立ち絵の高解像度差し替え対策）。

    Returns:
        (color, reason): 成功時は (float32 BGR(3,), "ok")。
        失敗時は (None, 理由文字列) — 呼び出し側はリングサンプリングへフォールバックする。
    """
    try:
        mouth = face_info.get("mouth") or {}
        bbox = mouth.get("bbox")
        if not bbox or len(bbox) != 4:
            return None, "face_info has no mouth bbox"
        if not image_path or not os.path.isfile(image_path):
            return None, f"image not found: {image_path}"

        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8),
                           cv2.IMREAD_UNCHANGED)
        if img is None:
            return None, "failed to decode image"
        if img.dtype != np.uint8:
            return None, f"unsupported dtype: {img.dtype}"
        if img.ndim != 3 or img.shape[2] not in (3, 4):
            return None, f"unsupported shape: {img.shape}"

        ph, pw = img.shape[:2]
        fw = int(face_info.get("image_width") or pw)
        fh = int(face_info.get("image_height") or ph)
        sx = pw / max(1, fw)
        sy = ph / max(1, fh)
        x0 = max(0, min(pw, int(round(bbox[0] * sx))))
        y0 = max(0, min(ph, int(round(bbox[1] * sy))))
        x1 = max(0, min(pw, int(round(bbox[2] * sx))))
        y1 = max(0, min(ph, int(round(bbox[3] * sy))))
        if x1 - x0 < 2 or y1 - y0 < 2:
            return None, "scaled bbox is empty"

        region = img[y0:y1, x0:x1]
        if img.shape[2] == 4:
            if float(region[:, :, 3].astype(np.float32).mean()) < FILL_COLOR_MIN_ALPHA:
                return None, "mouth bbox region is transparent"

        bgr = region[:, :, :3].reshape(-1, 3).astype(np.float32)
        if len(bgr) < FILL_COLOR_MIN_PIXELS:
            return None, "mouth bbox region too small"

        max_std = float(bgr.std(axis=0).max())
        if max_std >= FILL_COLOR_MAX_STD:
            return None, f"region not flat (std={max_std:.1f}); mouth may not be erased"

        dark_ratio = float((bgr.max(axis=1) < FILL_COLOR_DARK_V).mean())
        if dark_ratio >= FILL_COLOR_DARK_RATIO:
            return None, f"region too dark (dark={dark_ratio:.0%}); may not be skin"

        return np.median(bgr, axis=0).astype(np.float32), "ok"
    except Exception as e:
        return None, f"extract failed: {e}"


def fill_color_from_face_info(
    face_info: dict,
    base_dir: str,
) -> Tuple[Optional[np.ndarray], str]:
    """face_infoから立ち絵を解決して塗り色を抽出する（CLI・GUIテスト用）。

    face_info["image"] のファイル名を優先し、無ければ base_dir 内の
    単一画像ファイルを探す（リネーム済みでも追従できるように）。
    """
    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    candidates: list[str] = []
    name = face_info.get("image")
    if name:
        p = os.path.join(base_dir, str(name))
        if os.path.isfile(p):
            candidates.append(p)
    try:
        imgs = [os.path.join(base_dir, f) for f in sorted(os.listdir(base_dir))
                if f.lower().endswith(exts)]
    except OSError:
        imgs = []
    if len(imgs) == 1 and imgs[0] not in candidates:
        candidates.append(imgs[0])

    last_reason = "no base image found"
    for p in candidates:
        color, reason = extract_fill_color_from_image(p, face_info)
        if color is not None:
            return color, f"ok ({os.path.basename(p)})"
        last_reason = reason
    return None, last_reason


# ===================================================================
# Erase geometry (from mouth sprites)
# ===================================================================

SPRITE_FILE_CANDIDATES = (
    ("open.png", "mouth_open.png"),
    ("half.png", "mouth_half.png"),
    ("closed.png", "mouth_closed.png"),
)


@dataclass
class EraseGeometry:
    """口消し楕円の幾何（動画px単位、基準フレーム＝quad幅の中央値）。

    sprite_w / sprite_h: 口素材（open/half/closed のアルファ実寸の最大値）の幅・高さ
    eye_dist: 目間距離（動画px）。膨張・フェザーの大きさとクランプの基準
    source: "sprites"（口素材から取得）/ "default"（既定比率にフォールバック）
    """
    sprite_w: float
    sprite_h: float
    eye_dist: float
    source: str = "sprites"
    reason: str = ""

    def params(self, scale: float = 1.0) -> Tuple[float, float, int, int]:
        """(semi_w, semi_h, dilate_px, feather_px) を返す。scale はフレーム毎の拡縮比。

        semi_w/semi_h は膨張前の楕円半径。膨張込みの半径が目間距離×上限比率を
        超えないように切り詰める。
        """
        eye = self.eye_dist * scale
        dilate = int(np.clip(round(eye * ERASE_DILATE_RATIO),
                             ERASE_DILATE_MIN_PX, ERASE_DILATE_MAX_PX))
        feather = int(np.clip(round(eye * ERASE_FEATHER_RATIO),
                              ERASE_FEATHER_MIN_PX, ERASE_FEATHER_MAX_PX))
        semi_w = self.sprite_w * ERASE_SPRITE_W_SCALE / 2.0 * scale
        semi_h = self.sprite_h * ERASE_SPRITE_H_SCALE / 2.0 * scale
        semi_w = max(2.0, min(semi_w, ERASE_SEMI_W_MAX_NORM * eye - dilate))
        semi_h = max(2.0, min(semi_h, ERASE_SEMI_H_MAX_NORM * eye - dilate))
        return semi_w, semi_h, dilate, feather

    def describe(self) -> str:
        sw, sh, d, f = self.params()
        note = f": {self.reason}" if self.reason else ""
        return (f"sprite={self.sprite_w:.0f}x{self.sprite_h:.0f}px "
                f"eye_dist={self.eye_dist:.0f}px -> ellipse semi={sw:.0f}x{sh:.0f}px "
                f"dilate={d}px feather={f}px ({self.source}{note})")


def measure_sprite_extent(path: str) -> Optional[Tuple[int, int]]:
    """口素材PNGの不透明部分（最大連結成分）の幅・高さを返す。

    キャンバスにはパディングが含まれるので画像サイズは使わない。
    読めない・アルファ無し・不透明画素が無い場合は None。
    """
    try:
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    except Exception:
        return None
    if img is None or img.ndim != 3 or img.shape[2] != 4:
        return None
    alpha = (img[:, :, 3] >= SPRITE_ALPHA_THRESHOLD).astype(np.uint8)
    n, _labels, stats, _ = cv2.connectedComponentsWithStats(alpha, connectivity=8)
    if n <= 1:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    w = int(stats[k, cv2.CC_STAT_WIDTH])
    h = int(stats[k, cv2.CC_STAT_HEIGHT])
    if w < 2 or h < 2:
        return None
    return w, h


def compute_erase_geometry(
    mouth_dir: Optional[str],
    face_info: dict,
    video_w: int,
    video_h: int,
) -> Optional[EraseGeometry]:
    """口素材と face_info から消去楕円の幾何を決める。

    - 大きさ: mouth_dir の open/half/closed のアルファ実寸の最大値
      （ラベルを取り違えても最大の口が基準になる）× ERASE_SPRITE_W/H_SCALE
    - 口素材は準備動画（立ち絵と同じ解像度）から抽出されているので、
      face_info 座標と同じ video_w/image_width 比でスケールする
    - 目間距離に対する比率でクランプし、口素材が読めなければ既定比率を使う
    - face_info に目間距離が無い場合は None（呼び出し側は旧方式で続行）
    """
    try:
        eye_base = float(face_info.get("eye_distance") or 0.0)
        img_w = float(face_info.get("image_width") or 0.0)
    except (TypeError, ValueError):
        return None
    if eye_base <= 0:
        return None
    scale = (float(video_w) / img_w) if (img_w > 0 and video_w > 0) else 1.0
    eye = eye_base * scale

    w_max = 0
    h_max = 0
    used: list[str] = []
    if mouth_dir and os.path.isdir(mouth_dir):
        for candidates in SPRITE_FILE_CANDIDATES:
            for name in candidates:
                p = os.path.join(mouth_dir, name)
                if not os.path.isfile(p):
                    continue
                ext = measure_sprite_extent(p)
                if ext is not None:
                    w_max = max(w_max, ext[0])
                    h_max = max(h_max, ext[1])
                    used.append(name)
                break

    notes: list[str] = []
    if w_max > 0 and h_max > 0:
        w_norm = w_max * scale / eye
        h_norm = h_max * scale / eye
        w_c = float(np.clip(w_norm, ERASE_SPRITE_W_NORM_MIN, ERASE_SPRITE_W_NORM_MAX))
        h_c = float(np.clip(h_norm, ERASE_SPRITE_H_NORM_MIN, ERASE_SPRITE_H_NORM_MAX))
        notes.append(",".join(used))
        if w_c != w_norm or h_c != h_norm:
            notes.append(f"clamped from w={w_norm:.2f},h={h_norm:.2f} x eye_dist")
        w_norm, h_norm = w_c, h_c
        source = "sprites"
    else:
        w_norm, h_norm = ERASE_SPRITE_W_NORM_DEFAULT, ERASE_SPRITE_H_NORM_DEFAULT
        source = "default"
        notes.append("no readable mouth sprite"
                     + (f" in {mouth_dir}" if mouth_dir else ""))

    return EraseGeometry(
        sprite_w=w_norm * eye,
        sprite_h=h_norm * eye,
        eye_dist=eye,
        source=source,
        reason="; ".join(notes),
    )


# ===================================================================
# Phase 2: Mouth Erasure
# ===================================================================

def phase2_mouth_erasure(
    video_path: str,
    quads: np.ndarray,
    valid: np.ndarray,
    out_video: str,
    video_w: int,
    video_h: int,
    fps: float,
    keep_audio: bool = True,
    use_alpha: bool = False,
    bg_color_specs: Optional[list] = None,
    bg_tolerance: int = 65,
    fill_color: Optional[np.ndarray] = None,
    erase_geom: Optional[EraseGeometry] = None,
):
    """追跡済みの口位置を楕円で肌色に塗りつぶす（SAM3不使用）。

    消去楕円の大きさは erase_geom（口素材の実寸基準、compute_erase_geometry）
    で決め、フレーム毎に quad幅/中央値 の比で拡縮して、face_info 由来の
    口中心（quad中心）に置く。下方向へのずらしは顎に当たるので行わない。
    erase_geom が None の場合は旧方式（quadサイズ＝face_infoの口bbox基準）。

    fill_color: 塗り色(BGR)。口消し済み立ち絵から抽出した色を渡す。
    None の場合は従来どおり毎フレーム外側リングからサンプリングする。

    背景透過のアルファは口消し前のフレームから作る。楕円が顔の外に
    はみ出しても、塗った肌色は背景として透過される。
    """
    total_frames = len(quads)

    if fill_color is not None:
        fill_color = np.asarray(fill_color, dtype=np.float32).reshape(3)
        print(f"[phase2] Fill color from base image: "
              f"BGR={fill_color.astype(int).tolist()}")
    else:
        print("[phase2] Fill color: per-frame ring sampling (fallback)")

    # quad から期待口中心・サイズを取得
    centers_arr, widths_arr, heights_arr, angles_arr = decompose_quads_vectorized(quads)

    if erase_geom is not None:
        median_w = float(np.median(widths_arr)) if len(widths_arr) else 0.0
        print(f"[phase2] Erase geometry: {erase_geom.describe()}")
    else:
        median_w = 0.0
        print("[phase2] Erase geometry: legacy (face_info mouth bbox)")

    # 動画を開く
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    is_webm = out_video.lower().endswith(".webm")
    writer = None
    try:
        # ライターを準備
        if use_alpha and is_webm:
            writer = FFmpegAlphaWriter(out_video, fps, video_w, video_h)
        else:
            writer = _FFmpegH264Writer(out_video, fps, video_w, video_h)

        # 背景色スペック（WebMアルファ用）
        parsed_colors = []
        if bg_color_specs:
            parsed_colors = [parse_color_spec(s) for s in bg_color_specs]

        print(f"[phase2] Mouth erasure: {total_frames} frames, output={out_video}")

        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            _progress(2, i + 1, total_frames, "Mouth erasure")

            cx = float(centers_arr[i, 0])
            cy = float(centers_arr[i, 1])
            w = float(widths_arr[i])
            h = float(heights_arr[i])
            angle = float(angles_arr[i])

            # ── 背景透過アルファ（口消し前のフレームから作る）──
            alpha = None
            if use_alpha and is_webm:
                if parsed_colors:
                    alpha = create_alpha_mask_multi_color(frame, parsed_colors)
                else:
                    alpha = create_alpha_hybrid(frame, tolerance=bg_tolerance,
                                                 edge_width=5)

            # ── 口位置の楕円範囲を塗りつぶし ──
            if erase_geom is not None:
                scale_f = (w / median_w) if median_w > 1e-6 else 1.0
                scale_f = float(np.clip(scale_f, 0.5, 2.0))
                semi_w, semi_h, dil, fea = erase_geom.params(scale_f)
                frame = _erase_ellipse(frame, cx, cy, semi_w, semi_h, angle,
                                       fill_color=fill_color,
                                       dilate_px=dil, feather_px=fea)
            else:
                frame = _erase_mouth_quad(frame, cx, cy, w, h, angle,
                                          fill_color=fill_color)

            # ── WebMアルファ出力 ──
            if alpha is not None:
                if not parsed_colors:
                    frame = despill_green(frame, alpha)
                bgra = np.dstack([frame, alpha])
                writer.write(bgra)
            else:
                writer.write(frame)

        release_ok = writer.release()
        writer = None
        if not release_ok:
            raise RuntimeError(
                f"[phase2] video writer (ffmpeg) reported failure; "
                f"output discarded: {out_video}"
            )
    except BaseException:
        # 例外時: ffmpegプロセスをkillし、書きかけの出力と一時ログを後始末
        if writer is not None:
            abort_ffmpeg_writer(writer)
            writer = None
        try:
            if os.path.isfile(out_video):
                os.unlink(out_video)
        except OSError:
            pass
        raise
    finally:
        cap.release()

    _progress(2, total_frames, total_frames, "Mouth erasure done")

    # 音声mux
    if keep_audio:
        print("[phase2] Muxing audio...")
        base, ext = os.path.splitext(out_video)
        final_path = f"{base}_final{ext}"
        mux_fn = mux_audio_ffmpeg_webm if is_webm else mux_audio_ffmpeg
        if mux_fn(out_video, video_path, final_path):
            os.replace(final_path, out_video)
            print("[phase2] Audio muxed successfully")
        else:
            # mux失敗時: 壊れかけのfinalファイルを削除（無音のout_videoは残す）
            try:
                if os.path.isfile(final_path):
                    os.unlink(final_path)
            except OSError:
                pass
            print("[phase2] Audio mux failed (continuing without audio)")


def _erase_mouth_quad(
    frame: np.ndarray,
    cx: float, cy: float,
    mouth_w: float, mouth_h: float,
    angle_deg: float,
    fill_color: Optional[np.ndarray] = None,
) -> np.ndarray:
    """旧方式: mouth quad（face_infoの口bbox基準）の楕円範囲を塗る。

    erase_geom を渡せない呼び出し元の互換用。口を閉じた立ち絵では
    quad が線状になり消去範囲が足りないので、通常は _erase_ellipse を
    compute_erase_geometry の幾何で使う。
    """
    return _erase_ellipse(
        frame, cx, cy,
        mouth_w * ERASE_W_SCALE / 2.0, mouth_h * ERASE_H_SCALE / 2.0,
        angle_deg, fill_color=fill_color,
        dilate_px=MOUTH_DILATE_PX, feather_px=MOUTH_FEATHER_PX,
    )


def _erase_ellipse(
    frame: np.ndarray,
    cx: float, cy: float,
    semi_w: float, semi_h: float,
    angle_deg: float,
    fill_color: Optional[np.ndarray] = None,
    dilate_px: int = MOUTH_DILATE_PX,
    feather_px: int = MOUTH_FEATHER_PX,
) -> np.ndarray:
    """(cx, cy) を中心とする半径 (semi_w, semi_h) の楕円範囲を肌色で塗りつぶす。

    1. 楕円マスクを作成
    2. dilate_px 膨張してマージンを確保
    3. 塗り色を決定（fill_color指定時はそれを使用、
       未指定時は外側リングから肌色をサンプリング）
    4. 塗りつぶし + feather_px のフェザーブレンド
    """
    fh, fw = frame.shape[:2]

    if semi_w < 2 or semi_h < 2:
        return frame

    # 楕円マスク
    erase_mask = np.zeros((fh, fw), dtype=np.uint8)
    cv2.ellipse(
        erase_mask,
        (int(cx), int(cy)),
        (int(semi_w), int(semi_h)),
        angle_deg, 0, 360, 255, -1,
    )

    # 膨張してマージン確保
    dilate_px = max(0, int(dilate_px))
    if dilate_px > 0:
        dilate_k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (dilate_px * 2 + 1, dilate_px * 2 + 1))
        erase_mask = cv2.dilate(erase_mask, dilate_k, iterations=1)

    if fill_color is not None:
        # 口消し済み立ち絵から抽出した色で塗る（リングサンプリング不要）
        skin_color = np.clip(fill_color, 0, 255).astype(np.uint8)
    else:
        # リングマスク（消去範囲の外側から肌色をサンプリング）
        # 上・右・左のみ（下は顎やグリーンバックに接触するため除外）
        ring_k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (MOUTH_RING_PX * 2 + 1, MOUTH_RING_PX * 2 + 1))
        outer = cv2.dilate(erase_mask, ring_k, iterations=1)
        ring = outer & ~erase_mask

        # 楕円中心より下の部分をカット（顎・グリーンバック回避）
        # cyが負でもインデックス反転しないよう0でクランプ
        center_y = max(0, int(cy))
        ring[center_y:, :] = 0

        # リングから肌色をサンプリング
        ring_pixels = frame[ring > 0]
        if len(ring_pixels) == 0:
            return frame
        skin_color = ring_pixels.mean(axis=0).astype(np.uint8)

    # 塗りつぶし + フェザー
    # 消去範囲（erase_mask）の境界を中心に、幅 feather_px の帯で肌色と元フレームを
    # 線形に混ぜる（境界の内側 feather/2 で塗り 100%、外側 feather/2 で 0%）。
    # マスクを幅広くぼかす従来方式（ガウス、遷移幅≒±7px）は境界の内側が半塗りに
    # なり、楕円の縁にかかる口の端（横に広い笑顔の口角など）が薄く残っていた。
    # 帯を外側だけに付けると小顔キャラで顎の輪郭や襟まで塗るため、中心に置く。
    feather_px = max(0, int(feather_px))
    ys, xs = np.nonzero(erase_mask)
    if len(ys) == 0:
        return frame
    margin = feather_px + 1
    y0, y1 = max(0, int(ys.min()) - margin), min(fh, int(ys.max()) + 1 + margin)
    x0, x1 = max(0, int(xs.min()) - margin), min(fw, int(xs.max()) + 1 + margin)
    roi = frame[y0:y1, x0:x1]
    core_roi = (erase_mask[y0:y1, x0:x1] > 0).astype(np.uint8)

    if feather_px > 0:
        # 境界からの符号付き距離（内側が正）。ROI の外周は「消去範囲の外」として扱う
        inside = cv2.distanceTransform(
            np.pad(core_roi, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
        outside = cv2.distanceTransform(
            np.pad(1 - core_roi, 1, constant_values=1), cv2.DIST_L2, 5)[1:-1, 1:-1]
        signed = inside - outside
        blend = np.clip((signed + feather_px / 2.0) / float(feather_px),
                        0.0, 1.0).astype(np.float32)
    else:
        blend = core_roi.astype(np.float32)

    painted = np.empty_like(roi)
    painted[:] = skin_color
    blend = blend[:, :, None]
    result = frame.copy()
    result[y0:y1, x0:x1] = (painted * blend + roi * (1.0 - blend)).astype(np.uint8)

    return result


# ===================================================================
# Main pipeline
# ===================================================================

def run_pipeline(
    video_path: str,
    face_info_path: str,
    out_track: str,
    out_video: str,
    model_path: str = "",
    device: str = "auto",
    smooth_cutoff: float = 3.0,
    keep_audio: bool = True,
    use_alpha: bool = False,
    bg_color_specs: Optional[list[str]] = None,
    bg_tolerance: int = 65,
    fill_color_hex: Optional[str] = None,
) -> int:
    """パイプライン全体を実行。import時にも呼び出し可能。

    Returns:
        0=成功, 1=エラー
    """
    # ── face_info.json読み込み ──
    if not os.path.isfile(face_info_path):
        print(f"[error] face_info.json not found: {face_info_path}")
        return 1

    with open(face_info_path, "r", encoding="utf-8") as f:
        face_info = json.load(f)

    required_keys = ["left_eye", "right_eye", "mouth", "eye_distance",
                     "mouth_norm_x", "mouth_norm_y", "mouth_w_norm", "mouth_h_norm",
                     "image_width", "image_height"]
    missing = [k for k in required_keys if k not in face_info]
    if missing:
        print(f"[error] face_info.json is missing keys: {missing}")
        print("       Re-run asset_preparer.py のタブ③（目・口検出） to regenerate face_info.json")
        return 1

    if not os.path.isfile(video_path):
        print(f"[error] Video not found: {video_path}")
        return 1

    base_dir = os.path.dirname(os.path.abspath(face_info_path))

    # ── 口 PNG のキャンバスサイズ（quad の大きさの基準。3 枚同一が必須）──
    mouth_dir = os.path.join(base_dir, "mouth")
    try:
        from track_utils import sprite_canvas_size, TRACK_FORMAT_VERSION, TRACK_SMOOTH_VERSION
        sprite_size = sprite_canvas_size(mouth_dir)
    except ValueError as e:
        print(f"[error] Mouth sprites unusable ({e}); "
              f"put open.png/half.png/closed.png of the same size in {mouth_dir}")
        return 1
    print(f"[init] Mouth sprite canvas: {sprite_size[0]}x{sprite_size[1]}px")

    # ── 塗り色の決定（明示指定 > 口消し済み立ち絵 > リングサンプリング）──
    fill_color: Optional[np.ndarray] = None
    if fill_color_hex:
        try:
            fill_color = hex_to_bgr(fill_color_hex).astype(np.float32)
            print(f"[init] Fill color (manual): {fill_color_hex}")
        except ValueError as e:
            print(f"[error] Invalid --fill-color: {e}")
            return 1
    else:
        fill_color, reason = fill_color_from_face_info(face_info, base_dir)
        if fill_color is not None:
            print(f"[init] Fill color from erased base image: "
                  f"BGR={fill_color.astype(int).tolist()} {reason}")
        else:
            print(f"[warn] Fill color unavailable ({reason}); "
                  f"falling back to ring sampling")

    # ── SAM3モデルロード（1回のみ）──
    print("[init] Loading SAM3 model...")
    try:
        detector = SAM3MouthDetector(model_path=model_path, device=device)
    except Exception as e:
        print(f"[error] SAM3 initialization failed: {e}")
        return 1

    # ── Phase 1: Eye Tracking ──
    print("=" * 60)
    print("[phase1] Starting eye tracking...")
    print("=" * 60)

    track_stats: dict = {}
    try:
        quads, valid, confidence, video_w, video_h, fps, raw_valid = \
            phase1_eye_tracking(video_path, face_info, detector, smooth_cutoff,
                                sprite_size=sprite_size, stats=track_stats)
    except Exception as e:
        print(f"[error] Eye tracking failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # ── NPZ保存 ──
    np.savez_compressed(
        out_track,
        quad=quads,
        valid=valid,
        confidence=confidence,
        w=np.int64(video_w),
        h=np.int64(video_h),
        fps=np.float64(fps),
        # convert_npz_to_json.py 互換フィールド。refSpriteSize = 口 PNG のキャンバス
        ref_sprite_w=np.int64(sprite_size[0]),
        ref_sprite_h=np.int64(sprite_size[1]),
        calib_offset=np.array([0.0, 0.0], dtype=np.float64),
        calib_scale=np.float64(1.0),
        calib_rotation=np.float64(0.0),
        # 位置データの版と平滑化の版（track_utils.TRACK_FORMAT_VERSION / TRACK_SMOOTH_VERSION）
        track_format=np.int64(TRACK_FORMAT_VERSION),
        track_smooth=np.int64(TRACK_SMOOTH_VERSION),
    )
    print(f"[phase1] Saved track: {out_track}")

    # calibrated は同一コピー
    track_base, track_ext = os.path.splitext(out_track)
    calib_path = f"{track_base}_calibrated{track_ext}"
    if out_track != calib_path:
        shutil.copy2(out_track, calib_path)
        print(f"[phase1] Copied calibrated: {calib_path}")

    # メトリクス保存（validは全1化されるため、補間前の生validを使う）
    metrics = calc_track_metrics(quads, raw_valid, confidence, min_conf=0.5)
    metrics.update(track_stats)
    metrics_path = f"{track_base}_metrics.json"
    save_metrics_json(metrics_path, metrics)

    # ── Phase 2: Mouth Erasure ──
    print("=" * 60)
    print("[phase2] Starting mouth erasure...")
    print("=" * 60)

    # 消去楕円の幾何: face_info と同じフォルダの mouth/ にある口素材の実寸基準
    erase_geom = compute_erase_geometry(mouth_dir, face_info, video_w, video_h)
    if erase_geom is None:
        print("[warn] face_info has no eye_distance; using legacy erase geometry")
    elif erase_geom.source != "sprites":
        print(f"[warn] Erase geometry fallback ({erase_geom.reason}); "
              f"put open.png/half.png/closed.png in {mouth_dir}")

    try:
        phase2_mouth_erasure(
            video_path, quads, valid,
            out_video, video_w, video_h, fps,
            keep_audio=keep_audio,
            use_alpha=use_alpha,
            bg_color_specs=bg_color_specs,
            bg_tolerance=bg_tolerance,
            fill_color=fill_color,
            erase_geom=erase_geom,
        )
    except Exception as e:
        print(f"[error] Mouth erasure failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("=" * 60)
    print("[done] Pipeline complete!")
    print(f"  Track: {out_track}")
    print(f"  Video: {out_video}")
    print("=" * 60)
    return 0


# ===================================================================
# CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Eye-based mouth tracking and erasure pipeline")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--face-info", required=True, help="face_info.json path")
    parser.add_argument("--out-track", required=True, help="Output NPZ path")
    parser.add_argument("--out-video", required=True, help="Output mouthless video path")
    parser.add_argument("--model", default="", help="SAM3 model path")
    parser.add_argument("--device", default="auto", help="GPU device (auto/cuda/cuda:0/cpu)")
    parser.add_argument("--smooth-cutoff", type=float, default=3.0,
                        help="Smoothing cutoff frequency (Hz)")
    parser.add_argument("--keep-audio", action="store_true", help="Keep audio from source")
    parser.add_argument("--alpha", action="store_true", help="Output WebM with alpha")
    parser.add_argument("--bg-color", action="append", default=[],
                        help="Background color spec (#RRGGBB:tolerance:feather)")
    parser.add_argument("--bg-tolerance", type=int, default=65,
                        help="Background tolerance for hybrid chroma key (default: 65)")
    parser.add_argument("--fill-color", default=None,
                        help="Mouth fill color '#RRGGBB' (default: auto from "
                             "mouth-erased base image, fallback to ring sampling)")

    args = parser.parse_args()

    rc = run_pipeline(
        video_path=args.video,
        face_info_path=args.face_info,
        out_track=args.out_track,
        out_video=args.out_video,
        model_path=args.model,
        device=args.device,
        smooth_cutoff=args.smooth_cutoff,
        keep_audio=args.keep_audio,
        use_alpha=args.alpha,
        bg_color_specs=args.bg_color if args.bg_color else None,
        bg_tolerance=args.bg_tolerance,
        fill_color_hex=args.fill_color,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
