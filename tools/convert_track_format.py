#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""convert_track_format.py

旧形式の口の位置データ（NPZ / JSON）を、口PNG全体を quad に写す新形式
（track_format = 2）へ変換する。SAM3・GPU・動画は不要。

旧形式（2026-09-13 より前の出力）は quad の大きさが立ち絵の口 bbox で、
ArtificialGirlfriend の MotionPNGPlayer は口PNGをこの枠に押し込んで描くため
口が小さく薄くなる。新形式では
  - quad の大きさ = 口PNGのキャンバスサイズ × 目間距離比
      比率 r = 旧 quad 幅 ÷ (mouth_w_norm × eye_distance)
      （旧 quad 幅 = mouth_w_norm × そのフレームの目間距離[動画px]。PNG も立ち絵の
        目間距離も元画像 px なので、動画/画像の解像度比は約分される）
      高さも同じ r を使う（旧 quad 高さは 8〜9px しかなく比を取ると精度が落ちる）
  - 角度 = 旧角度 − 立ち絵の目の角度（相対角。静止時 0 度）
  - refSpriteSize = 口PNGのキャンバスサイズ
  - track_format / trackFormat = 2

対象: <親フォルダ>/<キャラ>/ の {id}.npz と {id}_calibrated.npz（両方書き換え）。
      {id}.json は _calibrated.npz から convert_npz_to_json で作り直す。
      {id}_metrics.json は quad 幅に依存する 2 項目
      （quad_width_median_px, jitter_p95_over_width）を更新する。
      track_format >= 2 のものは読み飛ばす（何度実行してもよい）。
必要: <キャラ>_face_info.json（または face_info.json）と mouth/ の口PNG 3 枚（同一サイズ）。
      無いフォルダは読み飛ばして報告する。

Usage:
    python tools/convert_track_format.py "Y:\\Assets\\Auto Gen Works" --dry-run
    python tools/convert_track_format.py "Y:\\Assets\\Auto Gen Works"
    python tools/convert_track_format.py "Y:\\Assets\\Auto Gen Works" --chars Aya,Shiori
    python tools/convert_track_format.py "Y:\\Assets\\Auto Gen Works" --verbose   # 動画ごとの行も出す

変換前に位置データ（.npz / _calibrated.npz / .json / _metrics.json）を退避すること。
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from pathlib import Path

# プロジェクトルートを import パスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from convert_npz_to_json import convert_npz_to_json  # noqa: E402
from track_utils import (  # noqa: E402
    TRACK_FORMAT_VERSION,
    compose_quads_vectorized,
    decompose_quads_vectorized,
    sprite_canvas_size,
)

EXCLUDED_DIRS = {"prompts", "miss_data", "backup", "original", "mouth"}


def wrap_angle_deg(angle):
    return ((angle + 180.0) % 360.0) - 180.0


def load_face_info(char_dir: str, name: str):
    for fname in (f"{name}_face_info.json", "face_info.json"):
        path = os.path.join(char_dir, fname)
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    return None


def reference_angle_deg(face_info: dict) -> float:
    """立ち絵の目の角度（eye_pipeline の quad 角度と同じ変換: right−left の向き + 180 度）。"""
    le = face_info["left_eye"]
    re_ = face_info["right_eye"]
    raw = np.degrees(np.arctan2(re_["cy"] - le["cy"], re_["cx"] - le["cx"]))
    return float(wrap_angle_deg(raw + 180.0))


def convert_quads(quads: np.ndarray, face_info: dict, sprite_size, ref_angle: float):
    centers, widths, heights, angles = decompose_quads_vectorized(quads.astype(np.float32))
    mouth_w_norm = float(face_info["mouth_w_norm"])
    eye_distance = float(face_info["eye_distance"])
    if mouth_w_norm <= 0 or eye_distance <= 0:
        raise ValueError(f"invalid face_info: mouth_w_norm={mouth_w_norm}, eye_distance={eye_distance}")
    ratio = widths / (mouth_w_norm * eye_distance)         # = 目間距離[動画px] / 立ち絵の目間距離
    new_w = (float(sprite_size[0]) * ratio).astype(np.float32)
    new_h = (float(sprite_size[1]) * ratio).astype(np.float32)
    new_angles = wrap_angle_deg(angles - ref_angle).astype(np.float32)
    new_quads = compose_quads_vectorized(centers, new_w, new_h, new_angles)
    return new_quads.astype(quads.dtype), widths, new_w, angles, new_angles


def npz_format(path: str) -> int:
    with np.load(path, allow_pickle=False) as d:
        return int(d["track_format"]) if "track_format" in d.files else 1


def convert_npz(path: str, face_info: dict, sprite_size, ref_angle: float, dry_run: bool):
    """1 つの NPZ を変換。戻り値は表示用の要約 dict。"""
    with np.load(path, allow_pickle=False) as d:
        arrays = {k: d[k] for k in d.files}
    quads = arrays["quad"]
    new_quads, old_w, new_w, old_a, new_a = convert_quads(quads, face_info, sprite_size, ref_angle)
    summary = {
        "old_w_med": float(np.median(old_w)),
        "new_w_med": float(np.median(new_w)),
        "old_angle_med": float(np.median(old_a)),
        "new_angle_med": float(np.median(new_a)),
        "old_ref": (int(arrays["ref_sprite_w"]), int(arrays["ref_sprite_h"])),
    }
    if not dry_run:
        arrays["quad"] = new_quads
        arrays["ref_sprite_w"] = np.int64(sprite_size[0])
        arrays["ref_sprite_h"] = np.int64(sprite_size[1])
        arrays["track_format"] = np.int64(TRACK_FORMAT_VERSION)
        tmp = path + ".tmp"
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp + ".npz" if os.path.exists(tmp + ".npz") else tmp, path)
    return summary


def regenerate_json(calib_path: str, char_dir: str, job_id: str):
    """_calibrated.npz から {id}.json を作り直す（batch_engine と同じ経路）。"""
    with contextlib.redirect_stdout(io.StringIO()):
        convert_npz_to_json(Path(calib_path), Path(char_dir))
    generic = os.path.join(char_dir, "mouth_track.json")
    target = os.path.join(char_dir, f"{job_id}.json")
    if os.path.isfile(generic):
        os.replace(generic, target)


def update_metrics(metrics_path: str, new_w_med: float):
    if not os.path.isfile(metrics_path):
        return
    with open(metrics_path, "r", encoding="utf-8") as f:
        m = json.load(f)
    m["quad_width_median_px"] = float(new_w_med)
    jitter = float(m.get("jitter_p95_px") or 0.0)
    m["jitter_p95_over_width"] = float(jitter / max(1.0, new_w_med)) if new_w_med > 0 else jitter
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)


def process_character(char_dir: str, dry_run: bool, verbose: bool) -> dict:
    name = os.path.basename(char_dir)
    result = {"name": name, "converted": 0, "skipped": 0, "errors": [], "lines": []}

    face_info = load_face_info(char_dir, name)
    if face_info is None:
        result["errors"].append("face_info.json なし（キャラフォルダではない）")
        return result
    for key in ("mouth_w_norm", "eye_distance", "left_eye", "right_eye"):
        if key not in face_info:
            result["errors"].append(f"face_info に {key} が無い")
            return result
    try:
        sprite_size = sprite_canvas_size(os.path.join(char_dir, "mouth"))
    except ValueError as e:
        result["errors"].append(f"口PNG: {e}")
        return result
    ref_angle = reference_angle_deg(face_info)
    result["sprite_size"] = sprite_size
    result["ref_angle"] = ref_angle

    ids = sorted(
        f[:-4] for f in os.listdir(char_dir)
        if f.endswith(".npz") and not f.endswith("_calibrated.npz")
    )
    for job_id in ids:
        main_path = os.path.join(char_dir, f"{job_id}.npz")
        calib_path = os.path.join(char_dir, f"{job_id}_calibrated.npz")
        targets = [p for p in (main_path, calib_path) if os.path.isfile(p)]
        try:
            todo = [p for p in targets if npz_format(p) < TRACK_FORMAT_VERSION]
        except Exception as e:
            result["errors"].append(f"{job_id}: NPZ を読めない ({e})")
            continue
        if not todo:
            result["skipped"] += 1
            continue
        try:
            summary = None
            for p in todo:
                summary = convert_npz(p, face_info, sprite_size, ref_angle, dry_run)
            # JSON は _calibrated.npz から作り直す（無ければ本体 NPZ から）
            src = calib_path if os.path.isfile(calib_path) else main_path
            if not dry_run:
                regenerate_json(src, char_dir, job_id)
                update_metrics(os.path.join(char_dir, f"{job_id}_metrics.json"), summary["new_w_med"])
            result["converted"] += 1
            result["last_summary"] = summary
            if verbose:
                result["lines"].append(
                    f"    {job_id}: quad幅 {summary['old_w_med']:.1f} → {summary['new_w_med']:.1f} px, "
                    f"角度 {summary['old_angle_med']:+.1f} → {summary['new_angle_med']:+.1f} 度, "
                    f"refSpriteSize {list(summary['old_ref'])} → [{sprite_size[0]}, {sprite_size[1]}]"
                    f" ({len(todo)} NPZ)")
        except Exception as e:
            result["errors"].append(f"{job_id}: {e}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="口の位置データを新形式（quad = 口PNG全体の枠, track_format=2）へ変換する")
    parser.add_argument("parent", help="キャラクターフォルダが並ぶ親フォルダ（ワークフォルダ）")
    parser.add_argument("--chars", default="", help="対象キャラ名（カンマ区切り。省略時は全部）")
    parser.add_argument("--dry-run", action="store_true", help="書き換えずに内容を一覧する")
    parser.add_argument("--verbose", action="store_true", help="動画ごとの行も出す")
    args = parser.parse_args()

    parent = os.path.abspath(args.parent)
    if not os.path.isdir(parent):
        print(f"[error] フォルダがありません: {parent}")
        return 1
    only = {c.strip() for c in args.chars.split(",") if c.strip()}

    mode = "DRY-RUN（書き換えなし）" if args.dry_run else "変換"
    print(f"[{mode}] {parent}")
    total_conv = total_skip = 0
    had_error = False
    for entry in sorted(os.listdir(parent)):
        char_dir = os.path.join(parent, entry)
        if not os.path.isdir(char_dir) or entry.startswith(".") or entry.lower() in EXCLUDED_DIRS:
            continue
        if only and entry not in only:
            continue
        r = process_character(char_dir, args.dry_run, args.verbose)
        if r["errors"] and r["converted"] == 0 and r["skipped"] == 0:
            print(f"  - {entry}: 対象外（{'; '.join(r['errors'])}）")
            continue
        head = f"  {entry}: {r['converted']} 本を{'変換予定' if args.dry_run else '変換'}, {r['skipped']} 本は変換済み"
        if "sprite_size" in r:
            head += f" | 口PNG {r['sprite_size'][0]}x{r['sprite_size'][1]}, 角度基準 {r['ref_angle']:+.2f} 度"
        if r.get("last_summary"):
            s = r["last_summary"]
            head += f" | quad幅 中央値 {s['old_w_med']:.1f} → {s['new_w_med']:.1f} px（例: 最後の動画）"
        print(head)
        for line in r["lines"]:
            print(line)
        for err in r["errors"]:
            print(f"    [error] {err}")
            had_error = True
        total_conv += r["converted"]
        total_skip += r["skipped"]
    print(f"[done] {'変換予定' if args.dry_run else '変換'} {total_conv} 本, 変換済み {total_skip} 本"
          + ("（エラーあり）" if had_error else ""))
    return 1 if had_error else 0


if __name__ == "__main__":
    sys.exit(main())
