#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""smooth_track_median.py

既存の口の位置データ（NPZ / JSON）に移動中央値の平滑化をかける後処理ツール。
SAM3・GPU・動画は不要。

まばたきの半開きでは、上まぶたに隠れた分だけ目マスクの重心が下にずれ、口の中心が
縦に 1〜2 px 跳ねる（完全閉眼は瞬き判定で補間されるが、半開きは面積が 70% を超えて
素通りする）。中心 x/y・幅・高さ・角度の各時系列に 9 コマの移動中央値（端は端の値で
延長）をかけると、4 コマ以内に行って戻る跳ねだけが消え、傾ける・寄る・戻すといった
単調な動きは値そのまま通る（2026-09-13 稜裁定。実測: 通常フレームの変化 p99 0.66 px、
頭が 93 px 動く動画でも最大 0.8 px）。

パイプライン（eye_pipeline.phase1_eye_tracking）も同じ処理を EMA の前段で行うので、
本ツールは平滑化前に作られた素材を揃えるためのもの。処理は track_utils.smooth_quads_moving_median
を共用する。

対象: <親フォルダ>/<キャラ>/ の {id}.npz と {id}_calibrated.npz（両方書き換え）。
      {id}.json は _calibrated.npz から convert_npz_to_json で作り直す。
      {id}_metrics.json は quad 幅・ジッタの 3 項目を更新する。
      NPZ の track_smooth >= 1 のものは読み飛ばす（何度実行してもよい）。
      trackFormat（プレイヤーの契約）は変えない。valid も変えない。

Usage:
    python tools/smooth_track_median.py "Y:\\Assets\\Auto Gen Works" --dry-run
    python tools/smooth_track_median.py "Y:\\Assets\\Auto Gen Works"
    python tools/smooth_track_median.py "Y:\\Assets\\Auto Gen Works" --chars Hina,Yuzuki --verbose

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
    SMOOTH_MEDIAN_SIZE,
    TRACK_SMOOTH_VERSION,
    smooth_quads_moving_median,
)

EXCLUDED_DIRS = {"prompts", "miss_data", "backup", "original", "mouth"}


def npz_smooth_version(path: str) -> int:
    with np.load(path, allow_pickle=False) as d:
        return int(d["track_smooth"]) if "track_smooth" in d.files else 0


def center_change_stats(old_quads: np.ndarray, new_quads: np.ndarray) -> dict:
    """平滑化で口の中心がどれだけ動くか（表示用）。"""
    d = np.linalg.norm(new_quads.mean(axis=1) - old_quads.mean(axis=1), axis=1)
    return {
        "max_change": float(d.max()) if len(d) else 0.0,
        "n_over_1px": int((d > 1.0).sum()),
        "n_over_0_5px": int((d > 0.5).sum()),
    }


def smooth_npz(path: str, dry_run: bool) -> dict:
    with np.load(path, allow_pickle=False) as d:
        arrays = {k: d[k] for k in d.files}
    old = arrays["quad"]
    new = smooth_quads_moving_median(old, SMOOTH_MEDIAN_SIZE)
    stats = center_change_stats(old, new)
    if not dry_run:
        arrays["quad"] = new.astype(old.dtype)
        arrays["track_smooth"] = np.int64(TRACK_SMOOTH_VERSION)
        tmp = path + ".tmp.npz"
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
    stats["quads"] = new
    return stats


def regenerate_json(calib_path: str, char_dir: str, job_id: str):
    """_calibrated.npz から {id}.json を作り直す（batch_engine と同じ経路）。"""
    with contextlib.redirect_stdout(io.StringIO()):
        convert_npz_to_json(Path(calib_path), Path(char_dir))
    generic = os.path.join(char_dir, "mouth_track.json")
    target = os.path.join(char_dir, f"{job_id}.json")
    if os.path.isfile(generic):
        os.replace(generic, target)


def update_metrics(metrics_path: str, quads: np.ndarray):
    """quad 幅とジッタを平滑化後の値に更新する。

    元の jitter_p95 は補間前の valid フレームだけで取っていたが、NPZ の valid は
    出力用に全 1 化されているため、ここでは全フレームで取る。
    """
    if not os.path.isfile(metrics_path):
        return
    with open(metrics_path, "r", encoding="utf-8") as f:
        m = json.load(f)
    w = np.linalg.norm(quads[:, 1] - quads[:, 0], axis=1)
    w_med = float(np.median(w)) if len(w) else 0.0
    centers = quads.mean(axis=1)
    jitter = np.linalg.norm(np.diff(centers, axis=0), axis=1) if len(centers) > 1 else np.zeros(0)
    j95 = float(np.quantile(jitter, 0.95)) if jitter.size else 0.0
    m["quad_width_median_px"] = w_med
    m["jitter_p95_px"] = j95
    m["jitter_p95_over_width"] = float(j95 / max(1.0, w_med)) if w_med > 0 else j95
    m["smooth_median_size"] = int(SMOOTH_MEDIAN_SIZE)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)


def process_character(char_dir: str, dry_run: bool, verbose: bool) -> dict:
    name = os.path.basename(char_dir)
    result = {"name": name, "done": 0, "skipped": 0, "errors": [], "lines": [], "max_changes": []}
    ids = sorted(
        f[:-4] for f in os.listdir(char_dir)
        if f.endswith(".npz") and not f.endswith("_calibrated.npz")
    )
    if not ids:
        result["errors"].append("位置データ（.npz）なし")
        return result
    for job_id in ids:
        main_path = os.path.join(char_dir, f"{job_id}.npz")
        calib_path = os.path.join(char_dir, f"{job_id}_calibrated.npz")
        targets = [p for p in (main_path, calib_path) if os.path.isfile(p)]
        try:
            todo = [p for p in targets if npz_smooth_version(p) < TRACK_SMOOTH_VERSION]
        except Exception as e:
            result["errors"].append(f"{job_id}: NPZ を読めない ({e})")
            continue
        if not todo:
            result["skipped"] += 1
            continue
        try:
            stats = None
            for p in todo:
                stats = smooth_npz(p, dry_run)
            if not dry_run:
                src = calib_path if os.path.isfile(calib_path) else main_path
                regenerate_json(src, char_dir, job_id)
                update_metrics(os.path.join(char_dir, f"{job_id}_metrics.json"), stats["quads"])
            result["done"] += 1
            result["max_changes"].append(stats["max_change"])
            if verbose:
                result["lines"].append(
                    f"    {job_id}: 中心の変化 最大 {stats['max_change']:.2f} px, "
                    f"0.5px超 {stats['n_over_0_5px']} コマ, 1px超 {stats['n_over_1px']} コマ ({len(todo)} NPZ)")
        except Exception as e:
            result["errors"].append(f"{job_id}: {e}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=f"口の位置データに {SMOOTH_MEDIAN_SIZE} コマの移動中央値をかける（まばたきの微振動除去）")
    parser.add_argument("parent", help="キャラクターフォルダが並ぶ親フォルダ（ワークフォルダ）")
    parser.add_argument("--chars", default="", help="対象キャラ名（カンマ区切り。省略時は全部）")
    parser.add_argument("--dry-run", action="store_true", help="書き換えずに効果を一覧する")
    parser.add_argument("--verbose", action="store_true", help="動画ごとの行も出す")
    args = parser.parse_args()

    parent = os.path.abspath(args.parent)
    if not os.path.isdir(parent):
        print(f"[error] フォルダがありません: {parent}")
        return 1
    only = {c.strip() for c in args.chars.split(",") if c.strip()}

    mode = "DRY-RUN（書き換えなし）" if args.dry_run else "平滑化"
    print(f"[{mode}] {parent}  移動中央値 {SMOOTH_MEDIAN_SIZE} コマ")
    total_done = total_skip = 0
    had_error = False
    for entry in sorted(os.listdir(parent)):
        char_dir = os.path.join(parent, entry)
        if not os.path.isdir(char_dir) or entry.startswith(".") or entry.lower() in EXCLUDED_DIRS:
            continue
        if only and entry not in only:
            continue
        r = process_character(char_dir, args.dry_run, args.verbose)
        if r["errors"] and r["done"] == 0 and r["skipped"] == 0:
            print(f"  - {entry}: 対象外（{'; '.join(r['errors'])}）")
            continue
        head = f"  {entry}: {r['done']} 本を{'平滑化予定' if args.dry_run else '平滑化'}, {r['skipped']} 本は平滑化済み"
        if r["max_changes"]:
            mc = np.array(r["max_changes"])
            head += (f" | 中心の変化（動画ごとの最大）中央値 {np.median(mc):.2f} px, 最大 {mc.max():.2f} px")
        print(head)
        for line in r["lines"]:
            print(line)
        for err in r["errors"]:
            print(f"    [error] {err}")
            had_error = True
        total_done += r["done"]
        total_skip += r["skipped"]
    print(f"[done] {'平滑化予定' if args.dry_run else '平滑化'} {total_done} 本, 平滑化済み {total_skip} 本"
          + ("（エラーあり）" if had_error else ""))
    return 1 if had_error else 0


if __name__ == "__main__":
    sys.exit(main())
