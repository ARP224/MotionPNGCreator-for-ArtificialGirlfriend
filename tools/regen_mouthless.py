#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""regen_mouthless.py

既存のトラッキングNPZを再利用して、口消しWebMだけを高速に再生成する。
SAM3・GPU不要（ffmpegエンコードのみ）。塗り色仕様の変更後に、
既存キャラのWebMを一括で作り直す用途。

塗り色は口消し済み立ち絵（face_infoのmouth bbox領域）から取得し、
取得できない場合は従来のリングサンプリングで塗る（本番Phase2と同一ロジック）。
消去楕円の大きさは {キャラ}/mouth/ の口素材（open/half/closed）の実寸から
決める（compute_erase_geometry。本番Phase2と同一ロジック）。

既存WebMは処理前に .bak.タイムスタンプ へ一時リネームする。
このリネームはファイルロック検知を兼ねる（プレイヤーが開いている等で
リネームできない場合はそのジョブをスキップ）。成功時は一時ファイルを
削除し、失敗時は元のWebMに復元するため、.bak は残らない。

注意: video_generator のPhase2実行中には使わないこと（WebM書き込みが競合する）。

Usage:
    python tools/regen_mouthless.py --work-folder "assets/Auto Gen works"
    python tools/regen_mouthless.py --work-folder "assets/Auto Gen works" --chars Sara,Rinka
    python tools/regen_mouthless.py --work-folder "assets/Auto Gen works" --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# プロジェクトルートをimportパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

EXCLUDED_DIRS = {"prompts", "miss_data", "backup", "original", "mouth"}


def find_face_info(char_dir: str, char_name: str) -> str | None:
    for fn in (f"{char_name}_face_info.json", "face_info.json"):
        p = os.path.join(char_dir, fn)
        if os.path.isfile(p):
            return p
    return None


def find_jobs(char_dir: str) -> list[tuple[str, str, str]]:
    """(job_id, npz_path, original_mp4_path) のリストを返す。"""
    jobs = []
    for f in sorted(os.listdir(char_dir)):
        if not f.endswith(".npz") or f.endswith("_calibrated.npz"):
            continue
        stem = f[:-len(".npz")]
        orig = os.path.join(char_dir, "original", f"{stem}_original.mp4")
        if os.path.isfile(orig):
            jobs.append((stem, os.path.join(char_dir, f), orig))
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Regenerate mouthless WebMs from existing track NPZs "
                    "(no SAM3 required)")
    ap.add_argument("--work-folder", required=True, help="work folder path")
    ap.add_argument("--chars", default="",
                    help="comma-separated character names (default: all)")
    ap.add_argument("--bg-tolerance", type=int, default=65,
                    help="chroma key tolerance; must match original generation "
                         "(default: 65)")
    ap.add_argument("--dry-run", action="store_true",
                    help="list planned jobs and fill colors without writing")
    args = ap.parse_args()

    work = args.work_folder
    if not os.path.isdir(work):
        print(f"[error] work folder not found: {work}")
        return 1

    only = {c.strip() for c in args.chars.split(",") if c.strip()}

    # eye_pipeline は SAM3 モジュール経由で torch をロードするため
    # importに数秒かかるが、SAM3自体は使用しない
    print("[init] loading pipeline modules...")
    from eye_pipeline import (
        phase2_mouth_erasure,
        fill_color_from_face_info,
        compute_erase_geometry,
    )

    n_done = 0
    n_skip = 0
    n_fail = 0

    for name in sorted(os.listdir(work)):
        char_dir = os.path.join(work, name)
        if not os.path.isdir(char_dir) or name.startswith("."):
            continue
        if name.lower() in EXCLUDED_DIRS:
            continue
        if only and name not in only:
            continue

        face_p = find_face_info(char_dir, name)
        if not face_p:
            print(f"[skip] {name}: face_info.json not found")
            continue
        try:
            with open(face_p, "r", encoding="utf-8") as f:
                face_info = json.load(f)
        except Exception as e:
            print(f"[skip] {name}: failed to read face_info ({e})")
            continue

        jobs = find_jobs(char_dir)
        if not jobs:
            print(f"[skip] {name}: no npz+original pairs")
            continue

        fill_color, reason = fill_color_from_face_info(face_info, char_dir)
        if fill_color is not None:
            print(f"[char] {name}: fill color "
                  f"BGR={fill_color.astype(int).tolist()} {reason} "
                  f"({len(jobs)} clips)")
        else:
            print(f"[char] {name}: fill color FALLBACK to ring sampling "
                  f"({reason}) ({len(jobs)} clips)")

        # 消去楕円の幾何: 口素材の実寸基準。動画サイズ（NPZのw,h）ごとに1回計算
        mouth_dir = os.path.join(char_dir, "mouth")
        geom_cache: dict = {}

        def get_geom(w: int, h: int):
            key = (w, h)
            if key not in geom_cache:
                geom = compute_erase_geometry(mouth_dir, face_info, w, h)
                geom_cache[key] = geom
                if geom is None:
                    print(f"[char] {name}: erase geometry LEGACY "
                          f"(face_info has no eye_distance)")
                else:
                    tag = "" if geom.source == "sprites" else " FALLBACK"
                    print(f"[char] {name}: erase geometry{tag} {geom.describe()}")
            return geom_cache[key]

        if args.dry_run:
            try:
                with np.load(jobs[0][1], allow_pickle=False) as z:
                    get_geom(int(z["w"]), int(z["h"]))
            except Exception as e:
                print(f"       (could not read {jobs[0][1]}: {e})")
            for stem, _npz, _orig in jobs:
                print(f"       would regen: {stem}.webm")
            continue

        for stem, npz_path, orig_path in jobs:
            out_webm = os.path.join(char_dir, f"{stem}.webm")
            bak_path = ""

            # 既存WebMを一時リネーム（ロック検知＋失敗時復元用。成功時は削除）
            if os.path.isfile(out_webm):
                ts = time.strftime("%Y%m%d-%H%M%S")
                bak_path = f"{out_webm}.bak.{ts}"
                try:
                    os.replace(out_webm, bak_path)
                except OSError as e:
                    print(f"[skip] {name}/{stem}: cannot rename existing webm "
                          f"(locked by player?): {e}")
                    n_skip += 1
                    continue

            try:
                with np.load(npz_path, allow_pickle=False) as z:
                    quads = np.asarray(z["quad"], dtype=np.float32)
                    valid = np.asarray(z["valid"], dtype=np.uint8)
                    w = int(z["w"])
                    h = int(z["h"])
                    fps = float(z["fps"])

                phase2_mouth_erasure(
                    orig_path, quads, valid,
                    out_webm, w, h, fps,
                    keep_audio=True,
                    use_alpha=True,
                    bg_tolerance=args.bg_tolerance,
                    fill_color=fill_color,
                    erase_geom=get_geom(w, h),
                )

                if bak_path:
                    try:
                        os.unlink(bak_path)
                    except OSError:
                        pass
                n_done += 1
                print(f"[done] {name}/{stem}.webm")

            except Exception as e:
                n_fail += 1
                print(f"[fail] {name}/{stem}: {e}")
                # 失敗時はバックアップを復元して資産を守る
                if bak_path and os.path.isfile(bak_path) and not os.path.isfile(out_webm):
                    try:
                        os.replace(bak_path, out_webm)
                        print(f"       restored previous webm from backup")
                    except OSError as e2:
                        print(f"       WARNING: backup restore failed: {e2} "
                              f"(backup remains at {bak_path})")

    print(f"\n[summary] regenerated={n_done} skipped={n_skip} failed={n_fail}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
