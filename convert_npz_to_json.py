#!/usr/bin/env python3
#
# Origin / 出自:
#   - rotejin/MotionPNGTuber (MIT License, Copyright (c) 2025 rotejin)
#     https://github.com/rotejin/MotionPNGTuber
# Modified by Ryo (ARP224) in 2026 (パス処理とエラーハンドリングの修正、trackFormat / trackSmooth の出力).
# License: AGPL-3.0-only. 原著作権表示とMITの許諾表示はリポジトリ直下の
#          NOTICE.md に保持されています。ライセンス全文は LICENSE を参照。
#
"""
npz → JSON 変換ツール
mouth_track_calibrated.npz をブラウザで読み込めるJSONに変換します。
使い方:
    python convert_npz_to_json.py <npzファイル> [出力先]

例:
    python convert_npz_to_json.py ../assets/assets03/mouth_track_calibrated.npz ./data/
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def convert_npz_to_json(npz_path: Path, output_dir: Path) -> Path:
    """npzファイルをJSONに変換"""
    # with文でclose（Windowsのファイルロック残り対策）
    with np.load(npz_path, allow_pickle=False) as data:
        # メタデータ
        result = {
            "fps": float(data["fps"]),
            "width": int(data["w"]),
            "height": int(data["h"]),
            "refSpriteSize": [int(data["ref_sprite_w"]), int(data["ref_sprite_h"])],
            "calibration": {
                "offset": data["calib_offset"].tolist(),
                "scale": float(data["calib_scale"]),
                "rotation": float(data["calib_rotation"]),
            },
            "calibrationApplied": False,
            "frames": [],
        }
        # 位置データの版（2 = quad が口PNG全体を写す枠。track_utils.TRACK_FORMAT_VERSION）。
        # 旧NPZには無い。frames の前に置いて読みやすくする
        if "track_format" in data.files:
            frames = result.pop("frames")
            result["trackFormat"] = int(data["track_format"])
            result["frames"] = frames
        # 平滑化の版（track_utils.TRACK_SMOOTH_VERSION）。同じく frames の前に置く
        if "track_smooth" in data.files:
            frames = result.pop("frames")
            result["trackSmooth"] = int(data["track_smooth"])
            result["frames"] = frames

        # フレームデータ
        quads = data["quad"]
        valids = data["valid"]

    for i in range(len(quads)):
        frame = {
            "quad": quads[i].tolist(),  # [[x,y], [x,y], [x,y], [x,y]]
            "valid": bool(valids[i]),
        }
        result["frames"].append(frame)

    # 出力
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "mouth_track.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"変換完了: {output_path}")
    print(f"  フレーム数: {len(result['frames'])}")
    print(f"  FPS: {result['fps']}")
    print(f"  動画サイズ: {result['width']}x{result['height']}")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="npz → JSON 変換ツール")
    parser.add_argument("npz_file", type=Path, help="入力npzファイル")
    parser.add_argument("output_dir", type=Path, nargs="?", default=Path("."), help="出力ディレクトリ (デフォルト: カレント)")

    args = parser.parse_args()

    if not args.npz_file.exists():
        print(f"エラー: ファイルが見つかりません: {args.npz_file}", file=sys.stderr)
        sys.exit(1)

    convert_npz_to_json(args.npz_file, args.output_dir)


if __name__ == "__main__":
    main()
