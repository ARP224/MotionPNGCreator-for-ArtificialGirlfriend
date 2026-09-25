#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_lipsync_identity.py

MotionPNGTuber_Player/lipsync.js が ArtificialGirlfriend の MotionPNGPlayer/lipsync.js
と byte 同一かを確認する。プレイヤーの描画エンジンは AG 側を正として同一ファイルを
持つ約束（2026-09-13）。差分があれば先頭の相違行を示して終了コード 1。

AG リポジトリの場所は --ag-root、環境変数 AG_ROOT、既定 C:\\ArtificialGirlfriend の順。

Usage:
    python tools/check_lipsync_identity.py
    python tools/check_lipsync_identity.py --ag-root D:\\repos\\ArtificialGirlfriend
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys

DEFAULT_AG_ROOT = r"C:\ArtificialGirlfriend"
MPC_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "MotionPNGTuber_Player", "lipsync.js")
AG_REL = os.path.join("MotionPNGPlayer", "lipsync.js")


def main() -> int:
    parser = argparse.ArgumentParser(description="lipsync.js の AG との同一性確認")
    parser.add_argument("--ag-root", default=os.environ.get("AG_ROOT", DEFAULT_AG_ROOT),
                        help="ArtificialGirlfriend リポジトリのルート")
    args = parser.parse_args()

    ag_file = os.path.join(args.ag_root, AG_REL)
    for label, path in (("MPC", MPC_FILE), ("AG", ag_file)):
        if not os.path.isfile(path):
            print(f"[NG] {label} のファイルがありません: {path}")
            return 1

    a = open(MPC_FILE, "rb").read()
    b = open(ag_file, "rb").read()
    ha, hb = hashlib.sha256(a).hexdigest(), hashlib.sha256(b).hexdigest()
    print(f"MPC: {MPC_FILE}\n     {len(a)} bytes, sha256 {ha[:16]}…")
    print(f"AG:  {ag_file}\n     {len(b)} bytes, sha256 {hb[:16]}…")
    if a == b:
        print("[OK] byte 同一")
        return 0

    la, lb = a.split(b"\n"), b.split(b"\n")
    for i, (x, y) in enumerate(zip(la, lb), 1):
        if x != y:
            print(f"[NG] 先頭の相違: {i} 行目")
            print(f"     MPC: {x[:100]!r}")
            print(f"     AG:  {y[:100]!r}")
            break
    else:
        print(f"[NG] 行数が違います: MPC {len(la)} 行 / AG {len(lb)} 行")
    if (b"\r\n" in a) != (b"\r\n" in b):
        print("     改行コードが違います（CRLF/LF）。.gitattributes の eol=lf を確認")
    return 1


if __name__ == "__main__":
    sys.exit(main())
