# NOTICE

このファイルは、本プロジェクト「MotionPNGCreator for ArtificialGirlfriend」に含まれる
第三者著作物の帰属表示（attribution）です。
プロジェクト全体に適用されるライセンス条件は [`LICENSE`](LICENSE) を参照してください。

---

## 1. 本プロジェクトのライセンス

本プロジェクトは **GNU Affero General Public License v3.0 only（AGPL-3.0-only）** の
単一ライセンスで配布されます。ディレクトリごとの例外はありません。

Copyright (C) 2026 Ryo (ARP224)
GitHub: https://github.com/ARP224

AGPL-3.0 を採用する理由は、SAM3 の推論に使用している以下のライブラリが
AGPL-3.0 で提供されているためです。

- [Ultralytics](https://github.com/ultralytics/ultralytics)（AGPL-3.0）
- [ultralytics-thop](https://pypi.org/project/ultralytics-thop/)（AGPL-3.0）
- [ultralytics/CLIP](https://github.com/ultralytics/CLIP)（AGPL-3.0。OpenAI CLIP のフォークで、
  Ultralytics により AGPL-3.0 で再配布されているもの）

> **補足:** 以前のバージョンでは `MotionPNGTuber_Player/` ディレクトリのみ MIT License で
> 配布していましたが、ライセンス構成を単純化するため AGPL-3.0-only に統一しました。
> MIT License はサブライセンスを許諾しているため、この統一は MIT の条件に適合します。
> なお、これによって原著作者や利用者の権利が失われることはありません。
> 各上流の原典は、下記の上流リポジトリから**元のライセンスのまま**入手できます。

---

## 2. 本プロジェクトの由来

本プロジェクトは、以下の **3 つの上流**を持つ派生プロジェクトです。

| # | 上流 | ライセンス | 受領条件 |
|---|------|-----------|---------|
| 1 | [rotejin/MotionPNGTuber](https://github.com/rotejin/MotionPNGTuber) | MIT License | Copyright (c) 2025 rotejin |
| 2 | [rotejin/MotionPNGTuber_Player](https://github.com/rotejin/MotionPNGTuber_Player) | MIT License | Copyright (c) 2026 rotejin |
| 3 | [kazuya-bros/MouthSpriteExtractor-SAM3](https://github.com/kazuya-bros/MouthSpriteExtractor-SAM3) | **AGPL-3.0** | Copyright (c) 2026 kazuya-bros |

上流 3 は、上流 1 の口スプライト抽出機能を SAM3 で再実装した派生版です。
同リポジトリには LICENSE ファイルは置かれていませんが、`README.md`、`pyproject.toml` の
`license = "AGPL-3.0"`、および各ソースファイルの docstring において
AGPL-3.0 での配布が明示的に宣言されています。本プロジェクトは、
**上流 3 に由来するコードを AGPL-3.0 の条件で受領**しています。

本プロジェクトが上流に追加した主な要素は、Vidu API によるループ動画の自動生成、
バッチ処理エンジン、素材準備 GUI、目の処理パイプライン、日英 i18n、
ワンクリックインストーラーです。

---

## 3. 第三者著作物を含むファイル

以下のファイルには上流の著作物が残存しています。各ファイルの冒頭にも同趣旨の
出自表示を記載しています（AGPL-3.0 §5(a) に基づく改変表示を含む）。
例外は `MotionPNGTuber_Player/lipsync.js` で、ArtificialGirlfriend の
`MotionPNGPlayer/lipsync.js` と byte 同一に保つため冒頭表示を持たず、本表で表示します。

| ファイル | 由来 | 改変 |
|---------|------|------|
| `convert_npz_to_json.py` | 上流 1（MIT / rotejin） | 改変あり（2026 / Ryo (ARP224)） |
| `erase_mouth_offline.py` | 上流 1（MIT / rotejin） | 改変あり（2026 / Ryo (ARP224)） |
| `mouth_sprite_extractor.py` | **上流 1（MIT / rotejin）＋ 上流 3（AGPL-3.0 / kazuya-bros）** | 改変あり（2026 / Ryo (ARP224)） |
| `sam3_mouth_detector.py` | 上流 3（AGPL-3.0 / kazuya-bros） | 改変あり（2026 / Ryo (ARP224)） |
| `MotionPNGTuber_Player/lipsync.js` | 上流 2（MIT / rotejin） | 改変あり（2026 / Ryo (ARP224)）。[ArtificialGirlfriend](https://github.com/ARP224/ArtificialGirlfriend) の `MotionPNGPlayer/lipsync.js`（MIT License）と同一ファイル（2026-09-13〜、`tools/check_lipsync_identity.py` で検証） |
| `MotionPNGTuber_Player/style.css` | 上流 2（MIT / rotejin） | 無改変 |

`.gitignore` および `pyproject.toml` / `uv.lock` は上流 1 に起源を持ちますが、
現在残存しているのは設定項目としての事実的記述のみです。

---

## 4. rotejin 氏の MIT License 表示

MIT License の条件に従い、上流 1 および上流 2 の著作権表示と許諾表示を以下に保持します。

### 4-1. MotionPNGTuber（上流 1）

```
MIT License

Copyright (c) 2025 rotejin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### 4-2. MotionPNGTuber_Player（上流 2）

```
MIT License

Copyright (c) 2026 rotejin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 5. サードパーティに関する注記

### Ultralytics 系（AGPL-3.0）

- [Ultralytics](https://github.com/ultralytics/ultralytics) — SAM3 モデルの推論に使用
- ultralytics-thop — Ultralytics の依存パッケージ
- [ultralytics/CLIP](https://github.com/ultralytics/CLIP) — SAM3 のテキストプロンプト処理に使用

これらが本プロジェクトを AGPL-3.0 とする直接の根拠です。
Ultralytics を AGPL-3.0 以外の条件で利用したい場合は、Ultralytics の
[Enterprise License](https://www.ultralytics.com/license) が必要になる場合があります。

### SAM3 モデル（Meta）

モデルウェイト（`Sam3/sam3.pt`）は**本リポジトリに含まれていません**（`.gitignore` 済み）。
入手には [HuggingFace](https://huggingface.co/facebook/sam3) でのアクセス承認が必要で、
利用条件は Meta の定める [SAM License](https://github.com/facebookresearch/sam3/blob/main/LICENSE)
に従います。同ライセンスは主に以下の用途を禁止しています。

- 軍事・戦争目的での使用
- 原子力産業・原子力応用での使用
- 武器開発、スパイ活動
- 輸出規制（ITAR 等）対象となる活動

上記の制限の範囲内であれば商用利用は可能です。

### その他の依存パッケージ

PyTorch / torchvision（BSD-3-Clause）、OpenCV（Apache-2.0）、Pillow（MIT-CMU）、
NumPy・SciPy（BSD）、timm（Apache-2.0）、tkinterdnd2（MIT）、websockets（BSD-3-Clause）、
Electron（MIT）ほか。いずれも AGPL-3.0 と互換のライセンスです。

CUDA 関連ライブラリ（`nvidia-*`）は NVIDIA の独自ライセンスですが、
**本リポジトリはこれらを再配布しておらず**、`uv sync` の実行時に利用者の環境へ
直接取得されます。

### 外部 API・外部ツール

- **Vidu API** — 動画生成に外部 API（有料）を使用します。利用は Vidu の利用規約に従います。
  API キーは本リポジトリに含まれません。
- **ffmpeg / Git / Node.js / uv** — インストーラは winget 等を通じて利用者の環境へ
  導入を行うのみで、本リポジトリはこれらのバイナリを同梱していません。
  ffmpeg は別プロセスとして呼び出しており、本プログラムと結合していません。

### 名称について

「MotionPNGTuber」は上流プロジェクトの名称です。本プロジェクトは由来と互換性を
示すためにこの名称に言及していますが、上流プロジェクトおよびその作者と
提携・後援関係にはありません。「Vidu」「SAM」は各権利者の名称です。
