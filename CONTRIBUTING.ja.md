<p align="right">
  <strong>日本語</strong> | <a href="./CONTRIBUTING.md">English</a>
</p>

# 開発に参加する

MotionPNGCreator for ArtificialGirlfriend（MPC）に興味を持ってくれてありがとうございます。バグ報告・機能の要望・Pull Request、どれも歓迎です。**日本語・英語どちらでも構いません。**

個人プロジェクトのため、返答やレビューが遅れることがあります。気長に待ってもらえると助かります。

## 報告・相談の場所

- **[Issues](https://github.com/ARP224/MotionPNGCreator-for-ArtificialGirlfriend/issues)** — バグ報告と機能の要望はこちらへ。テンプレートを用意していますが、項目は目安です。書ける範囲で自由に書いてください
- **[Discussions](https://github.com/ARP224/MotionPNGCreator-for-ArtificialGirlfriend/discussions)** — 使い方の質問・雑談・アイデアの相談はこちらへ

バグ報告に、OSとGPU・どの工程か（Step 1のタブ / Step 2のページ）・何をしたら何が起きたか・`uv run python asset_preparer.py`（または `uv run python video_generator.py`）で起動したときのコンソール出力を添えてもらえると、調査が速く進みます。**スクリーンショットやログを貼る前に、Vidu APIキーが写り込んでいないか確認してください。**

## Pull Request の方針

- **小さな修正（typoや明らかなバグの修正など）は、そのまま送ってもらって大丈夫です**
- **大きな変更は、先にIssueで相談してください。** 方向性が合わないまま作業してもらうと、お互いの時間がもったいないので
- このリポジトリの `main` は配信チャネルです。ユーザーは `git pull` で直接 `main` を受け取るため、**マージはそのままリリースになります**。取り込みは慎重に行います
- 挙動に触る変更は、メンテナが実際にGUIを動かして確認します。動画生成に触る変更はViduの課金とGPU処理を伴うため、確認に時間がかかることがあります。**「この操作をすると、こうなるはず」という確認手順を一言添えてもらえると、確認が速く進みます**

## 送る前のチェック

開発環境は、READMEのセットアップで作った環境がそのまま使えます（特別なセットアップはありません）。自動テストは無いので、そのうえで:

- 触った工程を、実際にGUIで動かして確認してください（課金なしで確認できるのは、Step 1の①②③⑤⑥⑦です。④とStep 2の生成はViduのクレジットを消費します）
- UIの文言（`locales/`）を変更した場合は、日英の整合を確認してください: `uv run python tools/check_locales.py`

## コードの決まりごと

最低限、次の4つを守ってもらえれば大丈夫です。

### 1. UIに出す文字列はハードコードしない

画面に出す文言は `tr("キー")` で書き、`locales/ja.json` と `locales/en.json` の**両方**にキーを追加してください。整合は `uv run python tools/check_locales.py` で確認できます。

### 2. 課金に触る処理は「見積り→確認」を必ず通す

Vidu APIを呼ぶ処理は、開始前に本数と推定クレジットを表示して確認ダイアログを挟む、既存の型に合わせてください。ユーザーに黙って課金が走る変更は取り込めません。

### 3. .bat / .vbs はCRLF・ASCIIのみで書く

起動スクリプトはcmdの仕様上CRLFが必須で、文字化け防止のためASCIIのみで記述しています（`.gitattributes` と各ファイル冒頭のNOTE参照）。エディタの自動変換に注意してください。

### 4. 構造の変更と挙動の変更を混ぜない

ファイルの移動・リネーム・分割と、動作を変える修正は、別のコミットに分けてください。混ざっていると、あとから「どのコミットで挙動が変わったのか」を追えなくなります。

## 貢献のライセンス

Pull Requestを送った時点で、あなたの貢献はこのプロジェクトと同じ **AGPL-3.0-only**（GNU Affero General Public License version 3・`LICENSE` 参照）で提供されることに同意したものとします。著作権はあなた自身に残ります。CLA（貢献者ライセンス同意書）はありません。

リポジトリ全体（`MotionPNGTuber_Player/` を含む）が単一のライセンスです。ファイル別の由来と原著作権表示は [NOTICE.md](NOTICE.md) を参照してください。
