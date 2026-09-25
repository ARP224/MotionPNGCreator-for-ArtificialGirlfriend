# MotionPNGTuber Player

> **MotionPNGCreator for ArtificialGirlfriend** の成果物確認用プレイヤー（Electron）

バッチ生成した口消し動画・トラッキングデータを、デスクトップ上の透過オーバーレイ
ウィンドウで再生確認するためのプレイヤーです。口パクはダミー音声で自動駆動されるため、
**マイクは使用しません**。

- **本体（リポジトリルート）**: 画像からの動画生成・口消し・口スプライト抽出などの**アセット作成**
- **このパッケージ**: 作成済みアセットの**再生確認**

アセットの作成方法は [本体のREADME](../README.md) を参照してください。

本パッケージは [MotionPNGTuber](https://github.com/rotejin/MotionPNGTuber)（rotejin氏）の
プレイヤーが原型です。**マイク入力によるリアルタイム口パク配信**（PNGTuber用途）を
したい場合は、原型のMotionPNGTuberを使用してください。

---

## 使い方

### 通常: video_generator から起動

本体の video_generator「成果物確認」ページから自動起動されます
（プレイリスト・WebSocket連携つき）。通常はこちらを使ってください。

### 手動起動（単体アセットの再生）

フォルダ選択UIはないため、引数でアセットフォルダを指定します:

```powershell
cd MotionPNGTuber_Player
npm install          # 初回のみ（インストーラー実行済みなら不要）
npx electron . --character-folder <アセットフォルダのパス>
```

ウィンドウは右クリックメニューからサイズ変更・最前面固定・終了ができます。

---

## アセット構成

キャラクターフォルダは以下の構成が必要です（本体のGUIで一括生成可能）:

```
character_folder/
├── *mouthless*.webm       # 口なし動画（必須。WebM VP9+アルファ推奨、H.264 MP4も可）
├── mouth_track.json       # トラッキングデータ（必須）
└── mouth/
    ├── closed.png         # 口を閉じた状態（必須）
    ├── open.png           # 口を開けた状態（必須）
    ├── half.png           # 半開き（任意）
    ├── e.png              # 「え」の口（任意）
    └── u.png              # 「う」の口（任意）
```

### 動画ファイルの要件

| 項目 | 要件 |
|------|------|
| コーデック | WebM（VP9、アルファチャンネル対応）または H.264 MP4 |
| フレームレート | CFR（固定フレームレート）必須 |
| ファイル名 | `mouthless` を含むこと（例: `loop_mouthless.webm`） |

> **重要**: VFR（可変フレームレート）だとトラッキングと同期がズレます

> **注**: 上記の命名規則は単体アセットとして読み込む場合のものです。本体の
> video_generator「成果物確認」から起動する場合はプレイリスト経由でパスが直接
> 渡されるため、バッチ出力のファイル名（`{ジョブID}.webm`）のままで再生できます。

### 口スプライト画像

- **形式**: PNG（透過推奨）
- **サイズ**: トラッキングデータの `refSpriteSize` に合わせる（例: 128x85px）
- **内容**: 口部分のみを透過背景で作成

---

## トラブルシューティング

### 口の位置がズレる

- 動画がCFR（固定フレームレート）か確認
- `mouth_track.json` の `fps` と動画のfpsが一致しているか確認
- `calibration` パラメータで微調整

### 動画が再生されない

- WebM (VP9) または H.264 でエンコードされているか確認
- ファイル名に `mouthless` が含まれているか確認（単体アセット読み込みの場合）

### 起動しない

- Node.js がインストールされ `npx` が使えるか確認
- `npm install` を実行済みか確認（本体のインストーラーが自動実行します）

---

## ファイル構成

```
MotionPNGTuber_Player/
├── index-electron.html   # Electron版HTML（透過オーバーレイ）
├── main.js               # Electronメインプロセス
├── preload.js            # Electronプリロード
├── lipsync.js            # リップシンクエンジン（LipsyncEngineクラス）
├── multi-motion.js       # 複数モーション切替（プレイリスト再生）
├── ws-review.js          # 本体との連携用WebSocketクライアント
├── i18n.js               # 日英UI翻訳
├── style.css             # スタイルシート
├── package.json          # Electron依存関係
├── package-lock.json     # 動作確認済みバージョンの固定
├── settings.json         # ウィンドウ設定（実行時に自動保存）
└── README.md             # このファイル
```

---

## 開発者向け: LipsyncEngine の組み込み

LipsyncEngine（`lipsync.js`）は他のWebアプリケーションに組み込んで使用できます。
音声解析データを外部から `processAudioData()` に渡すことで、TTS等の任意の音声
ソースで口パクを駆動できます。

```javascript
const engine = new LipsyncEngine({
    elements: { video, mouthCanvas, stage },
    assets: {
        video: './assets/character/mouthless_h264.mp4',
        track: './assets/character/mouth_track.json',
        mouth_closed: './assets/character/mouth/closed.png',
        mouth_open: './assets/character/mouth/open.png',
        mouth_half: './assets/character/mouth/half.png',  // 任意
        mouth_e: './assets/character/mouth/e.png',        // 任意
        mouth_u: './assets/character/mouth/u.png'         // 任意
    },
    options: {
        debug: true,           // デバッグログ出力
        sensitivity: 50        // 感度（0-100）
    }
});

// 動画の再生は呼び出し側で行う（start() は描画ループの開始のみ）
video.play();

// 外部から音声データを入力（マイク不使用）
engine.processAudioData({
    rms: 0.15,   // 音量（RMS値）
    high: 0.08,  // 高周波成分
    low: 0.12    // 低周波成分
});
```

### API一覧

| メソッド | 説明 |
|---------|------|
| `loadFiles(files)` | File配列からアセットを読み込み |
| `start()` | 描画ループ開始（動画の再生は呼び出し側で `video.play()` を行う） |
| `stop()` | 描画ループ停止（動画も一時停止） |
| `processAudioData(data)` | 外部から音声データを入力 |
| `setSensitivity(value)` | 感度設定（0-100） |
| `resetAudioStats()` | 音声統計値をリセット |
| `cleanup()` | リソース解放 |

---

## ライセンス

[GNU Affero General Public License v3.0 only](../LICENSE)（AGPL-3.0-only）

Copyright (C) 2026 Ryo (ARP224)

このプレイヤーは [MotionPNGTuber_Player](https://github.com/rotejin/MotionPNGTuber_Player)
（MIT License, Copyright (c) 2026 rotejin）の派生物です。
`lipsync.js` と `style.css` に rotejin 氏の著作物が含まれます。
MIT License の許諾表示を含む帰属情報は [NOTICE.md](../NOTICE.md) を参照してください。

`lipsync.js` は [ArtificialGirlfriend](https://github.com/ARP224/ArtificialGirlfriend) の
`MotionPNGPlayer/lipsync.js`（MIT License、rotejin 氏原作・Ryo 改変）と同一ファイルです
（2026-09-13〜）。口の位置データの契約（quad は口PNG全体を写す枠）はこのファイル内の
`drawMouthSprite()` のコメントが正で、両リポジトリで自動的に同じ文面になります。
同一性は `tools/check_lipsync_identity.py` で検証します。ファイル内に出自ヘッダは
置きません（byte 同一を保つため）。

> このディレクトリは本体リポジトリの一部です。単体で再配布する場合も
> `LICENSE`（AGPL-3.0-only）と `NOTICE.md` を必ず同梱してください。
