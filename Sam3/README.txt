SAM3 モデルの配置 / SAM3 model placement
=========================================

このフォルダに SAM3 のモデルファイル sam3.pt を置いてください。
Put the SAM3 model file sam3.pt in this folder.

    Sam3/
    ├── README.txt   ← このファイル / this file
    └── sam3.pt      ← ここに配置 / place the model here

入手方法 / How to get the model:

1. HuggingFace でアクセス承認をリクエストします（承認が必要なモデルです）
   Request access on HuggingFace (this is a gated model):
   https://huggingface.co/facebook/sam3

2. 承認後、ファイル一覧の sam3.pt をダウンロードし、このフォルダに置きます。
   After approval, download sam3.pt from the file list and put it in this folder.

注意 / Notes:

- *.pt ファイルは .gitignore により git 管理から除外されます。
  *.pt files are excluded from git via .gitignore.
