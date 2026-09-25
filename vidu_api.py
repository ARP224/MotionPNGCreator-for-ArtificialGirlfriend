#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vidu_api.py

Vidu API クライアント共通モジュール。
batch_engine.py / utility.py から共用する。

- vidu_post / vidu_get: エラーボディから err_code を抽出して ViduAPIError を送出
- vidu_download: リトライ付きダウンロード
- image_to_base64_uri: 画像ファイル → data:URI base64
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path

VIDU_API_BASE = "https://api.vidu.com/ent/v2"

VIDU_MODELS = [
    "viduq3-pro", "viduq3-turbo",
    "viduq2-pro", "viduq2-pro-fast", "viduq2-turbo",
    "viduq1", "viduq1-classic", "vidu2.0",
]

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


class ViduAPIError(Exception):
    """Vidu APIエラー。err_codeフィールドを保持。"""
    def __init__(self, message: str, err_code: str = "", status_code: int = 0):
        super().__init__(message)
        self.err_code = err_code
        self.status_code = status_code


def _raise_vidu_error(e: urllib.error.HTTPError):
    err_body = ""
    err_code = ""
    try:
        err_body = e.read().decode("utf-8", errors="replace")
        err_json = json.loads(err_body)
        err_code = err_json.get("err_code", "") or err_json.get("code", "")
    except Exception:
        pass
    raise ViduAPIError(
        f"HTTP {e.code}: {err_code or err_body or e.reason}",
        err_code=err_code, status_code=e.code,
    ) from e


def vidu_post(api_key: str, endpoint: str, data: dict) -> dict:
    url = f"{VIDU_API_BASE}/{endpoint}"
    req = urllib.request.Request(url, method="POST")
    req.add_header("Authorization", f"Token {api_key}")
    req.add_header("Content-Type", "application/json")
    body = json.dumps(data).encode()
    try:
        with urllib.request.urlopen(req, body, timeout=120) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        _raise_vidu_error(e)


def vidu_get(api_key: str, endpoint: str) -> dict:
    url = f"{VIDU_API_BASE}/{endpoint}"
    req = urllib.request.Request(url, method="GET")
    req.add_header("Authorization", f"Token {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        _raise_vidu_error(e)


def vidu_download(url: str, save_path: str, max_retries: int = 3):
    """生成動画をダウンロード。失敗時はリトライ（指数的バックオフ）。

    全体をメモリに載せず、<save_path>.part にストリーム書き込みしてから
    os.replace で確定する（中断時に部分ファイルを残さない）。
    """
    part_path = f"{save_path}.part"
    for attempt in range(max_retries + 1):
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=300) as resp, \
                    open(part_path, "wb") as f:
                shutil.copyfileobj(resp, f)
            os.replace(part_path, save_path)
            return
        except Exception:
            if attempt >= max_retries:
                try:
                    os.unlink(part_path)
                except OSError:
                    pass
                raise
            time.sleep(5 * (attempt + 1))


def image_to_base64_uri(image_path: str) -> str:
    """画像ファイルをdata:URI base64文字列に変換。"""
    ext = Path(image_path).suffix.lower()
    mime_map = {".png": "image/png", ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg", ".webp": "image/webp"}
    mime = mime_map.get(ext, "image/png")
    with open(image_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"
