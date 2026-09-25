#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
video_generator.py

MotionPNGCreator for ArtificialGirlfriend バッチ自動生成 GUI。

ページ構成:
0. インストラクション: 全体フローと各ページの説明
1. メインページ: フォルダ読み込み、キャラクター/プロンプト割り当て、開始/停止
2. キューモニター: Phase1/Phase2 の進捗表示
3. プロンプトリスト: List A〜E の編集・保存
4. 設定: モデル、パラメータ（APIキーは asset_preparer で設定）
5. 成果物確認: 背景透過プレビュー(動的グリッド) / キャラクタープレイヤー
6. 再生成: Phase2再生成

Usage:
    uv run python video_generator.py
"""

from __future__ import annotations

import json
import logging
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

from batch_engine import (
    BatchConfig,
    BatchEngine,
    CharacterInfo,
    JobInfo,
    JOB_STATE_PENDING,
    JOB_STATE_PHASE1_SUBMITTED,
    JOB_STATE_PHASE1_DONE,
    JOB_STATE_PHASE2_RUNNING,
    JOB_STATE_COMPLETED,
    JOB_STATE_FAILED,
    BATCH_STATUS_IDLE,
    BATCH_STATUS_RUNNING,
    BATCH_STATUS_PAUSED,
    BATCH_STATUS_STOPPED,
    BATCH_STATUS_COMPLETED,
    calc_credits,
    get_duration_range,
    path_ansi_safe,
    scan_characters,
    load_prompt_list,
    save_prompt_list,
    MIN_CONF_THRESHOLD,
)
from i18n import tr, current_language, catalog_load_errors

# pythonw（コンソール非表示）起動では stdout/stderr が None になり print() が
# 例外を出すため、ダミー出力へ差し替える。ログを見たい場合はコンソールから
# `uv run python video_generator.py` で起動する
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")

# リダイレクト時（> log.txt 等）のUnicodeEncodeError防止。
# コンソール表示のエンコーディング自体は変更しない
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(errors="replace")
        except Exception:
            pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
SETTINGS_PATH = os.path.join(HERE, ".batch_settings.json")
PROMPT_LISTS_DIR = os.path.join(HERE, "prompt_lists")

DEFAULT_PROMPTS_PATH = os.path.join(PROMPT_LISTS_DIR, "defaults.json")

PROMPT_LIST_NAMES = ["list_a", "list_b", "list_c", "list_d", "list_e"]
PROMPT_LIST_LABELS = ["List A", "List B", "List C", "List D", "List E"]
MAX_PROMPTS_PER_LIST = 50

from vidu_api import VIDU_MODELS

# プロンプトのテキストファイル読み込み時に行頭の番号・箇条書き記号を除去
_PROMPT_LINE_PREFIX_RE = re.compile(r"^\s*(?:\d+[\.\)]\s*|[-*・]\s*)")

# 説明文中のリテラル波括弧（{ジョブID} 等）はJSON側で {{ }} にエスケープされている
GEN_INSTRUCTION_TEXT = tr("gen.instruction.text")

# 言語コード → 表示名（各言語のネイティブ表記のため翻訳しない）
LANG_DISPLAY = {"ja": "日本語", "en": "English"}

# Electron用プレイリスト一時ファイルの接頭辞（%TEMP%配下に作成）
PLAYLIST_TMP_PREFIX = "mpngc_playlist_"


def _cleanup_stale_playlists():
    """前回セッションが残したプレイリスト一時ファイルを削除。

    別インスタンスが今まさに使用中のファイルを消さないよう、
    1日以上前のものだけを対象にする。%TEMP% の走査コストが
    起動をブロックしないよう、呼び出し側はバックグラウンド実行する。
    """
    tmp_dir = tempfile.gettempdir()
    try:
        entries = os.listdir(tmp_dir)
    except OSError:
        return
    cutoff = time.time() - 24 * 3600
    for name in entries:
        if name.startswith(PLAYLIST_TMP_PREFIX) and name.endswith(".json"):
            path = os.path.join(tmp_dir, name)
            try:
                if os.path.getmtime(path) < cutoff:
                    os.unlink(path)
            except OSError:
                pass


_FFMPEG_STATUS: Optional[str] = None


def _ffmpeg_problem() -> Optional[str]:
    """ffmpeg の有無と libvpx-vp9 エンコーダを検査する。

    問題があればエラーメッセージの locale キーを返し、問題なければ None。
    結果はプロセス内でキャッシュされる（初回のみ subprocess 実行）。
    検査自体に失敗した場合は誤ブロックを避けるため None を返す。
    """
    global _FFMPEG_STATUS
    if _FFMPEG_STATUS is None:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            _FFMPEG_STATUS = "gen.msg.ffmpeg_missing"
        else:
            try:
                proc = subprocess.run(
                    [ffmpeg, "-hide_banner", "-encoders"],
                    capture_output=True, timeout=15,
                    creationflags=subprocess.CREATE_NO_WINDOW
                    if sys.platform == "win32" else 0)
                if proc.returncode == 0 and b"libvpx-vp9" not in proc.stdout:
                    _FFMPEG_STATUS = "gen.msg.ffmpeg_no_vp9"
                else:
                    _FFMPEG_STATUS = ""
            except Exception:
                _FFMPEG_STATUS = ""
    return _FFMPEG_STATUS or None


def _seed_default_prompts(settings: dict) -> bool:
    """初回起動時のみ、defaults.json からデフォルトプロンプトをスロットに配置。

    既にスロットのJSONがある場合は触らない。一度シードしたら設定の
    prompts_seeded フラグで記録し、以後は実行しない（ユーザーが意図的に
    スロットのファイルを削除しても勝手に復活させないため）。

    戻り値: シードを実行し、フラグの保存が必要なら True。
    """
    if settings.get("prompts_seeded"):
        return False
    try:
        with open(DEFAULT_PROMPTS_PATH, "r", encoding="utf-8-sig") as f:
            defaults = json.load(f)
    except Exception:
        return False
    if not isinstance(defaults, dict):
        return False
    os.makedirs(PROMPT_LISTS_DIR, exist_ok=True)
    for list_name, prompts in defaults.items():
        if list_name not in PROMPT_LIST_NAMES or not isinstance(prompts, list):
            continue
        path = os.path.join(PROMPT_LISTS_DIR, f"{list_name}.json")
        if not os.path.isfile(path):
            save_prompt_list(path, [str(p) for p in prompts])
    return True


# ---------------------------------------------------------------------------
# Settings persistence
# ---------------------------------------------------------------------------

def _load_settings() -> dict:
    if os.path.isfile(SETTINGS_PATH):
        try:
            # utf-8-sig: 外部エディタで編集されBOMが付いたファイルも読めるように
            with open(SETTINGS_PATH, "r", encoding="utf-8-sig") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_settings(data: dict):
    """既存の設定とマージして保存（asset_preparer と共有するキーを消さない）。"""
    merged = _load_settings()
    merged.update(data)
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------

class BatchGeneratorApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("MotionPNGCreator for ArtificialGirlfriend - Video Generator / Step2")
        self.geometry("1200x800")
        self.minsize(900, 600)

        self._settings = _load_settings()
        self._engine = BatchEngine()
        # %TEMP% の走査で起動をブロックしないようバックグラウンドで掃除
        threading.Thread(target=_cleanup_stale_playlists, daemon=True).start()
        self._work_folder = ""
        self._characters: list[CharacterInfo] = []

        # checkbox vars: {char_name: {list_name: BooleanVar}}
        self._assign_vars: dict[str, dict[str, tk.BooleanVar]] = {}

        # Build UI
        self._build_ui()
        self._load_settings_to_ui()
        self._setup_engine_callbacks()

        # asset_preparer で選択したワークフォルダを復元（全ページに反映）
        wf = self._settings.get("work_folder", "")
        if wf and os.path.isdir(wf):
            self._apply_work_folder(wf)

        # 初回起動ならデフォルトプロンプトをスロットに配置してから自動ロード
        if _seed_default_prompts(self._settings):
            self._settings["prompts_seeded"] = True
            _save_settings({"prompts_seeded": True})
        self._auto_load_prompts_on_startup()

        # ウィンドウ閉じ時のクリーンアップ
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Poll for UI updates from engine threads
        self._ui_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self._poll_ui_queue()

    # ================================================================
    # UI Construction
    # ================================================================

    def _build_ui(self):
        # --- Left sidebar ---
        sidebar = ttk.Frame(self, width=180)
        sidebar.pack(side="left", fill="y", padx=(5, 0), pady=5)
        sidebar.pack_propagate(False)

        ttk.Label(sidebar, text=tr("gen.sidebar.title"), font=("", 11, "bold")).pack(pady=(5, 10))

        self._page_buttons: list[ttk.Button] = []
        pages = [tr("gen.page.instruction"), tr("gen.page.main"),
                 tr("gen.page.queue"), tr("gen.page.prompts"),
                 tr("gen.page.settings"), tr("gen.page.review"),
                 tr("gen.page.regen")]
        for i, name in enumerate(pages):
            btn = ttk.Button(sidebar, text=name, command=lambda idx=i: self._show_page(idx))
            btn.pack(fill="x", padx=5, pady=2)
            self._page_buttons.append(btn)

        # --- Main content area ---
        self._content = ttk.Frame(self)
        self._content.pack(side="left", fill="both", expand=True, padx=5, pady=5)

        self._pages: list[ttk.Frame] = []
        for _ in pages:
            frame = ttk.Frame(self._content)
            self._pages.append(frame)

        self._build_page_instruction(self._pages[0])
        self._build_page_main(self._pages[1])
        self._build_page_queue_monitor(self._pages[2])
        self._build_page_prompts(self._pages[3])
        self._build_page_settings(self._pages[4])
        self._build_page_review(self._pages[5])
        self._build_page_regen(self._pages[6])

        # ページ表示時の更新フック（バッチ実行で original/ 等が増えた後の陳腐化対策）
        self._page_show_hooks = {
            5: self._on_review_page_shown,
            6: self._on_regen_page_shown,
        }

        self._show_page(0)

    def _show_page(self, idx: int):
        for p in self._pages:
            p.pack_forget()
        self._pages[idx].pack(fill="both", expand=True)
        hook = self._page_show_hooks.get(idx)
        if hook:
            hook()

    def _on_review_page_shown(self):
        """成果物確認ページ表示時: フォルダ引き継ぎ＋キャラ一覧を最新化。"""
        if self._work_folder and not self._review_work_folder:
            self._review_work_folder = self._work_folder
            self._review_folder_var.set(self._work_folder)
        self._review_refresh_chars()

    def _on_regen_page_shown(self):
        """再生成ページ表示時: フォルダ引き継ぎ＋対象一覧を最新化。"""
        if self._work_folder and not self._regen_folder_var.get():
            self._regen_folder_var.set(self._work_folder)
        self._regen_scan()

    # ================================================================
    # Page 0: Instruction
    # ================================================================

    def _build_page_instruction(self, parent: ttk.Frame):
        frame = ttk.LabelFrame(parent, text=tr("gen.instruction.frame"), padding=5)
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        txt = tk.Text(frame, wrap="word")
        sb = ttk.Scrollbar(frame, command=txt.yview)
        txt.config(yscrollcommand=sb.set)
        txt.insert("1.0", GEN_INSTRUCTION_TEXT)
        txt.config(state="disabled")
        txt.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

    # ================================================================
    # Page 1: Main
    # ================================================================

    def _build_page_main(self, parent: ttk.Frame):
        note = ttk.Label(
            parent, foreground="#555",
            text=tr("gen.main.note"))
        note.pack(anchor="w", padx=8, pady=(8, 0))

        # --- Top: folder selection ---
        top = ttk.LabelFrame(parent, text=tr("gen.common.work_folder"), padding=5)
        top.pack(fill="x", padx=5, pady=5)

        row = ttk.Frame(top)
        row.pack(fill="x")
        ttk.Button(row, text=tr("gen.main.select_folder"), command=self._select_work_folder).pack(side="left", padx=2)
        self._var_work_folder = tk.StringVar()
        ttk.Entry(row, textvariable=self._var_work_folder, state="readonly").pack(
            side="left", fill="x", expand=True, padx=2)

        # --- Character list with prompt assignment ---
        mid = ttk.LabelFrame(parent, text=tr("gen.main.char_list_frame"), padding=5)
        mid.pack(fill="both", expand=True, padx=5, pady=5)

        # Scrollable frame for characters
        canvas = tk.Canvas(mid, highlightthickness=0)
        scrollbar = ttk.Scrollbar(mid, orient="vertical", command=canvas.yview)
        self._char_frame = ttk.Frame(canvas)
        self._char_frame.bind("<Configure>",
                              lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self._char_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        # --- Bottom: controls ---
        bot = ttk.LabelFrame(parent, text=tr("gen.main.control_frame"), padding=5)
        bot.pack(fill="x", padx=5, pady=5)

        info_row = ttk.Frame(bot)
        info_row.pack(fill="x", pady=2)
        self._var_job_count = tk.StringVar(value=tr("gen.main.job_count_default"))
        self._var_credit_est = tk.StringVar(value=tr("gen.main.credit_est_default"))
        self._var_credit_remain = tk.StringVar(value=tr("gen.main.credit_remain_default"))
        ttk.Label(info_row, textvariable=self._var_job_count).pack(side="left", padx=10)
        ttk.Label(info_row, textvariable=self._var_credit_est).pack(side="left", padx=10)
        ttk.Label(info_row, textvariable=self._var_credit_remain).pack(side="left", padx=10)

        # Material check button
        check_row = ttk.Frame(bot)
        check_row.pack(fill="x", pady=2)
        ttk.Button(check_row, text=tr("gen.main.preflight_btn"),
                   command=self._run_preflight).pack(side="left", padx=5)
        self._var_preflight_status = tk.StringVar(value="")
        ttk.Label(check_row, textvariable=self._var_preflight_status,
                  foreground="gray").pack(side="left", padx=5)

        # Progress
        prog_row = ttk.Frame(bot)
        prog_row.pack(fill="x", pady=2)
        ttk.Label(prog_row, text=tr("gen.main.progress_label")).pack(side="left", padx=5)
        self._progress_bar = ttk.Progressbar(prog_row, mode="determinate", length=400)
        self._progress_bar.pack(side="left", fill="x", expand=True, padx=5)
        self._var_progress_text = tk.StringVar(value="0 / 0")
        ttk.Label(prog_row, textvariable=self._var_progress_text).pack(side="left", padx=5)

        # Buttons
        btn_row = ttk.Frame(bot)
        btn_row.pack(fill="x", pady=5)

        # 配置: [開始] | [一時停止][再開] | [前回の続行][中止]
        # 一時停止⇔再開、前回の続行/中止をそれぞれ横並びのセットにする
        self._btn_start = ttk.Button(btn_row, text=tr("gen.main.start"), command=self._on_start)
        self._btn_start.pack(side="left", padx=(5, 20))
        self._btn_pause = ttk.Button(btn_row, text=tr("gen.main.pause"), command=self._on_pause,
                                     state="disabled")
        self._btn_pause.pack(side="left", padx=2)
        self._btn_resume = ttk.Button(btn_row, text=tr("gen.main.resume"), command=self._on_resume,
                                      state="disabled")
        self._btn_resume.pack(side="left", padx=(2, 20))
        self._btn_resume_state = ttk.Button(btn_row, text=tr("gen.main.resume_state"),
                                            command=self._on_resume_state)
        self._btn_resume_state.pack(side="left", padx=2)
        self._btn_stop = ttk.Button(btn_row, text=tr("gen.main.stop"), command=self._on_stop,
                                    state="disabled")
        self._btn_stop.pack(side="left", padx=2)

        # Status label
        self._var_batch_status = tk.StringVar(value=tr("gen.status.idle"))
        ttk.Label(btn_row, textvariable=self._var_batch_status,
                  font=("", 10, "bold")).pack(side="right", padx=10)

    def _select_work_folder(self):
        folder = filedialog.askdirectory(title=tr("gen.dialog.select_work_folder"))
        if not folder:
            return
        self._warn_if_path_not_ansi(folder)
        self._apply_work_folder(folder)

    def _warn_if_path_not_ansi(self, folder: str):
        if not path_ansi_safe(folder):
            messagebox.showwarning(tr("gen.msgbox.warning_title"),
                                   tr("common.warn.path_not_ansi"))

    def _apply_work_folder(self, folder: str):
        """ワークフォルダをメイン/成果物確認/再生成の3ページへ一括反映。"""
        self._work_folder = folder
        self._var_work_folder.set(folder)
        _save_settings({"work_folder": folder})
        self._load_characters()
        # 成果物確認ページ
        self._review_work_folder = folder
        self._review_folder_var.set(folder)
        self._review_refresh_chars()
        # 再生成ページ
        self._regen_folder_var.set(folder)
        self._regen_scan()

    def _load_characters(self):
        """ワークフォルダからキャラクターをスキャン。

        再スキャン（素材チェック等）で既存のチェック状態が失われないよう、
        同名キャラクターの割り当ては引き継ぐ。
        """
        prev_assign = {
            name: {ln: var.get() for ln, var in vars_dict.items()}
            for name, vars_dict in self._assign_vars.items()
        }
        # Clear existing widgets
        for w in self._char_frame.winfo_children():
            w.destroy()
        self._assign_vars.clear()
        self._characters = scan_characters(self._work_folder)

        if not self._characters:
            ttk.Label(self._char_frame, text=tr("gen.main.no_chars")).pack(pady=20)
            return

        # Grid layout for aligned columns
        grid = ttk.Frame(self._char_frame)
        grid.pack(fill="x", padx=5, pady=5)

        # Column widths
        col_widths = [18, 6] + [6] * len(PROMPT_LIST_LABELS)

        # Header row
        ttk.Label(grid, text=tr("gen.common.character"), width=col_widths[0], anchor="w",
                  font=("", 9, "bold")).grid(row=0, column=0, sticky="w", padx=(0, 5))
        ttk.Label(grid, text=tr("gen.common.state"), width=col_widths[1], anchor="center",
                  font=("", 9, "bold")).grid(row=0, column=1, padx=5)
        for col_idx, label in enumerate(PROMPT_LIST_LABELS):
            ttk.Label(grid, text=label, width=col_widths[2 + col_idx], anchor="center",
                      font=("", 9, "bold")).grid(row=0, column=2 + col_idx, padx=5)

        ttk.Separator(self._char_frame, orient="horizontal").pack(fill="x", padx=5)

        for row_idx, char in enumerate(self._characters, start=1):
            ttk.Label(grid, text=char.name, width=col_widths[0], anchor="w").grid(
                row=row_idx, column=0, sticky="w", padx=(0, 5), pady=1)

            if char.is_valid:
                ttk.Label(grid, text="OK", width=col_widths[1], anchor="center",
                          foreground="green").grid(row=row_idx, column=1, padx=5, pady=1)
            else:
                err_text = "; ".join(char.errors)
                lbl = ttk.Label(grid, text="NG", width=col_widths[1], anchor="center",
                                foreground="red")
                lbl.grid(row=row_idx, column=1, padx=5, pady=1)
                lbl.bind("<Enter>", lambda e, t=err_text: self._var_preflight_status.set(t))
                lbl.bind("<Leave>", lambda e: self._var_preflight_status.set(""))

            # Prompt list checkboxes
            self._assign_vars[char.name] = {}
            for col_idx, list_name in enumerate(PROMPT_LIST_NAMES):
                initial = (char.is_valid and
                           prev_assign.get(char.name, {}).get(list_name, False))
                var = tk.BooleanVar(value=initial)
                self._assign_vars[char.name][list_name] = var
                cb = ttk.Checkbutton(grid, variable=var, command=self._update_job_count)
                cb.grid(row=row_idx, column=2 + col_idx, padx=5, pady=1)
                if not char.is_valid:
                    cb.configure(state="disabled")

        self._update_job_count()

    def _get_prompt_assignments(self) -> dict[str, list[str]]:
        """UIのチェックボックスからプロンプト割り当てを収集。"""
        assignments = {}
        for char_name, vars_dict in self._assign_vars.items():
            selected = [ln for ln, var in vars_dict.items() if var.get()]
            if selected:
                assignments[char_name] = selected
        return assignments

    def _update_job_count(self):
        """生成本数と推定クレジットを更新。"""
        assignments = self._get_prompt_assignments()

        total = 0
        for char_name, lists in assignments.items():
            for list_name in lists:
                path = os.path.join(PROMPT_LISTS_DIR, f"{list_name}.json")
                prompts = load_prompt_list(path) if os.path.isfile(path) else []
                total += len(prompts)

        self._var_job_count.set(tr("gen.main.job_count", n=total))

        # Credit estimate
        settings = self._get_settings_from_ui()
        per_job = calc_credits(
            settings.get("model", "viduq2-pro-fast"),
            settings.get("resolution", "720p"),
            settings.get("duration", 5),
        )
        total_credits = per_job * total
        cost_usd = total_credits * 0.005
        self._var_credit_est.set(tr("gen.main.credit_est", credits=total_credits, cost=cost_usd))

    def _run_preflight(self):
        """素材チェック＋API残高確認。"""
        if not self._work_folder:
            messagebox.showwarning(tr("gen.msgbox.warning_title"), tr("gen.msg.select_work_folder"))
            return

        # Reload characters
        self._load_characters()

        # Check API
        settings = self._get_settings_from_ui()
        api_key = settings.get("api_key", "")
        if not api_key:
            self._var_preflight_status.set(tr("gen.main.api_key_missing_status"))
            return

        self._var_preflight_status.set(tr("gen.main.checking_credits"))
        self.update_idletasks()

        def _check():
            engine = BatchEngine()
            engine._config.api_key = api_key
            result = engine.check_api_credits()
            self.after(0, lambda: self._on_preflight_result(result))

        threading.Thread(target=_check, daemon=True).start()

    def _on_preflight_result(self, result: dict):
        if "error" in result:
            self._var_preflight_status.set(tr("gen.main.api_error", error=result['error']))
            self._var_credit_remain.set(tr("gen.main.credit_check_failed"))
        else:
            remain = result.get("credit_remain", 0)
            limit = result.get("concurrency_limit", 5)
            self._var_credit_remain.set(tr("gen.main.credit_remain", remain=remain, limit=limit))
            valid_count = sum(1 for c in self._characters if c.is_valid)
            total_count = len(self._characters)
            self._var_preflight_status.set(
                tr("gen.main.preflight_done", valid=valid_count, total=total_count))

    # ================================================================
    # Start / Pause / Stop
    # ================================================================

    def _on_start(self):
        """バッチ開始。"""
        if not self._work_folder:
            messagebox.showwarning(tr("gen.msgbox.warning_title"), tr("gen.msg.select_work_folder"))
            return

        # Phase2（口消しWebM出力）に必須のffmpegを課金前に検査
        ffmpeg_problem = _ffmpeg_problem()
        if ffmpeg_problem:
            messagebox.showerror(tr("gen.msgbox.error_title"), tr(ffmpeg_problem))
            return

        settings = self._get_settings_from_ui()
        api_key = settings.get("api_key", "")
        if not api_key:
            messagebox.showwarning(tr("gen.msgbox.warning_title"), tr("gen.msg.api_key_missing_settings"))
            return

        assignments = self._get_prompt_assignments()
        if not assignments:
            messagebox.showwarning(tr("gen.msgbox.warning_title"),
                                   tr("gen.msg.assign_prompts"))
            return

        # Build config
        config = BatchConfig(
            api_key=api_key,
            model=settings.get("model", "viduq2-pro-fast"),
            duration=settings.get("duration", 5),
            resolution=settings.get("resolution", "720p"),
            bg_tolerance=settings.get("bg_tolerance", 65),
            smooth_cutoff=settings.get("smooth_cutoff", 3.0),
            gpu_rest_count=settings.get("gpu_rest_count", 10),
            gpu_rest_seconds=settings.get("gpu_rest_seconds", 60),
        )

        # Build jobs (prompt_lists/ フォルダから読み込み)
        os.makedirs(PROMPT_LISTS_DIR, exist_ok=True)
        valid_chars = [c for c in self._characters if c.is_valid]
        jobs = self._engine.build_jobs(valid_chars, assignments, PROMPT_LISTS_DIR)

        if not jobs:
            messagebox.showwarning(tr("gen.msgbox.warning_title"), tr("gen.msg.no_jobs"))
            return

        # Confirmation（料金を算出できない組み合わせは開始しない）
        per_job = calc_credits(config.model, config.resolution, config.duration)
        if per_job <= 0:
            messagebox.showerror(tr("gen.msgbox.error_title"),
                                 tr("gen.msg.price_unknown"))
            return
        total_credits = per_job * len(jobs)
        cost_usd = total_credits * 0.005
        msg = tr("gen.msg.confirm_start_body",
                 n=len(jobs), credits=total_credits, cost=cost_usd,
                 model=config.model, duration=config.duration,
                 resolution=config.resolution)
        if not messagebox.askyesno(tr("gen.msgbox.confirm_title"), msg):
            return

        self._persist_settings()
        self._engine.start(self._work_folder, jobs, config)
        self._update_control_buttons(BATCH_STATUS_RUNNING)
        self._regen_update_buttons(BATCH_STATUS_RUNNING)

    def _on_resume_state(self):
        """前回のbatch_state.jsonから復帰。"""
        if not self._work_folder:
            messagebox.showwarning(tr("gen.msgbox.warning_title"), tr("gen.msg.select_work_folder"))
            return

        state_path = os.path.join(self._work_folder, "batch_state.json")
        if not os.path.isfile(state_path):
            messagebox.showinfo(tr("gen.msgbox.info_title"), tr("gen.msg.no_state_file"))
            return

        # Phase2（口消しWebM出力）に必須のffmpegを再開前に検査
        ffmpeg_problem = _ffmpeg_problem()
        if ffmpeg_problem:
            messagebox.showerror(tr("gen.msgbox.error_title"), tr(ffmpeg_problem))
            return

        # APIキーを現在の設定から注入
        settings = self._get_settings_from_ui()
        api_key = settings.get("api_key", "")
        if not api_key:
            messagebox.showwarning(tr("gen.msgbox.warning_title"), tr("gen.msg.api_key_missing"))
            return

        self._persist_settings()
        try:
            self._engine.resume_from_state(self._work_folder, api_key=api_key)
            self._update_control_buttons(BATCH_STATUS_RUNNING)
        except Exception as e:
            messagebox.showerror(tr("gen.msgbox.error_title"), tr("gen.msg.resume_failed", error=e))

    def _on_pause(self):
        self._engine.pause()
        self._update_control_buttons(BATCH_STATUS_PAUSED)
        self._regen_update_buttons(BATCH_STATUS_PAUSED)

    def _on_resume(self):
        self._engine.resume()
        self._update_control_buttons(BATCH_STATUS_RUNNING)
        self._regen_update_buttons(BATCH_STATUS_RUNNING)

    def _on_stop(self):
        if messagebox.askyesno(tr("gen.msgbox.confirm_title"), tr("gen.msg.confirm_stop")):
            self._engine.stop()
            self._update_control_buttons(BATCH_STATUS_STOPPED)
            self._regen_update_buttons(BATCH_STATUS_STOPPED)

    def _update_control_buttons(self, status: str):
        self._var_batch_status.set({
            BATCH_STATUS_IDLE: tr("gen.status.idle"),
            BATCH_STATUS_RUNNING: tr("gen.status.running"),
            BATCH_STATUS_PAUSED: tr("gen.status.paused"),
            BATCH_STATUS_STOPPED: tr("gen.status.stopped"),
            BATCH_STATUS_COMPLETED: tr("gen.common.completed"),
        }.get(status, status))

        is_running = status == BATCH_STATUS_RUNNING
        is_paused = status == BATCH_STATUS_PAUSED
        is_idle = status in (BATCH_STATUS_IDLE, BATCH_STATUS_STOPPED, BATCH_STATUS_COMPLETED)

        self._btn_start.configure(state="normal" if is_idle else "disabled")
        self._btn_resume_state.configure(state="normal" if is_idle else "disabled")
        self._btn_pause.configure(state="normal" if is_running else "disabled")
        self._btn_resume.configure(state="normal" if is_paused else "disabled")
        self._btn_stop.configure(state="normal" if (is_running or is_paused) else "disabled")

    # ================================================================
    # Page 2: Queue Monitor
    # ================================================================

    def _build_page_queue_monitor(self, parent: ttk.Frame):
        note = ttk.Label(
            parent, foreground="#555",
            text=tr("gen.queue.note"))
        note.pack(anchor="w", padx=8, pady=(8, 0))

        nb = ttk.Notebook(parent)
        nb.pack(fill="both", expand=True, padx=5, pady=5)

        # --- Phase1 tab ---
        tab1 = ttk.Frame(nb)
        nb.add(tab1, text=tr("gen.queue.tab_phase1"))

        # Slots display
        self._phase1_slot_frame = ttk.LabelFrame(tab1, text=tr("gen.queue.slots_frame"), padding=5)
        self._phase1_slot_frame.pack(fill="x", padx=5, pady=5)

        # 現在の活動（残高確認・タスク投入中など）。スロットは投入完了まで
        # 埋まらないため、開始直後の「何も起きていない」ように見える間の表示
        self._var_phase1_activity = tk.StringVar(value="")
        ttk.Label(self._phase1_slot_frame, textvariable=self._var_phase1_activity,
                  foreground="#0066CC", anchor="w").pack(fill="x", padx=5)

        self._phase1_slot_labels: list[ttk.Label] = []
        for i in range(5):
            lbl = ttk.Label(self._phase1_slot_frame, text=f"Slot {i+1}: -", anchor="w")
            lbl.pack(fill="x", padx=5, pady=1)
            self._phase1_slot_labels.append(lbl)

        # Phase1 job list
        p1_list_frame = ttk.Frame(tab1)
        p1_list_frame.pack(fill="both", expand=True, padx=5, pady=5)

        cols = ("id", "character", "prompt", "state", "retries")
        self._p1_tree = ttk.Treeview(p1_list_frame, columns=cols, show="headings", height=15)
        self._p1_tree.heading("id", text="ID")
        self._p1_tree.heading("character", text=tr("gen.common.character"))
        self._p1_tree.heading("prompt", text=tr("gen.queue.col_prompt"))
        self._p1_tree.heading("state", text=tr("gen.common.state"))
        self._p1_tree.heading("retries", text=tr("gen.queue.col_retries"))
        self._p1_tree.column("id", width=150)
        self._p1_tree.column("character", width=100)
        self._p1_tree.column("prompt", width=300)
        self._p1_tree.column("state", width=120)
        self._p1_tree.column("retries", width=60)

        p1_scroll = ttk.Scrollbar(p1_list_frame, orient="vertical",
                                  command=self._p1_tree.yview)
        self._p1_tree.configure(yscrollcommand=p1_scroll.set)
        self._p1_tree.pack(side="left", fill="both", expand=True)
        p1_scroll.pack(side="right", fill="y")

        # --- Phase2 tab ---
        tab2 = ttk.Frame(nb)
        nb.add(tab2, text=tr("gen.queue.tab_phase2"))

        # Current job
        cur_frame = ttk.LabelFrame(tab2, text=tr("gen.queue.processing"), padding=5)
        cur_frame.pack(fill="x", padx=5, pady=5)
        self._var_p2_current = tk.StringVar(value="-")
        ttk.Label(cur_frame, textvariable=self._var_p2_current, font=("", 10)).pack(
            fill="x", padx=5)

        # Phase2 stats
        stats_frame = ttk.Frame(tab2)
        stats_frame.pack(fill="x", padx=5, pady=5)
        self._var_p2_waiting = tk.StringVar(value=tr("gen.queue.waiting", n=0))
        self._var_p2_completed = tk.StringVar(value=tr("gen.queue.completed", n=0))
        self._var_p2_failed = tk.StringVar(value=tr("gen.queue.failed", n=0))
        ttk.Label(stats_frame, textvariable=self._var_p2_waiting).pack(side="left", padx=10)
        ttk.Label(stats_frame, textvariable=self._var_p2_completed).pack(side="left", padx=10)
        ttk.Label(stats_frame, textvariable=self._var_p2_failed).pack(side="left", padx=10)

        # Phase2 job list
        p2_list_frame = ttk.Frame(tab2)
        p2_list_frame.pack(fill="both", expand=True, padx=5, pady=5)

        cols2 = ("id", "character", "state", "retries", "output")
        self._p2_tree = ttk.Treeview(p2_list_frame, columns=cols2, show="headings", height=15)
        self._p2_tree.heading("id", text="ID")
        self._p2_tree.heading("character", text=tr("gen.common.character"))
        self._p2_tree.heading("state", text=tr("gen.common.state"))
        self._p2_tree.heading("retries", text=tr("gen.queue.col_retries"))
        self._p2_tree.heading("output", text=tr("gen.queue.col_output"))
        self._p2_tree.column("id", width=150)
        self._p2_tree.column("character", width=100)
        self._p2_tree.column("state", width=120)
        self._p2_tree.column("retries", width=60)
        self._p2_tree.column("output", width=350)

        p2_scroll = ttk.Scrollbar(p2_list_frame, orient="vertical",
                                  command=self._p2_tree.yview)
        self._p2_tree.configure(yscrollcommand=p2_scroll.set)
        self._p2_tree.pack(side="left", fill="both", expand=True)
        p2_scroll.pack(side="right", fill="y")

    # ================================================================
    # Page 3: Prompt Lists
    # ================================================================

    def _build_page_prompts(self, parent: ttk.Frame):
        note = ttk.Label(
            parent, foreground="#555",
            text=tr("gen.prompts.note"))
        note.pack(anchor="w", padx=8, pady=(8, 0))

        nb = ttk.Notebook(parent)
        nb.pack(fill="both", expand=True, padx=5, pady=5)

        self._prompt_texts: dict[str, list[tk.Text]] = {}
        # 各リストに紐づいたファイルパス
        self._prompt_file_paths: dict[str, tk.StringVar] = {}

        for list_name, label in zip(PROMPT_LIST_NAMES, PROMPT_LIST_LABELS):
            tab = ttk.Frame(nb)
            nb.add(tab, text=label)

            # File path display
            file_row = ttk.Frame(tab)
            file_row.pack(fill="x", padx=5, pady=(5, 0))
            self._prompt_file_paths[list_name] = tk.StringVar(value=tr("gen.prompts.not_loaded"))
            ttk.Label(file_row, text=tr("gen.prompts.file_label")).pack(side="left")
            ttk.Label(file_row, textvariable=self._prompt_file_paths[list_name],
                      foreground="gray").pack(side="left", padx=5)

            # Toolbar
            toolbar = ttk.Frame(tab)
            toolbar.pack(fill="x", padx=5, pady=5)
            ttk.Button(toolbar, text=tr("gen.prompts.load_from_file"),
                       command=lambda ln=list_name: self._load_prompt_from_file(ln)).pack(
                side="left", padx=2)
            ttk.Button(toolbar, text=tr("gen.common.save"),
                       command=lambda ln=list_name: self._save_prompt_list(ln)).pack(
                side="left", padx=2)
            ttk.Button(toolbar, text=tr("gen.prompts.clear"),
                       command=lambda ln=list_name: self._clear_prompt_list(ln)).pack(
                side="left", padx=2)
            ttk.Label(toolbar, text=tr("gen.prompts.max_note", max=MAX_PROMPTS_PER_LIST)).pack(
                side="left", padx=10)

            # Scrollable prompt entries
            canvas = tk.Canvas(tab, highlightthickness=0)
            scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
            inner = ttk.Frame(canvas)
            inner.bind("<Configure>",
                       lambda e, c=canvas: c.configure(scrollregion=c.bbox("all")))
            canvas.create_window((0, 0), window=inner, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            scrollbar.pack(side="right", fill="y")
            canvas.pack(side="left", fill="both", expand=True)

            texts = []
            for i in range(MAX_PROMPTS_PER_LIST):
                row = ttk.Frame(inner)
                row.pack(fill="x", padx=5, pady=1)
                ttk.Label(row, text=f"{i+1:02d}", width=4).pack(side="left")
                txt = tk.Text(row, height=1, width=80, wrap="none")
                txt.pack(side="left", fill="x", expand=True, padx=2)
                texts.append(txt)

            self._prompt_texts[list_name] = texts

    def _load_prompt_from_file(self, list_name: str):
        """ファイルダイアログでJSON/テキストを選択して読み込み。"""
        path = filedialog.askopenfilename(
            title=tr("gen.prompts.select_file_title", name=list_name),
            filetypes=[(tr("gen.filetype.prompt"), "*.json *.txt"), (tr("gen.filetype.all"), "*.*")])
        if not path:
            return
        prompts = self._parse_prompt_file(path)
        if not prompts:
            messagebox.showwarning(tr("gen.msgbox.warning_title"), tr("gen.msg.no_prompts_found"))
            return
        self._fill_prompt_ui(list_name, prompts)
        self._prompt_file_paths[list_name].set(path)

        # prompt_lists/ フォルダにも保存（次回起動時に自動ロード）
        self._save_prompt_to_app_dir(list_name, prompts)
        self._update_job_count()

    @staticmethod
    def _parse_prompt_file(path: str) -> list[str]:
        """プロンプトファイルを読み込む。JSON文字列配列を優先し、
        だめならテキスト（1行1件、行頭の番号・箇条書き記号は除去）として解釈。"""
        prompts = load_prompt_list(path)
        if prompts:
            return prompts
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()
        except Exception:
            return []
        result = []
        for line in lines:
            line = _PROMPT_LINE_PREFIX_RE.sub("", line).strip()
            if line:
                result.append(line)
        return result

    def _save_prompt_list(self, list_name: str):
        """UIの内容をprompt_lists/フォルダに保存。"""
        prompts = []
        for txt in self._prompt_texts[list_name]:
            text = txt.get("1.0", "end-1c").strip()
            if text:
                prompts.append(text)
        if not prompts:
            messagebox.showwarning(tr("gen.msgbox.warning_title"), tr("gen.msg.no_prompts_entered"))
            return
        self._save_prompt_to_app_dir(list_name, prompts)
        self._prompt_file_paths[list_name].set(
            os.path.join(PROMPT_LISTS_DIR, f"{list_name}.json"))
        messagebox.showinfo(tr("gen.msgbox.save_done_title"),
                            tr("gen.msg.prompts_saved", name=list_name, n=len(prompts)))
        self._update_job_count()

    def _save_prompt_to_app_dir(self, list_name: str, prompts: list[str]):
        """prompt_lists/ フォルダにJSONを保存。"""
        os.makedirs(PROMPT_LISTS_DIR, exist_ok=True)
        path = os.path.join(PROMPT_LISTS_DIR, f"{list_name}.json")
        save_prompt_list(path, prompts)

    def _auto_load_prompts_on_startup(self):
        """起動時にprompt_lists/からプロンプトを自動ロード。"""
        if not os.path.isdir(PROMPT_LISTS_DIR):
            return
        for list_name in PROMPT_LIST_NAMES:
            path = os.path.join(PROMPT_LISTS_DIR, f"{list_name}.json")
            if os.path.isfile(path):
                prompts = load_prompt_list(path)
                if prompts:
                    self._fill_prompt_ui(list_name, prompts)
                    self._prompt_file_paths[list_name].set(path)

    def _clear_prompt_list(self, list_name: str):
        """リストの全入力欄をクリア。"""
        for txt in self._prompt_texts[list_name]:
            txt.delete("1.0", "end")
        self._prompt_file_paths[list_name].set(tr("gen.prompts.not_loaded"))
        self._update_job_count()

    def _fill_prompt_ui(self, list_name: str, prompts: list[str]):
        """プロンプトリストをUI入力欄に反映。"""
        texts = self._prompt_texts[list_name]
        for i, txt in enumerate(texts):
            txt.delete("1.0", "end")
            if i < len(prompts):
                txt.insert("1.0", prompts[i])

    # ================================================================
    # Page 4: Settings
    # ================================================================

    def _build_page_settings(self, parent: ttk.Frame):
        canvas = tk.Canvas(parent, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        # --- API Key（asset_preparerで設定、ここでは表示のみ） ---
        api_frame = ttk.LabelFrame(inner, text="Vidu API", padding=10)
        api_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(api_frame,
                  text=tr("gen.settings.api_key_note")).pack(anchor="w")
        self._var_api_key = tk.StringVar()
        ttk.Entry(api_frame, textvariable=self._var_api_key, show="*", width=60,
                  state="readonly").pack(fill="x", padx=5, pady=2)

        # --- Language ---
        lang_frame = ttk.LabelFrame(inner, text=tr("common.lang.frame"), padding=10)
        lang_frame.pack(fill="x", padx=10, pady=5)

        row_lang = ttk.Frame(lang_frame)
        row_lang.pack(fill="x", pady=2)
        ttk.Label(row_lang, text=tr("common.lang.label"), width=15, anchor="w").pack(side="left")
        self._var_language = tk.StringVar(
            value=LANG_DISPLAY.get(current_language(), LANG_DISPLAY["en"]))
        lang_cb = ttk.Combobox(row_lang, textvariable=self._var_language,
                               values=list(LANG_DISPLAY.values()),
                               state="readonly", width=12)
        lang_cb.pack(side="left", padx=5)
        lang_cb.bind("<<ComboboxSelected>>", self._on_language_change)
        ttk.Label(row_lang, text=tr("common.lang.restart_hint")).pack(side="left", padx=8)

        # --- Model ---
        model_frame = ttk.LabelFrame(inner, text=tr("gen.settings.video_frame"), padding=10)
        model_frame.pack(fill="x", padx=10, pady=5)

        row1 = ttk.Frame(model_frame)
        row1.pack(fill="x", pady=2)
        ttk.Label(row1, text=tr("gen.settings.model"), width=15, anchor="w").pack(side="left")
        self._var_model = tk.StringVar(value="viduq2-pro-fast")
        model_cb = ttk.Combobox(row1, textvariable=self._var_model,
                                values=VIDU_MODELS, state="readonly", width=20)
        model_cb.pack(side="left", padx=5)
        model_cb.bind("<<ComboboxSelected>>", self._on_model_change)

        row2 = ttk.Frame(model_frame)
        row2.pack(fill="x", pady=2)
        ttk.Label(row2, text=tr("gen.settings.duration"), width=15, anchor="w").pack(side="left")
        self._var_duration = tk.IntVar(value=5)
        self._spin_duration = ttk.Spinbox(row2, from_=1, to=8,
                                          textvariable=self._var_duration, width=5)
        self._spin_duration.pack(side="left", padx=5)

        row3 = ttk.Frame(model_frame)
        row3.pack(fill="x", pady=2)
        ttk.Label(row3, text=tr("gen.settings.resolution_label"), width=15, anchor="w").pack(side="left")
        # 素材準備①のリスケール(720x1280)と合わせるため720p固定
        self._var_resolution = tk.StringVar(value="720p")
        ttk.Label(row3, text=tr("gen.settings.resolution_fixed")).pack(
            side="left", padx=5)

        # --- Background transparency ---
        bg_frame = ttk.LabelFrame(inner, text=tr("gen.settings.bg_frame"), padding=10)
        bg_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(bg_frame, text=tr("gen.settings.bg_note")).pack(
            anchor="w", pady=2)

        row_tol = ttk.Frame(bg_frame)
        row_tol.pack(fill="x", pady=2)
        ttk.Label(row_tol, text=tr("gen.settings.bg_tolerance"), width=20, anchor="w").pack(side="left")
        self._var_bg_tolerance = tk.DoubleVar(value=65.0)
        ttk.Scale(row_tol, from_=1, to=100, variable=self._var_bg_tolerance,
                  orient="horizontal", length=200).pack(side="left", padx=5)
        self._bg_tolerance_label = ttk.Label(row_tol, text="65", width=5)
        self._bg_tolerance_label.pack(side="left")
        # command= はスライダー操作でしか発火しないため、起動時の設定読み込み
        # （プログラムからのset）にも反応する変数トレースでスナップする
        self._var_bg_tolerance.trace_add(
            "write", lambda *_: self._on_bg_tolerance_change())

        # --- GPU Rest ---
        gpu_frame = ttk.LabelFrame(inner, text=tr("gen.settings.gpu_frame"), padding=10)
        gpu_frame.pack(fill="x", padx=10, pady=5)

        row5 = ttk.Frame(gpu_frame)
        row5.pack(fill="x", pady=2)
        ttk.Label(row5, text=tr("gen.settings.gpu_rest_interval"), width=15, anchor="w").pack(side="left")
        self._var_gpu_rest_count = tk.IntVar(value=10)
        ttk.Spinbox(row5, from_=1, to=100, textvariable=self._var_gpu_rest_count,
                     width=5).pack(side="left", padx=5)
        ttk.Label(row5, text=tr("gen.settings.gpu_rest_unit")).pack(side="left")

        row6 = ttk.Frame(gpu_frame)
        row6.pack(fill="x", pady=2)
        ttk.Label(row6, text=tr("gen.settings.gpu_rest_duration"), width=15, anchor="w").pack(side="left")
        self._var_gpu_rest_seconds = tk.IntVar(value=60)
        ttk.Spinbox(row6, from_=10, to=600, textvariable=self._var_gpu_rest_seconds,
                     width=5).pack(side="left", padx=5)
        ttk.Label(row6, text=tr("gen.settings.seconds_unit")).pack(side="left")

        # --- Smoothing ---
        smooth_frame = ttk.LabelFrame(inner, text=tr("gen.settings.smooth_frame"), padding=10)
        smooth_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(smooth_frame, text=tr("gen.settings.smooth_note"),
                  justify="left").pack(anchor="w", pady=(0, 4))

        row7 = ttk.Frame(smooth_frame)
        row7.pack(fill="x", pady=2)
        ttk.Label(row7, text=tr("gen.settings.smooth_cutoff_label"), width=15, anchor="w").pack(side="left")
        self._var_smooth_cutoff = tk.DoubleVar(value=3.0)
        ttk.Spinbox(row7, from_=0.0, to=10.0, increment=0.5,
                     textvariable=self._var_smooth_cutoff, width=5).pack(side="left", padx=5)
        ttk.Label(row7, text="Hz").pack(side="left")


    def _on_language_change(self, event=None):
        """言語変更時: 設定に保存し、再起動後に反映される旨を案内。"""
        display = self._var_language.get()
        code = next((c for c, d in LANG_DISPLAY.items() if d == display), "en")
        if code == current_language():
            return
        _save_settings({"language": code})
        messagebox.showinfo(tr("common.lang.changed_title"), tr("common.lang.restart_note"))

    def _on_model_change(self, event=None):
        """モデル変更時にduration制限を更新（解像度は720p固定）。"""
        model = self._var_model.get()
        dur_min, dur_max = get_duration_range(model)
        self._spin_duration.configure(from_=dur_min, to=dur_max)
        if self._var_duration.get() < dur_min:
            self._var_duration.set(dur_min)
        if self._var_duration.get() > dur_max:
            self._var_duration.set(dur_max)

    def _get_settings_from_ui(self) -> dict:
        """UIから設定を辞書として取得（Spinboxの非数値入力は既定値にフォールバック）。"""
        def _num(var, default):
            try:
                return var.get()
            except tk.TclError:
                return default
        return {
            "api_key": self._var_api_key.get().strip(),
            "model": self._var_model.get(),
            "duration": _num(self._var_duration, 5),
            "resolution": self._var_resolution.get(),
            "bg_tolerance": _num(self._var_bg_tolerance, 65),
            "gpu_rest_count": _num(self._var_gpu_rest_count, 10),
            "gpu_rest_seconds": _num(self._var_gpu_rest_seconds, 60),
            "smooth_cutoff": _num(self._var_smooth_cutoff, 3.0),
        }

    def _on_bg_tolerance_change(self):
        """許容値スライダーを0.5刻みにスナップし、表示を整える。

        ttk.Scaleは変数へ生の浮動小数（例: 63.7254...）を書き込むため、
        そのまま表示・保存すると端数が出る。Tclはトレース実行中の
        再発火を抑止するため、set()しても本メソッドは再帰しない。
        ラベルはここで直接更新する。
        """
        try:
            raw = self._var_bg_tolerance.get()
        except tk.TclError:
            return
        snapped = round(raw * 2) / 2
        if abs(raw - snapped) > 1e-9:
            self._var_bg_tolerance.set(snapped)
        self._bg_tolerance_label.config(text=f"{snapped:g}")

    def _persist_settings(self):
        """現在のUI設定を .batch_settings.json へ保存する。

        手動の保存ボタンは無く、バッチ開始・再生成開始・前回の続行・
        ウィンドウを閉じる時に自動保存される（スライダー等の変更イベント
        には紐づけない）。保存失敗で元の操作を妨げない。
        """
        try:
            _save_settings(self._get_settings_from_ui())
        except Exception:
            pass

    def _load_settings_to_ui(self):
        s = _load_settings()
        if not s:
            return
        self._var_api_key.set(s.get("api_key", ""))
        self._var_model.set(s.get("model", "viduq2-pro-fast"))
        self._var_duration.set(s.get("duration", 5))
        self._var_resolution.set("720p")  # 720p固定
        self._var_bg_tolerance.set(s.get("bg_tolerance", 65))
        self._var_gpu_rest_count.set(s.get("gpu_rest_count", 10))
        self._var_gpu_rest_seconds.set(s.get("gpu_rest_seconds", 60))
        self._var_smooth_cutoff.set(s.get("smooth_cutoff", 3.0))

        self._on_model_change()

    # ================================================================
    # Page 5: Review
    # ================================================================

    _REVIEW_PANEL_MIN_W = 180   # minimum panel width for column calc
    _REVIEW_THUMB_MAX_H = 160
    _REVIEW_DEBOUNCE_MS = 150
    _REVIEW_PAGE_SIZE = 30      # max videos per page

    def _build_page_review(self, parent: ttk.Frame):
        """成果物確認ページの構築。"""
        self._review_work_folder = ""
        self._review_char_name = ""
        self._review_videos: list[dict] = []
        self._review_page_idx = 0
        self._review_panels: list[dict] = []
        self._review_debounce_ids: list[Optional[str]] = []
        self._review_cur_cols = 0
        self._review_ws_server: Optional[object] = None
        self._review_electron_proc: Optional[subprocess.Popen] = None
        # 削除機能の状態
        self._review_now_playing = ""                       # プレイヤーが再生中の動画名
        self._review_pending_delete: Optional[str] = None   # プレイヤーの応答待ち中の削除対象
        self._review_delete_timeout_id: Optional[str] = None

        note = ttk.Label(
            parent, foreground="#555",
            text=tr("gen.review.note"))
        note.pack(anchor="w", padx=10, pady=(8, 0))

        # --- Top controls ---
        top = ttk.LabelFrame(parent, text=tr("gen.review.target_frame"), padding=5)
        top.pack(fill="x", padx=10, pady=(5, 3))

        row1 = ttk.Frame(top)
        row1.pack(fill="x", pady=2)
        ttk.Label(row1, text=tr("gen.review.folder_label")).pack(side="left", padx=(0, 5))
        self._review_folder_var = tk.StringVar()
        ttk.Entry(row1, textvariable=self._review_folder_var, state="readonly"
                  ).pack(side="left", fill="x", expand=True, padx=(0, 5))
        ttk.Button(row1, text=tr("gen.common.select"), command=self._review_select_folder).pack(side="left")

        row2 = ttk.Frame(top)
        row2.pack(fill="x", pady=2)
        ttk.Label(row2, text=tr("gen.review.char_label")).pack(side="left", padx=(0, 5))
        self._review_char_combo = ttk.Combobox(row2, state="readonly", width=30)
        self._review_char_combo.pack(side="left", padx=(0, 10))
        ttk.Button(row2, text=tr("gen.review.load"), command=self._review_load).pack(side="left")

        # --- Notebook ---
        self._review_nb = ttk.Notebook(parent)
        self._review_nb.pack(fill="both", expand=True, padx=10, pady=5)

        tab_preview = ttk.Frame(self._review_nb)
        self._review_nb.add(tab_preview, text=tr("gen.review.tab_preview"))
        self._build_review_preview_tab(tab_preview)

        tab_player = ttk.Frame(self._review_nb)
        self._review_nb.add(tab_player, text=tr("gen.review.tab_player"))
        self._build_review_player_tab(tab_player)

    # ----------------------------------------------------------------
    # Review: folder / character / load
    # ----------------------------------------------------------------

    def _review_select_folder(self):
        folder = filedialog.askdirectory(title=tr("gen.dialog.select_work_folder"))
        if not folder:
            return
        self._warn_if_path_not_ansi(folder)
        self._apply_work_folder(folder)

    def _review_refresh_chars(self):
        """レビュー対象キャラクター（original/ を持つフォルダ）を列挙。"""
        folder = self._review_work_folder
        chars = []
        if folder and os.path.isdir(folder):
            for entry in sorted(os.listdir(folder)):
                d = os.path.join(folder, entry)
                if os.path.isdir(d) and os.path.isdir(os.path.join(d, "original")):
                    chars.append(entry)
        # 既存の選択が残っていれば維持する（ページ再表示時の更新でリセットしない）
        current = self._review_char_combo.get()
        self._review_char_combo["values"] = chars
        if current and current in chars:
            self._review_char_combo.set(current)
        elif chars:
            self._review_char_combo.current(0)
        else:
            self._review_char_combo.set("")

    def _review_load(self):
        char_name = self._review_char_combo.get()
        folder = self._review_work_folder
        if not char_name or not folder:
            messagebox.showwarning(tr("gen.msgbox.warning_title"), tr("gen.msg.select_folder_char"))
            return

        if char_name != self._review_char_name:
            self._review_stop_player()
        self._review_char_name = char_name

        char_dir = os.path.join(folder, char_name)

        # Scan videos: need .webm to preview alpha
        self._review_videos = []
        for f in sorted(os.listdir(char_dir)):
            if not f.endswith(".webm"):
                continue
            job_id = f.replace(".webm", "")
            webm_path = os.path.join(char_dir, f)
            npz_path = os.path.join(char_dir, f"{job_id}.npz")
            json_path = os.path.join(char_dir, f"{job_id}.json")
            self._review_videos.append({
                "job_id": job_id,
                "webm_path": webm_path,
                "npz_path": npz_path,
                "has_json": os.path.isfile(json_path),
            })

        self._review_page_idx = 0
        self._review_populate_grid()
        self._review_populate_player_list()

    # ----------------------------------------------------------------
    # Review: Alpha Preview Tab (dynamic grid)
    # ----------------------------------------------------------------

    def _build_review_preview_tab(self, parent: ttk.Frame):
        # Navigation
        nav = ttk.Frame(parent)
        nav.pack(fill="x", padx=5, pady=3)
        self._review_btn_prev = ttk.Button(nav, text=tr("gen.review.prev_page"),
                                           command=self._review_prev_page)
        self._review_btn_prev.pack(side="left", padx=5)
        self._review_page_label_var = tk.StringVar(value="")
        ttk.Label(nav, textvariable=self._review_page_label_var,
                  font=("", 10)).pack(side="left", padx=10)
        self._review_btn_next = ttk.Button(nav, text=tr("gen.review.next_page"),
                                           command=self._review_next_page)
        self._review_btn_next.pack(side="left", padx=5)
        self._review_video_count_var = tk.StringVar(value="")
        ttk.Label(nav, textvariable=self._review_video_count_var).pack(side="right", padx=5)

        # Scrollable grid
        container = ttk.Frame(parent)
        container.pack(fill="both", expand=True)

        canvas = tk.Canvas(container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        self._review_grid_frame = ttk.Frame(canvas)
        self._review_grid_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        self._review_canvas_win = canvas.create_window((0, 0), window=self._review_grid_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self._review_canvas = canvas

        # Stretch grid_frame width to match canvas
        def _on_canvas_configure(event):
            canvas.itemconfigure(self._review_canvas_win, width=event.width)
            self._review_check_reflow(event.width)
        canvas.bind("<Configure>", _on_canvas_configure)

        def _on_mousewheel(event):
            # bind_all はウィジェット破棄後も残り得るため存在確認でガード
            if not canvas.winfo_exists():
                return
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _on_mousewheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

    def _review_calc_cols(self, available_width: int) -> int:
        """利用可能な幅から列数を計算。"""
        if available_width <= 0:
            return 4
        cols = max(1, available_width // self._REVIEW_PANEL_MIN_W)
        return cols

    def _review_check_reflow(self, width: int):
        """幅変更時にグリッドのリフロー（列数変更）をチェック。"""
        new_cols = self._review_calc_cols(width)
        if new_cols != self._review_cur_cols and self._review_panels:
            self._review_cur_cols = new_cols
            self._review_reflow_grid()

    def _review_reflow_grid(self):
        """既存パネルを新しい列数で再配置（再ロードなし）。"""
        cols = self._review_cur_cols
        for c in range(cols):
            self._review_grid_frame.columnconfigure(c, weight=1, uniform="rv")
        # Clear leftover column weights
        for c in range(cols, cols + 10):
            self._review_grid_frame.columnconfigure(c, weight=0, uniform="")
        for i, panel in enumerate(self._review_panels):
            r = i // cols
            c = i % cols
            panel["frame"].grid(row=r, column=c, padx=3, pady=3, sticky="nsew")

    def _review_create_panel(self, parent: ttk.Frame) -> dict:
        """1パネル分のウィジェットを作成（位置はreflowで設定）。"""
        frame = ttk.LabelFrame(parent, text="", padding=3)

        name_var = tk.StringVar()
        ttk.Label(frame, textvariable=name_var, font=("", 8)).pack(anchor="w")

        thumb_label = ttk.Label(frame)
        thumb_label.pack(pady=2)

        slider_frame = ttk.Frame(frame)
        slider_frame.pack(fill="x")
        ttk.Label(slider_frame, text="F:", font=("", 7)).pack(side="left")
        frame_var = tk.IntVar(value=0)
        frame_slider = ttk.Scale(slider_frame, from_=0, to=1, variable=frame_var,
                                 orient="horizontal")
        frame_slider.pack(side="left", fill="x", expand=True)
        frame_num_var = tk.StringVar(value="0/0")
        ttk.Label(slider_frame, textvariable=frame_num_var,
                  font=("", 7), width=8).pack(side="left")

        return {
            "frame": frame,
            "name_var": name_var,
            "thumb_label": thumb_label,
            "photo": None,
            "frame_var": frame_var,
            "frame_slider": frame_slider,
            "frame_num_var": frame_num_var,
            "video_idx": -1,
            "total_frames": 0,
            "video_w": 0,
            "video_h": 0,
        }

    def _review_populate_grid(self):
        """現ページの動画をグリッドに配置。"""
        for w in self._review_grid_frame.winfo_children():
            w.destroy()
        self._review_panels.clear()
        self._review_debounce_ids = []

        total = len(self._review_videos)
        page_size = self._REVIEW_PAGE_SIZE
        total_pages = max(1, (total + page_size - 1) // page_size)
        self._review_page_idx = min(self._review_page_idx, total_pages - 1)

        start = self._review_page_idx * page_size
        end = min(start + page_size, total)

        self._review_page_label_var.set(
            tr("gen.review.page_label", page=self._review_page_idx + 1, total=total_pages))
        self._review_video_count_var.set(tr("gen.review.video_count", n=total))
        self._review_btn_prev.configure(
            state="normal" if self._review_page_idx > 0 else "disabled")
        self._review_btn_next.configure(
            state="normal" if self._review_page_idx < total_pages - 1 else "disabled")

        canvas_w = self._review_canvas.winfo_width()
        self._review_cur_cols = self._review_calc_cols(canvas_w)

        for i in range(end - start):
            video_idx = start + i
            panel = self._review_create_panel(self._review_grid_frame)
            panel["video_idx"] = video_idx

            idx_in_panels = len(self._review_panels)
            self._review_panels.append(panel)
            self._review_debounce_ids.append(None)

            panel["frame_slider"].configure(
                command=lambda val, pi=idx_in_panels: self._review_on_frame_slider(pi, val))

        self._review_reflow_grid()
        self._review_progressive_load(0)

    def _review_progressive_load(self, idx: int):
        if idx >= len(self._review_panels):
            return
        panel = self._review_panels[idx]
        video_info = self._review_videos[panel["video_idx"]]
        self._review_load_panel(idx, video_info)
        self.after(50, lambda: self._review_progressive_load(idx + 1))

    def _review_load_panel(self, panel_idx: int, video_info: dict):
        """WebMからフレーム0のα付きプレビューをロード。"""
        panel = self._review_panels[panel_idx]
        panel["name_var"].set(video_info["job_id"])

        webm_path = video_info["webm_path"]
        if not os.path.isfile(webm_path):
            return

        # Get video info via cv2 (frame count, dimensions)
        cap = cv2.VideoCapture(webm_path)
        if not cap.isOpened():
            return
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        panel["total_frames"] = total_frames
        panel["video_w"] = w
        panel["video_h"] = h
        panel["frame_slider"].configure(to=max(1, total_frames - 1))
        panel["frame_var"].set(0)
        panel["frame_num_var"].set(f"0/{total_frames}")

        self._review_render_frame(panel_idx, 0)

    def _review_render_frame(self, panel_idx: int, frame_idx: int):
        """FFmpegでWebMの指定フレームをBGRAデコードしてプレビュー表示。"""
        panel = self._review_panels[panel_idx]
        video_info = self._review_videos[panel["video_idx"]]
        webm_path = video_info["webm_path"]
        w, h = panel["video_w"], panel["video_h"]
        if w == 0 or h == 0:
            return

        fps = 24.0  # fallback
        try:
            cap = cv2.VideoCapture(webm_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
            cap.release()
        except Exception:
            pass
        seek_time = frame_idx / fps

        # FFmpeg: decode single frame with VP9 alpha
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            # 黙って空のパネルを並べず、一度だけ理由を通知する
            if not getattr(self, "_review_ffmpeg_warned", False):
                self._review_ffmpeg_warned = True
                messagebox.showwarning(tr("gen.msgbox.warning_title"),
                                       tr("gen.review.ffmpeg_missing"))
            return
        cmd = [
            ffmpeg,
            "-c:v", "libvpx-vp9",
            "-ss", f"{seek_time:.4f}",
            "-i", webm_path,
            "-vframes", "1",
            "-pix_fmt", "bgra",
            "-f", "rawvideo",
            "pipe:1",
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=10,
                                  creationflags=subprocess.CREATE_NO_WINDOW
                                  if sys.platform == "win32" else 0)
        except Exception:
            return

        expected = w * h * 4
        if len(proc.stdout) != expected:
            return

        frame_bgra = np.frombuffer(proc.stdout, dtype=np.uint8).reshape(h, w, 4)
        bgr = frame_bgra[:, :, :3]
        alpha = frame_bgra[:, :, 3]

        # Build preview: magenta for transparent, blend for semi-transparent
        preview = bgr.copy()
        mask_full = alpha == 0
        mask_semi = (alpha > 0) & (alpha < 255)
        preview[mask_full] = [255, 0, 255]
        if np.any(mask_semi):
            a = alpha[mask_semi].astype(np.float32) / 255.0
            magenta = np.array([255, 0, 255], dtype=np.float32)
            preview[mask_semi] = (bgr[mask_semi] * a[:, None]
                                  + magenta * (1.0 - a[:, None])).astype(np.uint8)

        # Resize thumbnail
        max_h = self._REVIEW_THUMB_MAX_H
        scale = min(1.0, max_h / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        preview = cv2.resize(preview, (new_w, new_h), interpolation=cv2.INTER_AREA)

        rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        panel["photo"] = photo
        panel["thumb_label"].configure(image=photo)

    def _review_on_frame_slider(self, panel_idx: int, _val: str):
        if panel_idx < len(self._review_debounce_ids):
            old_id = self._review_debounce_ids[panel_idx]
            if old_id is not None:
                self.after_cancel(old_id)
        aid = self.after(self._REVIEW_DEBOUNCE_MS,
                         lambda: self._review_seek_frame(panel_idx))
        self._review_debounce_ids[panel_idx] = aid

    def _review_seek_frame(self, panel_idx: int):
        panel = self._review_panels[panel_idx]
        frame_idx = panel["frame_var"].get()
        panel["frame_num_var"].set(f"{frame_idx}/{panel['total_frames']}")
        self._review_render_frame(panel_idx, frame_idx)

    def _review_prev_page(self):
        if self._review_page_idx > 0:
            self._review_page_idx -= 1
            self._review_populate_grid()

    def _review_next_page(self):
        total = len(self._review_videos)
        total_pages = max(1, (total + self._REVIEW_PAGE_SIZE - 1) // self._REVIEW_PAGE_SIZE)
        if self._review_page_idx < total_pages - 1:
            self._review_page_idx += 1
            self._review_populate_grid()

    # ----------------------------------------------------------------
    # Review: Character Player Tab
    # ----------------------------------------------------------------

    def _build_review_player_tab(self, parent: ttk.Frame):
        # Controls
        ctrl = ttk.Frame(parent)
        ctrl.pack(fill="x", padx=5, pady=5)
        self._review_btn_play = ttk.Button(ctrl, text=tr("gen.review.play"),
                                           command=self._review_start_player)
        self._review_btn_play.pack(side="left", padx=5)
        self._review_btn_stop = ttk.Button(ctrl, text=tr("gen.common.stop"),
                                           command=self._review_stop_player,
                                           state="disabled")
        self._review_btn_stop.pack(side="left", padx=5)
        self._review_playing_var = tk.StringVar(value="")
        ttk.Label(ctrl, textvariable=self._review_playing_var,
                  font=("", 9)).pack(side="left", padx=10)

        # Video list
        list_frame = ttk.Frame(parent)
        list_frame.pack(fill="both", expand=True, padx=5, pady=5)

        cols = ("del", "name", "webm")
        self._review_player_tree = ttk.Treeview(
            list_frame, columns=cols, show="headings", selectmode="browse")
        self._review_player_tree.heading("del", text=tr("gen.review.col_delete"))
        self._review_player_tree.heading("name", text=tr("gen.review.col_name"))
        self._review_player_tree.heading("webm", text="JSON")
        self._review_player_tree.column("del", width=90, anchor="center",
                                        stretch=False)
        self._review_player_tree.column("name", width=300)
        self._review_player_tree.column("webm", width=60, anchor="center")

        tree_scroll = ttk.Scrollbar(list_frame, orient="vertical",
                                    command=self._review_player_tree.yview)
        self._review_player_tree.configure(yscrollcommand=tree_scroll.set)
        self._review_player_tree.pack(side="left", fill="both", expand=True)
        tree_scroll.pack(side="right", fill="y")

        self._review_player_tree.bind(
            "<<TreeviewSelect>>", self._review_on_video_select)
        # 削除列のクリック（widgetバインドはクラスバインドより先に発火するため、
        # "break" を返すことで選択変更→goto送信を抑止できる）
        self._review_player_tree.bind("<Button-1>", self._review_on_tree_click)

    def _review_populate_player_list(self):
        """動画リストをTreeviewに反映。"""
        tree = self._review_player_tree
        tree.delete(*tree.get_children())
        for v in self._review_videos:
            json_status = "OK" if v.get("has_json") else "-"
            tree.insert("", "end", iid=v["job_id"],
                        values=(self._review_del_cell_text(v["job_id"]),
                                v["job_id"], json_status))

    def _review_del_cell_text(self, job_id: str) -> str:
        """削除列のセル表示。再生中の動画は削除不可を示す▶表示。"""
        if self._review_electron_proc is not None and job_id == self._review_now_playing:
            return "▶"
        return tr("gen.review.btn_delete")

    def _review_refresh_del_cells(self):
        """削除列の表示を最新の再生状態に合わせて更新。"""
        tree = self._review_player_tree
        for iid in tree.get_children():
            tree.set(iid, "del", self._review_del_cell_text(iid))

    # ----------------------------------------------------------------
    # Review: 不良動画の削除
    # ----------------------------------------------------------------

    def _review_on_tree_click(self, event):
        """削除列（#1）のクリックを処理。それ以外は通常の選択動作に任せる。"""
        tree = self._review_player_tree
        if tree.identify_region(event.x, event.y) != "cell":
            return None
        if tree.identify_column(event.x) != "#1":
            return None
        job_id = tree.identify_row(event.y)
        if not job_id:
            return None
        self._review_request_delete(job_id)
        return "break"  # 選択変更（→goto送信）を抑止

    def _review_request_delete(self, job_id: str):
        """削除要求のエントリポイント。プレイヤー再生中は先にプレイリストから外す。"""
        # 再生中の動画はグレーアウト扱い（無反応）
        if self._review_electron_proc is not None and job_id == self._review_now_playing:
            return
        # 再生開始直後などで再生中動画が未確定の間は、誤って再生中の
        # 動画を削除しないよう全件ブロックする
        if self._review_electron_proc is not None and not self._review_now_playing:
            messagebox.showwarning(tr("gen.msgbox.warning_title"),
                                   tr("gen.review.delete_unknown_playing"))
            return
        if self._review_pending_delete is not None:
            messagebox.showwarning(tr("gen.msgbox.warning_title"),
                                   tr("gen.review.delete_busy"))
            return
        if not messagebox.askyesno(tr("gen.msgbox.confirm_title"),
                                   tr("gen.review.delete_confirm", name=job_id)):
            return

        video = next((v for v in self._review_videos if v["job_id"] == job_id), None)
        if video is None:
            return

        # プレイヤー再生中でプレイリストに載っている動画は、先にプレイヤー側で
        # プレイリストから除去させ、ファイルハンドル解放の応答を待ってから移動する
        if (self._review_electron_proc is not None and video.get("has_json")
                and self._review_ws_server is not None):
            if self._review_ws_server.send_command({"type": "remove", "name": job_id}):
                self._review_pending_delete = job_id
                self._review_delete_timeout_id = self.after(
                    4000, self._review_on_delete_timeout)
                return
        # プレイヤー未起動 / プレイリスト外 / 送信失敗 → 直接移動
        self._review_delete_files(job_id)

    def _review_on_removed(self, msg: dict):
        """プレイヤーからの removed 応答（UIスレッドで実行）。"""
        name = msg.get("name", "")
        if name != self._review_pending_delete:
            return
        self._review_pending_delete = None
        if self._review_delete_timeout_id:
            self.after_cancel(self._review_delete_timeout_id)
            self._review_delete_timeout_id = None

        if msg.get("ok"):
            # ファイルハンドル解放に少し猶予を与えてから移動
            self.after(200, lambda: self._review_delete_files(name))
        else:
            reason = msg.get("reason", "")
            key = {
                "playing": "gen.review.delete_playing",
                "last": "gen.review.delete_last",
            }.get(reason, "gen.review.delete_retry")
            messagebox.showwarning(tr("gen.msgbox.warning_title"), tr(key))

    def _review_on_delete_timeout(self):
        """プレイヤーからの removed 応答が来なかった場合。"""
        self._review_delete_timeout_id = None
        if self._review_pending_delete is None:
            return
        self._review_pending_delete = None
        messagebox.showwarning(tr("gen.msgbox.warning_title"),
                               tr("gen.review.delete_timeout"))

    def _review_delete_files(self, job_id: str):
        """クリップ一式を完全に削除し、リストを更新する。"""
        char_dir = os.path.join(self._review_work_folder, self._review_char_name)

        # webm（プレイヤーにロックされやすい）を最初に処理する
        targets = [
            os.path.join(char_dir, f"{job_id}.webm"),
            os.path.join(char_dir, f"{job_id}.json"),
            os.path.join(char_dir, f"{job_id}.npz"),
            os.path.join(char_dir, f"{job_id}_calibrated.npz"),
            os.path.join(char_dir, f"{job_id}_metrics.json"),
            os.path.join(char_dir, "original", f"{job_id}_original.mp4"),
        ]
        errors = []
        for path in targets:
            if not os.path.isfile(path):
                continue
            deleted = False
            for _attempt in range(3):
                try:
                    os.unlink(path)
                    deleted = True
                    break
                except OSError:
                    time.sleep(0.3)
            if not deleted:
                errors.append(os.path.basename(path))

        # リストから除去して両タブを更新
        self._review_videos = [v for v in self._review_videos
                               if v["job_id"] != job_id]
        self._review_populate_player_list()
        self._review_populate_grid()

        if errors:
            messagebox.showwarning(
                tr("gen.msgbox.warning_title"),
                tr("gen.review.delete_failed", name=job_id, files=", ".join(errors)))

    def _review_start_player(self):
        """Electronプレイヤーを起動。"""
        if not self._review_videos or not self._review_char_name:
            messagebox.showwarning(tr("gen.msgbox.warning_title"), tr("gen.msg.load_char"))
            return

        char_dir = os.path.join(self._review_work_folder, self._review_char_name)
        mouth_dir = os.path.join(char_dir, "mouth")
        if not os.path.isdir(mouth_dir):
            messagebox.showwarning(tr("gen.msgbox.warning_title"),
                                   tr("gen.msg.mouth_dir_missing", dir=mouth_dir))
            return

        # Build playlist with videos that have .json (track data)
        playlist_videos = []
        for v in self._review_videos:
            if not v.get("has_json"):
                continue
            char_dir_path = os.path.dirname(v["webm_path"])
            json_path = os.path.join(char_dir_path, f"{v['job_id']}.json")
            playlist_videos.append({
                "name": v["job_id"],
                "videoPath": v["webm_path"].replace("\\", "/"),
                "trackPath": json_path.replace("\\", "/"),
                "mouthPath": mouth_dir.replace("\\", "/"),
            })

        if not playlist_videos:
            messagebox.showwarning(tr("gen.msgbox.warning_title"), tr("gen.msg.no_playable_videos"))
            return

        playlist = {"videos": playlist_videos}

        # Write playlist to temp file (%TEMP%配下・残骸は起動時に掃除)
        tmp = tempfile.NamedTemporaryFile(
            mode="w", prefix=PLAYLIST_TMP_PREFIX, suffix=".json",
            delete=False, encoding="utf-8")
        json.dump(playlist, tmp, ensure_ascii=False, indent=2)
        tmp.close()
        self._review_playlist_path = tmp.name

        # Start WebSocket server
        try:
            from review_ws_server import ReviewWSServer
            self._review_ws_server = ReviewWSServer(port=0)
            ws_port = self._review_ws_server.start()
            self._review_ws_server.on_message = self._review_on_ws_message
        except Exception as e:
            messagebox.showerror(tr("gen.msgbox.error_title"), tr("gen.msg.ws_server_failed", error=e))
            os.unlink(self._review_playlist_path)
            return

        # Launch Electron
        player_dir = os.path.join(HERE, "MotionPNGTuber_Player")
        npx = shutil.which("npx")
        if not npx:
            messagebox.showerror(tr("gen.msgbox.error_title"), tr("gen.msg.npx_missing"))
            self._review_ws_server.stop()
            self._review_ws_server = None
            os.unlink(self._review_playlist_path)
            return
        # npm install 未実行だと npx が黙ってElectronのダウンロードを試みて
        # ハングしたように見えるため、事前に存在を確認する
        if not os.path.isdir(os.path.join(player_dir, "node_modules", "electron")):
            messagebox.showerror(tr("gen.msgbox.error_title"),
                                 tr("gen.msg.electron_not_installed"))
            self._review_ws_server.stop()
            self._review_ws_server = None
            os.unlink(self._review_playlist_path)
            return
        cmd = [
            npx, "electron", ".",
            "--character-folder", char_dir,
            "--playlist", self._review_playlist_path,
            "--ws-port", str(ws_port),
        ]
        try:
            self._review_electron_proc = subprocess.Popen(
                cmd, cwd=player_dir,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW
                if sys.platform == "win32" else 0)
        except Exception as e:
            messagebox.showerror(tr("gen.msgbox.error_title"), tr("gen.msg.player_launch_failed", error=e))
            self._review_ws_server.stop()
            self._review_ws_server = None
            os.unlink(self._review_playlist_path)
            return

        self._review_btn_play.configure(state="disabled")
        self._review_btn_stop.configure(state="normal")
        self._review_playing_var.set(tr("gen.review.playing"))

        # Poll for process alive
        self._review_poll_player()

    def _review_stop_player(self):
        """Electronプレイヤーを停止。"""
        if self._review_electron_proc is not None:
            pid = self._review_electron_proc.pid
            if sys.platform.startswith("win"):
                try:
                    subprocess.run(
                        ["taskkill", "/T", "/F", "/PID", str(pid)],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        creationflags=subprocess.CREATE_NO_WINDOW)
                except Exception:
                    try:
                        self._review_electron_proc.kill()
                    except Exception:
                        pass
            else:
                self._review_electron_proc.terminate()
            try:
                self._review_electron_proc.wait(timeout=3)
            except Exception:
                pass
            self._review_electron_proc = None
        if self._review_ws_server is not None:
            self._review_ws_server.stop()
            self._review_ws_server = None
        # Cleanup temp file
        if hasattr(self, "_review_playlist_path") and self._review_playlist_path:
            try:
                os.unlink(self._review_playlist_path)
            except OSError:
                pass
            self._review_playlist_path = ""
        self._review_btn_play.configure(state="normal")
        self._review_btn_stop.configure(state="disabled")
        self._review_playing_var.set("")
        # 削除機能の状態をリセット（応答待ちはプレイヤー消滅により完了しない）
        self._review_now_playing = ""
        self._review_pending_delete = None
        if self._review_delete_timeout_id:
            self.after_cancel(self._review_delete_timeout_id)
            self._review_delete_timeout_id = None
        self._review_refresh_del_cells()

    def _review_poll_player(self):
        """Electronプロセスの生存を定期チェック。"""
        if self._review_electron_proc is None:
            return
        if self._review_electron_proc.poll() is not None:
            # Process has exited
            self._review_stop_player()
            return
        self.after(1000, self._review_poll_player)

    def _review_on_video_select(self, _event):
        """TreeviewクリックでElectronに切替コマンドを送信。"""
        sel = self._review_player_tree.selection()
        if not sel or self._review_ws_server is None:
            return
        name = sel[0]  # iid = job_id
        self._review_ws_server.send_command({"type": "goto", "name": name})

    def _review_on_ws_message(self, msg: dict):
        """Electronからのメッセージを処理（WSスレッドから呼ばれる）。"""
        if msg.get("type") == "now_playing":
            name = msg.get("name", "")
            # Debounce: cancel pending highlight and schedule new one
            if hasattr(self, "_review_highlight_after_id") and self._review_highlight_after_id:
                self.after_cancel(self._review_highlight_after_id)
            self._review_highlight_after_id = self.after(
                300, lambda: self._review_highlight_playing(name))
        elif msg.get("type") == "removed":
            self.after(0, lambda m=dict(msg): self._review_on_removed(m))

    def _review_highlight_playing(self, name: str):
        """現在再生中の動画をTreeviewでハイライト。"""
        self._review_highlight_after_id = None
        self._review_now_playing = name
        tree = self._review_player_tree
        if tree.exists(name):
            tree.selection_set(name)
            tree.see(name)
        self._review_playing_var.set(tr("gen.review.playing_name", name=name))
        self._review_refresh_del_cells()

    # ================================================================
    # Engine callbacks
    # ================================================================

    def _setup_engine_callbacks(self):
        """エンジンからのコールバックをUIスレッドに転送。"""
        self._engine.on_progress = lambda t, d: self._enqueue_ui("progress", d)
        self._engine.on_job_update = lambda j: self._enqueue_ui("job_update", j)
        self._engine.on_error = lambda p, m: self._enqueue_ui("error", (p, m))
        self._engine.on_auto_pause = lambda m: self._enqueue_ui("auto_pause", m)
        self._engine.on_batch_complete = lambda: self._enqueue_ui("batch_complete", None)

    def _on_close(self):
        """ウィンドウ閉じ時のクリーンアップ。"""
        # 実行中エンジンは停止してから閉じる（stateファイル書き込み中の即死防止）
        if self._engine.status in (BATCH_STATUS_RUNNING, BATCH_STATUS_PAUSED):
            if not messagebox.askyesno(tr("gen.msgbox.confirm_title"),
                                       tr("gen.msg.confirm_close_running")):
                return
            self._engine.stop()
        self._persist_settings()
        # Review page cleanup
        self._review_stop_player()
        self.destroy()

    def _enqueue_ui(self, event_type: str, data):
        """UIスレッドのキューにイベントを追加（エンジンスレッドから呼ばれる）。"""
        self._ui_queue.put((event_type, data))

    def _poll_ui_queue(self):
        """UIスレッドでキューを定期的にポーリング。"""
        while True:
            try:
                event_type, data = self._ui_queue.get_nowait()
            except queue.Empty:
                break
            try:
                if event_type == "progress":
                    self._handle_progress(data)
                elif event_type == "job_update":
                    self._handle_job_update(data)
                elif event_type == "error":
                    self._handle_error(data)
                elif event_type == "auto_pause":
                    self._handle_auto_pause(data)
                elif event_type == "batch_complete":
                    self._handle_batch_complete()
            except Exception:
                logging.exception("UIイベント処理でエラー: %s", event_type)

        # Also refresh queue monitor periodically
        if self._engine.status in (BATCH_STATUS_RUNNING, BATCH_STATUS_PAUSED,
                                    BATCH_STATUS_COMPLETED, BATCH_STATUS_STOPPED):
            self._refresh_queue_monitor()

        self.after(500, self._poll_ui_queue)

    def _handle_progress(self, stats: dict):
        total = stats.get("total", 0)
        completed = stats.get("completed", 0)
        failed = stats.get("failed", 0)
        done = completed + failed

        self._progress_bar["maximum"] = max(total, 1)
        self._progress_bar["value"] = done
        self._var_progress_text.set(tr("gen.main.progress_text", done=done, total=total, failed=failed))

    def _handle_job_update(self, job: JobInfo):
        # Queue monitor will refresh periodically
        pass

    def _handle_error(self, data: tuple):
        phase, message = data
        # Show in status bar
        self._var_preflight_status.set(f"[{phase}] {message}")

    def _handle_auto_pause(self, message: str):
        self._update_control_buttons(BATCH_STATUS_PAUSED)
        messagebox.showwarning(tr("gen.msgbox.auto_pause_title"), message)

    def _handle_batch_complete(self):
        self._update_control_buttons(BATCH_STATUS_COMPLETED)
        self._regen_update_buttons(BATCH_STATUS_COMPLETED)
        self._refresh_queue_monitor()
        messagebox.showinfo(tr("gen.common.completed"), tr("gen.msg.batch_done"))

    def _refresh_queue_monitor(self):
        """キューモニターの表示を差分更新（ちらつき防止）。"""
        stats = self._engine.get_stats()

        # Phase1の現在の活動（エンジンが投入中などにセット、待機中は空文字）
        self._var_phase1_activity.set(self._engine.phase1_activity)

        # Phase1 slots (ラベルのテキスト変更のみ = ちらつかない)
        active_jobs = self._engine.get_phase1_active_jobs()
        num_slots = max(len(self._phase1_slot_labels), len(active_jobs))
        while len(self._phase1_slot_labels) < num_slots:
            idx = len(self._phase1_slot_labels)
            lbl = ttk.Label(self._phase1_slot_frame, text=f"Slot {idx+1}: -", anchor="w")
            lbl.pack(fill="x", padx=5, pady=1)
            self._phase1_slot_labels.append(lbl)
        for i, lbl in enumerate(self._phase1_slot_labels):
            if i < len(active_jobs):
                j = active_jobs[i]
                new_text = f"Slot {i+1}: {j.id} ({j.character})"
            else:
                new_text = f"Slot {i+1}: -"
            if lbl.cget("text") != new_text:
                lbl.configure(text=new_text)

        # Phase1 tree — 差分更新
        self._update_tree_diff(self._p1_tree, self._build_p1_rows())

        # Phase2 current
        p2_running = [j for j in self._engine.jobs
                      if j.state == JOB_STATE_PHASE2_RUNNING]
        new_p2_text = (f"{p2_running[0].id} ({p2_running[0].character})"
                       if p2_running else "-")
        if self._var_p2_current.get() != new_p2_text:
            self._var_p2_current.set(new_p2_text)

        # Phase2 stats
        self._var_p2_waiting.set(tr("gen.queue.waiting", n=stats.get('phase1_done', 0)))
        self._var_p2_completed.set(tr("gen.queue.completed", n=stats.get('completed', 0)))
        self._var_p2_failed.set(tr("gen.queue.failed", n=stats.get('failed', 0)))

        # Phase2 tree — 差分更新
        self._update_tree_diff(self._p2_tree, self._build_p2_rows())

    def _build_p1_rows(self) -> dict[str, tuple]:
        """Phase1ツリーの行データを構築。{job_id: (values...)}"""
        rows = {}
        for job in self._engine.jobs:
            if job.state in (JOB_STATE_PENDING, JOB_STATE_PHASE1_SUBMITTED):
                state_text = {
                    JOB_STATE_PENDING: tr("gen.queue.state_pending"),
                    JOB_STATE_PHASE1_SUBMITTED: tr("gen.queue.state_generating"),
                }.get(job.state, job.state)
                prompt_short = (job.prompt_text[:40] + "..."
                                if len(job.prompt_text) > 40 else job.prompt_text)
                rows[job.id] = (job.id, job.character, prompt_short,
                                state_text, job.phase1_retries)
        return rows

    def _build_p2_rows(self) -> dict[str, tuple]:
        """Phase2ツリーの行データを構築。{job_id: (values...)}"""
        rows = {}
        for job in self._engine.jobs:
            if job.state in (JOB_STATE_PHASE1_DONE, JOB_STATE_PHASE2_RUNNING,
                             JOB_STATE_COMPLETED, JOB_STATE_FAILED):
                state_text = {
                    JOB_STATE_PHASE1_DONE: tr("gen.queue.state_waiting"),
                    JOB_STATE_PHASE2_RUNNING: tr("gen.queue.processing"),
                    JOB_STATE_COMPLETED: tr("gen.common.completed"),
                    JOB_STATE_FAILED: tr("gen.common.failed"),
                }.get(job.state, job.state)
                output = os.path.basename(job.output_webm) if job.output_webm else "-"
                rows[job.id] = (job.id, job.character, state_text,
                                job.phase2_retries, output)
        return rows

    @staticmethod
    def _update_tree_diff(tree: ttk.Treeview, new_rows: dict[str, tuple]):
        """Treeviewを差分更新。追加・変更・削除のみ反映し、ちらつきを防止。"""
        existing = {}
        for iid in tree.get_children():
            existing[iid] = tuple(tree.item(iid, "values"))

        new_ids = set(new_rows.keys())
        old_ids = set(existing.keys())

        # 削除: 新データに無い行
        for iid in old_ids - new_ids:
            tree.delete(iid)

        # 追加 or 更新
        for job_id, values in new_rows.items():
            str_values = tuple(str(v) for v in values)
            if job_id in existing:
                if existing[job_id] != str_values:
                    tree.item(job_id, values=values)
            else:
                tree.insert("", "end", iid=job_id, values=values)

    # ================================================================
    # Page 6: 再生成（Phase2）
    # ================================================================

    def _build_page_regen(self, parent: ttk.Frame):
        """再生成ページの構築。"""
        note = ttk.Label(
            parent, foreground="#555",
            text=tr("gen.regen.note"))
        note.pack(anchor="w", padx=10, pady=(8, 0))

        # --- ワークフォルダ選択 ---
        folder_frame = ttk.LabelFrame(parent, text=tr("gen.common.work_folder"), padding=5)
        folder_frame.pack(fill="x", padx=10, pady=(10, 5))

        row = ttk.Frame(folder_frame)
        row.pack(fill="x")
        self._regen_folder_var = tk.StringVar()
        ttk.Entry(row, textvariable=self._regen_folder_var, state="readonly"
                  ).pack(side="left", fill="x", expand=True, padx=(0, 5))
        ttk.Button(row, text=tr("gen.common.select"), command=self._regen_select_folder
                   ).pack(side="left")

        # --- フィルター ---
        filter_frame = ttk.LabelFrame(parent, text=tr("gen.regen.filter_frame"), padding=5)
        filter_frame.pack(fill="x", padx=10, pady=5)

        self._regen_filter_var = tk.StringVar(value="failed_only")
        ttk.Radiobutton(filter_frame, text=tr("gen.regen.filter_all"),
                        variable=self._regen_filter_var, value="all",
                        command=self._regen_update_counts
                        ).pack(side="left", padx=10)
        ttk.Radiobutton(filter_frame, text=tr("gen.regen.filter_failed"),
                        variable=self._regen_filter_var, value="failed_only",
                        command=self._regen_update_counts
                        ).pack(side="left", padx=10)

        # --- キャラクター選択（スクロール可能） ---
        char_lf = ttk.LabelFrame(parent, text=tr("gen.regen.char_select_frame"), padding=5)
        char_lf.pack(fill="both", expand=True, padx=10, pady=5)

        canvas = tk.Canvas(char_lf, highlightthickness=0)
        scrollbar = ttk.Scrollbar(char_lf, orient="vertical", command=canvas.yview)
        self._regen_char_frame = ttk.Frame(canvas)

        self._regen_char_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self._regen_char_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # マウスホイールスクロール
        def _on_mousewheel(event):
            # bind_all はウィジェット破棄後も残り得るため存在確認でガード
            if not canvas.winfo_exists():
                return
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _on_mousewheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

        self._regen_char_vars: dict[str, tk.BooleanVar] = {}
        self._regen_data: dict[str, dict] = {}

        # --- 対象合計 + ボタン ---
        bottom_frame = ttk.Frame(parent)
        bottom_frame.pack(fill="x", padx=10, pady=(5, 10))

        self._regen_total_var = tk.StringVar(value=tr("gen.regen.total", n=0))
        ttk.Label(bottom_frame, textvariable=self._regen_total_var,
                  font=("", 12, "bold")).pack(side="left", padx=10)

        self._regen_btn_stop = ttk.Button(
            bottom_frame, text=tr("gen.common.stop"), command=self._regen_stop, state="disabled")
        self._regen_btn_stop.pack(side="right", padx=5)

        self._regen_btn_start = ttk.Button(
            bottom_frame, text=tr("gen.regen.start"), command=self._regen_start)
        self._regen_btn_start.pack(side="right", padx=5)

    def _regen_select_folder(self):
        """再生成用ワークフォルダ選択（全ページに反映）。"""
        folder = filedialog.askdirectory(title=tr("gen.dialog.select_work_folder"))
        if not folder:
            return
        self._warn_if_path_not_ansi(folder)
        self._apply_work_folder(folder)

    def _regen_scan(self):
        """ワークフォルダをスキャンし、再生成対象を取得。"""
        folder = self._regen_folder_var.get()
        if not folder or not os.path.isdir(folder):
            return

        self._regen_data = {}

        for entry in sorted(os.listdir(folder)):
            char_dir = os.path.join(folder, entry)
            if not os.path.isdir(char_dir):
                continue
            original_dir = os.path.join(char_dir, "original")
            if not os.path.isdir(original_dir):
                continue

            # face_info 存在チェック
            has_face_info = False
            for name in [f"{entry}_face_info.json", "face_info.json"]:
                if os.path.isfile(os.path.join(char_dir, name)):
                    has_face_info = True
                    break
            if not has_face_info:
                continue

            videos = []
            for f in sorted(os.listdir(original_dir)):
                if not f.endswith("_original.mp4"):
                    continue
                job_id = f.replace("_original.mp4", "")
                mp4_path = os.path.join(original_dir, f)

                # メトリクスチェック
                metrics_path = os.path.join(char_dir, f"{job_id}_metrics.json")
                is_failed = True  # メトリクス無し → 失敗扱い
                mean_conf = -1.0
                if os.path.isfile(metrics_path):
                    try:
                        with open(metrics_path, "r", encoding="utf-8") as mf:
                            metrics = json.load(mf)
                        mean_conf = metrics.get("mean_conf", 0.0)
                        is_failed = mean_conf < MIN_CONF_THRESHOLD
                    except Exception:
                        pass

                videos.append({
                    "path": mp4_path,
                    "job_id": job_id,
                    "character": entry,
                    "failed": is_failed,
                })

            if videos:
                total = len(videos)
                failed = sum(1 for v in videos if v["failed"])
                self._regen_data[entry] = {
                    "total": total,
                    "failed": failed,
                    "videos": videos,
                }

        self._regen_build_char_list()
        self._regen_update_counts()

    def _regen_build_char_list(self):
        """キャラクター選択リストを再構築。"""
        # 既存ウィジェットをクリア
        for w in self._regen_char_frame.winfo_children():
            w.destroy()
        self._regen_char_vars.clear()

        if not self._regen_data:
            ttk.Label(self._regen_char_frame,
                      text=tr("gen.regen.no_chars")).pack(pady=20)
            return

        # ヘッダー
        header = ttk.Frame(self._regen_char_frame)
        header.pack(fill="x", padx=5, pady=(0, 5))
        ttk.Label(header, text="", width=3).pack(side="left")
        ttk.Label(header, text=tr("gen.common.character"), width=15, anchor="w",
                  font=("", 9, "bold")).pack(side="left")
        ttk.Label(header, text=tr("gen.regen.col_videos"), width=8, anchor="center",
                  font=("", 9, "bold")).pack(side="left")
        ttk.Label(header, text=tr("gen.regen.col_failed"), width=8, anchor="center",
                  font=("", 9, "bold")).pack(side="left")
        ttk.Label(header, text=tr("gen.regen.col_target"), width=8, anchor="center",
                  font=("", 9, "bold")).pack(side="left")

        for char_name, data in sorted(self._regen_data.items()):
            row = ttk.Frame(self._regen_char_frame)
            row.pack(fill="x", padx=5, pady=1)

            var = tk.BooleanVar(value=data["failed"] > 0)
            self._regen_char_vars[char_name] = var

            cb = ttk.Checkbutton(row, variable=var,
                                 command=self._regen_update_counts)
            cb.pack(side="left")
            ttk.Label(row, text=char_name, width=15, anchor="w").pack(side="left")
            ttk.Label(row, text=str(data["total"]), width=8,
                      anchor="center").pack(side="left")

            failed_label = ttk.Label(row, text=str(data["failed"]), width=8,
                                     anchor="center")
            failed_label.pack(side="left")
            if data["failed"] > 0:
                failed_label.configure(foreground="red")

            # 対象数ラベル（フィルター依存）
            target_var = tk.StringVar()
            ttk.Label(row, textvariable=target_var, width=8,
                      anchor="center").pack(side="left")
            # target_var を保持して _regen_update_counts で更新
            data["_target_var"] = target_var

    def _regen_update_counts(self):
        """フィルターとチェック状態に基づき対象本数を更新。"""
        filter_mode = self._regen_filter_var.get()
        total_target = 0

        for char_name, data in self._regen_data.items():
            var = self._regen_char_vars.get(char_name)
            is_checked = var.get() if var else False

            if is_checked:
                if filter_mode == "all":
                    count = data["total"]
                else:
                    count = data["failed"]
            else:
                count = 0

            target_var = data.get("_target_var")
            if target_var:
                target_var.set(str(count))
            total_target += count

        self._regen_total_var.set(tr("gen.regen.total", n=total_target))

    def _regen_get_target_videos(self) -> list[dict]:
        """選択されたキャラ×フィルターに基づき対象動画リストを取得。"""
        filter_mode = self._regen_filter_var.get()
        targets = []

        for char_name, data in self._regen_data.items():
            var = self._regen_char_vars.get(char_name)
            if not var or not var.get():
                continue
            for video in data["videos"]:
                if filter_mode == "all" or video["failed"]:
                    targets.append(video)

        return targets

    def _regen_start(self):
        """再生成（Phase2のみ）を開始。"""
        # 一時停止中も含めてブロックする。実行中/一時停止中のバッチと
        # 同じ batch_state.json を共有するため、上書きすると復帰情報が壊れる。
        if self._engine.status in (BATCH_STATUS_RUNNING, BATCH_STATUS_PAUSED):
            messagebox.showwarning(tr("gen.msgbox.warning_title"), tr("gen.msg.batch_busy"))
            return

        folder = self._regen_folder_var.get()
        if not folder:
            messagebox.showwarning(tr("gen.msgbox.warning_title"), tr("gen.msg.select_work_folder"))
            return

        targets = self._regen_get_target_videos()
        if not targets:
            messagebox.showwarning(tr("gen.msgbox.warning_title"), tr("gen.msg.no_regen_targets"))
            return

        # 確認
        msg = tr("gen.msg.confirm_regen_body", n=len(targets))
        if not messagebox.askyesno(tr("gen.msgbox.confirm_title"), msg):
            return

        # 設定ページからPhase2関連パラメータを取得
        self._persist_settings()
        settings = self._get_settings_from_ui()
        config = BatchConfig(
            bg_tolerance=settings.get("bg_tolerance", 65),
            smooth_cutoff=settings.get("smooth_cutoff", 3.0),
            gpu_rest_count=settings.get("gpu_rest_count", 10),
            gpu_rest_seconds=settings.get("gpu_rest_seconds", 60),
        )

        # ジョブ作成
        jobs = []
        for video in targets:
            job = JobInfo(
                id=video["job_id"],
                character=video["character"],
                prompt_list="",
                prompt_index=0,
                prompt_text="",
                original_path=video["path"],
            )
            jobs.append(job)

        # 開始
        self._engine.start_phase2_only(folder, jobs, config)
        self._update_control_buttons(BATCH_STATUS_RUNNING)
        self._regen_update_buttons(BATCH_STATUS_RUNNING)

        # キューモニターに自動遷移
        self._show_page(2)

    def _regen_stop(self):
        """再生成を停止。"""
        if messagebox.askyesno(tr("gen.msgbox.confirm_title"), tr("gen.msg.confirm_stop_regen")):
            self._engine.stop()
            self._update_control_buttons(BATCH_STATUS_STOPPED)
            self._regen_update_buttons(BATCH_STATUS_STOPPED)

    def _regen_update_buttons(self, status: str):
        """再生成ページのボタン状態を更新。"""
        is_idle = status in (BATCH_STATUS_IDLE, BATCH_STATUS_STOPPED,
                             BATCH_STATUS_COMPLETED)
        is_active = status in (BATCH_STATUS_RUNNING, BATCH_STATUS_PAUSED)

        self._regen_btn_start.configure(state="normal" if is_idle else "disabled")
        self._regen_btn_stop.configure(state="normal" if is_active else "disabled")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app = BatchGeneratorApp()
    _warn_if_catalog_broken()
    app.mainloop()


def _warn_if_catalog_broken():
    """翻訳カタログが読めていない場合に警告する。

    この状態では tr() が生のキー名を返すため、メッセージは日英ハードコード。
    """
    errors = catalog_load_errors()
    if errors:
        messagebox.showwarning(
            "Translation files / 翻訳ファイル",
            "翻訳ファイルの読み込みに失敗しました。UIがキー名のまま表示されます。\n"
            "locales フォルダを含めてリポジトリを再取得してください。\n\n"
            "Failed to load translation files. UI text will appear as raw keys.\n"
            "Please re-download the repository including the locales folder.\n\n"
            + "\n".join(errors))


if __name__ == "__main__":
    main()
