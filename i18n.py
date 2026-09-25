"""JSON辞書ベースの軽量i18nモジュール。

使い方:
    from i18n import tr
    label = tr("gen.main.start")
    msg = tr("gen.log.jobs_done", n=3)   # locales側は "{n} 件..." 形式

言語の決定（初回tr()呼び出し時に一度だけ・スレッドセーフ）:
  1. .batch_settings.json の "language" キー（GUIのドロップダウンでのみ書き込まれる）
  2. OSロケール（WindowsはGetUserDefaultUILanguageを最優先）
  3. "en"

このモジュールは設定ファイルを読むだけで、書き込みは一切しない。
キーが見つからない場合は 選択言語 → en → キー名そのまま の順でフォールバックする。
翻訳文字列の書式エラー（波括弧の不整合など）でも例外を出さず未整形のまま返す。
"""

import json
import locale
import os
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
SETTINGS_PATH = os.path.join(HERE, ".batch_settings.json")
LOCALES_DIR = os.path.join(HERE, "locales")

SUPPORTED_LANGUAGES = ["ja", "en"]
DEFAULT_LANGUAGE = "en"

_LANG_ID_JAPANESE = 0x11  # Windows LANGID の primary language

_lock = threading.Lock()
_lang = None
_catalog = {}
_fallback = {}
_load_errors: list[str] = []


def _detect_os_language() -> str:
    if sys.platform == "win32":
        try:
            import ctypes
            lang_id = ctypes.windll.kernel32.GetUserDefaultUILanguage()
            if (lang_id & 0x3FF) == _LANG_ID_JAPANESE:
                return "ja"
            return "en"
        except Exception:
            pass
    try:
        loc = locale.getlocale()[0] or ""
    except Exception:
        loc = ""
    if not loc:
        loc = os.environ.get("LANG", "")
    low = loc.lower()
    # "ja_JP" のほか、Windowsが返す "Japanese_Japan" 形式も ja とみなす
    if low.startswith("ja") or low.startswith("japanese"):
        return "ja"
    return "en"


def _read_settings_language():
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8-sig") as f:
            lang = json.load(f).get("language")
        return lang if lang in SUPPORTED_LANGUAGES else None
    except Exception:
        return None


def _load_catalog(lang: str) -> dict:
    path = os.path.join(LOCALES_DIR, f"{lang}.json")
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("catalog root is not a JSON object")
        return data
    except FileNotFoundError:
        return {}  # 未対応言語の追加時などは静かにフォールバック
    except Exception as e:
        # 破損・読み込み不能は記録しておき、GUI側が起動時に警告できるようにする
        _load_errors.append(f"{os.path.basename(path)}: {e}")
        return {}


def _init():
    global _lang, _catalog, _fallback
    if _lang is not None:
        return
    with _lock:
        if _lang is not None:
            return
        lang = _read_settings_language() or _detect_os_language()
        _catalog = _load_catalog(lang)
        if lang == DEFAULT_LANGUAGE:
            _fallback = _catalog
        else:
            _fallback = _load_catalog(DEFAULT_LANGUAGE)
        _lang = lang  # 最後に代入（他スレッドから見て初期化完了の印）


def current_language() -> str:
    """現在の表示言語コードを返す（"ja" / "en"）。"""
    _init()
    return _lang


def catalog_load_errors() -> list[str]:
    """翻訳カタログの読み込み失敗一覧を返す（正常時は空リスト）。

    失敗すると全UIが生のキー名表示になるため、GUIは起動時にこれを確認して
    ユーザーへ警告することを推奨する。
    """
    _init()
    return list(_load_errors)


def tr(key: str, **kwargs) -> str:
    """キーに対応する翻訳文字列を返す。kwargs は str.format で埋め込む。

    kwargs が無くても常に format を試みる（訳文中の {{ }} エスケープを
    { } に展開するため）。書式エラー時は未整形のまま返し、例外は出さない。
    """
    _init()
    text = _catalog.get(key)
    if text is None:
        text = _fallback.get(key)
    if text is None:
        return key
    try:
        return text.format(**kwargs)
    except (KeyError, IndexError, ValueError):
        return text
