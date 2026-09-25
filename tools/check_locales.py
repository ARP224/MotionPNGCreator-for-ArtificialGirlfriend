"""locales/*.json の整合性チェック。

実行: uv run python tools/check_locales.py

検査項目:
  1. 全ロケール間でキー集合が一致すること
  2. 各キーの {placeholder} 集合がロケール間で一致すること
     （波括弧の不整合 = 未エスケープの literal { } も書式パース時に検出）
  3. *.red_phrase キー: 対応する *.text キーが存在し、フレーズが
     改行を含まず、本文の同一行内に完全一致で含まれること
     （tk.Text の search は改行をまたいでマッチしないため）
"""

import json
import string
import sys
from pathlib import Path

LOCALES_DIR = Path(__file__).resolve().parent.parent / "locales"


def load_locales() -> dict:
    locales = {}
    for path in sorted(LOCALES_DIR.glob("*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"[NG] {path.name}: JSONパースエラー: {e}")
            sys.exit(1)
        if not isinstance(data, dict):
            print(f"[NG] {path.name}: トップレベルがオブジェクトではありません")
            sys.exit(1)
        locales[path.stem] = data
    return locales


def placeholders(text: str):
    """str.format のフィールド名集合を返す。パース不能なら ValueError。"""
    fields = set()
    for literal, field, spec, conv in string.Formatter().parse(text):
        if field is not None:
            # "{0}" や "{}" の位置引数は使わない方針なのでフィールド名をそのまま集める
            fields.add(field)
    return fields


def main() -> int:
    locales = load_locales()
    if not locales:
        print(f"[NG] {LOCALES_DIR} に *.json がありません")
        return 1

    errors = []
    names = sorted(locales.keys())

    # 1. キー集合の一致
    all_keys = {name: set(cat.keys()) for name, cat in locales.items()}
    union = set().union(*all_keys.values())
    for name in names:
        missing = union - all_keys[name]
        for key in sorted(missing):
            others = [n for n in names if key in all_keys[n]]
            errors.append(f"キー欠落: {name}.json に '{key}' がない（{', '.join(others)} には存在）")

    # 2. プレースホルダの一致（全ロケールに存在するキーのみ比較）
    common = set.intersection(*all_keys.values()) if all_keys else set()
    fields_by_locale = {}
    for name in names:
        fields_by_locale[name] = {}
        for key, text in locales[name].items():
            if not isinstance(text, str):
                errors.append(f"型エラー: {name}.json の '{key}' が文字列ではありません")
                continue
            try:
                fields_by_locale[name][key] = placeholders(text)
            except ValueError as e:
                errors.append(
                    f"波括弧エラー: {name}.json の '{key}' が書式として不正です（{e}）。"
                    f"リテラルの波括弧は {{{{ }}}} にエスケープしてください"
                )
    for key in sorted(common):
        sets = {}
        for name in names:
            if key in fields_by_locale[name]:
                sets[name] = fields_by_locale[name][key]
        values = list(sets.values())
        if values and any(v != values[0] for v in values[1:]):
            detail = ", ".join(f"{n}={sorted(s) or '[]'}" for n, s in sets.items())
            errors.append(f"プレースホルダ不一致: '{key}' → {detail}")

    # 3. red_phrase の包含・改行チェック
    for name in names:
        cat = locales[name]
        for key, phrase in cat.items():
            if not key.endswith(".red_phrase") or not isinstance(phrase, str):
                continue
            text_key = key[: -len(".red_phrase")] + ".text"
            if not phrase:
                errors.append(f"red_phrase空: {name}.json の '{key}'")
                continue
            if "\n" in phrase:
                errors.append(f"red_phrase改行: {name}.json の '{key}' に改行が含まれています")
                continue
            text = cat.get(text_key)
            if not isinstance(text, str):
                errors.append(f"red_phrase対応欠落: {name}.json に '{text_key}' がありません")
                continue
            if not any(phrase in line for line in text.split("\n")):
                errors.append(
                    f"red_phrase不包含: {name}.json の '{key}' が "
                    f"'{text_key}' の同一行内に見つかりません"
                )

    if errors:
        for e in errors:
            print(f"[NG] {e}")
        print(f"\n{len(errors)} 件の問題があります")
        return 1

    total = len(union)
    print(f"[OK] {', '.join(n + '.json' for n in names)}: {total} キー、整合性に問題なし")
    return 0


if __name__ == "__main__":
    sys.exit(main())
