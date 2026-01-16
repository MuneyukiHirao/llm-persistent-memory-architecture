#!/usr/bin/env python3
"""
learnings (JSONB) → learning (TEXT NULL) 変換スクリプト

既存の memory/*.json ファイルの観点別 learnings を
単一の learning テキストに変換する。

変換ルール:
1. learnings が存在し、価値ある内容がある場合 → 最も重要な学びを learning に設定
2. learnings が報告や数値のみの場合 → learning = null

実行方法:
    python scripts/convert_learnings_to_learning.py

出力:
    - 各ファイルの変換結果を表示
    - --apply オプションで実際にファイルを更新
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def is_exceptional_learning(text: str) -> bool:
    """学びが「例外的なイベント」かどうかを判定"""
    # 報告や数値のみのパターン
    report_patterns = [
        r"^\d+件?の?テスト",  # "36件のテスト"
        r"^\d+テスト",  # "36テスト"
        r"^\d+秒",  # "0.10秒"
        r"^全テスト.*パス",  # "全テストがパス"
        r"^テスト.*成功",  # "テストが成功"
        r"^完了",  # "完了"
        r"^作成した$",  # "作成した"
        r"^実装した$",  # "実装した"
    ]

    for pattern in report_patterns:
        if re.match(pattern, text):
            return False

    # 価値ある学びのパターン
    exceptional_patterns = [
        r"エラー",  # エラー解決
        r"問題",   # 問題発見
        r"発見",   # 発見
        r"判明",   # 判明
        r"重要",   # 重要
        r"注意",   # 注意点
        r"回避",   # 回避策
        r"防止",   # 防止策
        r"改善",   # 改善
        r"効率",   # 効率化
        r"なぜなら",  # 理由説明
        r"ため",   # 理由説明
    ]

    for pattern in exceptional_patterns:
        if re.search(pattern, text):
            return True

    # デフォルトは保持（手動確認用）
    return True


def extract_best_learning(learnings: Dict[str, str]) -> Optional[str]:
    """learnings辞書から最も価値のある学びを抽出"""
    if not learnings:
        return None

    # 例外的な学びをフィルタリング
    exceptional = []
    for perspective, text in learnings.items():
        if text and is_exceptional_learning(text):
            exceptional.append(f"[{perspective}] {text}")

    if not exceptional:
        return None

    # 最も長い（詳細な）学びを選択
    exceptional.sort(key=len, reverse=True)

    # 複数あれば最大3つまで結合
    if len(exceptional) > 3:
        exceptional = exceptional[:3]

    return "\n".join(exceptional)


def convert_memory(memory: Dict[str, Any]) -> Dict[str, Any]:
    """単一メモリエントリを変換"""
    # learnings を learning に変換
    learnings = memory.pop("learnings", None)

    if learnings:
        learning = extract_best_learning(learnings)
        memory["learning"] = learning
    else:
        memory["learning"] = None

    return memory


def convert_file(file_path: Path, apply: bool = False) -> Dict[str, Any]:
    """ファイルを変換"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    converted_count = 0
    null_count = 0

    # memories 配列を処理
    if "memories" in data:
        for memory in data["memories"]:
            if "learnings" in memory:
                convert_memory(memory)
                if memory["learning"]:
                    converted_count += 1
                else:
                    null_count += 1

    # learnings 配列も処理（memory_core_agent の古い形式）
    if "learnings" in data and isinstance(data["learnings"], list):
        for memory in data["learnings"]:
            if "learnings" in memory:
                convert_memory(memory)
                if memory.get("learning"):
                    converted_count += 1
                else:
                    null_count += 1

    result = {
        "file": str(file_path.name),
        "converted": converted_count,
        "null": null_count,
        "total": converted_count + null_count,
    }

    if apply:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        result["applied"] = True

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="learnings → learning 変換")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="実際にファイルを更新する",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="特定のファイルのみ変換",
    )
    args = parser.parse_args()

    memory_dir = Path(__file__).parent.parent / "memory"

    if args.file:
        files = [memory_dir / args.file]
    else:
        files = list(memory_dir.glob("*_memory.json"))

    print("=" * 60)
    print("learnings → learning 変換")
    print("=" * 60)

    if not args.apply:
        print("⚠️  プレビューモード（--apply で実際に更新）")
    print()

    total_converted = 0
    total_null = 0

    for file_path in sorted(files):
        if not file_path.exists():
            continue

        result = convert_file(file_path, apply=args.apply)

        status = "✅" if args.apply else "📋"
        print(f"{status} {result['file']}")
        print(f"   変換: {result['converted']}件, NULL: {result['null']}件")

        total_converted += result["converted"]
        total_null += result["null"]

    print()
    print("=" * 60)
    print(f"合計: 変換 {total_converted}件, NULL {total_null}件")

    if not args.apply:
        print()
        print("実際に変換するには: python scripts/convert_learnings_to_learning.py --apply")


if __name__ == "__main__":
    main()
