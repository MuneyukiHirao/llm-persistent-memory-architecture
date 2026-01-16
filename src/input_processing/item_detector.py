"""論点検出モジュール

仕様書参照: docs/phase2-implementation-spec.ja.md セクション5.1
"""

import re
from typing import List

from src.config.phase2_config import Phase2Config


class ItemDetector:
    """論点（項目）検出クラス

    ユーザー入力から論点（タスク項目）を検出する。
    箇条書き、番号付きリスト、改行区切りなどを認識。

    Phase 2 MVP: 簡易実装（正規表現ベース、LLM呼び出しなし）
    """

    def __init__(self, config: Phase2Config):
        self.config = config

        # 箇条書きパターン（-, *, •, ・, ■, □, ◆, ◇ など）
        self._bullet_pattern = re.compile(
            r'^\s*[-*•・■□◆◇▪▫►▸➤➢→]\s*(.+)$',
            re.MULTILINE
        )

        # 番号付きリストパターン（1. 1) (1) ① など）
        self._numbered_pattern = re.compile(
            r'^\s*(?:\d+[.)）]|\(\d+\)|[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳])\s*(.+)$',
            re.MULTILINE
        )

        # 「第N項」「項目N」パターン
        self._heading_pattern = re.compile(
            r'^\s*(?:第\d+[項条章節]|項目\d+|タスク\d+|要件\d+)[：:]\s*(.+)$',
            re.MULTILINE
        )

    def detect(self, user_input: str) -> List[str]:
        """ユーザー入力から論点を検出

        Args:
            user_input: ユーザー入力テキスト

        Returns:
            検出された論点のリスト
        """
        if not user_input or not user_input.strip():
            return []

        items: List[str] = []

        # 1. 箇条書きを検出
        bullet_items = self._bullet_pattern.findall(user_input)
        items.extend([item.strip() for item in bullet_items if item.strip()])

        # 2. 番号付きリストを検出
        numbered_items = self._numbered_pattern.findall(user_input)
        items.extend([item.strip() for item in numbered_items if item.strip()])

        # 3. 見出し形式を検出
        heading_items = self._heading_pattern.findall(user_input)
        items.extend([item.strip() for item in heading_items if item.strip()])

        # 明示的なリストが見つからない場合、改行区切りでチェック
        if not items:
            items = self._detect_by_newlines(user_input)

        # 重複を除去しつつ順序を保持
        seen = set()
        unique_items = []
        for item in items:
            if item not in seen:
                seen.add(item)
                unique_items.append(item)

        return unique_items

    def _detect_by_newlines(self, user_input: str) -> List[str]:
        """改行区切りで論点を検出（フォールバック）

        複数行に渡る入力で、各行が独立した論点の可能性がある場合に使用。
        ただし、単純な改行は論点とは見なさない。
        """
        lines = user_input.strip().split('\n')

        # 1行のみの場合は論点リストではない
        if len(lines) <= 1:
            return []

        items = []
        for line in lines:
            line = line.strip()
            # 空行、短すぎる行、接続詞で始まる行は除外
            if not line or len(line) < 3:
                continue
            # 接続詞・副詞で始まる行は前の行の続きと判断
            if self._is_continuation(line):
                continue
            items.append(line)

        # 3行以上で初めて論点リストとみなす
        if len(items) >= 3:
            return items

        return []

    def _is_continuation(self, line: str) -> bool:
        """行が前の行の続きかどうかを判定"""
        continuation_markers = [
            'そして', 'また', 'さらに', 'ただし', 'しかし', 'なお',
            'つまり', 'すなわち', 'よって', 'したがって', 'ゆえに',
            'ところで', 'ちなみに', 'あるいは', 'もしくは',
        ]
        for marker in continuation_markers:
            if line.startswith(marker):
                return True
        return False
