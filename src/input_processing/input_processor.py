"""入力処理層メインモジュール

仕様書参照: docs/phase2-implementation-spec.ja.md セクション5.1
"""

import uuid
from dataclasses import dataclass, field
from typing import List, Optional

from src.config.phase2_config import Phase2Config
from src.input_processing.item_detector import ItemDetector
from src.input_processing.summarizer import Summarizer


@dataclass
class ProcessedInput:
    """入力処理層の出力

    Attributes:
        summary: 概要（summary_max_tokens以内）。入力が小さい場合は元の入力そのまま。
        detail_refs: 詳細へのポインタ（入力が大きい場合に元データを参照するためのID）
        items: 検出された論点（タスク項目）のリスト
        item_count: 論点数
        original_size_tokens: 元の入力の推定トークン数
        needs_negotiation: 論点数が閾値を超え、ユーザーとの交渉が必要か
        negotiation_options: 交渉時に提示する選択肢
    """

    summary: str
    detail_refs: List[str] = field(default_factory=list)
    items: List[str] = field(default_factory=list)
    item_count: int = 0
    original_size_tokens: int = 0
    needs_negotiation: bool = False
    negotiation_options: List[str] = field(default_factory=list)


class InputProcessor:
    """入力処理層

    ユーザー入力を前処理してオーケストレーターに渡す。

    責務:
    - 論点数の検出（ItemDetector）
    - 入力サイズの確認とサマリー生成（Summarizer）
    - オーケストレーターへの構造化された入力の提供

    重要な設計判断:
    - 「解釈」しない: 曖昧な指示もそのままオーケストレーターに渡す
    - 「曖昧さの解消」はオーケストレーターの責務
    """

    # 日本語テキストの概算: 1トークン ≒ 0.5〜1文字
    # 安全のため 1トークン = 1文字 で計算
    CHARS_PER_TOKEN = 1

    def __init__(self, config: Optional[Phase2Config] = None):
        """初期化

        Args:
            config: Phase2Config インスタンス。None の場合はデフォルト設定を使用。
        """
        self.config = config or Phase2Config()
        self.item_detector = ItemDetector(self.config)
        self.summarizer = Summarizer(self.config)

        # 詳細データの一時保存（Phase 2 MVPでは簡易実装）
        self._detail_storage: dict[str, str] = {}

    def process(self, user_input: str) -> ProcessedInput:
        """ユーザー入力を処理

        処理フロー:
        1. 論点数を検出
        2. 論点数が閾値以上なら交渉オプションを設定
        3. 入力サイズが閾値以上なら概要を生成
        4. 構造化された ProcessedInput を返却

        Args:
            user_input: ユーザー入力テキスト

        Returns:
            ProcessedInput: 処理された入力データ
        """
        if not user_input or not user_input.strip():
            return ProcessedInput(
                summary="",
                detail_refs=[],
                items=[],
                item_count=0,
                original_size_tokens=0,
                needs_negotiation=False,
                negotiation_options=[],
            )

        # 入力サイズを計算
        original_size_tokens = self._count_tokens(user_input)

        # 1. 論点数を検出
        items = self.item_detector.detect(user_input)
        item_count = len(items)

        # 2. 論点数チェック
        if item_count >= self.config.input_item_threshold:
            # 概要を生成
            summary = self.summarizer.summarize(user_input)

            # 詳細を保存
            detail_ref = self._store_detail(user_input)

            return ProcessedInput(
                summary=summary,
                detail_refs=[detail_ref],
                items=items,
                item_count=item_count,
                original_size_tokens=original_size_tokens,
                needs_negotiation=True,
                negotiation_options=[
                    f"優先度の高い{self.config.input_item_threshold}個を指定してください",
                    "全て処理します（時間がかかります）",
                    "カテゴリ別に分けて順次処理します",
                ],
            )

        # 3. 入力サイズチェック
        if original_size_tokens > self.config.input_size_threshold:
            # 概要を生成
            summary = self.summarizer.summarize(user_input)

            # 詳細を保存
            detail_ref = self._store_detail(user_input)

            return ProcessedInput(
                summary=summary,
                detail_refs=[detail_ref],
                items=items,
                item_count=item_count,
                original_size_tokens=original_size_tokens,
                needs_negotiation=False,
                negotiation_options=[],
            )

        # 4. 小さい入力はそのまま
        return ProcessedInput(
            summary=user_input,
            detail_refs=[],
            items=items,
            item_count=item_count,
            original_size_tokens=original_size_tokens,
            needs_negotiation=False,
            negotiation_options=[],
        )

    def get_detail(self, detail_ref: str) -> Optional[str]:
        """詳細データを取得

        Args:
            detail_ref: 詳細へのポインタ（_store_detail で返されたID）

        Returns:
            詳細データ。見つからない場合は None。
        """
        return self._detail_storage.get(detail_ref)

    def _count_tokens(self, text: str) -> int:
        """テキストのトークン数を推定

        Phase 2 MVP: 簡易実装（文字数ベース）
        将来的にはtiktokenなどを使用した正確なカウントを実装予定。

        Args:
            text: 入力テキスト

        Returns:
            推定トークン数
        """
        return len(text) // self.CHARS_PER_TOKEN

    def _store_detail(self, user_input: str) -> str:
        """詳細データを保存

        Phase 2 MVP: メモリ内の辞書に保存
        将来的にはRedisやDBへの永続化を実装予定。

        Args:
            user_input: 保存する入力データ

        Returns:
            詳細へのポインタ（参照用ID）
        """
        detail_ref = f"detail_{uuid.uuid4().hex[:12]}"
        self._detail_storage[detail_ref] = user_input
        return detail_ref
