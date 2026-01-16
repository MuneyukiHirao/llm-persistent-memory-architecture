# Spaced Repetition（間隔反復学習）スケジューラ
# 復習タイミングを最適化し、長期記憶への定着を促進
# 実装仕様: docs/architecture.ja.md「復習（Spaced Repetition）」セクション

"""
Spaced Repetitionモジュール

Ebbinghausの忘却曲線に基づく間隔反復学習のスケジュール計算を提供。
人間の学習で効果的な「間隔を空けた復習」を実装する。

設計方針（メモリ管理エージェント観点）:
- 強度の正確性: review_countの増加はcalculate_next_reviewでのみ行う
- 2段階強化との統合: 復習正解時は既存の強化メカニズムと組み合わせ可能
- テスト容易性: Phase1Configを依存性注入で受け取る
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List
from uuid import UUID

from src.config.phase1_config import Phase1Config
from src.models.memory import AgentMemory


@dataclass
class ReviewSchedule:
    """復習スケジュール情報

    Attributes:
        memory_id: 対象メモリのID
        next_review_at: 次回復習予定日時
        interval_days: 次回復習までの間隔（日数）
        review_count: 復習回数（正解回数）
    """

    memory_id: str
    next_review_at: datetime
    interval_days: int
    review_count: int

    def to_dict(self) -> dict:
        """辞書に変換

        Returns:
            全フィールドを含む辞書
        """
        return {
            "memory_id": self.memory_id,
            "next_review_at": self.next_review_at.isoformat(),
            "interval_days": self.interval_days,
            "review_count": self.review_count,
        }


class SpacedRepetitionScheduler:
    """Spaced Repetitionスケジューラ

    Ebbinghausの忘却曲線に基づき、復習タイミングを計算する。

    設計:
        - 正解時: 間隔を延長（interval * multiplier）
        - 不正解時: 間隔をリセット（初期値に戻る）
        - 最大間隔を超えないよう制限

    パラメータ（Phase1Config）:
        - initial_interval_days: 初回復習間隔（デフォルト: 1日）
        - interval_multiplier: 正解時の間隔倍率（デフォルト: 2.0）
        - max_interval_days: 最大間隔（デフォルト: 30日）

    使用例:
        scheduler = SpacedRepetitionScheduler(config)

        # 復習結果に基づいて次のスケジュールを計算
        schedule = scheduler.calculate_next_review(memory, is_correct=True)

        # 復習期限が来ているメモリを取得
        due_ids = scheduler.get_due_reviews(memories, datetime.now())

    Attributes:
        config: Phase1Config インスタンス
    """

    def __init__(self, config: Phase1Config | None = None):
        """SpacedRepetitionScheduler を初期化

        Args:
            config: Phase1Config インスタンス（省略時はデフォルト設定を使用）
        """
        self.config = config or Phase1Config()

    def calculate_next_review(
        self,
        memory: AgentMemory,
        is_correct: bool,
        current_time: datetime | None = None,
    ) -> ReviewSchedule:
        """次回復習スケジュールを計算

        Args:
            memory: 対象のAgentMemory
            is_correct: 復習テストで正解したか
            current_time: 現在時刻（テスト用、省略時はdatetime.now()）

        Returns:
            ReviewSchedule: 次回復習スケジュール

        計算ロジック:
            - 正解時:
                - 現在の間隔を取得（初回は initial_interval_days）
                - 新しい間隔 = 現在の間隔 * interval_multiplier
                - max_interval_days を超えないよう制限
                - review_count += 1
            - 不正解時:
                - 間隔を initial_interval_days にリセット
                - review_count は変更なし（正解回数を維持）

        Note:
            このメソッドは計算のみを行い、AgentMemoryの更新は行わない。
            実際の更新は呼び出し元が responsibility を持つ。
        """
        if current_time is None:
            current_time = datetime.now()

        # 現在の間隔を計算（review_countに基づく）
        # 初回（review_count=0）は initial_interval_days
        # 以降は前回の間隔 * multiplier（を積み重ねた値）
        current_interval = self._get_current_interval(memory.review_count)

        if is_correct:
            # 正解時: 間隔を延長
            new_interval = min(
                int(current_interval * self.config.interval_multiplier),
                self.config.max_interval_days,
            )
            new_review_count = memory.review_count + 1
        else:
            # 不正解時: 間隔をリセット
            new_interval = self.config.initial_interval_days
            new_review_count = memory.review_count  # 正解回数は維持

        next_review_at = current_time + timedelta(days=new_interval)

        return ReviewSchedule(
            memory_id=str(memory.id),
            next_review_at=next_review_at,
            interval_days=new_interval,
            review_count=new_review_count,
        )

    def get_due_reviews(
        self,
        memories: List[AgentMemory],
        current_time: datetime | None = None,
    ) -> List[str]:
        """復習期限が来ているメモリのIDリストを取得

        Args:
            memories: 検索対象のAgentMemoryリスト
            current_time: 現在時刻（テスト用、省略時はdatetime.now()）

        Returns:
            復習期限が来ているメモリのIDリスト（文字列）

        判定基準:
            - next_review_at が None の場合: 一度も復習スケジュールが設定されていないため除外
            - next_review_at <= current_time の場合: 期限到達として含める

        Note:
            返却順序は入力リストの順序に従う。
            優先度による並び替えは呼び出し元が responsibility を持つ。
        """
        if current_time is None:
            current_time = datetime.now()

        due_ids: List[str] = []

        for memory in memories:
            if memory.next_review_at is not None and memory.next_review_at <= current_time:
                due_ids.append(str(memory.id))

        return due_ids

    def _get_current_interval(self, review_count: int) -> int:
        """現在の復習間隔を計算（内部メソッド）

        Args:
            review_count: これまでの正解回数

        Returns:
            現在の復習間隔（日数）

        計算:
            interval = initial_interval_days * (multiplier ^ review_count)
            ただし max_interval_days を超えない

        例（initial=1, multiplier=2, max=30）:
            review_count=0: 1日
            review_count=1: 2日
            review_count=2: 4日
            review_count=3: 8日
            review_count=4: 16日
            review_count=5: 30日（max制限）
        """
        if review_count == 0:
            return self.config.initial_interval_days

        interval = self.config.initial_interval_days * (
            self.config.interval_multiplier ** review_count
        )
        return min(int(interval), self.config.max_interval_days)

    def schedule_initial_review(
        self,
        memory: AgentMemory,
        current_time: datetime | None = None,
    ) -> ReviewSchedule:
        """初回復習スケジュールを設定

        教育プロセスで記憶を作成した後、初回の復習スケジュールを設定する。

        Args:
            memory: 対象のAgentMemory
            current_time: 現在時刻（テスト用、省略時はdatetime.now()）

        Returns:
            ReviewSchedule: 初回復習スケジュール

        Note:
            これは calculate_next_review(memory, is_correct=True) と同等だが、
            意図を明確にするためのヘルパーメソッド。
        """
        if current_time is None:
            current_time = datetime.now()

        initial_interval = self.config.initial_interval_days
        next_review_at = current_time + timedelta(days=initial_interval)

        return ReviewSchedule(
            memory_id=str(memory.id),
            next_review_at=next_review_at,
            interval_days=initial_interval,
            review_count=0,  # 初回なのでまだ正解回数は0
        )
