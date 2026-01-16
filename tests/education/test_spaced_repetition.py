# SpacedRepetitionScheduler テスト
# 間隔反復学習のスケジュール計算が正しく機能することを確認

"""
SpacedRepetitionScheduler のユニットテスト

テスト観点:
- calculate_next_review: 正解/不正解時の間隔計算
- get_due_reviews: 復習期限判定
- パラメータ（multiplier, max_interval）の反映
"""

from datetime import datetime, timedelta
from uuid import uuid4

import pytest

from src.config.phase1_config import Phase1Config
from src.education.spaced_repetition import ReviewSchedule, SpacedRepetitionScheduler
from src.models.memory import AgentMemory


class TestReviewSchedule:
    """ReviewSchedule dataclass のテスト"""

    def test_create_review_schedule(self):
        """ReviewScheduleを正常に作成できる"""
        memory_id = str(uuid4())
        next_review = datetime(2026, 1, 20, 10, 0, 0)

        schedule = ReviewSchedule(
            memory_id=memory_id,
            next_review_at=next_review,
            interval_days=3,
            review_count=2,
        )

        assert schedule.memory_id == memory_id
        assert schedule.next_review_at == next_review
        assert schedule.interval_days == 3
        assert schedule.review_count == 2

    def test_to_dict(self):
        """to_dict()で辞書に変換できる"""
        memory_id = str(uuid4())
        next_review = datetime(2026, 1, 20, 10, 0, 0)

        schedule = ReviewSchedule(
            memory_id=memory_id,
            next_review_at=next_review,
            interval_days=3,
            review_count=2,
        )

        result = schedule.to_dict()

        assert result["memory_id"] == memory_id
        assert result["next_review_at"] == "2026-01-20T10:00:00"
        assert result["interval_days"] == 3
        assert result["review_count"] == 2


class TestSpacedRepetitionScheduler:
    """SpacedRepetitionScheduler のテスト"""

    @pytest.fixture
    def config(self) -> Phase1Config:
        """デフォルト設定のフィクスチャ"""
        return Phase1Config()

    @pytest.fixture
    def scheduler(self, config: Phase1Config) -> SpacedRepetitionScheduler:
        """スケジューラのフィクスチャ"""
        return SpacedRepetitionScheduler(config)

    @pytest.fixture
    def sample_memory(self) -> AgentMemory:
        """サンプルメモリのフィクスチャ"""
        return AgentMemory.create(
            agent_id="test_agent",
            content="テスト用メモリ",
        )

    def test_init_with_default_config(self):
        """デフォルト設定で初期化できる"""
        scheduler = SpacedRepetitionScheduler()
        assert scheduler.config is not None

    def test_init_with_custom_config(self, config: Phase1Config):
        """カスタム設定で初期化できる"""
        scheduler = SpacedRepetitionScheduler(config)
        assert scheduler.config == config

    # === calculate_next_review テスト ===

    def test_calculate_next_review_correct_first_time(
        self, scheduler: SpacedRepetitionScheduler, sample_memory: AgentMemory
    ):
        """初回正解時: interval = initial * multiplier"""
        current_time = datetime(2026, 1, 15, 10, 0, 0)

        schedule = scheduler.calculate_next_review(
            sample_memory, is_correct=True, current_time=current_time
        )

        # initial=1, multiplier=2 なので、初回正解で interval=2
        assert schedule.interval_days == 2
        assert schedule.next_review_at == current_time + timedelta(days=2)
        assert schedule.review_count == 1

    def test_calculate_next_review_correct_multiple_times(
        self, scheduler: SpacedRepetitionScheduler
    ):
        """複数回正解時: interval が指数的に増加"""
        current_time = datetime(2026, 1, 15, 10, 0, 0)

        # review_count=1 のメモリ（すでに1回正解済み）
        memory = AgentMemory.create(
            agent_id="test_agent",
            content="テスト用メモリ",
        )
        # review_countを手動で設定
        memory = memory.copy_with(review_count=1)

        schedule = scheduler.calculate_next_review(
            memory, is_correct=True, current_time=current_time
        )

        # 現在の間隔 = 1 * 2^1 = 2
        # 新しい間隔 = 2 * 2 = 4
        assert schedule.interval_days == 4
        assert schedule.review_count == 2

    def test_calculate_next_review_respects_max_interval(
        self, scheduler: SpacedRepetitionScheduler
    ):
        """max_interval_days を超えない"""
        current_time = datetime(2026, 1, 15, 10, 0, 0)

        # review_count=10 のメモリ（間隔が十分大きい）
        memory = AgentMemory.create(
            agent_id="test_agent",
            content="テスト用メモリ",
        )
        memory = memory.copy_with(review_count=10)

        schedule = scheduler.calculate_next_review(
            memory, is_correct=True, current_time=current_time
        )

        # max_interval_days=30 を超えない
        assert schedule.interval_days <= scheduler.config.max_interval_days
        assert schedule.interval_days == 30

    def test_calculate_next_review_incorrect_resets_interval(
        self, scheduler: SpacedRepetitionScheduler
    ):
        """不正解時: interval をリセット"""
        current_time = datetime(2026, 1, 15, 10, 0, 0)

        # review_count=3 のメモリ
        memory = AgentMemory.create(
            agent_id="test_agent",
            content="テスト用メモリ",
        )
        memory = memory.copy_with(review_count=3)

        schedule = scheduler.calculate_next_review(
            memory, is_correct=False, current_time=current_time
        )

        # 不正解時は initial_interval_days にリセット
        assert schedule.interval_days == scheduler.config.initial_interval_days
        # review_count は変更しない
        assert schedule.review_count == 3

    def test_calculate_next_review_uses_current_time(
        self, scheduler: SpacedRepetitionScheduler, sample_memory: AgentMemory
    ):
        """current_time が正しく使用される"""
        current_time = datetime(2026, 6, 15, 12, 30, 0)

        schedule = scheduler.calculate_next_review(
            sample_memory, is_correct=True, current_time=current_time
        )

        assert schedule.next_review_at > current_time
        expected = current_time + timedelta(days=schedule.interval_days)
        assert schedule.next_review_at == expected

    # === get_due_reviews テスト ===

    def test_get_due_reviews_empty_list(
        self, scheduler: SpacedRepetitionScheduler
    ):
        """空のリストを渡すと空のリストを返す"""
        result = scheduler.get_due_reviews([], datetime.now())
        assert result == []

    def test_get_due_reviews_no_due(
        self, scheduler: SpacedRepetitionScheduler
    ):
        """期限が来ていないメモリは含まれない"""
        current_time = datetime(2026, 1, 15, 10, 0, 0)
        future_time = datetime(2026, 1, 20, 10, 0, 0)

        memory = AgentMemory.create(
            agent_id="test_agent",
            content="テスト用メモリ",
        )
        memory = memory.copy_with(next_review_at=future_time)

        result = scheduler.get_due_reviews([memory], current_time)

        assert result == []

    def test_get_due_reviews_with_due_memories(
        self, scheduler: SpacedRepetitionScheduler
    ):
        """期限が来ているメモリのIDを返す"""
        current_time = datetime(2026, 1, 15, 10, 0, 0)
        past_time = datetime(2026, 1, 10, 10, 0, 0)

        memory = AgentMemory.create(
            agent_id="test_agent",
            content="テスト用メモリ",
        )
        memory = memory.copy_with(next_review_at=past_time)

        result = scheduler.get_due_reviews([memory], current_time)

        assert len(result) == 1
        assert result[0] == str(memory.id)

    def test_get_due_reviews_excludes_none_next_review(
        self, scheduler: SpacedRepetitionScheduler
    ):
        """next_review_at が None のメモリは除外"""
        current_time = datetime(2026, 1, 15, 10, 0, 0)

        memory = AgentMemory.create(
            agent_id="test_agent",
            content="テスト用メモリ",
        )
        # next_review_at は None（デフォルト）

        result = scheduler.get_due_reviews([memory], current_time)

        assert result == []

    def test_get_due_reviews_mixed_memories(
        self, scheduler: SpacedRepetitionScheduler
    ):
        """複数メモリで期限到達と未到達が混在"""
        current_time = datetime(2026, 1, 15, 10, 0, 0)
        past_time = datetime(2026, 1, 10, 10, 0, 0)
        future_time = datetime(2026, 1, 20, 10, 0, 0)

        # 期限到達
        due_memory = AgentMemory.create(
            agent_id="test_agent",
            content="期限到達メモリ",
        )
        due_memory = due_memory.copy_with(next_review_at=past_time)

        # 期限未到達
        not_due_memory = AgentMemory.create(
            agent_id="test_agent",
            content="期限未到達メモリ",
        )
        not_due_memory = not_due_memory.copy_with(next_review_at=future_time)

        # next_review_at が None
        no_schedule_memory = AgentMemory.create(
            agent_id="test_agent",
            content="スケジュールなしメモリ",
        )

        memories = [due_memory, not_due_memory, no_schedule_memory]
        result = scheduler.get_due_reviews(memories, current_time)

        assert len(result) == 1
        assert result[0] == str(due_memory.id)

    def test_get_due_reviews_exact_time(
        self, scheduler: SpacedRepetitionScheduler
    ):
        """current_time と next_review_at が等しい場合は期限到達"""
        current_time = datetime(2026, 1, 15, 10, 0, 0)

        memory = AgentMemory.create(
            agent_id="test_agent",
            content="テスト用メモリ",
        )
        memory = memory.copy_with(next_review_at=current_time)

        result = scheduler.get_due_reviews([memory], current_time)

        assert len(result) == 1
        assert result[0] == str(memory.id)

    # === schedule_initial_review テスト ===

    def test_schedule_initial_review(
        self, scheduler: SpacedRepetitionScheduler, sample_memory: AgentMemory
    ):
        """初回復習スケジュールを設定"""
        current_time = datetime(2026, 1, 15, 10, 0, 0)

        schedule = scheduler.schedule_initial_review(
            sample_memory, current_time=current_time
        )

        assert schedule.interval_days == scheduler.config.initial_interval_days
        assert schedule.next_review_at == current_time + timedelta(days=1)
        assert schedule.review_count == 0

    # === _get_current_interval テスト ===

    def test_get_current_interval_zero(
        self, scheduler: SpacedRepetitionScheduler
    ):
        """review_count=0 の場合は initial_interval_days"""
        result = scheduler._get_current_interval(0)
        assert result == scheduler.config.initial_interval_days

    def test_get_current_interval_progression(
        self, scheduler: SpacedRepetitionScheduler
    ):
        """review_count に応じて間隔が指数的に増加"""
        # initial=1, multiplier=2
        # review_count=0: 1日
        # review_count=1: 2日
        # review_count=2: 4日
        # review_count=3: 8日
        assert scheduler._get_current_interval(0) == 1
        assert scheduler._get_current_interval(1) == 2
        assert scheduler._get_current_interval(2) == 4
        assert scheduler._get_current_interval(3) == 8

    def test_get_current_interval_respects_max(
        self, scheduler: SpacedRepetitionScheduler
    ):
        """max_interval_days を超えない"""
        # review_count=10: 1 * 2^10 = 1024 -> max=30
        result = scheduler._get_current_interval(10)
        assert result == scheduler.config.max_interval_days


class TestSpacedRepetitionSchedulerWithCustomConfig:
    """カスタム設定でのテスト"""

    def test_custom_initial_interval(self):
        """カスタム initial_interval_days が反映される"""
        config = Phase1Config()
        config.initial_interval_days = 3

        scheduler = SpacedRepetitionScheduler(config)
        memory = AgentMemory.create(
            agent_id="test_agent",
            content="テスト用メモリ",
        )

        schedule = scheduler.schedule_initial_review(memory)

        assert schedule.interval_days == 3

    def test_custom_multiplier(self):
        """カスタム interval_multiplier が反映される"""
        config = Phase1Config()
        config.interval_multiplier = 3.0

        scheduler = SpacedRepetitionScheduler(config)
        memory = AgentMemory.create(
            agent_id="test_agent",
            content="テスト用メモリ",
        )

        schedule = scheduler.calculate_next_review(memory, is_correct=True)

        # initial=1, multiplier=3 なので、初回正解で interval=3
        assert schedule.interval_days == 3

    def test_custom_max_interval(self):
        """カスタム max_interval_days が反映される"""
        config = Phase1Config()
        config.max_interval_days = 14

        scheduler = SpacedRepetitionScheduler(config)
        memory = AgentMemory.create(
            agent_id="test_agent",
            content="テスト用メモリ",
        )
        memory = memory.copy_with(review_count=10)

        schedule = scheduler.calculate_next_review(memory, is_correct=True)

        assert schedule.interval_days == 14
