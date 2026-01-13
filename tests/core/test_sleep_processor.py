# 睡眠フェーズプロセッサのテスト
"""
SleepPhaseProcessor の単体テスト

テスト観点:
- apply_decay_all の正確性
- 空リストの処理
- 異なる consolidation_level での減衰率
- バッチ処理（ページネーション）
- 障害時のデータ整合性
"""

from datetime import datetime
from typing import List
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from src.config.phase1_config import Phase1Config
from src.core.sleep_processor import SleepPhaseProcessor
from src.models.memory import AgentMemory


class TestApplyDecayAll:
    """apply_decay_all メソッドのテスト"""

    @pytest.fixture
    def config(self) -> Phase1Config:
        """テスト用の設定"""
        return Phase1Config()

    @pytest.fixture
    def mock_db(self) -> MagicMock:
        """モック DatabaseConnection"""
        return MagicMock()

    @pytest.fixture
    def processor(self, mock_db: MagicMock, config: Phase1Config) -> SleepPhaseProcessor:
        """テスト用の SleepPhaseProcessor"""
        return SleepPhaseProcessor(mock_db, config)

    def _create_memory(
        self,
        strength: float = 1.0,
        consolidation_level: int = 0,
        access_count: int = 0,
    ) -> AgentMemory:
        """テスト用メモリを作成"""
        now = datetime.now()
        return AgentMemory(
            id=uuid4(),
            agent_id="test_agent",
            content="テスト用記憶",
            strength=strength,
            consolidation_level=consolidation_level,
            access_count=access_count,
            created_at=now,
            updated_at=now,
        )

    def test_apply_decay_all_empty_list(
        self, processor: SleepPhaseProcessor
    ):
        """空のメモリリストでエラーなし"""
        # Arrange
        processor.repository.get_memories_for_decay = MagicMock(return_value=[])

        # Act
        result = processor.apply_decay_all("test_agent")

        # Assert
        assert result == 0
        processor.repository.get_memories_for_decay.assert_called_once_with(
            "test_agent", 100
        )

    def test_apply_decay_all_single_memory(
        self, processor: SleepPhaseProcessor, config: Phase1Config
    ):
        """単一メモリの減衰"""
        # Arrange
        memory = self._create_memory(strength=1.0, consolidation_level=0)
        processor.repository.get_memories_for_decay = MagicMock(
            side_effect=[[memory], []]  # 最初に1件、次は空
        )
        processor.repository.batch_update_strength = MagicMock(return_value=1)

        # 期待される減衰率（Level 0: 0.95^0.1 ≒ 0.9949）
        expected_decay_rate = config.get_decay_rate(0)
        expected_new_strength = 1.0 * expected_decay_rate

        # Act
        result = processor.apply_decay_all("test_agent")

        # Assert
        assert result == 1
        processor.repository.batch_update_strength.assert_called_once()

        # batch_update_strength に渡された引数を検証
        call_args = processor.repository.batch_update_strength.call_args[0][0]
        assert len(call_args) == 1
        assert call_args[0][0] == memory.id
        assert call_args[0][1] == pytest.approx(expected_new_strength, rel=0.001)

    def test_apply_decay_all_multiple_consolidation_levels(
        self, processor: SleepPhaseProcessor, config: Phase1Config
    ):
        """異なる consolidation_level のメモリの減衰"""
        # Arrange
        memories = [
            self._create_memory(strength=1.0, consolidation_level=0),  # Level 0
            self._create_memory(strength=1.0, consolidation_level=2),  # Level 2
            self._create_memory(strength=1.0, consolidation_level=5),  # Level 5
        ]
        processor.repository.get_memories_for_decay = MagicMock(
            side_effect=[memories, []]
        )
        processor.repository.batch_update_strength = MagicMock(return_value=3)

        # 期待される減衰率
        expected_rates = {
            0: config.get_decay_rate(0),  # ≒ 0.9949
            2: config.get_decay_rate(2),  # ≒ 0.9980
            5: config.get_decay_rate(5),  # ≒ 0.9998
        }

        # Act
        result = processor.apply_decay_all("test_agent")

        # Assert
        assert result == 3

        # 各メモリの新しい強度を検証
        call_args = processor.repository.batch_update_strength.call_args[0][0]
        assert len(call_args) == 3

        for i, (memory, expected_level) in enumerate(zip(memories, [0, 2, 5])):
            expected_strength = 1.0 * expected_rates[expected_level]
            assert call_args[i][0] == memory.id
            assert call_args[i][1] == pytest.approx(expected_strength, rel=0.001)

    def test_apply_decay_all_pagination(
        self, processor: SleepPhaseProcessor
    ):
        """バッチ処理（100件以上）のページネーション"""
        # Arrange: 150件のメモリを3バッチに分けて処理
        batch_size = 50
        total_memories = 150

        # 3バッチ分のメモリを作成
        all_memories = [self._create_memory() for _ in range(total_memories)]

        # get_memories_for_decay が各バッチを返すように設定
        def mock_get_memories(agent_id, size):
            # 各呼び出しで異なるバッチを返す
            return []  # 実際には下のside_effectで制御

        processor.repository.get_memories_for_decay = MagicMock(
            side_effect=[
                all_memories[0:50],    # 1st batch
                all_memories[50:100],  # 2nd batch
                all_memories[100:150], # 3rd batch
                [],                    # End signal
            ]
        )
        processor.repository.batch_update_strength = MagicMock(return_value=50)

        # Act
        result = processor.apply_decay_all("test_agent", batch_size=50)

        # Assert
        assert result == 150
        assert processor.repository.batch_update_strength.call_count == 3

    def test_apply_decay_all_handles_duplicate_memory_in_batches(
        self, processor: SleepPhaseProcessor
    ):
        """同じメモリが複数バッチで返されても重複処理しない"""
        # Arrange
        memory1 = self._create_memory()
        memory2 = self._create_memory()

        # 同じメモリが再度返されるケース
        processor.repository.get_memories_for_decay = MagicMock(
            side_effect=[
                [memory1, memory2],  # 1st batch
                [memory1, memory2],  # Same memories returned (already processed)
            ]
        )
        processor.repository.batch_update_strength = MagicMock(return_value=2)

        # Act
        result = processor.apply_decay_all("test_agent", batch_size=10)

        # Assert: 2件のみ処理（重複なし）
        assert result == 2
        assert processor.repository.batch_update_strength.call_count == 1

    def test_apply_decay_all_strength_calculation(
        self, processor: SleepPhaseProcessor, config: Phase1Config
    ):
        """強度計算の正確性を検証"""
        # Arrange: 特定の強度を持つメモリ
        memory = self._create_memory(strength=0.8, consolidation_level=3)
        processor.repository.get_memories_for_decay = MagicMock(
            side_effect=[[memory], []]
        )
        processor.repository.batch_update_strength = MagicMock(return_value=1)

        # Level 3 の減衰率を取得
        decay_rate = config.get_decay_rate(3)  # 0.99^0.1 ≒ 0.9990
        expected_new_strength = 0.8 * decay_rate

        # Act
        processor.apply_decay_all("test_agent")

        # Assert
        call_args = processor.repository.batch_update_strength.call_args[0][0]
        assert call_args[0][1] == pytest.approx(expected_new_strength, rel=0.001)


class TestDecayRateCalculation:
    """減衰率計算の検証"""

    @pytest.fixture
    def config(self) -> Phase1Config:
        return Phase1Config()

    def test_decay_rate_level_0(self, config: Phase1Config):
        """Level 0 の減衰率 ≒ 0.9949"""
        rate = config.get_decay_rate(0)
        # 0.95^0.1 ≒ 0.9949
        assert rate == pytest.approx(0.9949, rel=0.001)

    def test_decay_rate_level_5(self, config: Phase1Config):
        """Level 5 の減衰率 ≒ 0.9998"""
        rate = config.get_decay_rate(5)
        # 0.998^0.1 ≒ 0.9998
        assert rate == pytest.approx(0.9998, rel=0.001)

    def test_decay_rate_progression(self, config: Phase1Config):
        """定着レベルが上がるほど減衰率が高くなる（減衰が小さい）"""
        rates = [config.get_decay_rate(level) for level in range(6)]

        # 各レベルで減衰率が上昇することを確認
        for i in range(5):
            assert rates[i] < rates[i + 1], f"Level {i} < Level {i+1}"

    def test_daily_decay_targets(self, config: Phase1Config):
        """日次減衰目標が正しく設定されている"""
        # 10タスク/日で累積すると日次目標に近づく
        for level in range(6):
            task_rate = config.get_decay_rate(level)
            daily_rate = task_rate ** config.expected_tasks_per_day
            expected_daily = config.daily_decay_targets[level]
            assert daily_rate == pytest.approx(expected_daily, rel=0.001)


class TestConsolidationLevel:
    """定着レベル計算の検証"""

    @pytest.fixture
    def config(self) -> Phase1Config:
        return Phase1Config()

    def test_consolidation_level_from_access_count(self, config: Phase1Config):
        """access_count から定着レベルを正しく計算"""
        # [0, 5, 15, 30, 60, 100]
        test_cases = [
            (0, 0),
            (4, 0),
            (5, 1),
            (14, 1),
            (15, 2),
            (29, 2),
            (30, 3),
            (59, 3),
            (60, 4),
            (99, 4),
            (100, 5),
            (1000, 5),
        ]

        for access_count, expected_level in test_cases:
            actual = config.get_consolidation_level(access_count)
            assert actual == expected_level, \
                f"access_count={access_count}: expected {expected_level}, got {actual}"


class TestArchiveWeakMemories:
    """archive_weak_memories メソッドのテスト"""

    @pytest.fixture
    def config(self) -> Phase1Config:
        """テスト用の設定"""
        return Phase1Config()

    @pytest.fixture
    def mock_db(self) -> MagicMock:
        """モック DatabaseConnection"""
        return MagicMock()

    @pytest.fixture
    def processor(self, mock_db: MagicMock, config: Phase1Config) -> SleepPhaseProcessor:
        """テスト用の SleepPhaseProcessor"""
        return SleepPhaseProcessor(mock_db, config)

    def _create_memory(
        self,
        strength: float = 1.0,
        status: str = "active",
    ) -> AgentMemory:
        """テスト用メモリを作成"""
        now = datetime.now()
        return AgentMemory(
            id=uuid4(),
            agent_id="test_agent",
            content="テスト用記憶 - これはテストのための長いコンテンツです。50文字を超える場合は省略されます。",
            strength=strength,
            status=status,
            created_at=now,
            updated_at=now,
        )

    def test_archive_weak_memories_no_active_memories(
        self, processor: SleepPhaseProcessor
    ):
        """アクティブメモリがない場合は0を返す"""
        # Arrange
        processor.repository.get_by_agent_id = MagicMock(return_value=[])

        # Act
        result = processor.archive_weak_memories("test_agent")

        # Assert
        assert result == 0
        processor.repository.get_by_agent_id.assert_called_once_with(
            "test_agent", status="active"
        )

    def test_archive_weak_memories_no_candidates(
        self, processor: SleepPhaseProcessor, config: Phase1Config
    ):
        """全メモリが閾値超の場合はアーカイブなし"""
        # Arrange: archive_threshold (0.1) より大きい強度のメモリ
        memories = [
            self._create_memory(strength=0.5),
            self._create_memory(strength=0.3),
            self._create_memory(strength=0.15),  # 閾値(0.1)より大きい
        ]
        processor.repository.get_by_agent_id = MagicMock(return_value=memories)
        processor.repository.batch_archive = MagicMock(return_value=0)

        # Act
        result = processor.archive_weak_memories("test_agent")

        # Assert
        assert result == 0
        processor.repository.batch_archive.assert_not_called()

    def test_archive_weak_memories_single_memory(
        self, processor: SleepPhaseProcessor, config: Phase1Config
    ):
        """単一メモリのアーカイブ"""
        # Arrange: 1件だけ閾値以下
        weak_memory = self._create_memory(strength=0.05)
        strong_memory = self._create_memory(strength=0.5)
        memories = [strong_memory, weak_memory]

        processor.repository.get_by_agent_id = MagicMock(return_value=memories)
        processor.repository.batch_archive = MagicMock(return_value=1)

        # Act
        result = processor.archive_weak_memories("test_agent")

        # Assert
        assert result == 1
        processor.repository.batch_archive.assert_called_once()

        # batch_archive に渡されたIDを検証
        call_args = processor.repository.batch_archive.call_args[0][0]
        assert len(call_args) == 1
        assert call_args[0] == weak_memory.id

    def test_archive_weak_memories_multiple_memories(
        self, processor: SleepPhaseProcessor, config: Phase1Config
    ):
        """複数メモリのアーカイブ"""
        # Arrange: 複数件が閾値以下
        weak_memories = [
            self._create_memory(strength=0.05),
            self._create_memory(strength=0.08),
            self._create_memory(strength=0.03),
        ]
        strong_memory = self._create_memory(strength=0.5)
        memories = [strong_memory] + weak_memories

        processor.repository.get_by_agent_id = MagicMock(return_value=memories)
        processor.repository.batch_archive = MagicMock(return_value=3)

        # Act
        result = processor.archive_weak_memories("test_agent")

        # Assert
        assert result == 3
        processor.repository.batch_archive.assert_called_once()

        # batch_archive に渡されたIDを検証（順序は保証しないのでsetで比較）
        call_args = processor.repository.batch_archive.call_args[0][0]
        assert len(call_args) == 3
        expected_ids = {m.id for m in weak_memories}
        actual_ids = set(call_args)
        assert actual_ids == expected_ids

    def test_archive_weak_memories_boundary_value_equal(
        self, processor: SleepPhaseProcessor, config: Phase1Config
    ):
        """境界値テスト: strength == archive_threshold (0.1) はアーカイブ対象"""
        # Arrange: ちょうど閾値と同じ強度
        threshold = config.archive_threshold  # 0.1
        boundary_memory = self._create_memory(strength=threshold)

        processor.repository.get_by_agent_id = MagicMock(return_value=[boundary_memory])
        processor.repository.batch_archive = MagicMock(return_value=1)

        # Act
        result = processor.archive_weak_memories("test_agent")

        # Assert: <= なので閾値ちょうどもアーカイブ対象
        assert result == 1
        processor.repository.batch_archive.assert_called_once()
        call_args = processor.repository.batch_archive.call_args[0][0]
        assert call_args[0] == boundary_memory.id

    def test_archive_weak_memories_boundary_value_just_above(
        self, processor: SleepPhaseProcessor, config: Phase1Config
    ):
        """境界値テスト: strength > archive_threshold はアーカイブ対象外"""
        # Arrange: 閾値をわずかに超える強度
        threshold = config.archive_threshold  # 0.1
        above_threshold_memory = self._create_memory(strength=threshold + 0.001)

        processor.repository.get_by_agent_id = MagicMock(return_value=[above_threshold_memory])
        processor.repository.batch_archive = MagicMock(return_value=0)

        # Act
        result = processor.archive_weak_memories("test_agent")

        # Assert: > なので閾値をわずかに超えるとアーカイブ対象外
        assert result == 0
        processor.repository.batch_archive.assert_not_called()

    def test_archive_weak_memories_calls_batch_archive(
        self, processor: SleepPhaseProcessor
    ):
        """batch_archive が正しく呼び出される（論理削除）"""
        # Arrange
        memory = self._create_memory(strength=0.05)
        processor.repository.get_by_agent_id = MagicMock(return_value=[memory])
        processor.repository.batch_archive = MagicMock(return_value=1)

        # Act
        processor.archive_weak_memories("test_agent")

        # Assert: batch_archive が UUIDリストで呼び出される
        processor.repository.batch_archive.assert_called_once()
        call_args = processor.repository.batch_archive.call_args[0][0]
        assert isinstance(call_args, list)
        assert all(isinstance(id, type(memory.id)) for id in call_args)

    def test_archive_weak_memories_returns_actual_archived_count(
        self, processor: SleepPhaseProcessor
    ):
        """batch_archive の戻り値（実際にアーカイブされた件数）を返す"""
        # Arrange: 3件が候補だが、2件のみアーカイブ成功（DBエラー等）
        weak_memories = [
            self._create_memory(strength=0.05),
            self._create_memory(strength=0.05),
            self._create_memory(strength=0.05),
        ]
        processor.repository.get_by_agent_id = MagicMock(return_value=weak_memories)
        processor.repository.batch_archive = MagicMock(return_value=2)  # 実際は2件成功

        # Act
        result = processor.archive_weak_memories("test_agent")

        # Assert: batch_archive の戻り値をそのまま返す
        assert result == 2


class TestConsolidateSimilar:
    """consolidate_similar メソッドのテスト（Phase 1 簡易版）"""

    @pytest.fixture
    def config(self) -> Phase1Config:
        """テスト用の設定"""
        return Phase1Config()

    @pytest.fixture
    def mock_db(self) -> MagicMock:
        """モック DatabaseConnection"""
        return MagicMock()

    @pytest.fixture
    def processor(self, mock_db: MagicMock, config: Phase1Config) -> SleepPhaseProcessor:
        """テスト用の SleepPhaseProcessor"""
        return SleepPhaseProcessor(mock_db, config)

    def test_consolidate_similar_returns_zero(
        self, processor: SleepPhaseProcessor
    ):
        """Phase 1 では常に0を返す"""
        # Act
        result = processor.consolidate_similar("test_agent")

        # Assert
        assert result == 0

    def test_consolidate_similar_no_error(
        self, processor: SleepPhaseProcessor
    ):
        """エラーが発生しないことを確認"""
        # Act & Assert: 例外が発生しないことを確認
        try:
            result = processor.consolidate_similar("test_agent")
            assert result == 0
        except Exception as e:
            pytest.fail(f"Unexpected exception raised: {e}")

    def test_consolidate_similar_different_agent_ids(
        self, processor: SleepPhaseProcessor
    ):
        """異なるエージェントIDでも同様に動作"""
        # Act & Assert
        assert processor.consolidate_similar("agent_001") == 0
        assert processor.consolidate_similar("agent_002") == 0
        assert processor.consolidate_similar("") == 0  # 空文字列でもエラーなし

    def test_consolidate_similar_logs_skip_message(
        self, processor: SleepPhaseProcessor, caplog
    ):
        """スキップ理由がログ出力されることを確認"""
        import logging

        # Arrange
        with caplog.at_level(logging.INFO):
            # Act
            processor.consolidate_similar("test_agent")

        # Assert: ログメッセージにスキップ理由が含まれる
        assert any(
            "consolidate_similar スキップ" in record.message
            and "test_agent" in record.message
            and "Phase 1 では未実装" in record.message
            for record in caplog.records
        )
