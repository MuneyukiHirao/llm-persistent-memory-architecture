# 強度管理マネージャーのテスト
"""
StrengthManager の単体テスト

テスト観点（検証エージェント視点）:
- テストカバレッジ: 2段階強化、インパクト反映、定着レベル、再活性化の網羅
- 再現性: モック使用による DB 非依存のテスト
- 境界値・異常系: 空リスト、存在しないID、無効なインパクトタイプ等
- パフォーマンス: バッチ処理の効率性
- 保守性: 各メソッドを独立してテスト可能な構造

仕様書参照:
- docs/phase1-implementation-spec.ja.md セクション4.1, 4.4
"""

from datetime import datetime
from typing import List, Optional
from unittest.mock import MagicMock, call
from uuid import UUID, uuid4

import pytest

from src.config.phase1_config import Phase1Config
from src.core.strength_manager import StrengthManager
from src.models.memory import AgentMemory


# =============================================================================
# テストヘルパー
# =============================================================================


def create_test_memory(
    memory_id: Optional[UUID] = None,
    agent_id: str = "test_agent",
    content: str = "テスト用の記憶",
    strength: float = 1.0,
    access_count: int = 0,
    candidate_count: int = 0,
    impact_score: float = 0.0,
    consolidation_level: int = 0,
    status: str = "active",
    strength_by_perspective: Optional[dict] = None,
) -> AgentMemory:
    """テスト用の AgentMemory を作成"""
    now = datetime.now()
    return AgentMemory(
        id=memory_id or uuid4(),
        agent_id=agent_id,
        content=content,
        strength=strength,
        strength_by_perspective=strength_by_perspective or {},
        access_count=access_count,
        candidate_count=candidate_count,
        impact_score=impact_score,
        consolidation_level=consolidation_level,
        status=status,
        created_at=now,
        updated_at=now,
        last_accessed_at=now,
    )


# =============================================================================
# mark_as_candidate（Stage 1: 候補ブースト）のテスト
# =============================================================================


class TestMarkAsCandidate:
    """mark_as_candidate メソッドのテスト

    2段階強化の第1段階: 検索候補になったメモリの
    candidate_count をインクリメント（strength は変更しない）
    """

    @pytest.fixture
    def config(self) -> Phase1Config:
        """テスト用の設定"""
        return Phase1Config()

    @pytest.fixture
    def mock_repository(self) -> MagicMock:
        """モック MemoryRepository"""
        mock = MagicMock()
        mock.batch_increment_candidate_count.return_value = 0
        return mock

    @pytest.fixture
    def manager(
        self, mock_repository: MagicMock, config: Phase1Config
    ) -> StrengthManager:
        """テスト用の StrengthManager"""
        return StrengthManager(mock_repository, config)

    def test_mark_as_candidate_single_memory(
        self, manager: StrengthManager, mock_repository: MagicMock
    ):
        """単一メモリの候補マーク"""
        # Arrange
        memory_id = uuid4()
        mock_repository.batch_increment_candidate_count.return_value = 1

        # Act
        result = manager.mark_as_candidate([memory_id])

        # Assert
        assert result == 1
        mock_repository.batch_increment_candidate_count.assert_called_once_with(
            [memory_id]
        )

    def test_mark_as_candidate_multiple_memories(
        self, manager: StrengthManager, mock_repository: MagicMock
    ):
        """複数メモリの候補マーク"""
        # Arrange
        memory_ids = [uuid4() for _ in range(5)]
        mock_repository.batch_increment_candidate_count.return_value = 5

        # Act
        result = manager.mark_as_candidate(memory_ids)

        # Assert
        assert result == 5
        mock_repository.batch_increment_candidate_count.assert_called_once_with(
            memory_ids
        )

    def test_mark_as_candidate_empty_list(
        self, manager: StrengthManager, mock_repository: MagicMock
    ):
        """空リストの場合は repository を呼ばない"""
        # Act
        result = manager.mark_as_candidate([])

        # Assert
        assert result == 0
        mock_repository.batch_increment_candidate_count.assert_not_called()

    def test_mark_as_candidate_returns_updated_count(
        self, manager: StrengthManager, mock_repository: MagicMock
    ):
        """repository からの戻り値をそのまま返す"""
        # Arrange: 3件のIDを渡すが、2件のみ更新成功（1件は存在しない等）
        memory_ids = [uuid4() for _ in range(3)]
        mock_repository.batch_increment_candidate_count.return_value = 2

        # Act
        result = manager.mark_as_candidate(memory_ids)

        # Assert
        assert result == 2


# =============================================================================
# mark_as_used（Stage 2: 使用ブースト）のテスト
# =============================================================================


class TestMarkAsUsed:
    """mark_as_used メソッドのテスト

    2段階強化の第2段階: 実際に使用されたメモリの
    access_count と strength を強化する。
    """

    @pytest.fixture
    def config(self) -> Phase1Config:
        """テスト用の設定"""
        return Phase1Config(
            strength_increment_on_use=0.1,
            perspective_strength_increment=0.15,
        )

    @pytest.fixture
    def sample_memory(self) -> AgentMemory:
        """テスト用のメモリ"""
        return create_test_memory(
            strength=1.0,
            access_count=0,
            consolidation_level=0,
        )

    @pytest.fixture
    def mock_repository(self, sample_memory: AgentMemory) -> MagicMock:
        """モック MemoryRepository"""
        mock = MagicMock()
        mock.get_by_id.return_value = sample_memory
        mock.update.side_effect = lambda m: m
        return mock

    @pytest.fixture
    def manager(
        self, mock_repository: MagicMock, config: Phase1Config
    ) -> StrengthManager:
        """テスト用の StrengthManager"""
        return StrengthManager(mock_repository, config)

    def test_mark_as_used_calls_increment_access_count(
        self, manager: StrengthManager, mock_repository: MagicMock, config: Phase1Config
    ):
        """increment_access_count が正しく呼ばれる"""
        # Arrange
        memory_id = uuid4()

        # Act
        manager.mark_as_used(memory_id)

        # Assert
        mock_repository.increment_access_count.assert_called_once_with(
            memory_id, config.strength_increment_on_use
        )

    def test_mark_as_used_with_perspective_updates_perspective_strength(
        self, manager: StrengthManager, mock_repository: MagicMock, config: Phase1Config
    ):
        """perspective 指定時に観点別強度も更新"""
        # Arrange
        memory_id = uuid4()

        # Act
        manager.mark_as_used(memory_id, perspective="コスト")

        # Assert
        mock_repository.update_perspective_strength.assert_called_once_with(
            memory_id, "コスト", config.perspective_strength_increment
        )

    def test_mark_as_used_without_perspective_skips_perspective_update(
        self, manager: StrengthManager, mock_repository: MagicMock
    ):
        """perspective なしの場合は観点別強度更新をスキップ"""
        # Arrange
        memory_id = uuid4()

        # Act
        manager.mark_as_used(memory_id)

        # Assert
        mock_repository.update_perspective_strength.assert_not_called()

    def test_mark_as_used_calls_update_consolidation_level(
        self, manager: StrengthManager, mock_repository: MagicMock
    ):
        """定着レベル更新が呼ばれる"""
        # Arrange
        memory_id = uuid4()

        # Act
        result = manager.mark_as_used(memory_id)

        # Assert: get_by_id が呼ばれる（update_consolidation_level 内で）
        mock_repository.get_by_id.assert_called_once_with(memory_id)

    def test_mark_as_used_returns_updated_memory(
        self, manager: StrengthManager, sample_memory: AgentMemory
    ):
        """更新後の AgentMemory を返す"""
        # Act
        result = manager.mark_as_used(sample_memory.id)

        # Assert
        assert result is not None
        assert isinstance(result, AgentMemory)

    def test_mark_as_used_returns_none_for_nonexistent_memory(
        self, manager: StrengthManager, mock_repository: MagicMock
    ):
        """存在しないメモリの場合は None を返す"""
        # Arrange
        mock_repository.get_by_id.return_value = None
        nonexistent_id = uuid4()

        # Act
        result = manager.mark_as_used(nonexistent_id)

        # Assert
        assert result is None


class TestMarkAsUsedStrengthValues:
    """mark_as_used の強度更新値のテスト"""

    @pytest.fixture
    def config(self) -> Phase1Config:
        """カスタム設定"""
        return Phase1Config(
            strength_increment_on_use=0.2,  # デフォルト 0.1 から変更
            perspective_strength_increment=0.3,  # デフォルト 0.15 から変更
        )

    @pytest.fixture
    def mock_repository(self) -> MagicMock:
        """モック MemoryRepository"""
        mock = MagicMock()
        mock.get_by_id.return_value = create_test_memory()
        mock.update.side_effect = lambda m: m
        return mock

    @pytest.fixture
    def manager(
        self, mock_repository: MagicMock, config: Phase1Config
    ) -> StrengthManager:
        return StrengthManager(mock_repository, config)

    def test_mark_as_used_uses_config_strength_increment(
        self, manager: StrengthManager, mock_repository: MagicMock, config: Phase1Config
    ):
        """設定値の strength_increment_on_use を使用"""
        # Arrange
        memory_id = uuid4()

        # Act
        manager.mark_as_used(memory_id)

        # Assert
        mock_repository.increment_access_count.assert_called_once_with(
            memory_id, 0.2  # カスタム設定値
        )

    def test_mark_as_used_uses_config_perspective_increment(
        self, manager: StrengthManager, mock_repository: MagicMock, config: Phase1Config
    ):
        """設定値の perspective_strength_increment を使用"""
        # Arrange
        memory_id = uuid4()

        # Act
        manager.mark_as_used(memory_id, perspective="納期")

        # Assert
        mock_repository.update_perspective_strength.assert_called_once_with(
            memory_id, "納期", 0.3  # カスタム設定値
        )


# =============================================================================
# apply_impact（インパクト反映）のテスト
# =============================================================================


class TestApplyImpact:
    """apply_impact メソッドのテスト

    インパクトスコアを加算し、強度に反映する。
    - user_positive: +2.0
    - task_success: +1.5
    - prevented_error: +2.0
    """

    @pytest.fixture
    def config(self) -> Phase1Config:
        """テスト用の設定"""
        return Phase1Config(
            impact_user_positive=2.0,
            impact_task_success=1.5,
            impact_prevented_error=2.0,
            impact_to_strength_ratio=0.2,
        )

    @pytest.fixture
    def sample_memory(self) -> AgentMemory:
        """テスト用のメモリ"""
        return create_test_memory(
            strength=1.0,
            impact_score=0.0,
        )

    @pytest.fixture
    def mock_repository(self, sample_memory: AgentMemory) -> MagicMock:
        """モック MemoryRepository"""
        mock = MagicMock()
        mock.get_by_id.return_value = sample_memory
        mock.update.side_effect = lambda m: m
        return mock

    @pytest.fixture
    def manager(
        self, mock_repository: MagicMock, config: Phase1Config
    ) -> StrengthManager:
        return StrengthManager(mock_repository, config)

    def test_apply_impact_user_positive(
        self, manager: StrengthManager, mock_repository: MagicMock
    ):
        """user_positive インパクトの適用"""
        # Arrange
        memory_id = uuid4()
        memory = create_test_memory(memory_id=memory_id, strength=1.0, impact_score=0.0)
        mock_repository.get_by_id.return_value = memory

        # Act
        result = manager.apply_impact(memory_id, "user_positive")

        # Assert
        assert result is not None
        # impact_score: 0.0 + 2.0 = 2.0
        assert result.impact_score == 2.0
        # strength: 1.0 + (2.0 * 0.2) = 1.4
        assert result.strength == pytest.approx(1.4, rel=0.001)

    def test_apply_impact_task_success(
        self, manager: StrengthManager, mock_repository: MagicMock
    ):
        """task_success インパクトの適用"""
        # Arrange
        memory_id = uuid4()
        memory = create_test_memory(memory_id=memory_id, strength=0.5, impact_score=1.0)
        mock_repository.get_by_id.return_value = memory

        # Act
        result = manager.apply_impact(memory_id, "task_success")

        # Assert
        assert result is not None
        # impact_score: 1.0 + 1.5 = 2.5
        assert result.impact_score == 2.5
        # strength: 0.5 + (1.5 * 0.2) = 0.8
        assert result.strength == pytest.approx(0.8, rel=0.001)

    def test_apply_impact_prevented_error(
        self, manager: StrengthManager, mock_repository: MagicMock
    ):
        """prevented_error インパクトの適用"""
        # Arrange
        memory_id = uuid4()
        memory = create_test_memory(memory_id=memory_id, strength=0.8, impact_score=0.0)
        mock_repository.get_by_id.return_value = memory

        # Act
        result = manager.apply_impact(memory_id, "prevented_error")

        # Assert
        assert result is not None
        # impact_score: 0.0 + 2.0 = 2.0
        assert result.impact_score == 2.0
        # strength: 0.8 + (2.0 * 0.2) = 1.2
        assert result.strength == pytest.approx(1.2, rel=0.001)

    def test_apply_impact_invalid_type_raises_error(
        self, manager: StrengthManager
    ):
        """無効な impact_type で ValueError が発生"""
        # Arrange
        memory_id = uuid4()

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            manager.apply_impact(memory_id, "invalid_type")

        assert "Invalid impact_type" in str(exc_info.value)
        assert "invalid_type" in str(exc_info.value)

    def test_apply_impact_returns_none_for_nonexistent_memory(
        self, manager: StrengthManager, mock_repository: MagicMock
    ):
        """存在しないメモリの場合は None を返す"""
        # Arrange
        mock_repository.get_by_id.return_value = None
        nonexistent_id = uuid4()

        # Act
        result = manager.apply_impact(nonexistent_id, "user_positive")

        # Assert
        assert result is None

    def test_apply_impact_calls_repository_update(
        self, manager: StrengthManager, mock_repository: MagicMock
    ):
        """repository.update が呼ばれる"""
        # Arrange
        memory_id = uuid4()
        memory = create_test_memory(memory_id=memory_id)
        mock_repository.get_by_id.return_value = memory

        # Act
        manager.apply_impact(memory_id, "task_success")

        # Assert
        mock_repository.update.assert_called_once()
        updated_memory = mock_repository.update.call_args[0][0]
        assert isinstance(updated_memory, AgentMemory)

    def test_apply_impact_preserves_other_fields(
        self, manager: StrengthManager, mock_repository: MagicMock
    ):
        """他のフィールドは変更されない"""
        # Arrange
        memory_id = uuid4()
        memory = create_test_memory(
            memory_id=memory_id,
            content="保持されるべきコンテンツ",
            access_count=5,
            candidate_count=10,
            consolidation_level=2,
        )
        mock_repository.get_by_id.return_value = memory

        # Act
        result = manager.apply_impact(memory_id, "user_positive")

        # Assert
        assert result is not None
        assert result.content == "保持されるべきコンテンツ"
        assert result.access_count == 5
        assert result.candidate_count == 10
        assert result.consolidation_level == 2


class TestApplyImpactMultipleTimes:
    """apply_impact の複数回適用テスト"""

    @pytest.fixture
    def config(self) -> Phase1Config:
        return Phase1Config()

    @pytest.fixture
    def mock_repository(self) -> MagicMock:
        mock = MagicMock()
        mock.update.side_effect = lambda m: m
        return mock

    @pytest.fixture
    def manager(
        self, mock_repository: MagicMock, config: Phase1Config
    ) -> StrengthManager:
        return StrengthManager(mock_repository, config)

    def test_apply_impact_accumulates(
        self, manager: StrengthManager, mock_repository: MagicMock
    ):
        """インパクトは累積する"""
        # Arrange
        memory_id = uuid4()
        initial_memory = create_test_memory(
            memory_id=memory_id, strength=1.0, impact_score=0.0
        )
        mock_repository.get_by_id.return_value = initial_memory

        # Act: 1回目の適用
        result1 = manager.apply_impact(memory_id, "user_positive")
        # impact_score: 0 + 2.0 = 2.0
        # strength: 1.0 + (2.0 * 0.2) = 1.4

        # 2回目の適用のために更新されたメモリを設定
        mock_repository.get_by_id.return_value = result1

        # Act: 2回目の適用
        result2 = manager.apply_impact(memory_id, "task_success")
        # impact_score: 2.0 + 1.5 = 3.5
        # strength: 1.4 + (1.5 * 0.2) = 1.7

        # Assert
        assert result2 is not None
        assert result2.impact_score == pytest.approx(3.5, rel=0.001)
        assert result2.strength == pytest.approx(1.7, rel=0.001)


# =============================================================================
# update_consolidation_level（定着レベル更新）のテスト
# =============================================================================


class TestUpdateConsolidationLevel:
    """update_consolidation_level メソッドのテスト

    access_count に基づいて定着レベルを更新する。
    閾値: [0, 5, 15, 30, 60, 100]
    """

    @pytest.fixture
    def config(self) -> Phase1Config:
        """テスト用の設定"""
        return Phase1Config(
            consolidation_thresholds=[0, 5, 15, 30, 60, 100]
        )

    @pytest.fixture
    def mock_repository(self) -> MagicMock:
        """モック MemoryRepository"""
        mock = MagicMock()
        mock.update.side_effect = lambda m: m
        return mock

    @pytest.fixture
    def manager(
        self, mock_repository: MagicMock, config: Phase1Config
    ) -> StrengthManager:
        return StrengthManager(mock_repository, config)

    @pytest.mark.parametrize(
        "access_count,expected_level",
        [
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
        ],
    )
    def test_update_consolidation_level_boundary_values(
        self,
        manager: StrengthManager,
        mock_repository: MagicMock,
        access_count: int,
        expected_level: int,
    ):
        """定着レベルの境界値テスト"""
        # Arrange
        memory_id = uuid4()
        memory = create_test_memory(
            memory_id=memory_id,
            access_count=access_count,
            consolidation_level=0,  # 初期レベル0
        )
        mock_repository.get_by_id.return_value = memory

        # Act
        result = manager.update_consolidation_level(memory_id)

        # Assert
        if expected_level != 0:
            # レベルが変わる場合は update が呼ばれる
            assert result is not None
            assert result.consolidation_level == expected_level
        else:
            # レベルが変わらない場合は元のメモリがそのまま返る
            assert result is not None

    def test_update_consolidation_level_no_change(
        self, manager: StrengthManager, mock_repository: MagicMock
    ):
        """レベルが変わらない場合は update を呼ばない"""
        # Arrange
        memory_id = uuid4()
        memory = create_test_memory(
            memory_id=memory_id,
            access_count=3,
            consolidation_level=0,  # 既に正しいレベル
        )
        mock_repository.get_by_id.return_value = memory

        # Act
        result = manager.update_consolidation_level(memory_id)

        # Assert
        assert result is not None
        mock_repository.update.assert_not_called()

    def test_update_consolidation_level_level_increases(
        self, manager: StrengthManager, mock_repository: MagicMock
    ):
        """レベルが上がる場合は update が呼ばれる"""
        # Arrange
        memory_id = uuid4()
        memory = create_test_memory(
            memory_id=memory_id,
            access_count=5,  # Level 1 の閾値
            consolidation_level=0,  # まだ Level 0
        )
        mock_repository.get_by_id.return_value = memory

        # Act
        result = manager.update_consolidation_level(memory_id)

        # Assert
        assert result is not None
        assert result.consolidation_level == 1
        mock_repository.update.assert_called_once()

    def test_update_consolidation_level_returns_none_for_nonexistent(
        self, manager: StrengthManager, mock_repository: MagicMock
    ):
        """存在しないメモリの場合は None を返す"""
        # Arrange
        mock_repository.get_by_id.return_value = None
        nonexistent_id = uuid4()

        # Act
        result = manager.update_consolidation_level(nonexistent_id)

        # Assert
        assert result is None


# =============================================================================
# reactivate（再活性化）のテスト
# =============================================================================


class TestReactivate:
    """reactivate メソッドのテスト

    アーカイブされたメモリをアクティブ状態に戻す。
    - status: 'archived' -> 'active'
    - strength: reactivation_strength (0.5) に設定
    """

    @pytest.fixture
    def config(self) -> Phase1Config:
        """テスト用の設定"""
        return Phase1Config(reactivation_strength=0.5)

    @pytest.fixture
    def mock_repository(self) -> MagicMock:
        """モック MemoryRepository"""
        mock = MagicMock()
        mock.update.side_effect = lambda m: m
        return mock

    @pytest.fixture
    def manager(
        self, mock_repository: MagicMock, config: Phase1Config
    ) -> StrengthManager:
        return StrengthManager(mock_repository, config)

    def test_reactivate_archived_memory(
        self, manager: StrengthManager, mock_repository: MagicMock, config: Phase1Config
    ):
        """アーカイブされたメモリを再活性化"""
        # Arrange
        memory_id = uuid4()
        archived_memory = create_test_memory(
            memory_id=memory_id,
            status="archived",
            strength=0.05,  # アーカイブ時の弱い強度
        )
        mock_repository.get_by_id.return_value = archived_memory

        # Act
        result = manager.reactivate(memory_id)

        # Assert
        assert result is not None
        assert result.status == "active"
        assert result.strength == config.reactivation_strength  # 0.5

    def test_reactivate_active_memory_raises_error(
        self, manager: StrengthManager, mock_repository: MagicMock
    ):
        """既にアクティブなメモリを再活性化しようとするとエラー"""
        # Arrange
        memory_id = uuid4()
        active_memory = create_test_memory(
            memory_id=memory_id,
            status="active",
        )
        mock_repository.get_by_id.return_value = active_memory

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            manager.reactivate(memory_id)

        assert "already active" in str(exc_info.value)

    def test_reactivate_returns_none_for_nonexistent(
        self, manager: StrengthManager, mock_repository: MagicMock
    ):
        """存在しないメモリの場合は None を返す"""
        # Arrange
        mock_repository.get_by_id.return_value = None
        nonexistent_id = uuid4()

        # Act
        result = manager.reactivate(nonexistent_id)

        # Assert
        assert result is None

    def test_reactivate_calls_repository_update(
        self, manager: StrengthManager, mock_repository: MagicMock
    ):
        """repository.update が呼ばれる"""
        # Arrange
        memory_id = uuid4()
        archived_memory = create_test_memory(
            memory_id=memory_id,
            status="archived",
        )
        mock_repository.get_by_id.return_value = archived_memory

        # Act
        manager.reactivate(memory_id)

        # Assert
        mock_repository.update.assert_called_once()

    def test_reactivate_preserves_other_fields(
        self, manager: StrengthManager, mock_repository: MagicMock
    ):
        """他のフィールドは変更されない"""
        # Arrange
        memory_id = uuid4()
        archived_memory = create_test_memory(
            memory_id=memory_id,
            status="archived",
            content="保持されるコンテンツ",
            access_count=50,
            impact_score=5.0,
            consolidation_level=3,
        )
        mock_repository.get_by_id.return_value = archived_memory

        # Act
        result = manager.reactivate(memory_id)

        # Assert
        assert result is not None
        assert result.content == "保持されるコンテンツ"
        assert result.access_count == 50
        assert result.impact_score == 5.0
        assert result.consolidation_level == 3

    def test_reactivate_uses_config_reactivation_strength(
        self, mock_repository: MagicMock
    ):
        """設定値の reactivation_strength を使用"""
        # Arrange
        custom_config = Phase1Config(reactivation_strength=0.7)
        manager = StrengthManager(mock_repository, custom_config)
        memory_id = uuid4()
        archived_memory = create_test_memory(
            memory_id=memory_id,
            status="archived",
        )
        mock_repository.get_by_id.return_value = archived_memory

        # Act
        result = manager.reactivate(memory_id)

        # Assert
        assert result is not None
        assert result.strength == 0.7


# =============================================================================
# get_impact_value（インパクト値取得）のテスト
# =============================================================================


class TestGetImpactValue:
    """get_impact_value メソッドのテスト"""

    @pytest.fixture
    def config(self) -> Phase1Config:
        """テスト用の設定"""
        return Phase1Config(
            impact_user_positive=2.0,
            impact_task_success=1.5,
            impact_prevented_error=2.0,
        )

    @pytest.fixture
    def manager(self, config: Phase1Config) -> StrengthManager:
        """テスト用の StrengthManager"""
        mock_repository = MagicMock()
        return StrengthManager(mock_repository, config)

    def test_get_impact_value_user_positive(
        self, manager: StrengthManager
    ):
        """user_positive のインパクト値"""
        assert manager.get_impact_value("user_positive") == 2.0

    def test_get_impact_value_task_success(
        self, manager: StrengthManager
    ):
        """task_success のインパクト値"""
        assert manager.get_impact_value("task_success") == 1.5

    def test_get_impact_value_prevented_error(
        self, manager: StrengthManager
    ):
        """prevented_error のインパクト値"""
        assert manager.get_impact_value("prevented_error") == 2.0

    def test_get_impact_value_invalid_type_raises_error(
        self, manager: StrengthManager
    ):
        """無効な impact_type で ValueError が発生"""
        with pytest.raises(ValueError) as exc_info:
            manager.get_impact_value("invalid")

        assert "Invalid impact_type" in str(exc_info.value)


# =============================================================================
# 設定パラメータ参照のテスト
# =============================================================================


class TestConfigParameterReference:
    """設定パラメータが正しく参照されているかのテスト"""

    def test_manager_uses_config_strength_increment_on_use(self):
        """strength_increment_on_use が Phase1Config から参照される"""
        # Arrange
        config = Phase1Config(strength_increment_on_use=0.25)
        mock_repository = MagicMock()
        mock_repository.get_by_id.return_value = create_test_memory()
        mock_repository.update.side_effect = lambda m: m
        manager = StrengthManager(mock_repository, config)

        # Act
        manager.mark_as_used(uuid4())

        # Assert
        mock_repository.increment_access_count.assert_called_once()
        call_args = mock_repository.increment_access_count.call_args
        assert call_args[0][1] == 0.25

    def test_manager_uses_config_perspective_strength_increment(self):
        """perspective_strength_increment が Phase1Config から参照される"""
        # Arrange
        config = Phase1Config(perspective_strength_increment=0.35)
        mock_repository = MagicMock()
        mock_repository.get_by_id.return_value = create_test_memory()
        mock_repository.update.side_effect = lambda m: m
        manager = StrengthManager(mock_repository, config)

        # Act
        manager.mark_as_used(uuid4(), perspective="テスト")

        # Assert
        mock_repository.update_perspective_strength.assert_called_once()
        call_args = mock_repository.update_perspective_strength.call_args
        assert call_args[0][2] == 0.35

    def test_manager_uses_config_reactivation_strength(self):
        """reactivation_strength が Phase1Config から参照される"""
        # Arrange
        config = Phase1Config(reactivation_strength=0.6)
        mock_repository = MagicMock()
        mock_repository.get_by_id.return_value = create_test_memory(status="archived")
        mock_repository.update.side_effect = lambda m: m
        manager = StrengthManager(mock_repository, config)

        # Act
        result = manager.reactivate(uuid4())

        # Assert
        assert result is not None
        assert result.strength == 0.6

    def test_manager_uses_config_impact_values(self):
        """インパクト値が Phase1Config から参照される"""
        # Arrange
        config = Phase1Config(
            impact_user_positive=3.0,
            impact_task_success=2.5,
            impact_prevented_error=3.5,
        )
        mock_repository = MagicMock()
        manager = StrengthManager(mock_repository, config)

        # Assert
        assert manager.get_impact_value("user_positive") == 3.0
        assert manager.get_impact_value("task_success") == 2.5
        assert manager.get_impact_value("prevented_error") == 3.5

    def test_manager_uses_config_impact_to_strength_ratio(self):
        """impact_to_strength_ratio が Phase1Config から参照される"""
        # Arrange
        config = Phase1Config(
            impact_user_positive=2.0,
            impact_to_strength_ratio=0.5,  # デフォルト0.2から変更
        )
        mock_repository = MagicMock()
        mock_repository.get_by_id.return_value = create_test_memory(
            strength=1.0, impact_score=0.0
        )
        mock_repository.update.side_effect = lambda m: m
        manager = StrengthManager(mock_repository, config)

        # Act
        result = manager.apply_impact(uuid4(), "user_positive")

        # Assert
        # strength: 1.0 + (2.0 * 0.5) = 2.0
        assert result is not None
        assert result.strength == pytest.approx(2.0, rel=0.001)


# =============================================================================
# 統合テスト（2段階強化の完全フロー）
# =============================================================================


class TestTwoStageReinforcementIntegration:
    """2段階強化の統合テスト"""

    @pytest.fixture
    def config(self) -> Phase1Config:
        return Phase1Config()

    @pytest.fixture
    def mock_repository(self) -> MagicMock:
        mock = MagicMock()
        mock.update.side_effect = lambda m: m
        return mock

    @pytest.fixture
    def manager(
        self, mock_repository: MagicMock, config: Phase1Config
    ) -> StrengthManager:
        return StrengthManager(mock_repository, config)

    def test_full_two_stage_flow(
        self, manager: StrengthManager, mock_repository: MagicMock, config: Phase1Config
    ):
        """Stage 1（候補）→ Stage 2（使用）の完全フロー"""
        # Arrange
        memory_ids = [uuid4() for _ in range(3)]
        memories = [create_test_memory(memory_id=mid) for mid in memory_ids]

        mock_repository.batch_increment_candidate_count.return_value = 3
        mock_repository.get_by_id.side_effect = lambda mid: next(
            (m for m in memories if m.id == mid), None
        )

        # Act: Stage 1 - 候補としてマーク
        candidate_count = manager.mark_as_candidate(memory_ids)

        # Assert: Stage 1
        assert candidate_count == 3
        mock_repository.batch_increment_candidate_count.assert_called_once_with(
            memory_ids
        )

        # Act: Stage 2 - 実際に使用（最初の1件のみ）
        used_memory = manager.mark_as_used(memory_ids[0], perspective="コスト")

        # Assert: Stage 2
        assert used_memory is not None
        mock_repository.increment_access_count.assert_called_once_with(
            memory_ids[0], config.strength_increment_on_use
        )
        mock_repository.update_perspective_strength.assert_called_once_with(
            memory_ids[0], "コスト", config.perspective_strength_increment
        )

    def test_stage1_only_does_not_change_strength(
        self, manager: StrengthManager, mock_repository: MagicMock
    ):
        """Stage 1 のみでは strength は変更されない"""
        # Arrange
        memory_ids = [uuid4() for _ in range(5)]
        mock_repository.batch_increment_candidate_count.return_value = 5

        # Act
        manager.mark_as_candidate(memory_ids)

        # Assert: increment_access_count は呼ばれない
        mock_repository.increment_access_count.assert_not_called()
        # update_perspective_strength も呼ばれない
        mock_repository.update_perspective_strength.assert_not_called()
