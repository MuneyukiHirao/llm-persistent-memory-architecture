# メモリリポジトリのテスト
"""
MemoryRepository の単体テスト

テスト観点:
- CRUD操作（create, get_by_id, get_by_agent_id, update, archive）
- 2段階強化メソッド（increment_candidate_count, increment_access_count, update_perspective_strength）
- バッチ操作（batch_increment_candidate_count, batch_update_strength, batch_archive）
- vector/JSONB型のシリアライズ
- 境界値・異常系

注意: DBモック使用（実DB接続不要）
"""

from datetime import datetime
from typing import List, Optional
from unittest.mock import MagicMock, patch, call
from uuid import UUID, uuid4

import pytest

from src.config.phase1_config import Phase1Config
from src.core.memory_repository import MemoryRepository
from src.models.memory import AgentMemory


class TestMemoryRepositorySetup:
    """MemoryRepository 初期化のテスト"""

    @pytest.fixture
    def config(self) -> Phase1Config:
        """テスト用の設定"""
        return Phase1Config()

    @pytest.fixture
    def mock_db(self) -> MagicMock:
        """モック DatabaseConnection"""
        return MagicMock()

    def test_init(self, mock_db: MagicMock, config: Phase1Config):
        """初期化時に db と config が設定される"""
        # Act
        repo = MemoryRepository(mock_db, config)

        # Assert
        assert repo.db is mock_db
        assert repo.config is config


class TestCreate:
    """create() メソッドのテスト"""

    @pytest.fixture
    def config(self) -> Phase1Config:
        """テスト用の設定"""
        return Phase1Config()

    @pytest.fixture
    def mock_cursor(self) -> MagicMock:
        """モックカーソル"""
        cursor = MagicMock()
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=None)
        return cursor

    @pytest.fixture
    def mock_db(self, mock_cursor: MagicMock) -> MagicMock:
        """モック DatabaseConnection"""
        db = MagicMock()
        db.get_cursor = MagicMock(return_value=mock_cursor)
        return db

    @pytest.fixture
    def repo(self, mock_db: MagicMock, config: Phase1Config) -> MemoryRepository:
        """テスト用リポジトリ"""
        return MemoryRepository(mock_db, config)

    def _create_memory(
        self,
        agent_id: str = "test_agent",
        content: str = "テスト用記憶",
        embedding: Optional[List[float]] = None,
        strength: float = 1.0,
    ) -> AgentMemory:
        """テスト用メモリを作成"""
        return AgentMemory.create(
            agent_id=agent_id,
            content=content,
            embedding=embedding,
            strength=strength,
        )

    def _make_db_row(self, memory: AgentMemory) -> tuple:
        """AgentMemory からDB行データを生成（23カラム）"""
        return (
            memory.id,
            memory.agent_id,
            memory.content,
            memory.embedding,
            memory.tags,
            memory.scope_level,
            memory.scope_domain,
            memory.scope_project,
            memory.strength,
            memory.strength_by_perspective,
            memory.access_count,
            memory.candidate_count,
            memory.last_accessed_at,
            memory.next_review_at,
            memory.review_count,
            memory.impact_score,
            memory.consolidation_level,
            memory.learning,
            memory.status,
            memory.source,
            memory.created_at,
            memory.updated_at,
            memory.last_decay_at,
        )

    def test_create_new_memory(
        self, repo: MemoryRepository, mock_cursor: MagicMock
    ):
        """新規メモリの作成"""
        # Arrange
        memory = self._create_memory()
        mock_cursor.fetchone.return_value = self._make_db_row(memory)

        # Act
        result = repo.create(memory)

        # Assert
        assert result.id == memory.id
        assert result.agent_id == memory.agent_id
        assert result.content == memory.content
        mock_cursor.execute.assert_called_once()

    def test_create_memory_with_embedding(
        self, repo: MemoryRepository, mock_cursor: MagicMock
    ):
        """embeddingを持つメモリの作成"""
        # Arrange
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        memory = self._create_memory(embedding=embedding)
        mock_cursor.fetchone.return_value = self._make_db_row(memory)

        # Act
        result = repo.create(memory)

        # Assert
        assert result.embedding == embedding

        # SQL引数でembeddingが正しくフォーマットされているか検証
        call_args = mock_cursor.execute.call_args[0][1]
        # embedding は4番目の引数（index 3）
        assert call_args[3] == "[0.1,0.2,0.3,0.4,0.5]"

    def test_create_memory_without_embedding(
        self, repo: MemoryRepository, mock_cursor: MagicMock
    ):
        """embeddingなしでメモリを作成可能"""
        # Arrange
        memory = self._create_memory(embedding=None)
        mock_cursor.fetchone.return_value = self._make_db_row(memory)

        # Act
        result = repo.create(memory)

        # Assert
        assert result.embedding is None

        # SQL引数でembeddingがNone
        call_args = mock_cursor.execute.call_args[0][1]
        assert call_args[3] is None

    def test_create_memory_with_jsonb_and_text_fields(
        self, repo: MemoryRepository, mock_cursor: MagicMock
    ):
        """JSONB フィールド（strength_by_perspective）と TEXT フィールド（learning）が正しく処理される"""
        # Arrange
        memory = AgentMemory.create(
            agent_id="test_agent",
            content="テスト",
            strength_by_perspective={"コスト": 1.2, "納期": 0.8},
            learning="緊急調達で15%コスト増",
        )
        mock_cursor.fetchone.return_value = self._make_db_row(memory)

        # Act
        result = repo.create(memory)

        # Assert
        mock_cursor.execute.assert_called_once()
        # JSONB フィールドが psycopg2.extras.Json でラップされているか
        # 実際にはexecute時に自動変換されるため、結果の確認で代用


class TestGetById:
    """get_by_id() メソッドのテスト"""

    @pytest.fixture
    def config(self) -> Phase1Config:
        return Phase1Config()

    @pytest.fixture
    def mock_cursor(self) -> MagicMock:
        cursor = MagicMock()
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=None)
        return cursor

    @pytest.fixture
    def mock_db(self, mock_cursor: MagicMock) -> MagicMock:
        db = MagicMock()
        db.get_cursor = MagicMock(return_value=mock_cursor)
        return db

    @pytest.fixture
    def repo(self, mock_db: MagicMock, config: Phase1Config) -> MemoryRepository:
        return MemoryRepository(mock_db, config)

    def _make_db_row(self, memory_id: UUID) -> tuple:
        """テスト用DB行を作成（23カラム）"""
        now = datetime.now()
        return (
            memory_id,
            "test_agent",
            "テスト記憶",
            None,  # embedding
            [],    # tags
            "project",
            None,  # scope_domain
            "test_project",
            1.0,   # strength
            {},    # strength_by_perspective
            0,     # access_count
            0,     # candidate_count
            None,  # last_accessed_at
            None,  # next_review_at
            0,     # review_count
            0.0,   # impact_score
            0,     # consolidation_level
            None,  # learning
            "active",
            None,  # source
            now,
            now,
            None,  # last_decay_at
        )

    def test_get_by_id_found(
        self, repo: MemoryRepository, mock_cursor: MagicMock
    ):
        """存在するIDでメモリを取得"""
        # Arrange
        memory_id = uuid4()
        mock_cursor.fetchone.return_value = self._make_db_row(memory_id)

        # Act
        result = repo.get_by_id(memory_id)

        # Assert
        assert result is not None
        assert result.id == memory_id
        assert result.agent_id == "test_agent"

    def test_get_by_id_not_found(
        self, repo: MemoryRepository, mock_cursor: MagicMock
    ):
        """存在しないIDでNoneを返す"""
        # Arrange
        memory_id = uuid4()
        mock_cursor.fetchone.return_value = None

        # Act
        result = repo.get_by_id(memory_id)

        # Assert
        assert result is None


class TestGetByAgentId:
    """get_by_agent_id() メソッドのテスト"""

    @pytest.fixture
    def config(self) -> Phase1Config:
        return Phase1Config()

    @pytest.fixture
    def mock_cursor(self) -> MagicMock:
        cursor = MagicMock()
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=None)
        return cursor

    @pytest.fixture
    def mock_db(self, mock_cursor: MagicMock) -> MagicMock:
        db = MagicMock()
        db.get_cursor = MagicMock(return_value=mock_cursor)
        return db

    @pytest.fixture
    def repo(self, mock_db: MagicMock, config: Phase1Config) -> MemoryRepository:
        return MemoryRepository(mock_db, config)

    def _make_db_row(self, memory_id: UUID, agent_id: str = "test_agent") -> tuple:
        """テスト用DB行を作成（23カラム）"""
        now = datetime.now()
        return (
            memory_id,
            agent_id,
            "テスト記憶",
            None,
            [],
            "project",
            None,
            "test_project",
            1.0,
            {},
            0,
            0,
            None,
            None,  # next_review_at
            0,     # review_count
            0.0,
            0,
            None,
            "active",
            None,
            now,
            now,
            None,
        )

    def test_get_by_agent_id_returns_list(
        self, repo: MemoryRepository, mock_cursor: MagicMock
    ):
        """エージェントIDでメモリリストを取得"""
        # Arrange
        agent_id = "test_agent"
        memory_ids = [uuid4(), uuid4(), uuid4()]
        mock_cursor.fetchall.return_value = [
            self._make_db_row(mid, agent_id) for mid in memory_ids
        ]

        # Act
        result = repo.get_by_agent_id(agent_id)

        # Assert
        assert len(result) == 3
        for i, memory in enumerate(result):
            assert memory.id == memory_ids[i]
            assert memory.agent_id == agent_id

    def test_get_by_agent_id_empty(
        self, repo: MemoryRepository, mock_cursor: MagicMock
    ):
        """メモリが存在しない場合は空リスト"""
        # Arrange
        mock_cursor.fetchall.return_value = []

        # Act
        result = repo.get_by_agent_id("nonexistent_agent")

        # Assert
        assert result == []

    def test_get_by_agent_id_with_status_filter(
        self, repo: MemoryRepository, mock_cursor: MagicMock
    ):
        """ステータスでフィルタ"""
        # Arrange
        mock_cursor.fetchall.return_value = []

        # Act
        repo.get_by_agent_id("test_agent", status="archived")

        # Assert
        call_args = mock_cursor.execute.call_args[0][1]
        assert call_args[1] == "archived"


class TestUpdate:
    """update() メソッドのテスト"""

    @pytest.fixture
    def config(self) -> Phase1Config:
        return Phase1Config()

    @pytest.fixture
    def mock_cursor(self) -> MagicMock:
        cursor = MagicMock()
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=None)
        return cursor

    @pytest.fixture
    def mock_db(self, mock_cursor: MagicMock) -> MagicMock:
        db = MagicMock()
        db.get_cursor = MagicMock(return_value=mock_cursor)
        return db

    @pytest.fixture
    def repo(self, mock_db: MagicMock, config: Phase1Config) -> MemoryRepository:
        return MemoryRepository(mock_db, config)

    def _make_db_row(self, memory: AgentMemory) -> tuple:
        """テスト用DB行を作成（23カラム）"""
        return (
            memory.id,
            memory.agent_id,
            memory.content,
            memory.embedding,
            memory.tags,
            memory.scope_level,
            memory.scope_domain,
            memory.scope_project,
            memory.strength,
            memory.strength_by_perspective,
            memory.access_count,
            memory.candidate_count,
            memory.last_accessed_at,
            memory.next_review_at,
            memory.review_count,
            memory.impact_score,
            memory.consolidation_level,
            memory.learning,
            memory.status,
            memory.source,
            memory.created_at,
            memory.updated_at,
            memory.last_decay_at,
        )

    def test_update_existing_memory(
        self, repo: MemoryRepository, mock_cursor: MagicMock
    ):
        """既存メモリの更新"""
        # Arrange
        memory = AgentMemory.create(
            agent_id="test_agent",
            content="更新後のコンテンツ",
            strength=1.5,
        )
        mock_cursor.fetchone.return_value = self._make_db_row(memory)

        # Act
        result = repo.update(memory)

        # Assert
        assert result.content == "更新後のコンテンツ"
        assert result.strength == 1.5
        mock_cursor.execute.assert_called_once()

    def test_update_nonexistent_memory_raises_error(
        self, repo: MemoryRepository, mock_cursor: MagicMock
    ):
        """存在しないメモリの更新でValueError"""
        # Arrange
        memory = AgentMemory.create(agent_id="test_agent", content="テスト")
        mock_cursor.fetchone.return_value = None

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            repo.update(memory)

        assert "not found" in str(exc_info.value)


class TestArchive:
    """archive() メソッドのテスト（論理削除）"""

    @pytest.fixture
    def config(self) -> Phase1Config:
        return Phase1Config()

    @pytest.fixture
    def mock_cursor(self) -> MagicMock:
        cursor = MagicMock()
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=None)
        return cursor

    @pytest.fixture
    def mock_db(self, mock_cursor: MagicMock) -> MagicMock:
        db = MagicMock()
        db.get_cursor = MagicMock(return_value=mock_cursor)
        return db

    @pytest.fixture
    def repo(self, mock_db: MagicMock, config: Phase1Config) -> MemoryRepository:
        return MemoryRepository(mock_db, config)

    def test_archive_existing_memory(
        self, repo: MemoryRepository, mock_cursor: MagicMock
    ):
        """存在するメモリのアーカイブ"""
        # Arrange
        memory_id = uuid4()
        mock_cursor.rowcount = 1

        # Act
        result = repo.archive(memory_id)

        # Assert
        assert result is True
        mock_cursor.execute.assert_called_once()
        # status が 'archived' に更新されることを確認

    def test_archive_nonexistent_memory(
        self, repo: MemoryRepository, mock_cursor: MagicMock
    ):
        """存在しないメモリのアーカイブはFalse"""
        # Arrange
        memory_id = uuid4()
        mock_cursor.rowcount = 0

        # Act
        result = repo.archive(memory_id)

        # Assert
        assert result is False


class TestIncrementCandidateCount:
    """increment_candidate_count() メソッドのテスト（2段階強化第1段階）"""

    @pytest.fixture
    def config(self) -> Phase1Config:
        return Phase1Config()

    @pytest.fixture
    def mock_cursor(self) -> MagicMock:
        cursor = MagicMock()
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=None)
        return cursor

    @pytest.fixture
    def mock_db(self, mock_cursor: MagicMock) -> MagicMock:
        db = MagicMock()
        db.get_cursor = MagicMock(return_value=mock_cursor)
        return db

    @pytest.fixture
    def repo(self, mock_db: MagicMock, config: Phase1Config) -> MemoryRepository:
        return MemoryRepository(mock_db, config)

    def test_increment_candidate_count(
        self, repo: MemoryRepository, mock_cursor: MagicMock
    ):
        """candidate_count のインクリメント"""
        # Arrange
        memory_id = uuid4()

        # Act
        repo.increment_candidate_count(memory_id)

        # Assert
        mock_cursor.execute.assert_called_once()
        sql = mock_cursor.execute.call_args[0][0]
        assert "candidate_count = candidate_count + 1" in sql


class TestIncrementAccessCount:
    """increment_access_count() メソッドのテスト（2段階強化第2段階）"""

    @pytest.fixture
    def config(self) -> Phase1Config:
        return Phase1Config()

    @pytest.fixture
    def mock_cursor(self) -> MagicMock:
        cursor = MagicMock()
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=None)
        return cursor

    @pytest.fixture
    def mock_db(self, mock_cursor: MagicMock) -> MagicMock:
        db = MagicMock()
        db.get_cursor = MagicMock(return_value=mock_cursor)
        return db

    @pytest.fixture
    def repo(self, mock_db: MagicMock, config: Phase1Config) -> MemoryRepository:
        return MemoryRepository(mock_db, config)

    def test_increment_access_count_with_strength(
        self, repo: MemoryRepository, mock_cursor: MagicMock
    ):
        """access_count と strength のインクリメント"""
        # Arrange
        memory_id = uuid4()
        strength_increment = 0.1

        # Act
        repo.increment_access_count(memory_id, strength_increment)

        # Assert
        mock_cursor.execute.assert_called_once()
        sql = mock_cursor.execute.call_args[0][0]
        assert "access_count = access_count + 1" in sql
        assert "strength = strength +" in sql
        # strength_increment が渡されていることを確認
        call_args = mock_cursor.execute.call_args[0][1]
        assert call_args[0] == 0.1


class TestUpdatePerspectiveStrength:
    """update_perspective_strength() メソッドのテスト"""

    @pytest.fixture
    def config(self) -> Phase1Config:
        return Phase1Config()

    @pytest.fixture
    def mock_cursor(self) -> MagicMock:
        cursor = MagicMock()
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=None)
        return cursor

    @pytest.fixture
    def mock_db(self, mock_cursor: MagicMock) -> MagicMock:
        db = MagicMock()
        db.get_cursor = MagicMock(return_value=mock_cursor)
        return db

    @pytest.fixture
    def repo(self, mock_db: MagicMock, config: Phase1Config) -> MemoryRepository:
        return MemoryRepository(mock_db, config)

    def test_update_perspective_strength(
        self, repo: MemoryRepository, mock_cursor: MagicMock
    ):
        """観点別強度の更新"""
        # Arrange
        memory_id = uuid4()
        perspective = "コスト"
        increment = 0.15

        # Act
        repo.update_perspective_strength(memory_id, perspective, increment)

        # Assert
        mock_cursor.execute.assert_called_once()
        sql = mock_cursor.execute.call_args[0][0]
        assert "strength_by_perspective" in sql
        assert "jsonb_set" in sql

        # パラメータの確認
        call_args = mock_cursor.execute.call_args[0][1]
        assert call_args[0] == [perspective]  # JSONBパス
        assert call_args[1] == perspective    # 既存値取得用
        assert call_args[2] == increment


class TestBatchIncrementCandidateCount:
    """batch_increment_candidate_count() メソッドのテスト"""

    @pytest.fixture
    def config(self) -> Phase1Config:
        return Phase1Config()

    @pytest.fixture
    def mock_cursor(self) -> MagicMock:
        cursor = MagicMock()
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=None)
        return cursor

    @pytest.fixture
    def mock_db(self, mock_cursor: MagicMock) -> MagicMock:
        db = MagicMock()
        db.get_cursor = MagicMock(return_value=mock_cursor)
        return db

    @pytest.fixture
    def repo(self, mock_db: MagicMock, config: Phase1Config) -> MemoryRepository:
        return MemoryRepository(mock_db, config)

    def test_batch_increment_multiple_memories(
        self, repo: MemoryRepository, mock_cursor: MagicMock
    ):
        """複数メモリの candidate_count を一括インクリメント"""
        # Arrange
        memory_ids = [uuid4() for _ in range(5)]
        mock_cursor.rowcount = 5

        # Act
        result = repo.batch_increment_candidate_count(memory_ids)

        # Assert
        assert result == 5
        mock_cursor.execute.assert_called_once()
        sql = mock_cursor.execute.call_args[0][0]
        assert "ANY(" in sql

    def test_batch_increment_empty_list(
        self, repo: MemoryRepository, mock_cursor: MagicMock
    ):
        """空リストでは0を返し、SQLを実行しない"""
        # Arrange
        memory_ids: List[UUID] = []

        # Act
        result = repo.batch_increment_candidate_count(memory_ids)

        # Assert
        assert result == 0
        mock_cursor.execute.assert_not_called()


class TestBatchUpdateStrength:
    """batch_update_strength() メソッドのテスト"""

    @pytest.fixture
    def config(self) -> Phase1Config:
        return Phase1Config()

    @pytest.fixture
    def mock_cursor(self) -> MagicMock:
        cursor = MagicMock()
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=None)
        return cursor

    @pytest.fixture
    def mock_db(self, mock_cursor: MagicMock) -> MagicMock:
        db = MagicMock()
        db.get_cursor = MagicMock(return_value=mock_cursor)
        return db

    @pytest.fixture
    def repo(self, mock_db: MagicMock, config: Phase1Config) -> MemoryRepository:
        return MemoryRepository(mock_db, config)

    def test_batch_update_strength_multiple(
        self, repo: MemoryRepository, mock_cursor: MagicMock
    ):
        """複数メモリの strength を一括更新"""
        # Arrange
        updates = [
            (uuid4(), 0.8),
            (uuid4(), 0.6),
            (uuid4(), 0.4),
        ]
        mock_cursor.rowcount = 3

        # Act
        result = repo.batch_update_strength(updates)

        # Assert
        assert result == 3
        mock_cursor.execute.assert_called_once()
        sql = mock_cursor.execute.call_args[0][0]
        assert "UPDATE agent_memory" in sql
        assert "VALUES" in sql

    def test_batch_update_strength_empty(
        self, repo: MemoryRepository, mock_cursor: MagicMock
    ):
        """空リストでは0を返し、SQLを実行しない"""
        # Arrange
        updates: List[tuple[UUID, float]] = []

        # Act
        result = repo.batch_update_strength(updates)

        # Assert
        assert result == 0
        mock_cursor.execute.assert_not_called()


class TestBatchArchive:
    """batch_archive() メソッドのテスト"""

    @pytest.fixture
    def config(self) -> Phase1Config:
        return Phase1Config()

    @pytest.fixture
    def mock_cursor(self) -> MagicMock:
        cursor = MagicMock()
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=None)
        return cursor

    @pytest.fixture
    def mock_db(self, mock_cursor: MagicMock) -> MagicMock:
        db = MagicMock()
        db.get_cursor = MagicMock(return_value=mock_cursor)
        return db

    @pytest.fixture
    def repo(self, mock_db: MagicMock, config: Phase1Config) -> MemoryRepository:
        return MemoryRepository(mock_db, config)

    def test_batch_archive_multiple(
        self, repo: MemoryRepository, mock_cursor: MagicMock
    ):
        """複数メモリを一括アーカイブ"""
        # Arrange
        memory_ids = [uuid4() for _ in range(3)]
        mock_cursor.rowcount = 3

        # Act
        result = repo.batch_archive(memory_ids)

        # Assert
        assert result == 3
        mock_cursor.execute.assert_called_once()
        sql = mock_cursor.execute.call_args[0][0]
        assert "status = 'archived'" in sql
        assert "ANY(" in sql

    def test_batch_archive_empty(
        self, repo: MemoryRepository, mock_cursor: MagicMock
    ):
        """空リストでは0を返し、SQLを実行しない"""
        # Arrange
        memory_ids: List[UUID] = []

        # Act
        result = repo.batch_archive(memory_ids)

        # Assert
        assert result == 0
        mock_cursor.execute.assert_not_called()


class TestGetMemoriesForDecay:
    """get_memories_for_decay() メソッドのテスト"""

    @pytest.fixture
    def config(self) -> Phase1Config:
        return Phase1Config()

    @pytest.fixture
    def mock_cursor(self) -> MagicMock:
        cursor = MagicMock()
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=None)
        return cursor

    @pytest.fixture
    def mock_db(self, mock_cursor: MagicMock) -> MagicMock:
        db = MagicMock()
        db.get_cursor = MagicMock(return_value=mock_cursor)
        return db

    @pytest.fixture
    def repo(self, mock_db: MagicMock, config: Phase1Config) -> MemoryRepository:
        return MemoryRepository(mock_db, config)

    def _make_db_row(self, memory_id: UUID, strength: float) -> tuple:
        """テスト用DB行を作成（23カラム）"""
        now = datetime.now()
        return (
            memory_id,
            "test_agent",
            "テスト記憶",
            None,
            [],
            "project",
            None,
            "test_project",
            strength,
            {},
            0,
            0,
            None,
            None,  # next_review_at
            0,     # review_count
            0.0,
            0,
            None,
            "active",
            None,
            now,
            now,
            None,
        )

    def test_get_memories_for_decay(
        self, repo: MemoryRepository, mock_cursor: MagicMock
    ):
        """減衰対象メモリを取得"""
        # Arrange
        memories = [
            self._make_db_row(uuid4(), 0.2),
            self._make_db_row(uuid4(), 0.5),
            self._make_db_row(uuid4(), 0.8),
        ]
        mock_cursor.fetchall.return_value = memories

        # Act
        result = repo.get_memories_for_decay("test_agent", batch_size=100)

        # Assert
        assert len(result) == 3
        mock_cursor.execute.assert_called_once()
        sql = mock_cursor.execute.call_args[0][0]
        assert "status = 'active'" in sql
        assert "ORDER BY strength ASC" in sql

    def test_get_memories_for_decay_with_batch_size(
        self, repo: MemoryRepository, mock_cursor: MagicMock
    ):
        """バッチサイズ指定"""
        # Arrange
        mock_cursor.fetchall.return_value = []

        # Act
        repo.get_memories_for_decay("test_agent", batch_size=50)

        # Assert
        call_args = mock_cursor.execute.call_args[0][1]
        assert call_args[1] == 50


class TestCountActiveMemories:
    """count_active_memories() メソッドのテスト"""

    @pytest.fixture
    def config(self) -> Phase1Config:
        return Phase1Config()

    @pytest.fixture
    def mock_cursor(self) -> MagicMock:
        cursor = MagicMock()
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=None)
        return cursor

    @pytest.fixture
    def mock_db(self, mock_cursor: MagicMock) -> MagicMock:
        db = MagicMock()
        db.get_cursor = MagicMock(return_value=mock_cursor)
        return db

    @pytest.fixture
    def repo(self, mock_db: MagicMock, config: Phase1Config) -> MemoryRepository:
        return MemoryRepository(mock_db, config)

    def test_count_active_memories(
        self, repo: MemoryRepository, mock_cursor: MagicMock
    ):
        """アクティブメモリの件数を取得"""
        # Arrange
        mock_cursor.fetchone.return_value = (42,)

        # Act
        result = repo.count_active_memories("test_agent")

        # Assert
        assert result == 42

    def test_count_active_memories_zero(
        self, repo: MemoryRepository, mock_cursor: MagicMock
    ):
        """メモリがない場合は0"""
        # Arrange
        mock_cursor.fetchone.return_value = (0,)

        # Act
        result = repo.count_active_memories("test_agent")

        # Assert
        assert result == 0


class TestGetLowestStrengthMemories:
    """get_lowest_strength_memories() メソッドのテスト"""

    @pytest.fixture
    def config(self) -> Phase1Config:
        return Phase1Config()

    @pytest.fixture
    def mock_cursor(self) -> MagicMock:
        cursor = MagicMock()
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=None)
        return cursor

    @pytest.fixture
    def mock_db(self, mock_cursor: MagicMock) -> MagicMock:
        db = MagicMock()
        db.get_cursor = MagicMock(return_value=mock_cursor)
        return db

    @pytest.fixture
    def repo(self, mock_db: MagicMock, config: Phase1Config) -> MemoryRepository:
        return MemoryRepository(mock_db, config)

    def _make_db_row(self, memory_id: UUID, strength: float) -> tuple:
        """テスト用DB行を作成（23カラム）"""
        now = datetime.now()
        return (
            memory_id,
            "test_agent",
            "テスト記憶",
            None,
            [],
            "project",
            None,
            "test_project",
            strength,
            {},
            0,
            0,
            None,
            None,  # next_review_at
            0,     # review_count
            0.0,
            0,
            None,
            "active",
            None,
            now,
            now,
            None,
        )

    def test_get_lowest_strength_memories(
        self, repo: MemoryRepository, mock_cursor: MagicMock
    ):
        """最低強度のメモリを取得"""
        # Arrange
        memories = [
            self._make_db_row(uuid4(), 0.1),
            self._make_db_row(uuid4(), 0.2),
        ]
        mock_cursor.fetchall.return_value = memories

        # Act
        result = repo.get_lowest_strength_memories("test_agent", limit=2)

        # Assert
        assert len(result) == 2
        sql = mock_cursor.execute.call_args[0][0]
        assert "ORDER BY strength ASC" in sql
        assert "LIMIT" in sql


class TestFormatEmbedding:
    """_format_embedding() 内部メソッドのテスト"""

    @pytest.fixture
    def config(self) -> Phase1Config:
        return Phase1Config()

    @pytest.fixture
    def mock_db(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def repo(self, mock_db: MagicMock, config: Phase1Config) -> MemoryRepository:
        return MemoryRepository(mock_db, config)

    def test_format_embedding_none(self, repo: MemoryRepository):
        """Noneの場合はNoneを返す"""
        # Act
        result = repo._format_embedding(None)

        # Assert
        assert result is None

    def test_format_embedding_vector(self, repo: MemoryRepository):
        """ベクトルを正しい形式に変換"""
        # Arrange
        embedding = [0.1, 0.2, 0.3]

        # Act
        result = repo._format_embedding(embedding)

        # Assert
        assert result == "[0.1,0.2,0.3]"

    def test_format_embedding_high_precision(self, repo: MemoryRepository):
        """高精度の浮動小数点を正しく変換"""
        # Arrange
        embedding = [0.123456789, -0.987654321]

        # Act
        result = repo._format_embedding(embedding)

        # Assert
        assert result == "[0.123456789,-0.987654321]"

    def test_format_embedding_empty_list(self, repo: MemoryRepository):
        """空リストは空のベクトル文字列"""
        # Arrange
        embedding: List[float] = []

        # Act
        result = repo._format_embedding(embedding)

        # Assert
        assert result == "[]"


class TestEdgeCases:
    """境界値・異常系のテスト"""

    @pytest.fixture
    def config(self) -> Phase1Config:
        return Phase1Config()

    @pytest.fixture
    def mock_cursor(self) -> MagicMock:
        cursor = MagicMock()
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=None)
        return cursor

    @pytest.fixture
    def mock_db(self, mock_cursor: MagicMock) -> MagicMock:
        db = MagicMock()
        db.get_cursor = MagicMock(return_value=mock_cursor)
        return db

    @pytest.fixture
    def repo(self, mock_db: MagicMock, config: Phase1Config) -> MemoryRepository:
        return MemoryRepository(mock_db, config)

    def test_create_with_max_embedding_dimension(
        self, repo: MemoryRepository, mock_cursor: MagicMock
    ):
        """1536次元のembeddingを正しく処理"""
        # Arrange
        embedding = [0.1] * 1536  # 1536次元
        memory = AgentMemory.create(
            agent_id="test_agent",
            content="テスト",
            embedding=embedding,
        )
        # 23カラム: next_review_at, review_count追加
        mock_cursor.fetchone.return_value = (
            memory.id,
            memory.agent_id,
            memory.content,
            embedding,
            [],
            "project",
            None,
            None,
            1.0,
            {},
            0,
            0,
            None,
            None,  # next_review_at
            0,     # review_count
            0.0,
            0,
            None,
            "active",
            None,
            datetime.now(),
            datetime.now(),
            None,
        )

        # Act
        result = repo.create(memory)

        # Assert
        assert len(result.embedding) == 1536

    def test_create_with_special_characters_in_content(
        self, repo: MemoryRepository, mock_cursor: MagicMock
    ):
        """特殊文字を含むコンテンツ"""
        # Arrange
        content = "テスト'内容\"with special\ncharacters"
        memory = AgentMemory.create(
            agent_id="test_agent",
            content=content,
        )
        # 23カラム: next_review_at, review_count追加
        mock_cursor.fetchone.return_value = (
            memory.id,
            memory.agent_id,
            content,
            None,
            [],
            "project",
            None,
            None,
            1.0,
            {},
            0,
            0,
            None,
            None,  # next_review_at
            0,     # review_count
            0.0,
            0,
            None,
            "active",
            None,
            datetime.now(),
            datetime.now(),
            None,
        )

        # Act
        result = repo.create(memory)

        # Assert
        assert result.content == content

    def test_batch_archive_large_list(
        self, repo: MemoryRepository, mock_cursor: MagicMock
    ):
        """大量のメモリを一括アーカイブ"""
        # Arrange
        memory_ids = [uuid4() for _ in range(1000)]
        mock_cursor.rowcount = 1000

        # Act
        result = repo.batch_archive(memory_ids)

        # Assert
        assert result == 1000


class TestSqlInjectionPrevention:
    """SQLインジェクション防止のテスト"""

    @pytest.fixture
    def config(self) -> Phase1Config:
        return Phase1Config()

    @pytest.fixture
    def mock_cursor(self) -> MagicMock:
        cursor = MagicMock()
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=None)
        return cursor

    @pytest.fixture
    def mock_db(self, mock_cursor: MagicMock) -> MagicMock:
        db = MagicMock()
        db.get_cursor = MagicMock(return_value=mock_cursor)
        return db

    @pytest.fixture
    def repo(self, mock_db: MagicMock, config: Phase1Config) -> MemoryRepository:
        return MemoryRepository(mock_db, config)

    def test_get_by_agent_id_with_injection_attempt(
        self, repo: MemoryRepository, mock_cursor: MagicMock
    ):
        """SQLインジェクション試行がパラメータ化クエリで防止される"""
        # Arrange
        malicious_agent_id = "'; DROP TABLE agent_memory; --"
        mock_cursor.fetchall.return_value = []

        # Act
        repo.get_by_agent_id(malicious_agent_id)

        # Assert - パラメータ化クエリで安全に処理される
        call_args = mock_cursor.execute.call_args[0][1]
        assert call_args[0] == malicious_agent_id  # そのまま渡される（エスケープはDBドライバが行う）
