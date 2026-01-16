# ベクトル検索モジュールのテスト
"""
VectorSearch の単体テスト

テスト観点:
- search_candidates() - コサイン類似度検索
- search_by_embedding() - 埋め込みベクトル直接検索
- similarity_threshold フィルタリング
- candidate_limit 制限
- AzureEmbeddingClient との連携（モック）
- 空結果、境界値のテスト

注意: DBモック使用（実DB接続不要）
"""

from datetime import datetime
from typing import List, Optional
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from src.config.phase1_config import Phase1Config
from src.embedding.azure_client import AzureEmbeddingClient, AzureEmbeddingError
from src.models.memory import AgentMemory
from src.search.vector_search import VectorSearch, VectorSearchError


# =============================================================================
# テストヘルパー
# =============================================================================


def _make_db_row(
    memory_id: UUID,
    agent_id: str = "test_agent",
    content: str = "テスト記憶",
    embedding: Optional[List[float]] = None,
    strength: float = 1.0,
    similarity: float = 0.85,
) -> tuple:
    """テスト用DB行データを作成（similarity込み、24カラム）"""
    now = datetime.now()
    return (
        memory_id,
        agent_id,
        content,
        embedding or [0.1] * 1536,
        [],  # tags
        "project",  # scope_level
        None,  # scope_domain
        "test_project",  # scope_project
        strength,
        {},  # strength_by_perspective
        0,  # access_count
        0,  # candidate_count
        None,  # last_accessed_at
        None,  # next_review_at
        0,  # review_count
        0.0,  # impact_score
        0,  # consolidation_level
        None,  # learning
        "active",  # status
        None,  # source
        now,  # created_at
        now,  # updated_at
        None,  # last_decay_at
        similarity,  # 検索結果の類似度（最後のカラム）
    )


# =============================================================================
# VectorSearch 初期化のテスト
# =============================================================================


class TestVectorSearchSetup:
    """VectorSearch 初期化のテスト"""

    @pytest.fixture
    def mock_db(self) -> MagicMock:
        """モック DatabaseConnection"""
        return MagicMock()

    @pytest.fixture
    def mock_embedding_client(self) -> MagicMock:
        """モック AzureEmbeddingClient"""
        mock = MagicMock(spec=AzureEmbeddingClient)
        mock.get_embedding.return_value = [0.1] * 1536
        return mock

    @pytest.fixture
    def config(self) -> Phase1Config:
        """テスト用の設定"""
        return Phase1Config(
            similarity_threshold=0.7,
            candidate_limit=50,
        )

    def test_init_with_explicit_config(
        self, mock_db: MagicMock, mock_embedding_client: MagicMock, config: Phase1Config
    ):
        """明示的な設定でインスタンス化できる"""
        # Act
        search = VectorSearch(
            db=mock_db,
            embedding_client=mock_embedding_client,
            config=config,
        )

        # Assert
        assert search.db is mock_db
        assert search.embedding_client is mock_embedding_client
        assert search.config.similarity_threshold == 0.7
        assert search.config.candidate_limit == 50

    def test_init_with_default_config(
        self, mock_db: MagicMock, mock_embedding_client: MagicMock
    ):
        """デフォルト設定でインスタンス化できる"""
        # Act
        search = VectorSearch(
            db=mock_db,
            embedding_client=mock_embedding_client,
        )

        # Assert
        assert search.config is not None
        assert hasattr(search.config, "similarity_threshold")
        assert hasattr(search.config, "candidate_limit")


# =============================================================================
# search_candidates() のテスト
# =============================================================================


class TestSearchCandidates:
    """search_candidates() メソッドのテスト"""

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
    def mock_embedding_client(self) -> MagicMock:
        """モック AzureEmbeddingClient"""
        mock = MagicMock(spec=AzureEmbeddingClient)
        mock.get_embedding.return_value = [0.1] * 1536
        return mock

    @pytest.fixture
    def config(self) -> Phase1Config:
        return Phase1Config(similarity_threshold=0.7, candidate_limit=50)

    @pytest.fixture
    def search(
        self, mock_db: MagicMock, mock_embedding_client: MagicMock, config: Phase1Config
    ) -> VectorSearch:
        return VectorSearch(mock_db, mock_embedding_client, config)

    def test_search_candidates_returns_results(
        self, search: VectorSearch, mock_cursor: MagicMock, mock_embedding_client: MagicMock
    ):
        """正常に候補を取得できる"""
        # Arrange
        memory_id = uuid4()
        mock_cursor.fetchall.return_value = [
            _make_db_row(memory_id, similarity=0.85)
        ]

        # Act
        results = search.search_candidates(query="テスト検索クエリ", agent_id="test_agent")

        # Assert
        assert len(results) == 1
        memory, similarity = results[0]
        assert memory.id == memory_id
        assert similarity == 0.85
        mock_embedding_client.get_embedding.assert_called_once_with("テスト検索クエリ")

    def test_search_candidates_empty_query_returns_empty_list(
        self, search: VectorSearch, mock_embedding_client: MagicMock
    ):
        """空のクエリの場合は空リストを返す"""
        # Act
        results = search.search_candidates(query="", agent_id="test_agent")

        # Assert
        assert results == []
        mock_embedding_client.get_embedding.assert_not_called()

    def test_search_candidates_whitespace_query_returns_empty_list(
        self, search: VectorSearch, mock_embedding_client: MagicMock
    ):
        """空白のみのクエリの場合は空リストを返す"""
        # Act
        results = search.search_candidates(query="   ", agent_id="test_agent")

        # Assert
        assert results == []
        mock_embedding_client.get_embedding.assert_not_called()

    def test_search_candidates_no_results(
        self, search: VectorSearch, mock_cursor: MagicMock
    ):
        """候補がない場合は空リストを返す"""
        # Arrange
        mock_cursor.fetchall.return_value = []

        # Act
        results = search.search_candidates(query="存在しないクエリ", agent_id="test_agent")

        # Assert
        assert results == []

    def test_search_candidates_multiple_results_sorted(
        self, search: VectorSearch, mock_cursor: MagicMock
    ):
        """複数の候補は類似度順にソートされて返される"""
        # Arrange
        memory_ids = [uuid4() for _ in range(3)]
        # DB側で既にソートされている想定（類似度降順）
        mock_cursor.fetchall.return_value = [
            _make_db_row(memory_ids[0], content="Memory 0", similarity=0.95),
            _make_db_row(memory_ids[1], content="Memory 1", similarity=0.80),
            _make_db_row(memory_ids[2], content="Memory 2", similarity=0.72),
        ]

        # Act
        results = search.search_candidates(query="テスト", agent_id="test_agent")

        # Assert
        assert len(results) == 3
        assert results[0][1] == 0.95
        assert results[1][1] == 0.80
        assert results[2][1] == 0.72

    def test_search_candidates_embedding_error(
        self, search: VectorSearch, mock_embedding_client: MagicMock
    ):
        """エンベディング取得エラー時は VectorSearchError を発生"""
        # Arrange
        mock_embedding_client.get_embedding.side_effect = AzureEmbeddingError("API error")

        # Act & Assert
        with pytest.raises(VectorSearchError) as exc_info:
            search.search_candidates(query="テスト", agent_id="test_agent")

        assert "エンベディング取得" in str(exc_info.value)

    def test_search_candidates_db_error(
        self, search: VectorSearch, mock_cursor: MagicMock
    ):
        """DB検索エラー時は VectorSearchError を発生"""
        # Arrange
        mock_cursor.execute.side_effect = Exception("Database error")

        # Act & Assert
        with pytest.raises(VectorSearchError) as exc_info:
            search.search_candidates(query="テスト", agent_id="test_agent")

        assert "ベクトル検索に失敗" in str(exc_info.value)


# =============================================================================
# search_by_embedding() のテスト
# =============================================================================


class TestSearchByEmbedding:
    """search_by_embedding() メソッドのテスト"""

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
    def mock_embedding_client(self) -> MagicMock:
        mock = MagicMock(spec=AzureEmbeddingClient)
        return mock

    @pytest.fixture
    def config(self) -> Phase1Config:
        return Phase1Config(similarity_threshold=0.7, candidate_limit=50)

    @pytest.fixture
    def search(
        self, mock_db: MagicMock, mock_embedding_client: MagicMock, config: Phase1Config
    ) -> VectorSearch:
        return VectorSearch(mock_db, mock_embedding_client, config)

    def test_search_by_embedding_returns_results(
        self, search: VectorSearch, mock_cursor: MagicMock
    ):
        """事前計算済みエンベディングで検索できる"""
        # Arrange
        memory_id = uuid4()
        mock_cursor.fetchall.return_value = [
            _make_db_row(memory_id, similarity=0.90)
        ]

        # Act
        query_embedding = [0.2] * 1536
        results = search.search_by_embedding(
            query_embedding=query_embedding,
            agent_id="test_agent",
        )

        # Assert
        assert len(results) == 1
        memory, similarity = results[0]
        assert memory.id == memory_id
        assert similarity == 0.90

    def test_search_by_embedding_custom_threshold(
        self, search: VectorSearch, mock_cursor: MagicMock
    ):
        """カスタム similarity_threshold が適用される"""
        # Arrange
        mock_cursor.fetchall.return_value = []

        # Act
        search.search_by_embedding(
            query_embedding=[0.2] * 1536,
            agent_id="test_agent",
            similarity_threshold=0.9,
        )

        # Assert - SQL パラメータに 0.9 が含まれる
        call_args = mock_cursor.execute.call_args[0][1]
        assert 0.9 in call_args

    def test_search_by_embedding_custom_limit(
        self, search: VectorSearch, mock_cursor: MagicMock
    ):
        """カスタム candidate_limit が適用される"""
        # Arrange
        mock_cursor.fetchall.return_value = []

        # Act
        search.search_by_embedding(
            query_embedding=[0.2] * 1536,
            agent_id="test_agent",
            candidate_limit=10,
        )

        # Assert - SQL パラメータに 10 が含まれる（LIMIT句）
        call_args = mock_cursor.execute.call_args[0][1]
        assert 10 in call_args

    def test_search_by_embedding_db_error(
        self, search: VectorSearch, mock_cursor: MagicMock
    ):
        """DB検索エラー時は VectorSearchError を発生"""
        # Arrange
        mock_cursor.execute.side_effect = Exception("Database error")

        # Act & Assert
        with pytest.raises(VectorSearchError) as exc_info:
            search.search_by_embedding(
                query_embedding=[0.1] * 1536,
                agent_id="test_agent",
            )

        assert "エンベディング検索に失敗" in str(exc_info.value)

    def test_search_by_embedding_empty_results(
        self, search: VectorSearch, mock_cursor: MagicMock
    ):
        """候補がない場合は空リストを返す"""
        # Arrange
        mock_cursor.fetchall.return_value = []

        # Act
        results = search.search_by_embedding(
            query_embedding=[0.1] * 1536,
            agent_id="test_agent",
        )

        # Assert
        assert results == []


# =============================================================================
# count_candidates() のテスト
# =============================================================================


class TestCountCandidates:
    """count_candidates() メソッドのテスト"""

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
    def mock_embedding_client(self) -> MagicMock:
        mock = MagicMock(spec=AzureEmbeddingClient)
        mock.get_embedding.return_value = [0.1] * 1536
        return mock

    @pytest.fixture
    def config(self) -> Phase1Config:
        return Phase1Config(similarity_threshold=0.7, candidate_limit=50)

    @pytest.fixture
    def search(
        self, mock_db: MagicMock, mock_embedding_client: MagicMock, config: Phase1Config
    ) -> VectorSearch:
        return VectorSearch(mock_db, mock_embedding_client, config)

    def test_count_candidates_returns_count(
        self, search: VectorSearch, mock_cursor: MagicMock
    ):
        """候補数を正しくカウントできる"""
        # Arrange
        mock_cursor.fetchone.return_value = (5,)

        # Act
        count = search.count_candidates(query="テスト", agent_id="test_agent")

        # Assert
        assert count == 5

    def test_count_candidates_empty_query_returns_zero(
        self, search: VectorSearch, mock_embedding_client: MagicMock
    ):
        """空のクエリの場合は 0 を返す"""
        # Act
        count = search.count_candidates(query="", agent_id="test_agent")

        # Assert
        assert count == 0
        mock_embedding_client.get_embedding.assert_not_called()

    def test_count_candidates_no_results(
        self, search: VectorSearch, mock_cursor: MagicMock
    ):
        """候補がない場合は 0 を返す"""
        # Arrange
        mock_cursor.fetchone.return_value = (0,)

        # Act
        count = search.count_candidates(query="存在しないクエリ", agent_id="test_agent")

        # Assert
        assert count == 0

    def test_count_candidates_fetchone_returns_none(
        self, search: VectorSearch, mock_cursor: MagicMock
    ):
        """fetchone が None を返す場合は 0"""
        # Arrange
        mock_cursor.fetchone.return_value = None

        # Act
        count = search.count_candidates(query="テスト", agent_id="test_agent")

        # Assert
        assert count == 0

    def test_count_candidates_embedding_error(
        self, search: VectorSearch, mock_embedding_client: MagicMock
    ):
        """エンベディング取得エラー時は VectorSearchError を発生"""
        # Arrange
        mock_embedding_client.get_embedding.side_effect = AzureEmbeddingError("API error")

        # Act & Assert
        with pytest.raises(VectorSearchError) as exc_info:
            search.count_candidates(query="テスト", agent_id="test_agent")

        assert "エンベディング取得" in str(exc_info.value)


# =============================================================================
# _format_embedding() のテスト
# =============================================================================


class TestFormatEmbedding:
    """_format_embedding() 内部メソッドのテスト"""

    @pytest.fixture
    def mock_db(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def mock_embedding_client(self) -> MagicMock:
        return MagicMock(spec=AzureEmbeddingClient)

    @pytest.fixture
    def search(self, mock_db: MagicMock, mock_embedding_client: MagicMock) -> VectorSearch:
        return VectorSearch(mock_db, mock_embedding_client)

    def test_format_embedding_basic(self, search: VectorSearch):
        """基本的なエンベディングのフォーマット変換"""
        # Arrange
        embedding = [0.1, 0.2, 0.3]

        # Act
        result = search._format_embedding(embedding)

        # Assert
        assert result == "[0.1,0.2,0.3]"

    def test_format_embedding_single_value(self, search: VectorSearch):
        """単一値のエンベディング"""
        # Arrange
        embedding = [0.5]

        # Act
        result = search._format_embedding(embedding)

        # Assert
        assert result == "[0.5]"

    def test_format_embedding_empty(self, search: VectorSearch):
        """空のエンベディング"""
        # Arrange
        embedding: List[float] = []

        # Act
        result = search._format_embedding(embedding)

        # Assert
        assert result == "[]"

    def test_format_embedding_full_dimension(self, search: VectorSearch):
        """1536次元のエンベディング"""
        # Arrange
        embedding = [0.1] * 1536

        # Act
        result = search._format_embedding(embedding)

        # Assert
        assert result.startswith("[")
        assert result.endswith("]")
        values = result[1:-1].split(",")
        assert len(values) == 1536

    def test_format_embedding_high_precision(self, search: VectorSearch):
        """高精度の浮動小数点を正しく変換"""
        # Arrange
        embedding = [0.123456789, -0.987654321]

        # Act
        result = search._format_embedding(embedding)

        # Assert
        assert result == "[0.123456789,-0.987654321]"


# =============================================================================
# similarity_threshold フィルタリングのテスト
# =============================================================================


class TestSimilarityThresholdFiltering:
    """similarity_threshold によるフィルタリングのテスト"""

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
    def mock_embedding_client(self) -> MagicMock:
        mock = MagicMock(spec=AzureEmbeddingClient)
        mock.get_embedding.return_value = [0.1] * 1536
        return mock

    def test_threshold_exact_match(
        self, mock_db: MagicMock, mock_embedding_client: MagicMock, mock_cursor: MagicMock
    ):
        """閾値ちょうどの場合も結果に含まれる（>=）"""
        # Arrange
        config = Phase1Config(similarity_threshold=0.7, candidate_limit=10)
        search = VectorSearch(mock_db, mock_embedding_client, config)

        memory_id = uuid4()
        mock_cursor.fetchall.return_value = [
            _make_db_row(memory_id, similarity=0.7)  # 閾値ちょうど
        ]

        # Act
        results = search.search_candidates(query="テスト", agent_id="test_agent")

        # Assert
        assert len(results) == 1
        assert results[0][1] == 0.7

    def test_high_threshold_filters_more(
        self, mock_db: MagicMock, mock_embedding_client: MagicMock, mock_cursor: MagicMock
    ):
        """高い閾値を設定するとSQLに渡される"""
        # Arrange
        config = Phase1Config(similarity_threshold=0.9, candidate_limit=10)
        search = VectorSearch(mock_db, mock_embedding_client, config)
        mock_cursor.fetchall.return_value = []

        # Act
        search.search_candidates(query="テスト", agent_id="test_agent")

        # Assert - SQL パラメータに 0.9 が含まれる
        call_args = mock_cursor.execute.call_args[0][1]
        assert 0.9 in call_args


# =============================================================================
# candidate_limit 制限のテスト
# =============================================================================


class TestCandidateLimitRestriction:
    """candidate_limit による制限のテスト"""

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
    def mock_embedding_client(self) -> MagicMock:
        mock = MagicMock(spec=AzureEmbeddingClient)
        mock.get_embedding.return_value = [0.1] * 1536
        return mock

    def test_limit_restricts_results(
        self, mock_db: MagicMock, mock_embedding_client: MagicMock, mock_cursor: MagicMock
    ):
        """candidate_limit で結果数が制限される"""
        # Arrange
        config = Phase1Config(similarity_threshold=0.5, candidate_limit=5)
        search = VectorSearch(mock_db, mock_embedding_client, config)

        memory_ids = [uuid4() for _ in range(5)]
        mock_cursor.fetchall.return_value = [
            _make_db_row(mid, similarity=0.9 - i * 0.05)
            for i, mid in enumerate(memory_ids)
        ]

        # Act
        results = search.search_candidates(query="テスト", agent_id="test_agent")

        # Assert - SQL パラメータに LIMIT 5 が含まれる
        call_args = mock_cursor.execute.call_args[0][1]
        assert 5 in call_args
        assert len(results) == 5

    def test_limit_one_returns_single_result(
        self, mock_db: MagicMock, mock_embedding_client: MagicMock, mock_cursor: MagicMock
    ):
        """candidate_limit=1 で単一結果のみ返される"""
        # Arrange
        config = Phase1Config(similarity_threshold=0.5, candidate_limit=1)
        search = VectorSearch(mock_db, mock_embedding_client, config)

        memory_id = uuid4()
        mock_cursor.fetchall.return_value = [
            _make_db_row(memory_id, similarity=0.95)
        ]

        # Act
        results = search.search_candidates(query="テスト", agent_id="test_agent")

        # Assert
        assert len(results) == 1


# =============================================================================
# AzureEmbeddingClient との連携テスト（モック）
# =============================================================================


class TestAzureEmbeddingClientIntegration:
    """AzureEmbeddingClient との連携テスト"""

    @pytest.fixture
    def mock_cursor(self) -> MagicMock:
        cursor = MagicMock()
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=None)
        cursor.fetchall.return_value = []
        return cursor

    @pytest.fixture
    def mock_db(self, mock_cursor: MagicMock) -> MagicMock:
        db = MagicMock()
        db.get_cursor = MagicMock(return_value=mock_cursor)
        return db

    @pytest.fixture
    def mock_embedding_client(self) -> MagicMock:
        mock = MagicMock(spec=AzureEmbeddingClient)
        mock.get_embedding.return_value = [0.5] * 1536
        return mock

    @pytest.fixture
    def search(self, mock_db: MagicMock, mock_embedding_client: MagicMock) -> VectorSearch:
        return VectorSearch(mock_db, mock_embedding_client)

    def test_embedding_client_called_with_query(
        self, search: VectorSearch, mock_embedding_client: MagicMock
    ):
        """検索時にエンベディングクライアントが正しく呼ばれる"""
        # Arrange
        query = "特定のクエリテキスト"

        # Act
        search.search_candidates(query=query, agent_id="test_agent")

        # Assert
        mock_embedding_client.get_embedding.assert_called_once_with(query)

    def test_embedding_used_in_sql(
        self, search: VectorSearch, mock_cursor: MagicMock, mock_embedding_client: MagicMock
    ):
        """エンベディングがSQL検索に使用される"""
        # Arrange
        test_embedding = [0.5] * 1536
        mock_embedding_client.get_embedding.return_value = test_embedding

        # Act
        search.search_candidates(query="テスト", agent_id="test_agent")

        # Assert - SQL パラメータにエンベディング文字列が含まれる
        call_args = mock_cursor.execute.call_args[0][1]
        embedding_str = search._format_embedding(test_embedding)
        assert embedding_str in call_args


# =============================================================================
# エッジケース・境界値のテスト
# =============================================================================


class TestEdgeCases:
    """エッジケースと境界値のテスト"""

    @pytest.fixture
    def mock_cursor(self) -> MagicMock:
        cursor = MagicMock()
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=None)
        cursor.fetchall.return_value = []
        return cursor

    @pytest.fixture
    def mock_db(self, mock_cursor: MagicMock) -> MagicMock:
        db = MagicMock()
        db.get_cursor = MagicMock(return_value=mock_cursor)
        return db

    @pytest.fixture
    def mock_embedding_client(self) -> MagicMock:
        mock = MagicMock(spec=AzureEmbeddingClient)
        mock.get_embedding.return_value = [0.1] * 1536
        return mock

    def test_very_low_similarity_threshold(
        self, mock_db: MagicMock, mock_embedding_client: MagicMock, mock_cursor: MagicMock
    ):
        """非常に低い閾値（0.0）の場合"""
        # Arrange
        config = Phase1Config(similarity_threshold=0.0, candidate_limit=100)
        search = VectorSearch(mock_db, mock_embedding_client, config)

        # Act - エラーなく実行できることを確認
        results = search.search_candidates(query="テスト", agent_id="test_agent")

        # Assert
        assert results == []

    def test_high_similarity_threshold(
        self, mock_db: MagicMock, mock_embedding_client: MagicMock, mock_cursor: MagicMock
    ):
        """非常に高い閾値（0.99）の場合"""
        # Arrange
        config = Phase1Config(similarity_threshold=0.99, candidate_limit=100)
        search = VectorSearch(mock_db, mock_embedding_client, config)

        # Act
        results = search.search_candidates(query="テスト", agent_id="test_agent")

        # Assert
        assert results == []

    def test_large_candidate_limit(
        self, mock_db: MagicMock, mock_embedding_client: MagicMock, mock_cursor: MagicMock
    ):
        """大きな candidate_limit（1000）の場合"""
        # Arrange
        config = Phase1Config(similarity_threshold=0.5, candidate_limit=1000)
        search = VectorSearch(mock_db, mock_embedding_client, config)

        # Act
        search.search_candidates(query="テスト", agent_id="test_agent")

        # Assert - SQL パラメータに 1000 が含まれる
        call_args = mock_cursor.execute.call_args[0][1]
        assert 1000 in call_args

    def test_unicode_query(
        self, mock_db: MagicMock, mock_embedding_client: MagicMock, mock_cursor: MagicMock
    ):
        """Unicode文字を含むクエリ"""
        # Arrange
        config = Phase1Config()
        search = VectorSearch(mock_db, mock_embedding_client, config)
        query = "日本語テスト 特殊文字!@#$%"

        # Act
        search.search_candidates(query=query, agent_id="test_agent")

        # Assert
        mock_embedding_client.get_embedding.assert_called_once_with(query)

    def test_very_long_query(
        self, mock_db: MagicMock, mock_embedding_client: MagicMock, mock_cursor: MagicMock
    ):
        """非常に長いクエリ"""
        # Arrange
        config = Phase1Config()
        search = VectorSearch(mock_db, mock_embedding_client, config)
        query = "あ" * 10000

        # Act
        search.search_candidates(query=query, agent_id="test_agent")

        # Assert
        mock_embedding_client.get_embedding.assert_called_once_with(query)

    def test_special_agent_id(
        self, mock_db: MagicMock, mock_embedding_client: MagicMock, mock_cursor: MagicMock
    ):
        """特殊文字を含むagent_id"""
        # Arrange
        config = Phase1Config()
        search = VectorSearch(mock_db, mock_embedding_client, config)
        agent_id = "agent_with_special-chars.v1.0"

        # Act
        search.search_candidates(query="テスト", agent_id=agent_id)

        # Assert - SQL パラメータに agent_id が含まれる
        call_args = mock_cursor.execute.call_args[0][1]
        assert agent_id in call_args


# =============================================================================
# SQLパラメータのテスト
# =============================================================================


class TestSqlParameters:
    """SQL パラメータが正しく渡されることのテスト"""

    @pytest.fixture
    def mock_cursor(self) -> MagicMock:
        cursor = MagicMock()
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=None)
        cursor.fetchall.return_value = []
        return cursor

    @pytest.fixture
    def mock_db(self, mock_cursor: MagicMock) -> MagicMock:
        db = MagicMock()
        db.get_cursor = MagicMock(return_value=mock_cursor)
        return db

    @pytest.fixture
    def mock_embedding_client(self) -> MagicMock:
        mock = MagicMock(spec=AzureEmbeddingClient)
        mock.get_embedding.return_value = [0.1] * 1536
        return mock

    def test_search_candidates_sql_params(
        self, mock_db: MagicMock, mock_embedding_client: MagicMock, mock_cursor: MagicMock
    ):
        """search_candidates の SQL パラメータが正しい"""
        # Arrange
        config = Phase1Config(similarity_threshold=0.7, candidate_limit=50)
        search = VectorSearch(mock_db, mock_embedding_client, config)

        # Act
        search.search_candidates(query="テスト", agent_id="my_agent")

        # Assert
        call_args = mock_cursor.execute.call_args[0]
        sql = call_args[0]
        params = call_args[1]

        # SQL に必要なプレースホルダーが含まれることを確認
        assert "agent_id = %s" in sql
        assert "status = 'active'" in sql
        assert "embedding IS NOT NULL" in sql
        assert "LIMIT %s" in sql

        # パラメータの確認
        assert "my_agent" in params
        assert 0.7 in params  # similarity_threshold
        assert 50 in params   # candidate_limit


# =============================================================================
# SQLインジェクション防止のテスト
# =============================================================================


class TestSqlInjectionPrevention:
    """SQLインジェクション防止のテスト"""

    @pytest.fixture
    def mock_cursor(self) -> MagicMock:
        cursor = MagicMock()
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=None)
        cursor.fetchall.return_value = []
        return cursor

    @pytest.fixture
    def mock_db(self, mock_cursor: MagicMock) -> MagicMock:
        db = MagicMock()
        db.get_cursor = MagicMock(return_value=mock_cursor)
        return db

    @pytest.fixture
    def mock_embedding_client(self) -> MagicMock:
        mock = MagicMock(spec=AzureEmbeddingClient)
        mock.get_embedding.return_value = [0.1] * 1536
        return mock

    def test_search_with_injection_attempt_in_agent_id(
        self, mock_db: MagicMock, mock_embedding_client: MagicMock, mock_cursor: MagicMock
    ):
        """SQLインジェクション試行がパラメータ化クエリで防止される（agent_id）"""
        # Arrange
        config = Phase1Config()
        search = VectorSearch(mock_db, mock_embedding_client, config)
        malicious_agent_id = "'; DROP TABLE agent_memory; --"

        # Act
        search.search_candidates(query="テスト", agent_id=malicious_agent_id)

        # Assert - パラメータ化クエリで安全に処理される
        call_args = mock_cursor.execute.call_args[0][1]
        assert malicious_agent_id in call_args
