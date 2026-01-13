# タスク実行フロー統合クラスのテスト
"""
TaskExecutor の search_memories メソッドの単体テスト

テスト観点:
- 検索フローの統合動作
- 2段階強化（candidate_count++）の実行
- エッジケースの処理（空クエリ、0件結果など）
- perspective パラメータの受け渡し
"""

from datetime import datetime
from typing import List, Optional, Tuple
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest

from src.config.phase1_config import Phase1Config
from src.core.task_executor import TaskExecutor
from src.models.memory import AgentMemory
from src.search.ranking import MemoryRanker, ScoredMemory


class TestSearchMemories:
    """search_memories メソッドのテスト"""

    @pytest.fixture
    def config(self) -> Phase1Config:
        """テスト用の設定"""
        return Phase1Config(
            score_weights={
                "similarity": 0.50,
                "strength": 0.30,
                "recency": 0.20,
            },
            top_k_results=10,
        )

    @pytest.fixture
    def sample_memories(self) -> List[AgentMemory]:
        """テスト用のメモリリスト"""
        now = datetime.now()
        return [
            AgentMemory(
                id=uuid4(),
                agent_id="test_agent",
                content="緊急調達では15%のコスト増を見込む必要がある",
                strength=1.0,
                strength_by_perspective={"コスト": 1.5},
                created_at=now,
                updated_at=now,
                last_accessed_at=now,
                candidate_count=0,
                access_count=0,
            ),
            AgentMemory(
                id=uuid4(),
                agent_id="test_agent",
                content="緊急調達の場合は納期を2週間短縮できる",
                strength=0.8,
                strength_by_perspective={"納期": 1.2},
                created_at=now,
                updated_at=now,
                last_accessed_at=now,
                candidate_count=0,
                access_count=0,
            ),
        ]

    @pytest.fixture
    def mock_vector_search(self, sample_memories: List[AgentMemory]) -> MagicMock:
        """モック VectorSearch"""
        mock = MagicMock()
        # search_candidates は (AgentMemory, similarity) のリストを返す
        mock.search_candidates.return_value = [
            (sample_memories[0], 0.85),
            (sample_memories[1], 0.72),
        ]
        return mock

    @pytest.fixture
    def mock_ranker(self, sample_memories: List[AgentMemory]) -> MagicMock:
        """モック MemoryRanker"""
        mock = MagicMock()
        # rank は ScoredMemory のリストを返す
        mock.rank.return_value = [
            ScoredMemory(
                memory=sample_memories[0],
                similarity=0.85,
                final_score=0.75,
                score_breakdown={
                    "similarity_raw": 0.85,
                    "similarity_weighted": 0.425,
                    "strength_raw": 1.0,
                    "strength_normalized": 0.5,
                    "strength_weighted": 0.15,
                    "recency_raw": 1.0,
                    "recency_weighted": 0.2,
                    "total": 0.775,
                },
            ),
            ScoredMemory(
                memory=sample_memories[1],
                similarity=0.72,
                final_score=0.65,
                score_breakdown={
                    "similarity_raw": 0.72,
                    "similarity_weighted": 0.36,
                    "strength_raw": 0.8,
                    "strength_normalized": 0.4,
                    "strength_weighted": 0.12,
                    "recency_raw": 1.0,
                    "recency_weighted": 0.2,
                    "total": 0.68,
                },
            ),
        ]
        return mock

    @pytest.fixture
    def mock_strength_manager(self) -> MagicMock:
        """モック StrengthManager"""
        mock = MagicMock()
        # mark_as_candidate は更新行数を返す
        mock.mark_as_candidate.return_value = 2
        return mock

    @pytest.fixture
    def mock_sleep_processor(self) -> MagicMock:
        """モック SleepPhaseProcessor"""
        return MagicMock()

    @pytest.fixture
    def mock_repository(self) -> MagicMock:
        """モック MemoryRepository"""
        return MagicMock()

    @pytest.fixture
    def task_executor(
        self,
        mock_vector_search: MagicMock,
        mock_ranker: MagicMock,
        mock_strength_manager: MagicMock,
        mock_sleep_processor: MagicMock,
        mock_repository: MagicMock,
        config: Phase1Config,
    ) -> TaskExecutor:
        """テスト用の TaskExecutor"""
        return TaskExecutor(
            vector_search=mock_vector_search,
            ranker=mock_ranker,
            strength_manager=mock_strength_manager,
            sleep_processor=mock_sleep_processor,
            repository=mock_repository,
            config=config,
        )

    def test_search_returns_scored_memories(
        self, task_executor: TaskExecutor, sample_memories: List[AgentMemory]
    ):
        """検索結果が ScoredMemory のリストとして返される"""
        result = task_executor.search_memories(
            query="緊急調達のコスト",
            agent_id="test_agent",
        )

        assert len(result) == 2
        assert all(isinstance(sm, ScoredMemory) for sm in result)
        assert result[0].memory.id == sample_memories[0].id
        assert result[1].memory.id == sample_memories[1].id

    def test_search_calls_vector_search(
        self, task_executor: TaskExecutor, mock_vector_search: MagicMock
    ):
        """VectorSearch.search_candidates が正しく呼ばれる"""
        task_executor.search_memories(
            query="緊急調達のコスト",
            agent_id="test_agent",
            perspective="コスト",
        )

        mock_vector_search.search_candidates.assert_called_once_with(
            query="緊急調達のコスト",
            agent_id="test_agent",
            perspective="コスト",
        )

    def test_search_calls_ranker(
        self,
        task_executor: TaskExecutor,
        mock_ranker: MagicMock,
        sample_memories: List[AgentMemory],
    ):
        """MemoryRanker.rank が正しく呼ばれる"""
        task_executor.search_memories(
            query="緊急調達のコスト",
            agent_id="test_agent",
            perspective="コスト",
        )

        mock_ranker.rank.assert_called_once()
        call_args = mock_ranker.rank.call_args
        # candidates 引数を検証
        candidates = call_args.kwargs.get("candidates") or call_args.args[0]
        assert len(candidates) == 2
        assert candidates[0][0].id == sample_memories[0].id
        # perspective 引数を検証
        perspective = call_args.kwargs.get("perspective")
        assert perspective == "コスト"

    def test_search_increments_candidate_count(
        self,
        task_executor: TaskExecutor,
        mock_strength_manager: MagicMock,
        sample_memories: List[AgentMemory],
    ):
        """検索候補の candidate_count が増加する（mark_as_candidate が呼ばれる）"""
        task_executor.search_memories(
            query="緊急調達のコスト",
            agent_id="test_agent",
        )

        # mark_as_candidate が呼ばれたことを確認
        mock_strength_manager.mark_as_candidate.assert_called_once()

        # 引数がメモリIDのリストであることを確認
        call_args = mock_strength_manager.mark_as_candidate.call_args
        memory_ids = call_args.args[0]
        assert len(memory_ids) == 2
        assert memory_ids[0] == sample_memories[0].id
        assert memory_ids[1] == sample_memories[1].id

    def test_search_empty_query_returns_empty(self, task_executor: TaskExecutor):
        """空のクエリの場合は空リストを返す"""
        result = task_executor.search_memories(
            query="",
            agent_id="test_agent",
        )

        assert result == []

    def test_search_whitespace_query_returns_empty(self, task_executor: TaskExecutor):
        """空白のみのクエリの場合も空リストを返す"""
        result = task_executor.search_memories(
            query="   ",
            agent_id="test_agent",
        )

        assert result == []

    def test_search_no_candidates_returns_empty(
        self, task_executor: TaskExecutor, mock_vector_search: MagicMock
    ):
        """VectorSearch が空リストを返す場合は空リストを返す"""
        mock_vector_search.search_candidates.return_value = []

        result = task_executor.search_memories(
            query="存在しないキーワード",
            agent_id="test_agent",
        )

        assert result == []

    def test_search_no_candidates_skips_reinforcement(
        self,
        task_executor: TaskExecutor,
        mock_vector_search: MagicMock,
        mock_strength_manager: MagicMock,
    ):
        """検索結果が空の場合は候補強化をスキップ"""
        mock_vector_search.search_candidates.return_value = []

        task_executor.search_memories(
            query="存在しないキーワード",
            agent_id="test_agent",
        )

        # mark_as_candidate が呼ばれないことを確認
        mock_strength_manager.mark_as_candidate.assert_not_called()

    def test_search_ranker_empty_result_returns_empty(
        self,
        task_executor: TaskExecutor,
        mock_ranker: MagicMock,
    ):
        """ランカーが空リストを返す場合は空リストを返す"""
        mock_ranker.rank.return_value = []

        result = task_executor.search_memories(
            query="緊急調達のコスト",
            agent_id="test_agent",
        )

        assert result == []

    def test_search_ranker_empty_result_skips_reinforcement(
        self,
        task_executor: TaskExecutor,
        mock_ranker: MagicMock,
        mock_strength_manager: MagicMock,
    ):
        """ランカーが空リストを返す場合は候補強化をスキップ"""
        mock_ranker.rank.return_value = []

        task_executor.search_memories(
            query="緊急調達のコスト",
            agent_id="test_agent",
        )

        # mark_as_candidate が呼ばれないことを確認
        mock_strength_manager.mark_as_candidate.assert_not_called()

    def test_search_perspective_passed_to_all_components(
        self,
        task_executor: TaskExecutor,
        mock_vector_search: MagicMock,
        mock_ranker: MagicMock,
    ):
        """perspective パラメータが VectorSearch と MemoryRanker の両方に渡される"""
        task_executor.search_memories(
            query="緊急調達のコスト",
            agent_id="test_agent",
            perspective="納期",
        )

        # VectorSearch への perspective 引数
        vs_call_args = mock_vector_search.search_candidates.call_args
        assert vs_call_args.kwargs.get("perspective") == "納期"

        # MemoryRanker への perspective 引数
        ranker_call_args = mock_ranker.rank.call_args
        assert ranker_call_args.kwargs.get("perspective") == "納期"

    def test_search_perspective_none_by_default(
        self,
        task_executor: TaskExecutor,
        mock_vector_search: MagicMock,
        mock_ranker: MagicMock,
    ):
        """perspective を指定しない場合は None が渡される"""
        task_executor.search_memories(
            query="緊急調達のコスト",
            agent_id="test_agent",
        )

        # VectorSearch への perspective 引数
        vs_call_args = mock_vector_search.search_candidates.call_args
        assert vs_call_args.kwargs.get("perspective") is None

        # MemoryRanker への perspective 引数
        ranker_call_args = mock_ranker.rank.call_args
        assert ranker_call_args.kwargs.get("perspective") is None


class TestSearchMemoriesFlowIntegrity:
    """search_memories の処理フローの整合性テスト"""

    @pytest.fixture
    def config(self) -> Phase1Config:
        return Phase1Config()

    @pytest.fixture
    def sample_memory(self) -> AgentMemory:
        now = datetime.now()
        return AgentMemory(
            id=uuid4(),
            agent_id="test_agent",
            content="テスト用の記憶",
            strength=1.0,
            created_at=now,
            updated_at=now,
            last_accessed_at=now,
            candidate_count=0,
        )

    def test_flow_order_vector_search_then_rank_then_reinforce(
        self, config: Phase1Config, sample_memory: AgentMemory
    ):
        """処理フロー: VectorSearch → Ranker → StrengthManager の順序を確認"""
        call_order = []

        # モックを作成し、呼び出し順序を記録
        mock_vector_search = MagicMock()
        mock_vector_search.search_candidates.side_effect = lambda **kwargs: (
            call_order.append("vector_search"),
            [(sample_memory, 0.8)],
        )[1]

        mock_ranker = MagicMock()
        mock_ranker.rank.side_effect = lambda **kwargs: (
            call_order.append("ranker"),
            [
                ScoredMemory(
                    memory=sample_memory,
                    similarity=0.8,
                    final_score=0.7,
                    score_breakdown={},
                )
            ],
        )[1]

        mock_strength_manager = MagicMock()
        mock_strength_manager.mark_as_candidate.side_effect = lambda ids: (
            call_order.append("strength_manager"),
            len(ids),
        )[1]

        executor = TaskExecutor(
            vector_search=mock_vector_search,
            ranker=mock_ranker,
            strength_manager=mock_strength_manager,
            sleep_processor=MagicMock(),
            repository=MagicMock(),
            config=config,
        )

        executor.search_memories(query="テスト", agent_id="test_agent")

        # 呼び出し順序を検証
        assert call_order == ["vector_search", "ranker", "strength_manager"]

    def test_only_ranked_memories_are_reinforced(
        self, config: Phase1Config
    ):
        """ランキング後の結果のみが候補強化される（フィルタ前の全候補ではない）"""
        now = datetime.now()

        # 5件の候補を作成
        all_memories = [
            AgentMemory(
                id=uuid4(),
                agent_id="test_agent",
                content=f"Memory {i}",
                strength=1.0,
                created_at=now,
                updated_at=now,
                last_accessed_at=now,
            )
            for i in range(5)
        ]

        # VectorSearch は5件返す
        mock_vector_search = MagicMock()
        mock_vector_search.search_candidates.return_value = [
            (mem, 0.8 - i * 0.1) for i, mem in enumerate(all_memories)
        ]

        # Ranker は上位3件のみ返す（top_k制限）
        mock_ranker = MagicMock()
        mock_ranker.rank.return_value = [
            ScoredMemory(
                memory=all_memories[i],
                similarity=0.8 - i * 0.1,
                final_score=0.7 - i * 0.1,
                score_breakdown={},
            )
            for i in range(3)
        ]

        mock_strength_manager = MagicMock()
        mock_strength_manager.mark_as_candidate.return_value = 3

        executor = TaskExecutor(
            vector_search=mock_vector_search,
            ranker=mock_ranker,
            strength_manager=mock_strength_manager,
            sleep_processor=MagicMock(),
            repository=MagicMock(),
            config=config,
        )

        result = executor.search_memories(query="テスト", agent_id="test_agent")

        # 結果は3件
        assert len(result) == 3

        # mark_as_candidate に渡されたIDは3件のみ
        call_args = mock_strength_manager.mark_as_candidate.call_args
        memory_ids = call_args.args[0]
        assert len(memory_ids) == 3
        # 上位3件のIDと一致
        for i in range(3):
            assert memory_ids[i] == all_memories[i].id


class TestSearchMemoriesErrorHandling:
    """search_memories のエラーハンドリングテスト"""

    @pytest.fixture
    def config(self) -> Phase1Config:
        return Phase1Config()

    def test_vector_search_error_propagates(self, config: Phase1Config):
        """VectorSearch のエラーは伝播する"""
        from src.search.vector_search import VectorSearchError

        mock_vector_search = MagicMock()
        mock_vector_search.search_candidates.side_effect = VectorSearchError(
            "Embedding API エラー"
        )

        executor = TaskExecutor(
            vector_search=mock_vector_search,
            ranker=MagicMock(),
            strength_manager=MagicMock(),
            sleep_processor=MagicMock(),
            repository=MagicMock(),
            config=config,
        )

        with pytest.raises(VectorSearchError) as exc_info:
            executor.search_memories(query="テスト", agent_id="test_agent")

        assert "Embedding API エラー" in str(exc_info.value)


# =============================================================================
# keyword 方式（identify_used_memories）のテスト
# =============================================================================


class TestExtractKeywords:
    """_extract_keywords メソッドのテスト"""

    @pytest.fixture
    def task_executor(self) -> TaskExecutor:
        """テスト用の TaskExecutor（最小構成）"""
        return TaskExecutor(
            vector_search=MagicMock(),
            ranker=MagicMock(),
            strength_manager=MagicMock(),
            sleep_processor=MagicMock(),
            repository=MagicMock(),
            config=Phase1Config(),
        )

    def test_extract_basic_words(self, task_executor: TaskExecutor):
        """基本的な単語分割"""
        result = task_executor._extract_keywords("hello world test")
        assert result == ["hello", "world", "test"]

    def test_extract_removes_stopwords(self, task_executor: TaskExecutor):
        """ストップワードが除外される"""
        result = task_executor._extract_keywords("the quick brown fox is a test")
        assert "the" not in result
        assert "is" not in result
        assert "quick" in result
        assert "brown" in result
        assert "fox" in result
        assert "test" in result

    def test_extract_removes_short_words(self, task_executor: TaskExecutor):
        """3文字未満の単語が除外される"""
        result = task_executor._extract_keywords("ab is cd testing ok")
        assert "ab" not in result
        assert "cd" not in result
        assert "ok" not in result
        assert "testing" in result

    def test_extract_case_insensitive(self, task_executor: TaskExecutor):
        """大文字小文字を区別せず小文字に変換"""
        result = task_executor._extract_keywords("Hello WORLD Test")
        assert result == ["hello", "world", "test"]

    def test_extract_with_punctuation(self, task_executor: TaskExecutor):
        """句読点が区切り文字として扱われる"""
        result = task_executor._extract_keywords("hello, world! testing? great.")
        assert "hello" in result
        assert "world" in result
        assert "testing" in result
        assert "great" in result
        # "yes" はストップワードに含まれるため除外される

    def test_extract_empty_content(self, task_executor: TaskExecutor):
        """空のコンテンツには空リストを返す"""
        assert task_executor._extract_keywords("") == []
        assert task_executor._extract_keywords("   ") == []
        assert task_executor._extract_keywords(None) == []

    def test_extract_removes_duplicates(self, task_executor: TaskExecutor):
        """重複が除去される"""
        result = task_executor._extract_keywords("test hello test world test")
        assert result.count("test") == 1
        assert result == ["test", "hello", "world"]

    def test_extract_with_japanese(self, task_executor: TaskExecutor):
        """日本語を含むテキスト（Phase 1 では分割されない）"""
        result = task_executor._extract_keywords("緊急調達 コスト cost 15%")
        # 日本語は分割されないが、英語は抽出される
        assert "cost" in result
        # 日本語は1つの「単語」として残る
        assert "緊急調達" in result
        assert "コスト" in result

    def test_extract_with_numbers(self, task_executor: TaskExecutor):
        """数字を含むテキスト"""
        result = task_executor._extract_keywords("test123 abc456 789")
        assert "test123" in result
        assert "abc456" in result
        assert "789" in result


class TestMatchesAnyKeyword:
    """_matches_any_keyword メソッドのテスト"""

    @pytest.fixture
    def task_executor(self) -> TaskExecutor:
        """テスト用の TaskExecutor（最小構成）"""
        return TaskExecutor(
            vector_search=MagicMock(),
            ranker=MagicMock(),
            strength_manager=MagicMock(),
            sleep_processor=MagicMock(),
            repository=MagicMock(),
            config=Phase1Config(),
        )

    def test_matches_exact_keyword(self, task_executor: TaskExecutor):
        """完全一致"""
        assert task_executor._matches_any_keyword(
            "This is a test", ["test"]
        ) is True

    def test_matches_partial_keyword(self, task_executor: TaskExecutor):
        """部分一致（キーワードがテキスト内に含まれる）"""
        assert task_executor._matches_any_keyword(
            "testing framework", ["test"]
        ) is True

    def test_matches_case_insensitive(self, task_executor: TaskExecutor):
        """大文字小文字を区別しない"""
        assert task_executor._matches_any_keyword(
            "This is a TEST", ["test"]
        ) is True
        assert task_executor._matches_any_keyword(
            "This is a test", ["TEST"]
        ) is True

    def test_matches_multiple_keywords(self, task_executor: TaskExecutor):
        """複数のキーワードのいずれかが含まれる"""
        assert task_executor._matches_any_keyword(
            "hello world", ["foo", "bar", "world"]
        ) is True

    def test_no_match(self, task_executor: TaskExecutor):
        """マッチしない場合"""
        assert task_executor._matches_any_keyword(
            "hello world", ["foo", "bar", "baz"]
        ) is False

    def test_empty_text(self, task_executor: TaskExecutor):
        """空のテキスト"""
        assert task_executor._matches_any_keyword(
            "", ["test"]
        ) is False

    def test_empty_keywords(self, task_executor: TaskExecutor):
        """空のキーワードリスト"""
        assert task_executor._matches_any_keyword(
            "hello world", []
        ) is False

    def test_none_text(self, task_executor: TaskExecutor):
        """None のテキスト"""
        assert task_executor._matches_any_keyword(
            None, ["test"]
        ) is False


class TestIdentifyUsedMemories:
    """identify_used_memories メソッドのテスト"""

    @pytest.fixture
    def config(self) -> Phase1Config:
        return Phase1Config()

    @pytest.fixture
    def sample_memories(self) -> List[AgentMemory]:
        """テスト用のメモリリスト（英語ベース - Phase 1 は日本語形態素解析未対応）"""
        now = datetime.now()
        return [
            AgentMemory(
                id=uuid4(),
                agent_id="test_agent",
                content="Emergency procurement requires additional cost increase of 15%",
                strength=1.0,
                created_at=now,
                updated_at=now,
                last_accessed_at=now,
            ),
            AgentMemory(
                id=uuid4(),
                agent_id="test_agent",
                content="Supplier Y has single location risk assessment needed",
                strength=0.8,
                created_at=now,
                updated_at=now,
                last_accessed_at=now,
            ),
            AgentMemory(
                id=uuid4(),
                agent_id="test_agent",
                content="Delivery schedule reduction incurs additional expenses",
                strength=0.9,
                created_at=now,
                updated_at=now,
                last_accessed_at=now,
            ),
        ]

    @pytest.fixture
    def scored_memories(
        self, sample_memories: List[AgentMemory]
    ) -> List[ScoredMemory]:
        """テスト用の ScoredMemory リスト"""
        return [
            ScoredMemory(
                memory=mem,
                similarity=0.8 - i * 0.1,
                final_score=0.7 - i * 0.1,
                score_breakdown={},
            )
            for i, mem in enumerate(sample_memories)
        ]

    @pytest.fixture
    def task_executor(self, config: Phase1Config) -> TaskExecutor:
        """テスト用の TaskExecutor"""
        return TaskExecutor(
            vector_search=MagicMock(),
            ranker=MagicMock(),
            strength_manager=MagicMock(),
            sleep_processor=MagicMock(),
            repository=MagicMock(),
            config=config,
        )

    def test_identify_basic_matching(
        self,
        task_executor: TaskExecutor,
        scored_memories: List[ScoredMemory],
    ):
        """基本的なマッチング"""
        # タスク結果に "procurement" "cost" "additional" が含まれる
        task_result = "Emergency procurement analysis shows additional cost increase"

        used_ids = task_executor.identify_used_memories(
            task_result=task_result,
            candidates=scored_memories,
        )

        # "procurement" "additional" "cost" を含む最初のメモリがマッチ
        assert scored_memories[0].memory.id in used_ids
        assert len(used_ids) >= 1

    def test_identify_no_match(
        self,
        task_executor: TaskExecutor,
        scored_memories: List[ScoredMemory],
    ):
        """マッチしない場合"""
        task_result = "Completely unrelated content. The weather is nice today."

        used_ids = task_executor.identify_used_memories(
            task_result=task_result,
            candidates=scored_memories,
        )

        assert used_ids == []

    def test_identify_empty_task_result(
        self,
        task_executor: TaskExecutor,
        scored_memories: List[ScoredMemory],
    ):
        """空の task_result"""
        used_ids = task_executor.identify_used_memories(
            task_result="",
            candidates=scored_memories,
        )

        assert used_ids == []

    def test_identify_none_task_result(
        self,
        task_executor: TaskExecutor,
        scored_memories: List[ScoredMemory],
    ):
        """None の task_result"""
        used_ids = task_executor.identify_used_memories(
            task_result=None,
            candidates=scored_memories,
        )

        assert used_ids == []

    def test_identify_empty_candidates(
        self,
        task_executor: TaskExecutor,
    ):
        """空の candidates"""
        used_ids = task_executor.identify_used_memories(
            task_result="test result",
            candidates=[],
        )

        assert used_ids == []

    def test_identify_multiple_matches(
        self,
        task_executor: TaskExecutor,
        scored_memories: List[ScoredMemory],
    ):
        """複数候補からの選択（複数がマッチ）"""
        # 複数のメモリに含まれるキーワードを含む結果
        task_result = "Emergency procurement with additional cost. Supplier risk assessment needed."

        used_ids = task_executor.identify_used_memories(
            task_result=task_result,
            candidates=scored_memories,
        )

        # 複数のメモリがマッチするはず
        assert len(used_ids) >= 2
        assert scored_memories[0].memory.id in used_ids  # procurement/additional/cost
        assert scored_memories[1].memory.id in used_ids  # supplier/risk/assessment

    def test_identify_case_insensitive(
        self,
        task_executor: TaskExecutor,
    ):
        """大文字小文字を区別しない"""
        now = datetime.now()
        memory = AgentMemory(
            id=uuid4(),
            agent_id="test_agent",
            content="The PROJECT requires DATABASE optimization",
            strength=1.0,
            created_at=now,
            updated_at=now,
            last_accessed_at=now,
        )
        scored = [
            ScoredMemory(
                memory=memory,
                similarity=0.8,
                final_score=0.7,
                score_breakdown={},
            )
        ]

        # 小文字でタスク結果
        task_result = "We optimized the database for the project."

        used_ids = task_executor.identify_used_memories(
            task_result=task_result,
            candidates=scored,
        )

        assert memory.id in used_ids

    def test_identify_returns_uuid_list(
        self,
        task_executor: TaskExecutor,
        scored_memories: List[ScoredMemory],
    ):
        """戻り値が UUID のリストである"""
        task_result = "Cost analysis completed for procurement"

        used_ids = task_executor.identify_used_memories(
            task_result=task_result,
            candidates=scored_memories,
        )

        # 型チェック
        assert isinstance(used_ids, list)
        for uid in used_ids:
            assert isinstance(uid, UUID)


# =============================================================================
# reinforce_used_memories（2段階強化 Stage 2）のテスト
# =============================================================================


class TestReinforceUsedMemories:
    """reinforce_used_memories メソッドのテスト

    2段階強化の Stage 2（使用強化）の単体テスト。
    - 各 memory_id に対して StrengthManager.mark_as_used() を呼び出す
    - 部分的な失敗時も継続して処理
    - 成功数を返却
    """

    @pytest.fixture
    def config(self) -> Phase1Config:
        """テスト用の設定"""
        return Phase1Config()

    @pytest.fixture
    def sample_memories(self) -> List[AgentMemory]:
        """テスト用のメモリリスト"""
        now = datetime.now()
        return [
            AgentMemory(
                id=uuid4(),
                agent_id="test_agent",
                content=f"Test memory content {i}",
                strength=1.0,
                access_count=0,
                candidate_count=1,
                created_at=now,
                updated_at=now,
                last_accessed_at=now,
            )
            for i in range(3)
        ]

    @pytest.fixture
    def mock_strength_manager(self, sample_memories: List[AgentMemory]) -> MagicMock:
        """モック StrengthManager"""
        mock = MagicMock()

        # mark_as_used は更新後の AgentMemory を返す
        def mark_as_used_side_effect(memory_id: UUID, perspective: Optional[str] = None):
            for mem in sample_memories:
                if mem.id == memory_id:
                    # 更新後のメモリを返す（strength と access_count を更新）
                    return AgentMemory(
                        id=mem.id,
                        agent_id=mem.agent_id,
                        content=mem.content,
                        strength=mem.strength + 0.1,  # strength_increment_on_use
                        access_count=mem.access_count + 1,
                        candidate_count=mem.candidate_count,
                        created_at=mem.created_at,
                        updated_at=datetime.now(),
                        last_accessed_at=datetime.now(),
                    )
            return None  # メモリが存在しない場合

        mock.mark_as_used.side_effect = mark_as_used_side_effect
        return mock

    @pytest.fixture
    def task_executor(
        self,
        mock_strength_manager: MagicMock,
        config: Phase1Config,
    ) -> TaskExecutor:
        """テスト用の TaskExecutor"""
        return TaskExecutor(
            vector_search=MagicMock(),
            ranker=MagicMock(),
            strength_manager=mock_strength_manager,
            sleep_processor=MagicMock(),
            repository=MagicMock(),
            config=config,
        )

    def test_reinforce_single_memory(
        self,
        task_executor: TaskExecutor,
        sample_memories: List[AgentMemory],
        mock_strength_manager: MagicMock,
    ):
        """単一メモリの強化"""
        memory_ids = [sample_memories[0].id]

        success_count = task_executor.reinforce_used_memories(
            memory_ids=memory_ids,
            agent_id="test_agent",
        )

        assert success_count == 1
        mock_strength_manager.mark_as_used.assert_called_once_with(
            memory_id=sample_memories[0].id,
            perspective=None,
        )

    def test_reinforce_multiple_memories(
        self,
        task_executor: TaskExecutor,
        sample_memories: List[AgentMemory],
        mock_strength_manager: MagicMock,
    ):
        """複数メモリの強化"""
        memory_ids = [mem.id for mem in sample_memories]

        success_count = task_executor.reinforce_used_memories(
            memory_ids=memory_ids,
            agent_id="test_agent",
        )

        assert success_count == 3
        assert mock_strength_manager.mark_as_used.call_count == 3

    def test_reinforce_with_perspective(
        self,
        task_executor: TaskExecutor,
        sample_memories: List[AgentMemory],
        mock_strength_manager: MagicMock,
    ):
        """perspective パラメータが mark_as_used に渡される"""
        memory_ids = [sample_memories[0].id]

        task_executor.reinforce_used_memories(
            memory_ids=memory_ids,
            agent_id="test_agent",
            perspective="コスト",
        )

        mock_strength_manager.mark_as_used.assert_called_once_with(
            memory_id=sample_memories[0].id,
            perspective="コスト",
        )

    def test_reinforce_empty_list(
        self,
        task_executor: TaskExecutor,
        mock_strength_manager: MagicMock,
    ):
        """空リストの場合は 0 を返し、mark_as_used は呼ばれない"""
        success_count = task_executor.reinforce_used_memories(
            memory_ids=[],
            agent_id="test_agent",
        )

        assert success_count == 0
        mock_strength_manager.mark_as_used.assert_not_called()

    def test_reinforce_partial_failure_nonexistent_memory(
        self,
        task_executor: TaskExecutor,
        sample_memories: List[AgentMemory],
        mock_strength_manager: MagicMock,
    ):
        """存在しないメモリIDが含まれる場合も継続して処理"""
        nonexistent_id = uuid4()  # 存在しないID
        memory_ids = [
            sample_memories[0].id,
            nonexistent_id,
            sample_memories[1].id,
        ]

        success_count = task_executor.reinforce_used_memories(
            memory_ids=memory_ids,
            agent_id="test_agent",
        )

        # 存在するメモリ2件のみ成功
        assert success_count == 2
        # 3回呼ばれる（失敗も含む）
        assert mock_strength_manager.mark_as_used.call_count == 3

    def test_reinforce_partial_failure_exception(
        self,
        task_executor: TaskExecutor,
        sample_memories: List[AgentMemory],
    ):
        """例外が発生しても継続して処理"""
        call_count = 0

        def mark_as_used_with_exception(memory_id: UUID, perspective=None):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                # 2回目の呼び出しで例外
                raise Exception("Database connection error")
            # 他は成功
            for mem in sample_memories:
                if mem.id == memory_id:
                    return AgentMemory(
                        id=mem.id,
                        agent_id=mem.agent_id,
                        content=mem.content,
                        strength=mem.strength + 0.1,
                        access_count=mem.access_count + 1,
                        candidate_count=mem.candidate_count,
                        created_at=mem.created_at,
                        updated_at=datetime.now(),
                        last_accessed_at=datetime.now(),
                    )
            return None

        mock_strength_manager = MagicMock()
        mock_strength_manager.mark_as_used.side_effect = mark_as_used_with_exception

        executor = TaskExecutor(
            vector_search=MagicMock(),
            ranker=MagicMock(),
            strength_manager=mock_strength_manager,
            sleep_processor=MagicMock(),
            repository=MagicMock(),
            config=Phase1Config(),
        )

        memory_ids = [mem.id for mem in sample_memories]

        success_count = executor.reinforce_used_memories(
            memory_ids=memory_ids,
            agent_id="test_agent",
        )

        # 2件成功（1件は例外で失敗）
        assert success_count == 2
        # 3回すべて呼ばれる
        assert mock_strength_manager.mark_as_used.call_count == 3

    def test_reinforce_returns_success_count(
        self,
        task_executor: TaskExecutor,
        sample_memories: List[AgentMemory],
    ):
        """戻り値が成功した強化の数である"""
        memory_ids = [sample_memories[0].id, sample_memories[1].id]

        result = task_executor.reinforce_used_memories(
            memory_ids=memory_ids,
            agent_id="test_agent",
        )

        assert isinstance(result, int)
        assert result == 2

    def test_reinforce_perspective_passed_to_all_memories(
        self,
        task_executor: TaskExecutor,
        sample_memories: List[AgentMemory],
        mock_strength_manager: MagicMock,
    ):
        """perspective が全メモリの強化に渡される"""
        memory_ids = [mem.id for mem in sample_memories]

        task_executor.reinforce_used_memories(
            memory_ids=memory_ids,
            agent_id="test_agent",
            perspective="テスト観点",
        )

        # 全呼び出しで同じ perspective が渡されることを確認
        for call in mock_strength_manager.mark_as_used.call_args_list:
            assert call.kwargs.get("perspective") == "テスト観点"


class TestReinforceUsedMemoriesFlowIntegrity:
    """reinforce_used_memories のフロー整合性テスト"""

    @pytest.fixture
    def config(self) -> Phase1Config:
        return Phase1Config()

    def test_reinforce_order_preserved(self, config: Phase1Config):
        """メモリの処理順序が保持される"""
        now = datetime.now()
        memories = [
            AgentMemory(
                id=uuid4(),
                agent_id="test_agent",
                content=f"Memory {i}",
                strength=1.0,
                access_count=0,
                created_at=now,
                updated_at=now,
                last_accessed_at=now,
            )
            for i in range(3)
        ]

        call_order: List[UUID] = []

        def track_calls(memory_id: UUID, perspective=None):
            call_order.append(memory_id)
            return memories[0]  # 任意の有効なメモリを返す

        mock_strength_manager = MagicMock()
        mock_strength_manager.mark_as_used.side_effect = track_calls

        executor = TaskExecutor(
            vector_search=MagicMock(),
            ranker=MagicMock(),
            strength_manager=mock_strength_manager,
            sleep_processor=MagicMock(),
            repository=MagicMock(),
            config=config,
        )

        memory_ids = [mem.id for mem in memories]
        executor.reinforce_used_memories(
            memory_ids=memory_ids,
            agent_id="test_agent",
        )

        # 呼び出し順序が入力順と一致
        assert call_order == memory_ids


# =============================================================================
# record_learning（学び記録）のテスト
# =============================================================================


class TestRecordLearning:
    """record_learning メソッドのテスト

    学び記録フローの単体テスト:
    - content の空チェック
    - エンベディング生成
    - learnings 構造の作成（perspective あり/なし）
    - strength_by_perspective の初期化
    - メモリの作成と保存
    - UUID の返却
    """

    @pytest.fixture
    def config(self) -> Phase1Config:
        """テスト用の設定"""
        return Phase1Config()

    @pytest.fixture
    def mock_embedding_client(self) -> MagicMock:
        """モック AzureEmbeddingClient"""
        mock = MagicMock()
        # 1536次元のダミーエンベディングを返す
        mock.get_embedding.return_value = [0.1] * 1536
        return mock

    @pytest.fixture
    def mock_vector_search(self, mock_embedding_client: MagicMock) -> MagicMock:
        """モック VectorSearch（embedding_client を保持）"""
        mock = MagicMock()
        mock.embedding_client = mock_embedding_client
        return mock

    @pytest.fixture
    def mock_repository(self) -> MagicMock:
        """モック MemoryRepository"""
        mock = MagicMock()

        # create は作成されたメモリを返す
        def create_side_effect(memory: AgentMemory) -> AgentMemory:
            return memory

        mock.create.side_effect = create_side_effect
        return mock

    @pytest.fixture
    def task_executor(
        self,
        mock_vector_search: MagicMock,
        mock_repository: MagicMock,
        config: Phase1Config,
    ) -> TaskExecutor:
        """テスト用の TaskExecutor"""
        return TaskExecutor(
            vector_search=mock_vector_search,
            ranker=MagicMock(),
            strength_manager=MagicMock(),
            sleep_processor=MagicMock(),
            repository=mock_repository,
            config=config,
        )

    def test_record_basic_learning(
        self,
        task_executor: TaskExecutor,
        mock_repository: MagicMock,
    ):
        """基本的な学び記録"""
        memory_id = task_executor.record_learning(
            agent_id="test_agent",
            content="緊急調達では15%のコスト増を見込む",
            learning="コスト増の具体的な割合を把握",
            perspective="コスト",
        )

        # UUID が返される
        assert isinstance(memory_id, UUID)

        # repository.create が呼ばれた
        mock_repository.create.assert_called_once()

    def test_record_with_perspective_creates_learnings_dict(
        self,
        task_executor: TaskExecutor,
        mock_repository: MagicMock,
    ):
        """perspective 指定時の learnings 構造"""
        task_executor.record_learning(
            agent_id="test_agent",
            content="納期は2週間バッファが必要",
            learning="バッファ期間の重要性",
            perspective="納期",
        )

        # create に渡された AgentMemory を検証
        call_args = mock_repository.create.call_args
        created_memory: AgentMemory = call_args.args[0]

        # learnings が {perspective: learning} の形式
        assert created_memory.learnings == {"納期": "バッファ期間の重要性"}

    def test_record_without_perspective_uses_general(
        self,
        task_executor: TaskExecutor,
        mock_repository: MagicMock,
    ):
        """perspective なしの場合は "general" を使用"""
        task_executor.record_learning(
            agent_id="test_agent",
            content="一般的な学び内容",
            learning="特定の観点に依存しない学び",
        )

        # create に渡された AgentMemory を検証
        call_args = mock_repository.create.call_args
        created_memory: AgentMemory = call_args.args[0]

        # learnings が {"general": learning} の形式
        assert created_memory.learnings == {"general": "特定の観点に依存しない学び"}

    def test_record_with_perspective_initializes_strength_by_perspective(
        self,
        task_executor: TaskExecutor,
        mock_repository: MagicMock,
    ):
        """perspective 指定時は strength_by_perspective が初期化される"""
        task_executor.record_learning(
            agent_id="test_agent",
            content="コスト関連の学び",
            learning="コストに関する重要な知見",
            perspective="コスト",
        )

        # create に渡された AgentMemory を検証
        call_args = mock_repository.create.call_args
        created_memory: AgentMemory = call_args.args[0]

        # strength_by_perspective に perspective が 1.0 で設定
        assert created_memory.strength_by_perspective == {"コスト": 1.0}

    def test_record_without_perspective_empty_strength_by_perspective(
        self,
        task_executor: TaskExecutor,
        mock_repository: MagicMock,
    ):
        """perspective なしの場合は strength_by_perspective が空"""
        task_executor.record_learning(
            agent_id="test_agent",
            content="一般的な学び",
            learning="観点なしの学び",
        )

        # create に渡された AgentMemory を検証
        call_args = mock_repository.create.call_args
        created_memory: AgentMemory = call_args.args[0]

        # strength_by_perspective が空
        assert created_memory.strength_by_perspective == {}

    def test_record_calls_embedding_client(
        self,
        task_executor: TaskExecutor,
        mock_embedding_client: MagicMock,
    ):
        """エンベディング生成が呼ばれる"""
        content = "テスト用のコンテンツ"

        task_executor.record_learning(
            agent_id="test_agent",
            content=content,
            learning="テスト用の学び",
        )

        # embedding_client.get_embedding が content で呼ばれた
        mock_embedding_client.get_embedding.assert_called_once_with(content)

    def test_record_sets_embedding_in_memory(
        self,
        task_executor: TaskExecutor,
        mock_repository: MagicMock,
        mock_embedding_client: MagicMock,
    ):
        """エンベディングがメモリに設定される"""
        expected_embedding = [0.5] * 1536
        mock_embedding_client.get_embedding.return_value = expected_embedding

        task_executor.record_learning(
            agent_id="test_agent",
            content="テスト",
            learning="学び",
        )

        # create に渡された AgentMemory を検証
        call_args = mock_repository.create.call_args
        created_memory: AgentMemory = call_args.args[0]

        assert created_memory.embedding == expected_embedding

    def test_record_sets_source_as_task(
        self,
        task_executor: TaskExecutor,
        mock_repository: MagicMock,
    ):
        """source が "task" に設定される"""
        task_executor.record_learning(
            agent_id="test_agent",
            content="テスト",
            learning="学び",
        )

        # create に渡された AgentMemory を検証
        call_args = mock_repository.create.call_args
        created_memory: AgentMemory = call_args.args[0]

        assert created_memory.source == "task"

    def test_record_sets_initial_strength(
        self,
        task_executor: TaskExecutor,
        mock_repository: MagicMock,
    ):
        """初期強度が 1.0 に設定される"""
        task_executor.record_learning(
            agent_id="test_agent",
            content="テスト",
            learning="学び",
        )

        # create に渡された AgentMemory を検証
        call_args = mock_repository.create.call_args
        created_memory: AgentMemory = call_args.args[0]

        assert created_memory.strength == 1.0

    def test_record_returns_uuid(
        self,
        task_executor: TaskExecutor,
    ):
        """戻り値が UUID である"""
        result = task_executor.record_learning(
            agent_id="test_agent",
            content="テスト",
            learning="学び",
        )

        assert isinstance(result, UUID)

    def test_record_empty_content_raises_value_error(
        self,
        task_executor: TaskExecutor,
    ):
        """空の content で ValueError が発生"""
        with pytest.raises(ValueError) as exc_info:
            task_executor.record_learning(
                agent_id="test_agent",
                content="",
                learning="学び",
            )

        assert "content を空にすることはできません" in str(exc_info.value)

    def test_record_whitespace_content_raises_value_error(
        self,
        task_executor: TaskExecutor,
    ):
        """空白のみの content で ValueError が発生"""
        with pytest.raises(ValueError) as exc_info:
            task_executor.record_learning(
                agent_id="test_agent",
                content="   ",
                learning="学び",
            )

        assert "content を空にすることはできません" in str(exc_info.value)

    def test_record_embedding_error_propagates(
        self,
        task_executor: TaskExecutor,
        mock_embedding_client: MagicMock,
    ):
        """エンベディング生成エラーは伝播する"""
        from src.embedding.azure_client import AzureEmbeddingError

        mock_embedding_client.get_embedding.side_effect = AzureEmbeddingError(
            "API エラー"
        )

        with pytest.raises(AzureEmbeddingError) as exc_info:
            task_executor.record_learning(
                agent_id="test_agent",
                content="テスト",
                learning="学び",
            )

        assert "API エラー" in str(exc_info.value)

    def test_record_sets_agent_id(
        self,
        task_executor: TaskExecutor,
        mock_repository: MagicMock,
    ):
        """agent_id がメモリに設定される"""
        task_executor.record_learning(
            agent_id="procurement_agent_01",
            content="テスト",
            learning="学び",
        )

        # create に渡された AgentMemory を検証
        call_args = mock_repository.create.call_args
        created_memory: AgentMemory = call_args.args[0]

        assert created_memory.agent_id == "procurement_agent_01"

    def test_record_sets_content(
        self,
        task_executor: TaskExecutor,
        mock_repository: MagicMock,
    ):
        """content がメモリに設定される"""
        content = "緊急調達のコスト増について"

        task_executor.record_learning(
            agent_id="test_agent",
            content=content,
            learning="学び",
        )

        # create に渡された AgentMemory を検証
        call_args = mock_repository.create.call_args
        created_memory: AgentMemory = call_args.args[0]

        assert created_memory.content == content


class TestRecordLearningIntegration:
    """record_learning の統合テスト"""

    @pytest.fixture
    def config(self) -> Phase1Config:
        return Phase1Config()

    def test_record_creates_complete_memory(self, config: Phase1Config):
        """完全なメモリオブジェクトが作成される"""
        mock_embedding_client = MagicMock()
        mock_embedding_client.get_embedding.return_value = [0.1] * 1536

        mock_vector_search = MagicMock()
        mock_vector_search.embedding_client = mock_embedding_client

        created_memories = []

        def capture_create(memory: AgentMemory) -> AgentMemory:
            created_memories.append(memory)
            return memory

        mock_repository = MagicMock()
        mock_repository.create.side_effect = capture_create

        executor = TaskExecutor(
            vector_search=mock_vector_search,
            ranker=MagicMock(),
            strength_manager=MagicMock(),
            sleep_processor=MagicMock(),
            repository=mock_repository,
            config=config,
        )

        memory_id = executor.record_learning(
            agent_id="test_agent",
            content="緊急調達では15%のコスト増を見込む",
            learning="コスト増の具体的な割合を把握",
            perspective="コスト",
        )

        # 作成されたメモリを検証
        assert len(created_memories) == 1
        memory = created_memories[0]

        assert memory.id == memory_id
        assert memory.agent_id == "test_agent"
        assert memory.content == "緊急調達では15%のコスト増を見込む"
        assert memory.embedding == [0.1] * 1536
        assert memory.learnings == {"コスト": "コスト増の具体的な割合を把握"}
        assert memory.strength_by_perspective == {"コスト": 1.0}
        assert memory.strength == 1.0
        assert memory.source == "task"
        assert memory.status == "active"
        assert memory.access_count == 0
        assert memory.candidate_count == 0


# =============================================================================
# execute_task（統合フロー）のテスト
# =============================================================================


class TestExecuteTask:
    """execute_task メソッドのテスト

    統合フロー（検索→タスク実行→使用判定→強化→学び記録）の単体テスト:
    - 各ステップが順序通りに呼ばれること
    - TaskExecutionResult の各フィールド確認
    - 部分的失敗時の継続動作（fail-soft）
    - エラー集約
    """

    @pytest.fixture
    def config(self) -> Phase1Config:
        """テスト用の設定"""
        return Phase1Config()

    @pytest.fixture
    def sample_memories(self) -> List[AgentMemory]:
        """テスト用のメモリリスト"""
        now = datetime.now()
        return [
            AgentMemory(
                id=uuid4(),
                agent_id="test_agent",
                content="Emergency procurement requires additional cost increase of 15%",
                strength=1.0,
                access_count=0,
                candidate_count=0,
                created_at=now,
                updated_at=now,
                last_accessed_at=now,
            ),
            AgentMemory(
                id=uuid4(),
                agent_id="test_agent",
                content="Supplier risk assessment needed for single location vendors",
                strength=0.8,
                access_count=0,
                candidate_count=0,
                created_at=now,
                updated_at=now,
                last_accessed_at=now,
            ),
        ]

    @pytest.fixture
    def scored_memories(
        self, sample_memories: List[AgentMemory]
    ) -> List[ScoredMemory]:
        """テスト用の ScoredMemory リスト"""
        return [
            ScoredMemory(
                memory=mem,
                similarity=0.85 - i * 0.1,
                final_score=0.75 - i * 0.1,
                score_breakdown={},
            )
            for i, mem in enumerate(sample_memories)
        ]

    @pytest.fixture
    def mock_vector_search(self, sample_memories: List[AgentMemory]) -> MagicMock:
        """モック VectorSearch"""
        mock = MagicMock()
        mock.search_candidates.return_value = [
            (sample_memories[0], 0.85),
            (sample_memories[1], 0.75),
        ]
        # embedding_client も設定（record_learning 用）
        mock.embedding_client = MagicMock()
        mock.embedding_client.get_embedding.return_value = [0.1] * 1536
        return mock

    @pytest.fixture
    def mock_ranker(self, scored_memories: List[ScoredMemory]) -> MagicMock:
        """モック MemoryRanker"""
        mock = MagicMock()
        mock.rank.return_value = scored_memories
        return mock

    @pytest.fixture
    def mock_strength_manager(self, sample_memories: List[AgentMemory]) -> MagicMock:
        """モック StrengthManager"""
        mock = MagicMock()
        mock.mark_as_candidate.return_value = 2

        def mark_as_used_side_effect(memory_id: UUID, perspective=None):
            for mem in sample_memories:
                if mem.id == memory_id:
                    return AgentMemory(
                        id=mem.id,
                        agent_id=mem.agent_id,
                        content=mem.content,
                        strength=mem.strength + 0.1,
                        access_count=mem.access_count + 1,
                        candidate_count=mem.candidate_count,
                        created_at=mem.created_at,
                        updated_at=datetime.now(),
                        last_accessed_at=datetime.now(),
                    )
            return None

        mock.mark_as_used.side_effect = mark_as_used_side_effect
        return mock

    @pytest.fixture
    def mock_repository(self) -> MagicMock:
        """モック MemoryRepository"""
        mock = MagicMock()
        mock.create.side_effect = lambda m: m
        return mock

    @pytest.fixture
    def task_executor(
        self,
        mock_vector_search: MagicMock,
        mock_ranker: MagicMock,
        mock_strength_manager: MagicMock,
        mock_repository: MagicMock,
        config: Phase1Config,
    ) -> TaskExecutor:
        """テスト用の TaskExecutor"""
        return TaskExecutor(
            vector_search=mock_vector_search,
            ranker=mock_ranker,
            strength_manager=mock_strength_manager,
            sleep_processor=MagicMock(),
            repository=mock_repository,
            config=config,
        )

    def test_execute_basic_flow(
        self,
        task_executor: TaskExecutor,
        scored_memories: List[ScoredMemory],
    ):
        """基本的なフロー実行"""
        # タスク関数：メモリの内容を使った結果を返す
        def task_func(memories: List[ScoredMemory]) -> str:
            return "Task completed with procurement cost analysis"

        result = task_executor.execute_task(
            query="procurement cost",
            agent_id="test_agent",
            task_func=task_func,
        )

        # 結果の検証
        assert result.task_result == "Task completed with procurement cost analysis"
        assert len(result.searched_memories) == 2
        assert len(result.errors) == 0
        assert result.recorded_memory_id is None  # 学び記録なし

    def test_execute_returns_task_execution_result(
        self,
        task_executor: TaskExecutor,
    ):
        """TaskExecutionResult が返される"""
        from src.core.task_executor import TaskExecutionResult

        def task_func(memories):
            return "result"

        result = task_executor.execute_task(
            query="test query",
            agent_id="test_agent",
            task_func=task_func,
        )

        assert isinstance(result, TaskExecutionResult)
        assert hasattr(result, "task_result")
        assert hasattr(result, "searched_memories")
        assert hasattr(result, "used_memory_ids")
        assert hasattr(result, "recorded_memory_id")
        assert hasattr(result, "executed_at")
        assert hasattr(result, "errors")

    def test_execute_identifies_used_memories(
        self,
        task_executor: TaskExecutor,
        scored_memories: List[ScoredMemory],
    ):
        """使用されたメモリが正しく判定される"""
        # タスク関数：メモリ内のキーワードを含む結果を返す
        def task_func(memories: List[ScoredMemory]) -> str:
            # "procurement" と "cost" を含む → 最初のメモリがマッチするはず
            return "We analyzed procurement and found additional cost increase"

        result = task_executor.execute_task(
            query="procurement cost",
            agent_id="test_agent",
            task_func=task_func,
        )

        # 使用されたメモリIDが含まれている
        assert len(result.used_memory_ids) >= 1
        assert scored_memories[0].memory.id in result.used_memory_ids

    def test_execute_reinforces_used_memories(
        self,
        task_executor: TaskExecutor,
        mock_strength_manager: MagicMock,
        scored_memories: List[ScoredMemory],
    ):
        """使用されたメモリが強化される"""
        def task_func(memories: List[ScoredMemory]) -> str:
            return "Procurement cost analysis completed"

        task_executor.execute_task(
            query="procurement cost",
            agent_id="test_agent",
            task_func=task_func,
        )

        # mark_as_used が呼ばれたことを確認
        assert mock_strength_manager.mark_as_used.called

    def test_execute_with_learning_content(
        self,
        task_executor: TaskExecutor,
        mock_repository: MagicMock,
    ):
        """学び記録が行われる（learning_content 指定時）"""
        def task_func(memories: List[ScoredMemory]) -> str:
            return "Task result"

        result = task_executor.execute_task(
            query="test query",
            agent_id="test_agent",
            task_func=task_func,
            learning_content="新しい学びの内容",
            learning_text="具体的な学び",
            perspective="テスト観点",
        )

        # 学び記録が行われた
        assert result.recorded_memory_id is not None
        mock_repository.create.assert_called_once()

    def test_execute_with_extract_learning(
        self,
        task_executor: TaskExecutor,
        mock_repository: MagicMock,
    ):
        """学び記録が行われる（extract_learning=True 時）"""
        def task_func(memories: List[ScoredMemory]) -> str:
            return "Important task result to learn from"

        result = task_executor.execute_task(
            query="test query",
            agent_id="test_agent",
            task_func=task_func,
            extract_learning=True,
        )

        # 学び記録が行われた
        assert result.recorded_memory_id is not None
        mock_repository.create.assert_called_once()

    def test_execute_without_learning(
        self,
        task_executor: TaskExecutor,
        mock_repository: MagicMock,
    ):
        """学び記録がスキップされる（extract_learning=False, learning_content なし）"""
        def task_func(memories: List[ScoredMemory]) -> str:
            return "Task result"

        result = task_executor.execute_task(
            query="test query",
            agent_id="test_agent",
            task_func=task_func,
        )

        # 学び記録が行われない
        assert result.recorded_memory_id is None
        mock_repository.create.assert_not_called()

    def test_execute_perspective_passed_to_components(
        self,
        task_executor: TaskExecutor,
        mock_vector_search: MagicMock,
        mock_ranker: MagicMock,
        mock_strength_manager: MagicMock,
    ):
        """perspective が各コンポーネントに渡される"""
        def task_func(memories: List[ScoredMemory]) -> str:
            return "Procurement cost analysis"

        task_executor.execute_task(
            query="cost analysis",
            agent_id="test_agent",
            task_func=task_func,
            perspective="コスト",
        )

        # VectorSearch への perspective
        vs_call = mock_vector_search.search_candidates.call_args
        assert vs_call.kwargs.get("perspective") == "コスト"

        # MemoryRanker への perspective
        ranker_call = mock_ranker.rank.call_args
        assert ranker_call.kwargs.get("perspective") == "コスト"

        # mark_as_used への perspective（呼ばれていれば）
        if mock_strength_manager.mark_as_used.called:
            for call in mock_strength_manager.mark_as_used.call_args_list:
                assert call.kwargs.get("perspective") == "コスト"


class TestExecuteTaskFlowOrder:
    """execute_task の処理フロー順序テスト"""

    @pytest.fixture
    def config(self) -> Phase1Config:
        return Phase1Config()

    @pytest.fixture
    def sample_memory(self) -> AgentMemory:
        now = datetime.now()
        return AgentMemory(
            id=uuid4(),
            agent_id="test_agent",
            content="Test memory content for flow verification",
            strength=1.0,
            access_count=0,
            candidate_count=0,
            created_at=now,
            updated_at=now,
            last_accessed_at=now,
        )

    def test_flow_order_step_by_step(
        self, config: Phase1Config, sample_memory: AgentMemory
    ):
        """処理フロー: 検索→タスク実行→使用判定→強化→結果返却 の順序確認"""
        call_order = []

        # モック作成
        mock_vector_search = MagicMock()
        mock_vector_search.search_candidates.side_effect = lambda **kwargs: (
            call_order.append("1_search_candidates"),
            [(sample_memory, 0.8)],
        )[1]
        mock_vector_search.embedding_client = MagicMock()

        mock_ranker = MagicMock()
        mock_ranker.rank.side_effect = lambda **kwargs: (
            call_order.append("2_rank"),
            [ScoredMemory(
                memory=sample_memory,
                similarity=0.8,
                final_score=0.7,
                score_breakdown={},
            )],
        )[1]

        mock_strength_manager = MagicMock()
        mock_strength_manager.mark_as_candidate.side_effect = lambda ids: (
            call_order.append("3_mark_as_candidate"),
            len(ids),
        )[1]
        mock_strength_manager.mark_as_used.side_effect = lambda **kwargs: (
            call_order.append("5_mark_as_used"),
            sample_memory,
        )[1]

        def task_func(memories):
            call_order.append("4_task_func")
            return "Test content with memory verification"

        executor = TaskExecutor(
            vector_search=mock_vector_search,
            ranker=mock_ranker,
            strength_manager=mock_strength_manager,
            sleep_processor=MagicMock(),
            repository=MagicMock(),
            config=config,
        )

        executor.execute_task(
            query="test",
            agent_id="test_agent",
            task_func=task_func,
        )

        # 呼び出し順序を検証
        assert call_order[0] == "1_search_candidates"
        assert call_order[1] == "2_rank"
        assert call_order[2] == "3_mark_as_candidate"
        assert call_order[3] == "4_task_func"
        # 使用判定後に強化が呼ばれる（使用メモリがある場合）
        if "5_mark_as_used" in call_order:
            assert call_order.index("5_mark_as_used") > call_order.index("4_task_func")


class TestExecuteTaskErrorHandling:
    """execute_task のエラーハンドリングテスト"""

    @pytest.fixture
    def config(self) -> Phase1Config:
        return Phase1Config()

    @pytest.fixture
    def sample_memory(self) -> AgentMemory:
        now = datetime.now()
        return AgentMemory(
            id=uuid4(),
            agent_id="test_agent",
            content="Test memory",
            strength=1.0,
            created_at=now,
            updated_at=now,
            last_accessed_at=now,
        )

    def test_search_error_propagates(self, config: Phase1Config):
        """検索エラーは例外として伝播する"""
        from src.search.vector_search import VectorSearchError

        mock_vector_search = MagicMock()
        mock_vector_search.search_candidates.side_effect = VectorSearchError(
            "Search failed"
        )

        executor = TaskExecutor(
            vector_search=mock_vector_search,
            ranker=MagicMock(),
            strength_manager=MagicMock(),
            sleep_processor=MagicMock(),
            repository=MagicMock(),
            config=config,
        )

        with pytest.raises(VectorSearchError):
            executor.execute_task(
                query="test",
                agent_id="test_agent",
                task_func=lambda m: "result",
            )

    def test_task_func_error_propagates(
        self, config: Phase1Config, sample_memory: AgentMemory
    ):
        """task_func のエラーは例外として伝播する"""
        mock_vector_search = MagicMock()
        mock_vector_search.search_candidates.return_value = [(sample_memory, 0.8)]

        mock_ranker = MagicMock()
        mock_ranker.rank.return_value = [
            ScoredMemory(
                memory=sample_memory,
                similarity=0.8,
                final_score=0.7,
                score_breakdown={},
            )
        ]

        executor = TaskExecutor(
            vector_search=mock_vector_search,
            ranker=mock_ranker,
            strength_manager=MagicMock(),
            sleep_processor=MagicMock(),
            repository=MagicMock(),
            config=config,
        )

        def failing_task(memories):
            raise ValueError("Task execution failed")

        with pytest.raises(ValueError) as exc_info:
            executor.execute_task(
                query="test",
                agent_id="test_agent",
                task_func=failing_task,
            )

        assert "Task execution failed" in str(exc_info.value)

    def test_reinforce_error_recorded_but_continues(
        self, config: Phase1Config, sample_memory: AgentMemory
    ):
        """強化エラーは記録されるが処理は継続する（fail-soft）"""
        mock_vector_search = MagicMock()
        mock_vector_search.search_candidates.return_value = [(sample_memory, 0.8)]
        mock_vector_search.embedding_client = MagicMock()

        mock_ranker = MagicMock()
        mock_ranker.rank.return_value = [
            ScoredMemory(
                memory=sample_memory,
                similarity=0.8,
                final_score=0.7,
                score_breakdown={},
            )
        ]

        mock_strength_manager = MagicMock()
        mock_strength_manager.mark_as_candidate.return_value = 1
        mock_strength_manager.mark_as_used.side_effect = Exception("DB error")

        executor = TaskExecutor(
            vector_search=mock_vector_search,
            ranker=mock_ranker,
            strength_manager=mock_strength_manager,
            sleep_processor=MagicMock(),
            repository=MagicMock(),
            config=config,
        )

        def task_func(memories):
            return "Test memory content"

        result = executor.execute_task(
            query="test",
            agent_id="test_agent",
            task_func=task_func,
        )

        # 処理は継続し、結果が返される
        assert result.task_result == "Test memory content"
        # エラーが記録されている
        assert len(result.errors) >= 1
        assert any("強化" in e or "reinforce" in e.lower() for e in result.errors)

    def test_learning_record_error_recorded_but_continues(
        self, config: Phase1Config, sample_memory: AgentMemory
    ):
        """学び記録エラーは記録されるが処理は継続する（fail-soft）"""
        mock_vector_search = MagicMock()
        mock_vector_search.search_candidates.return_value = [(sample_memory, 0.8)]
        mock_vector_search.embedding_client = MagicMock()
        mock_vector_search.embedding_client.get_embedding.side_effect = Exception(
            "Embedding API error"
        )

        mock_ranker = MagicMock()
        mock_ranker.rank.return_value = [
            ScoredMemory(
                memory=sample_memory,
                similarity=0.8,
                final_score=0.7,
                score_breakdown={},
            )
        ]

        mock_strength_manager = MagicMock()
        mock_strength_manager.mark_as_candidate.return_value = 1
        mock_strength_manager.mark_as_used.return_value = sample_memory

        executor = TaskExecutor(
            vector_search=mock_vector_search,
            ranker=mock_ranker,
            strength_manager=mock_strength_manager,
            sleep_processor=MagicMock(),
            repository=MagicMock(),
            config=config,
        )

        def task_func(memories):
            return "Task result"

        result = executor.execute_task(
            query="test",
            agent_id="test_agent",
            task_func=task_func,
            learning_content="学びの内容",
            learning_text="学びテキスト",
        )

        # 処理は継続し、結果が返される
        assert result.task_result == "Task result"
        # 学び記録は失敗している
        assert result.recorded_memory_id is None
        # エラーが記録されている
        assert len(result.errors) >= 1
        assert any("学び" in e for e in result.errors)


class TestExecuteTaskPartialFailure:
    """execute_task の部分的失敗テスト"""

    @pytest.fixture
    def config(self) -> Phase1Config:
        return Phase1Config()

    def test_empty_search_results_continues(self, config: Phase1Config):
        """検索結果が空でも処理は継続する"""
        mock_vector_search = MagicMock()
        mock_vector_search.search_candidates.return_value = []
        mock_vector_search.embedding_client = MagicMock()

        mock_ranker = MagicMock()
        mock_ranker.rank.return_value = []

        executor = TaskExecutor(
            vector_search=mock_vector_search,
            ranker=mock_ranker,
            strength_manager=MagicMock(),
            sleep_processor=MagicMock(),
            repository=MagicMock(),
            config=config,
        )

        def task_func(memories):
            return "Task completed without memories"

        result = executor.execute_task(
            query="test",
            agent_id="test_agent",
            task_func=task_func,
        )

        # 処理は継続
        assert result.task_result == "Task completed without memories"
        assert result.searched_memories == []
        assert result.used_memory_ids == []
        assert len(result.errors) == 0

    def test_no_used_memories_skips_reinforcement(self, config: Phase1Config):
        """使用メモリがない場合は強化をスキップする"""
        now = datetime.now()
        sample_memory = AgentMemory(
            id=uuid4(),
            agent_id="test_agent",
            content="Completely unrelated memory content",
            strength=1.0,
            created_at=now,
            updated_at=now,
            last_accessed_at=now,
        )

        mock_vector_search = MagicMock()
        mock_vector_search.search_candidates.return_value = [(sample_memory, 0.8)]

        mock_ranker = MagicMock()
        mock_ranker.rank.return_value = [
            ScoredMemory(
                memory=sample_memory,
                similarity=0.8,
                final_score=0.7,
                score_breakdown={},
            )
        ]

        mock_strength_manager = MagicMock()
        mock_strength_manager.mark_as_candidate.return_value = 1

        executor = TaskExecutor(
            vector_search=mock_vector_search,
            ranker=mock_ranker,
            strength_manager=mock_strength_manager,
            sleep_processor=MagicMock(),
            repository=MagicMock(),
            config=config,
        )

        def task_func(memories):
            # メモリのキーワードを全く含まない結果
            return "Weather is nice today"

        result = executor.execute_task(
            query="test",
            agent_id="test_agent",
            task_func=task_func,
        )

        # 使用メモリがない
        assert result.used_memory_ids == []
        # mark_as_used は呼ばれない
        mock_strength_manager.mark_as_used.assert_not_called()


class TestExtractLearningContent:
    """_extract_learning_content メソッドのテスト"""

    @pytest.fixture
    def task_executor(self) -> TaskExecutor:
        return TaskExecutor(
            vector_search=MagicMock(),
            ranker=MagicMock(),
            strength_manager=MagicMock(),
            sleep_processor=MagicMock(),
            repository=MagicMock(),
            config=Phase1Config(),
        )

    def test_extract_basic_string(self, task_executor: TaskExecutor):
        """基本的な文字列の抽出"""
        result = task_executor._extract_learning_content("Task result", "query")
        assert result == "Task result"

    def test_extract_none_returns_none(self, task_executor: TaskExecutor):
        """None の task_result には None を返す"""
        result = task_executor._extract_learning_content(None, "query")
        assert result is None

    def test_extract_empty_returns_none(self, task_executor: TaskExecutor):
        """空文字列には None を返す"""
        result = task_executor._extract_learning_content("", "query")
        assert result is None
        result = task_executor._extract_learning_content("   ", "query")
        assert result is None

    def test_extract_truncates_long_content(self, task_executor: TaskExecutor):
        """長すぎる content は切り詰められる"""
        long_result = "x" * 1000
        result = task_executor._extract_learning_content(long_result, "query")
        assert len(result) <= 503  # 500 + "..."
        assert result.endswith("...")

    def test_extract_converts_to_string(self, task_executor: TaskExecutor):
        """非文字列は文字列に変換される"""
        result = task_executor._extract_learning_content(12345, "query")
        assert result == "12345"

        result = task_executor._extract_learning_content({"key": "value"}, "query")
        assert "key" in result


# =============================================================================
# run_sleep_phase（睡眠フェーズ実行）のテスト
# =============================================================================


class TestRunSleepPhase:
    """run_sleep_phase メソッドのテスト

    睡眠フェーズ実行の単体テスト:
    - SleepPhaseProcessor.process_all() への委譲
    - SleepPhaseResult の返却確認
    - agent_id の受け渡し
    """

    @pytest.fixture
    def config(self) -> Phase1Config:
        """テスト用の設定"""
        return Phase1Config()

    @pytest.fixture
    def mock_sleep_result(self) -> "SleepPhaseResult":
        """モック SleepPhaseResult"""
        from src.core.sleep_processor import SleepPhaseResult

        return SleepPhaseResult(
            agent_id="test_agent",
            decayed_count=10,
            archived_count=3,
            consolidated_count=0,
            processed_at=datetime.now(),
            errors=[],
        )

    @pytest.fixture
    def mock_sleep_processor(self, mock_sleep_result: "SleepPhaseResult") -> MagicMock:
        """モック SleepPhaseProcessor"""
        mock = MagicMock()
        mock.process_all.return_value = mock_sleep_result
        return mock

    @pytest.fixture
    def task_executor(
        self,
        mock_sleep_processor: MagicMock,
        config: Phase1Config,
    ) -> TaskExecutor:
        """テスト用の TaskExecutor"""
        return TaskExecutor(
            vector_search=MagicMock(),
            ranker=MagicMock(),
            strength_manager=MagicMock(),
            sleep_processor=mock_sleep_processor,
            repository=MagicMock(),
            config=config,
        )

    def test_run_sleep_phase_returns_sleep_phase_result(
        self,
        task_executor: TaskExecutor,
    ):
        """SleepPhaseResult が返される"""
        from src.core.sleep_processor import SleepPhaseResult

        result = task_executor.run_sleep_phase(agent_id="test_agent")

        assert isinstance(result, SleepPhaseResult)

    def test_run_sleep_phase_calls_process_all(
        self,
        task_executor: TaskExecutor,
        mock_sleep_processor: MagicMock,
    ):
        """SleepPhaseProcessor.process_all が呼ばれる"""
        task_executor.run_sleep_phase(agent_id="test_agent")

        mock_sleep_processor.process_all.assert_called_once()

    def test_run_sleep_phase_passes_agent_id(
        self,
        task_executor: TaskExecutor,
        mock_sleep_processor: MagicMock,
    ):
        """agent_id が process_all に渡される"""
        task_executor.run_sleep_phase(agent_id="procurement_agent_01")

        mock_sleep_processor.process_all.assert_called_once_with("procurement_agent_01")

    def test_run_sleep_phase_returns_correct_counts(
        self,
        task_executor: TaskExecutor,
        mock_sleep_result: "SleepPhaseResult",
    ):
        """返却結果のカウント値が正しい"""
        result = task_executor.run_sleep_phase(agent_id="test_agent")

        assert result.decayed_count == mock_sleep_result.decayed_count
        assert result.archived_count == mock_sleep_result.archived_count
        assert result.consolidated_count == mock_sleep_result.consolidated_count
        assert result.agent_id == mock_sleep_result.agent_id

    def test_run_sleep_phase_returns_errors(
        self,
        config: Phase1Config,
    ):
        """エラーがある場合も SleepPhaseResult として返される"""
        from src.core.sleep_processor import SleepPhaseResult

        mock_result_with_errors = SleepPhaseResult(
            agent_id="test_agent",
            decayed_count=5,
            archived_count=0,
            consolidated_count=0,
            processed_at=datetime.now(),
            errors=["減衰処理でエラー: DB connection failed"],
        )

        mock_sleep_processor = MagicMock()
        mock_sleep_processor.process_all.return_value = mock_result_with_errors

        executor = TaskExecutor(
            vector_search=MagicMock(),
            ranker=MagicMock(),
            strength_manager=MagicMock(),
            sleep_processor=mock_sleep_processor,
            repository=MagicMock(),
            config=config,
        )

        result = executor.run_sleep_phase(agent_id="test_agent")

        assert len(result.errors) == 1
        assert "減衰処理でエラー" in result.errors[0]

    def test_run_sleep_phase_different_agent_ids(
        self,
        config: Phase1Config,
    ):
        """異なる agent_id での呼び出し"""
        from src.core.sleep_processor import SleepPhaseResult

        agent_ids = ["agent_01", "agent_02", "procurement_agent"]
        call_history = []

        def track_calls(agent_id: str) -> SleepPhaseResult:
            call_history.append(agent_id)
            return SleepPhaseResult(
                agent_id=agent_id,
                decayed_count=0,
                archived_count=0,
                consolidated_count=0,
                processed_at=datetime.now(),
                errors=[],
            )

        mock_sleep_processor = MagicMock()
        mock_sleep_processor.process_all.side_effect = track_calls

        executor = TaskExecutor(
            vector_search=MagicMock(),
            ranker=MagicMock(),
            strength_manager=MagicMock(),
            sleep_processor=mock_sleep_processor,
            repository=MagicMock(),
            config=config,
        )

        for agent_id in agent_ids:
            result = executor.run_sleep_phase(agent_id=agent_id)
            assert result.agent_id == agent_id

        # 呼び出し履歴を検証
        assert call_history == agent_ids
