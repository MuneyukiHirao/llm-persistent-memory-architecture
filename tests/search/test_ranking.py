# 優先度ランキングモジュールのテスト
"""
MemoryRanker の単体テスト

テスト観点:
- スコア計算の正確性
- 正規化ロジックの検証
- 時間減衰の検証
- 観点指定時の動作
- エッジケースの処理
"""

import math
from datetime import datetime, timedelta
from uuid import uuid4

import pytest

from src.config.phase1_config import Phase1Config
from src.models.memory import AgentMemory
from src.search.ranking import MemoryRanker, ScoredMemory


class TestMemoryRanker:
    """MemoryRanker のテスト"""

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
    def ranker(self, config: Phase1Config) -> MemoryRanker:
        """テスト用のランカー"""
        return MemoryRanker(config)

    @pytest.fixture
    def sample_memory(self) -> AgentMemory:
        """テスト用のメモリ"""
        return AgentMemory(
            id=uuid4(),
            agent_id="test_agent",
            content="テスト用の記憶内容",
            strength=1.0,
            strength_by_perspective={"コスト": 1.5, "納期": 0.8},
            created_at=datetime.now(),
            updated_at=datetime.now(),
            last_accessed_at=datetime.now(),
        )

    def test_rank_empty_candidates(self, ranker: MemoryRanker):
        """空の候補リストの場合、空リストを返す"""
        result = ranker.rank([])
        assert result == []

    def test_rank_single_candidate(
        self, ranker: MemoryRanker, sample_memory: AgentMemory
    ):
        """単一の候補の場合、スコア計算されて返される"""
        candidates = [(sample_memory, 0.8)]
        result = ranker.rank(candidates)

        assert len(result) == 1
        assert isinstance(result[0], ScoredMemory)
        assert result[0].memory == sample_memory
        assert result[0].similarity == 0.8
        assert result[0].final_score > 0

    def test_rank_multiple_candidates_sorted(self, ranker: MemoryRanker):
        """複数の候補はスコア降順でソートされる"""
        now = datetime.now()
        memories = [
            AgentMemory(
                id=uuid4(),
                agent_id="test_agent",
                content=f"Memory {i}",
                strength=1.0,
                created_at=now,
                updated_at=now,
                last_accessed_at=now,
            )
            for i in range(3)
        ]

        # 類似度を異なる値に設定
        candidates = [
            (memories[0], 0.5),
            (memories[1], 0.9),
            (memories[2], 0.7),
        ]

        result = ranker.rank(candidates)

        # スコア降順を確認
        assert len(result) == 3
        assert result[0].similarity == 0.9
        assert result[1].similarity == 0.7
        assert result[2].similarity == 0.5

    def test_rank_top_k_limit(self, ranker: MemoryRanker):
        """top_k_results を超える候補は制限される"""
        now = datetime.now()
        # 15件の候補を作成（top_k_results=10）
        candidates = [
            (
                AgentMemory(
                    id=uuid4(),
                    agent_id="test_agent",
                    content=f"Memory {i}",
                    strength=1.0,
                    created_at=now,
                    updated_at=now,
                    last_accessed_at=now,
                ),
                0.5 + i * 0.01,  # 類似度を少しずつ変える
            )
            for i in range(15)
        ]

        result = ranker.rank(candidates)

        assert len(result) == 10  # top_k_results
        # 最高スコアの候補が含まれていることを確認
        assert result[0].similarity >= 0.5 + 9 * 0.01

    def test_rank_with_perspective(
        self, ranker: MemoryRanker, sample_memory: AgentMemory
    ):
        """観点指定時は strength_by_perspective を使用"""
        candidates = [(sample_memory, 0.8)]

        # 観点 "コスト" を指定（strength_by_perspective["コスト"] = 1.5）
        result = ranker.rank(candidates, perspective="コスト")

        assert len(result) == 1
        # strength_raw が 1.5 であることを確認
        assert result[0].score_breakdown["strength_raw"] == 1.5

    def test_rank_with_missing_perspective(
        self, ranker: MemoryRanker, sample_memory: AgentMemory
    ):
        """存在しない観点の場合は全体の strength にフォールバック"""
        candidates = [(sample_memory, 0.8)]

        # 存在しない観点を指定
        result = ranker.rank(candidates, perspective="品質")

        assert len(result) == 1
        # strength_raw が全体の strength (1.0) であることを確認
        assert result[0].score_breakdown["strength_raw"] == 1.0


class TestNormalizeStrength:
    """_normalize_strength のテスト"""

    @pytest.fixture
    def ranker(self) -> MemoryRanker:
        return MemoryRanker()

    def test_normalize_strength_zero(self, ranker: MemoryRanker):
        """強度 0 の正規化"""
        assert ranker._normalize_strength(0.0) == 0.0

    def test_normalize_strength_negative(self, ranker: MemoryRanker):
        """負の強度は 0 にクリップ"""
        assert ranker._normalize_strength(-0.5) == 0.0

    def test_normalize_strength_one(self, ranker: MemoryRanker):
        """強度 1.0 の正規化"""
        result = ranker._normalize_strength(1.0)
        assert result == pytest.approx(0.5, rel=0.001)

    def test_normalize_strength_max(self, ranker: MemoryRanker):
        """強度 2.0（上限）の正規化"""
        result = ranker._normalize_strength(2.0)
        assert result == pytest.approx(1.0, rel=0.001)

    def test_normalize_strength_above_max(self, ranker: MemoryRanker):
        """強度が上限を超える場合は 1.0 にクリップ"""
        result = ranker._normalize_strength(3.0)
        assert result == pytest.approx(1.0, rel=0.001)


class TestCalculateRecency:
    """_calculate_recency のテスト"""

    @pytest.fixture
    def ranker(self) -> MemoryRanker:
        return MemoryRanker()

    def test_recency_just_accessed(self, ranker: MemoryRanker):
        """今アクセスしたばかりの場合、recency ≒ 1.0"""
        now = datetime.now()
        result = ranker._calculate_recency(now, now)
        assert result == pytest.approx(1.0, rel=0.01)

    def test_recency_30_days_ago(self, ranker: MemoryRanker):
        """30日前の場合、recency ≒ 1/e ≒ 0.368"""
        now = datetime.now()
        past = now - timedelta(days=30)
        result = ranker._calculate_recency(past, past)
        expected = math.exp(-1.0)  # exp(-30/30) = exp(-1) ≒ 0.368
        assert result == pytest.approx(expected, rel=0.01)

    def test_recency_60_days_ago(self, ranker: MemoryRanker):
        """60日前の場合、recency ≒ 1/e^2 ≒ 0.135"""
        now = datetime.now()
        past = now - timedelta(days=60)
        result = ranker._calculate_recency(past, past)
        expected = math.exp(-2.0)  # exp(-60/30) = exp(-2) ≒ 0.135
        assert result == pytest.approx(expected, rel=0.01)

    def test_recency_uses_last_accessed_at(self, ranker: MemoryRanker):
        """last_accessed_at が設定されている場合はそちらを使用"""
        now = datetime.now()
        created = now - timedelta(days=100)
        accessed = now - timedelta(days=10)

        result = ranker._calculate_recency(accessed, created)
        # 10日前の recency
        expected = math.exp(-10 / 30)
        assert result == pytest.approx(expected, rel=0.01)

    def test_recency_fallback_to_created_at(self, ranker: MemoryRanker):
        """last_accessed_at が None の場合は created_at を使用"""
        now = datetime.now()
        created = now - timedelta(days=15)

        result = ranker._calculate_recency(None, created)
        # 15日前の recency
        expected = math.exp(-15 / 30)
        assert result == pytest.approx(expected, rel=0.01)


class TestScoreBreakdown:
    """スコア内訳の検証"""

    @pytest.fixture
    def ranker(self) -> MemoryRanker:
        return MemoryRanker(
            Phase1Config(
                score_weights={
                    "similarity": 0.50,
                    "strength": 0.30,
                    "recency": 0.20,
                }
            )
        )

    def test_score_breakdown_components(self, ranker: MemoryRanker):
        """スコア内訳が正しく計算される"""
        now = datetime.now()
        memory = AgentMemory(
            id=uuid4(),
            agent_id="test_agent",
            content="Test memory",
            strength=1.0,
            created_at=now,
            updated_at=now,
            last_accessed_at=now,
        )

        candidates = [(memory, 0.8)]
        result = ranker.rank(candidates)

        assert len(result) == 1
        breakdown = result[0].score_breakdown

        # 必要なキーが含まれていることを確認
        assert "similarity_raw" in breakdown
        assert "similarity_weighted" in breakdown
        assert "strength_raw" in breakdown
        assert "strength_normalized" in breakdown
        assert "strength_weighted" in breakdown
        assert "recency_raw" in breakdown
        assert "recency_weighted" in breakdown
        assert "total" in breakdown

        # 重み付けスコアの検証
        assert breakdown["similarity_weighted"] == pytest.approx(0.8 * 0.50, rel=0.001)
        assert breakdown["strength_normalized"] == pytest.approx(0.5, rel=0.001)  # 1.0/2.0
        assert breakdown["strength_weighted"] == pytest.approx(0.5 * 0.30, rel=0.001)

        # total は各重み付けスコアの合計
        expected_total = (
            breakdown["similarity_weighted"]
            + breakdown["strength_weighted"]
            + breakdown["recency_weighted"]
        )
        assert breakdown["total"] == pytest.approx(expected_total, rel=0.001)


class TestIntegration:
    """統合テスト"""

    def test_realistic_scenario(self):
        """現実的なシナリオでのテスト"""
        config = Phase1Config()
        ranker = MemoryRanker(config)

        now = datetime.now()

        # 異なる特性を持つ記憶を作成
        memories = [
            # 高い類似度、低い強度、古い
            AgentMemory(
                id=uuid4(),
                agent_id="test_agent",
                content="高類似度・低強度・古い",
                strength=0.5,
                created_at=now - timedelta(days=60),
                updated_at=now - timedelta(days=60),
                last_accessed_at=now - timedelta(days=60),
            ),
            # 中程度の類似度、高い強度、最近
            AgentMemory(
                id=uuid4(),
                agent_id="test_agent",
                content="中類似度・高強度・最近",
                strength=2.0,
                created_at=now - timedelta(days=5),
                updated_at=now - timedelta(days=5),
                last_accessed_at=now - timedelta(days=5),
            ),
            # 低い類似度、中程度の強度、中程度の新鮮さ
            AgentMemory(
                id=uuid4(),
                agent_id="test_agent",
                content="低類似度・中強度・中新鮮",
                strength=1.2,
                created_at=now - timedelta(days=20),
                updated_at=now - timedelta(days=20),
                last_accessed_at=now - timedelta(days=20),
            ),
        ]

        candidates = [
            (memories[0], 0.9),
            (memories[1], 0.6),
            (memories[2], 0.4),
        ]

        result = ranker.rank(candidates)

        assert len(result) == 3

        # スコアが降順にソートされていることを確認
        for i in range(len(result) - 1):
            assert result[i].final_score >= result[i + 1].final_score

        # 各候補のスコア計算が行われていることを確認
        for scored in result:
            assert scored.final_score > 0
            assert "total" in scored.score_breakdown
