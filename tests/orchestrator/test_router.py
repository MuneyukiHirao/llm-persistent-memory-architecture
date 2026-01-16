# Router テスト
# 実装仕様: docs/phase2-implementation-spec.ja.md セクション5.3
"""
Router クラスのユニットテスト

テスト内容:
- ルーティング判断の動作確認
- スコア計算の正確性
- 選択理由の生成
- 負荷分散（activity_score）
- 確信度計算
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from src.orchestrator.router import Router, RoutingDecision
from src.agents.agent_registry import AgentDefinition


class MockAgentRegistry:
    """テスト用のモックAgentRegistry"""

    def __init__(self, agents: list = None):
        self.agents = agents or []

    def get_active_agents(self) -> list:
        return self.agents


@pytest.fixture
def sample_agents():
    """テスト用のサンプルエージェント"""
    return [
        AgentDefinition(
            agent_id="research_agent",
            name="調査エージェント",
            role="技術調査とリサーチを担当",
            perspectives=["正確性", "網羅性", "効率性", "信頼性", "関連性"],
            system_prompt="あなたは調査専門のエージェントです",
            capabilities=["research", "analysis", "documentation"],
            status="active",
        ),
        AgentDefinition(
            agent_id="implementation_agent",
            name="実装エージェント",
            role="コード実装と機能開発を担当",
            perspectives=["保守性", "効率性", "可読性", "拡張性", "テスト容易性"],
            system_prompt="あなたは実装専門のエージェントです",
            capabilities=["implementation", "coding", "debugging"],
            status="active",
        ),
        AgentDefinition(
            agent_id="test_agent",
            name="テストエージェント",
            role="テスト作成と品質検証を担当",
            perspectives=["網羅性", "正確性", "効率性", "再現性", "自動化"],
            system_prompt="あなたはテスト専門のエージェントです",
            capabilities=["testing", "debugging", "analysis"],
            status="active",
        ),
    ]


@pytest.fixture
def router(sample_agents):
    """テスト用のRouter"""
    registry = MockAgentRegistry(sample_agents)
    return Router(registry)


class TestRoutingDecision:
    """RoutingDecision のテスト"""

    def test_create_routing_decision(self):
        """RoutingDecision の作成"""
        decision = RoutingDecision(
            selected_agent_id="test_agent",
            selection_reason="テストに適しています",
            candidates=[
                {"agent_id": "test_agent", "score": 0.9, "reason": "テスト向け"},
            ],
            confidence=0.85,
        )

        assert decision.selected_agent_id == "test_agent"
        assert decision.selection_reason == "テストに適しています"
        assert len(decision.candidates) == 1
        assert decision.confidence == 0.85

    def test_to_dict(self):
        """to_dict メソッドのテスト"""
        decision = RoutingDecision(
            selected_agent_id="test_agent",
            selection_reason="テストに適しています",
            confidence=0.85,
        )

        result = decision.to_dict()
        assert result["selected_agent_id"] == "test_agent"
        assert result["selection_reason"] == "テストに適しています"
        assert result["confidence"] == 0.85


class TestRouterDecide:
    """Router.decide のテスト"""

    def test_decide_selects_best_agent(self, router):
        """最適なエージェントを選択"""
        decision = router.decide(
            task_summary="ユーザー認証機能を実装してください",
        )

        assert decision.selected_agent_id == "implementation_agent"
        assert decision.confidence > 0
        assert len(decision.candidates) <= 3

    def test_decide_for_research_task(self, router):
        """調査タスクで調査エージェントを選択"""
        decision = router.decide(
            task_summary="APIの仕様を調査して分析してください",
        )

        assert decision.selected_agent_id == "research_agent"

    def test_decide_for_test_task(self, router):
        """テストタスクでテストエージェントを選択"""
        decision = router.decide(
            task_summary="pytestでユニットテストを実行してE2Eテストも試験してください",
        )

        assert decision.selected_agent_id == "test_agent"

    def test_decide_with_items(self, router):
        """items を含むルーティング"""
        decision = router.decide(
            task_summary="機能を追加",
            items=["実装", "コーディング", "API開発"],
        )

        assert decision.selected_agent_id == "implementation_agent"

    def test_decide_no_active_agents(self):
        """アクティブなエージェントがいない場合"""
        registry = MockAgentRegistry([])
        router = Router(registry)

        decision = router.decide(task_summary="何かのタスク")

        assert decision.selected_agent_id == ""
        assert decision.confidence == 0.0
        assert "見つかりません" in decision.selection_reason

    def test_decide_with_past_experiences(self, router):
        """過去の経験を考慮したルーティング"""
        past_experiences = [
            {"agent_id": "implementation_agent", "success": True},
            {"agent_id": "implementation_agent", "success": True},
            {"agent_id": "implementation_agent", "success": True},
            {"agent_id": "test_agent", "success": False},
        ]

        decision = router.decide(
            task_summary="バグをデバッグして修正してください",
            past_experiences=past_experiences,
        )

        # implementation_agent と test_agent 両方が debugging を持つが、
        # past_experiences で implementation_agent の成功率が高い
        assert decision.selected_agent_id == "implementation_agent"


class TestRouterScoreCalculation:
    """Router のスコア計算テスト"""

    def test_capability_match(self, router, sample_agents):
        """capability マッチのスコア計算"""
        agent = sample_agents[0]  # research_agent

        # research キーワードを含むタスク
        score = router._match_capabilities(
            agent.capabilities,
            "技術を調査して分析する",
        )

        assert score > 0
        assert score <= 1.0

    def test_capability_match_no_match(self, router, sample_agents):
        """capability マッチなしのスコア"""
        agent = sample_agents[0]  # research_agent

        score = router._match_capabilities(
            agent.capabilities,
            "ランダムな無関係テキスト",
        )

        # 低いスコアになるはず
        assert score < 0.5

    def test_perspective_match(self, router, sample_agents):
        """観点マッチのスコア計算"""
        agent = sample_agents[0]  # research_agent

        score = router._match_perspectives(
            agent.perspectives,
            "正確で効率的な調査をお願いします",
        )

        assert score > 0
        assert score <= 1.0

    def test_past_success_rate_no_data(self, router):
        """過去データなしの成功率"""
        rate = router._get_past_success_rate("unknown_agent", None)

        assert rate == 0.5  # デフォルト値

    def test_past_success_rate_with_data(self, router):
        """過去データありの成功率"""
        experiences = [
            {"agent_id": "test_agent", "success": True},
            {"agent_id": "test_agent", "success": True},
            {"agent_id": "test_agent", "success": False},
        ]

        rate = router._get_past_success_rate("test_agent", experiences)

        assert rate == pytest.approx(2 / 3, rel=0.01)


class TestRouterActivityScore:
    """Router のアクティビティスコアテスト"""

    def test_activity_score_no_history(self, router):
        """履歴なしのアクティビティスコア"""
        score = router._get_activity_score("new_agent")

        assert score == 1.0  # 履歴なし = 最高優先度

    def test_activity_score_with_recent_activity(self, router):
        """最近のアクティビティがあるスコア"""
        router.record_execution("busy_agent", success=True)
        router.record_execution("busy_agent", success=True)
        router.record_execution("busy_agent", success=True)

        score = router._get_activity_score("busy_agent")

        # 3タスク実行済み → スコアが下がる
        assert score < 1.0
        assert score >= 0.0

    def test_record_execution_clears_old(self, router):
        """古い履歴がクリアされる"""
        # 古い履歴を追加
        old_time = datetime.now() - timedelta(days=2)
        router._execution_history["old_agent"] = [
            {"timestamp": old_time, "success": True},
        ]

        # 新しい履歴を追加（古い履歴がクリアされる）
        router.record_execution("old_agent", success=True)

        # 古い履歴は削除されている
        assert len(router._execution_history["old_agent"]) == 1


class TestRouterConfidence:
    """Router の確信度計算テスト"""

    def test_confidence_single_candidate(self, router):
        """候補が1つの場合の確信度"""
        candidates = [
            {"agent_id": "agent1", "score": 0.8},
        ]

        confidence = router._calculate_confidence(candidates)

        assert confidence == 1.0

    def test_confidence_high_diff(self, router):
        """スコア差が大きい場合の確信度"""
        candidates = [
            {"agent_id": "agent1", "score": 0.9},
            {"agent_id": "agent2", "score": 0.3},
        ]

        confidence = router._calculate_confidence(candidates)

        assert confidence == 1.0

    def test_confidence_low_diff(self, router):
        """スコア差が小さい場合の確信度"""
        candidates = [
            {"agent_id": "agent1", "score": 0.8},
            {"agent_id": "agent2", "score": 0.75},
        ]

        confidence = router._calculate_confidence(candidates)

        assert confidence < 1.0
        assert confidence > 0.0


class TestRouterSelectionReason:
    """Router の選択理由生成テスト"""

    def test_generate_selection_reason_with_match(self, router, sample_agents):
        """マッチする能力がある場合の理由生成"""
        agent = sample_agents[1]  # implementation_agent

        reason = router._generate_selection_reason(
            agent,
            "機能を実装してください",
        )

        assert agent.name in reason
        assert "implementation" in reason or "適しています" in reason

    def test_generate_selection_reason_no_match(self, router, sample_agents):
        """マッチする能力がない場合の理由生成"""
        agent = sample_agents[0]  # research_agent

        reason = router._generate_selection_reason(
            agent,
            "完全に無関係なタスク",
        )

        assert agent.name in reason


class TestRouterEdgeCases:
    """Router のエッジケーステスト"""

    def test_empty_capabilities(self, router):
        """空の capabilities"""
        score = router._match_capabilities([], "任意のタスク")
        assert score == 0.0

    def test_empty_perspectives(self, router):
        """空の perspectives"""
        score = router._match_perspectives([], "任意のタスク")
        assert score == 0.0

    def test_empty_task_summary(self, router):
        """空のタスク概要"""
        decision = router.decide(task_summary="")

        # エラーにならず、何らかの判断ができる
        assert decision is not None

    def test_clear_execution_history(self, router):
        """履歴クリアのテスト"""
        router.record_execution("agent1", success=True)
        router.record_execution("agent2", success=False)

        router.clear_execution_history()

        assert len(router._execution_history) == 0
