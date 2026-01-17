# Orchestrator テスト
"""
Orchestrator クラスの単体テスト

テスト対象:
- OrchestratorResult データクラス
- SessionContext データクラス
- Orchestrator クラス

テスト方針:
- Router, Evaluator, TaskExecutor をモック化
- 各メソッドの動作を独立してテスト
- フロー整合性（呼び出し順序）を検証
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from uuid import uuid4, UUID

from src.orchestrator.orchestrator import (
    Orchestrator,
    OrchestratorResult,
    SessionContext,
)
from src.orchestrator.router import Router, RoutingDecision
from src.orchestrator.evaluator import Evaluator, FeedbackResult
from src.config.phase2_config import Phase2Config


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_router():
    """Router のモック"""
    router = MagicMock(spec=Router)
    router.decide.return_value = RoutingDecision(
        selected_agent_id="test_agent_01",
        selection_reason="テスト用の選択理由",
        candidates=[
            {"agent_id": "test_agent_01", "score": 0.9, "reason": "最適"},
            {"agent_id": "test_agent_02", "score": 0.7, "reason": "次点"},
        ],
        confidence=0.85,
    )
    return router


@pytest.fixture
def mock_evaluator():
    """Evaluator のモック"""
    evaluator = MagicMock(spec=Evaluator)
    evaluator.evaluate.return_value = FeedbackResult(
        feedback_type="positive",
        confidence=0.9,
        detected_signals=["ありがとう"],
        raw_response="ありがとう",
    )
    return evaluator


@pytest.fixture
def mock_task_executor():
    """TaskExecutor のモック"""
    executor = MagicMock()
    executor.search_memories.return_value = []
    executor.record_learning.return_value = uuid4()
    executor.run_sleep_phase.return_value = MagicMock(
        decayed_count=5,
        archived_count=2,
        consolidated_count=1,
        errors=[],
    )
    return executor


@pytest.fixture
def orchestrator(mock_router, mock_evaluator, mock_task_executor):
    """Orchestrator のフィクスチャ"""
    return Orchestrator(
        agent_id="orchestrator_test",
        router=mock_router,
        evaluator=mock_evaluator,
        task_executor=mock_task_executor,
        config=Phase2Config(),
    )


# =============================================================================
# OrchestratorResult テスト
# =============================================================================

class TestOrchestratorResult:
    """OrchestratorResult データクラスのテスト"""

    def test_create_basic(self):
        """基本的な OrchestratorResult の作成"""
        session_id = uuid4()
        routing_decision = RoutingDecision(
            selected_agent_id="agent_01",
            selection_reason="テスト",
        )

        result = OrchestratorResult(
            session_id=session_id,
            routing_decision=routing_decision,
            agent_result={"output": "結果"},
            status="success",
        )

        assert result.session_id == session_id
        assert result.routing_decision.selected_agent_id == "agent_01"
        assert result.agent_result == {"output": "結果"}
        assert result.status == "success"
        assert result.error_message is None

    def test_to_dict(self):
        """to_dict メソッドのテスト"""
        session_id = uuid4()
        routing_decision = RoutingDecision(
            selected_agent_id="agent_01",
            selection_reason="テスト",
        )

        result = OrchestratorResult(
            session_id=session_id,
            routing_decision=routing_decision,
            status="success",
        )

        result_dict = result.to_dict()

        assert result_dict["session_id"] == str(session_id)
        assert result_dict["status"] == "success"
        assert "routing_decision" in result_dict
        assert "executed_at" in result_dict

    def test_is_success_property(self):
        """is_success プロパティのテスト"""
        routing_decision = RoutingDecision(
            selected_agent_id="agent_01",
            selection_reason="テスト",
        )

        # success
        result_success = OrchestratorResult(
            session_id=uuid4(),
            routing_decision=routing_decision,
            status="success",
        )
        assert result_success.is_success is True

        # partial_success
        result_partial = OrchestratorResult(
            session_id=uuid4(),
            routing_decision=routing_decision,
            status="partial_success",
        )
        assert result_partial.is_success is True

        # failure
        result_failure = OrchestratorResult(
            session_id=uuid4(),
            routing_decision=routing_decision,
            status="failure",
        )
        assert result_failure.is_success is False

    def test_is_failure_property(self):
        """is_failure プロパティのテスト"""
        routing_decision = RoutingDecision(
            selected_agent_id="agent_01",
            selection_reason="テスト",
        )

        # failure
        result = OrchestratorResult(
            session_id=uuid4(),
            routing_decision=routing_decision,
            status="failure",
        )
        assert result.is_failure is True

        # timeout
        result_timeout = OrchestratorResult(
            session_id=uuid4(),
            routing_decision=routing_decision,
            status="timeout",
        )
        assert result_timeout.is_failure is True

        # no_agent
        result_no_agent = OrchestratorResult(
            session_id=uuid4(),
            routing_decision=routing_decision,
            status="no_agent",
        )
        assert result_no_agent.is_failure is True


# =============================================================================
# SessionContext テスト
# =============================================================================

class TestSessionContext:
    """SessionContext データクラスのテスト"""

    def test_create_basic(self):
        """基本的な SessionContext の作成"""
        session_id = uuid4()
        session = SessionContext(
            session_id=session_id,
            task_summary="タスク概要",
            items=["項目1", "項目2"],
        )

        assert session.session_id == session_id
        assert session.task_summary == "タスク概要"
        assert session.items == ["項目1", "項目2"]
        assert session.routing_decision is None
        assert session.subtask_count == 0

    def test_default_values(self):
        """デフォルト値のテスト"""
        session = SessionContext(
            session_id=uuid4(),
            task_summary="タスク",
        )

        assert session.items == []
        assert session.routing_decision is None
        assert session.subtask_count == 0
        assert session.created_at is not None
        assert session.last_activity_at is not None


# =============================================================================
# Orchestrator テスト
# =============================================================================

class TestOrchestratorInit:
    """Orchestrator 初期化のテスト"""

    def test_init_basic(self, mock_router, mock_evaluator, mock_task_executor):
        """基本的な初期化"""
        orchestrator = Orchestrator(
            agent_id="test_orchestrator",
            router=mock_router,
            evaluator=mock_evaluator,
            task_executor=mock_task_executor,
        )

        assert orchestrator.agent_id == "test_orchestrator"
        assert orchestrator.router == mock_router
        assert orchestrator.evaluator == mock_evaluator
        assert orchestrator.task_executor == mock_task_executor
        assert orchestrator.config is not None

    def test_init_with_config(self, mock_router, mock_evaluator, mock_task_executor):
        """カスタム設定での初期化"""
        custom_config = Phase2Config(
            orchestrator_subtask_batch_size=10,
            orchestrator_idle_timeout_minutes=120,
        )

        orchestrator = Orchestrator(
            agent_id="test_orchestrator",
            router=mock_router,
            evaluator=mock_evaluator,
            task_executor=mock_task_executor,
            config=custom_config,
        )

        assert orchestrator.config.orchestrator_subtask_batch_size == 10
        assert orchestrator.config.orchestrator_idle_timeout_minutes == 120


class TestProcessRequest:
    """process_request メソッドのテスト"""

    def test_process_request_basic(self, orchestrator, mock_router):
        """基本的なリクエスト処理"""
        result = orchestrator.process_request(
            task_summary="テストタスク",
            items=["項目1", "項目2"],
        )

        assert result.is_success
        assert result.status == "success"
        assert result.routing_decision.selected_agent_id == "test_agent_01"
        assert result.session_id is not None

        # Router.decide が呼ばれたことを確認
        mock_router.decide.assert_called_once()

    def test_process_request_searches_past_experiences(
        self, orchestrator, mock_task_executor
    ):
        """過去の経験検索が呼ばれることを確認"""
        orchestrator.process_request(
            task_summary="テストタスク",
        )

        # TaskExecutor.search_memories が呼ばれたことを確認
        mock_task_executor.search_memories.assert_called_once_with(
            query="テストタスク",
            agent_id="orchestrator_test",
            perspective="エージェント適性",
        )

    def test_process_request_creates_new_session(self, orchestrator):
        """新規セッションが作成されることを確認"""
        result = orchestrator.process_request(
            task_summary="テストタスク",
        )

        session = orchestrator.get_session(result.session_id)
        assert session is not None
        assert session.task_summary == "テストタスク"

    def test_process_request_continues_existing_session(self, orchestrator):
        """既存セッションが継続されることを確認"""
        # 最初のリクエスト
        result1 = orchestrator.process_request(
            task_summary="タスク1",
        )
        session_id = result1.session_id

        # 同じセッションIDで2回目のリクエスト
        result2 = orchestrator.process_request(
            task_summary="タスク2",
            session_id=session_id,
        )

        assert result2.session_id == session_id

        # セッションのサブタスク数が増加
        session = orchestrator.get_session(session_id)
        assert session.subtask_count == 2

    def test_process_request_no_agent_found(self, orchestrator, mock_router):
        """エージェントが見つからない場合"""
        mock_router.decide.return_value = RoutingDecision(
            selected_agent_id="",
            selection_reason="エージェントなし",
        )

        result = orchestrator.process_request(
            task_summary="テストタスク",
        )

        assert result.status == "no_agent"
        assert result.is_failure
        assert "見つかりません" in result.error_message

    def test_process_request_handles_error(self, orchestrator, mock_router):
        """エラーハンドリング"""
        mock_router.decide.side_effect = Exception("ルーティングエラー")

        result = orchestrator.process_request(
            task_summary="テストタスク",
        )

        assert result.status == "failure"
        assert result.is_failure
        assert "ルーティングエラー" in result.error_message


class TestReceiveFeedback:
    """receive_feedback メソッドのテスト"""

    def test_receive_feedback_positive(self, orchestrator, mock_evaluator):
        """positive フィードバックの処理"""
        # まずリクエストを処理
        result = orchestrator.process_request(
            task_summary="テストタスク",
        )

        # フィードバック受信
        feedback = orchestrator.receive_feedback(
            session_id=result.session_id,
            user_response="ありがとう、良いです",
        )

        assert feedback.feedback_type == "positive"
        mock_evaluator.evaluate.assert_called_once_with("ありがとう、良いです")

    def test_receive_feedback_negative_records_learning(
        self, orchestrator, mock_evaluator, mock_task_executor
    ):
        """negative フィードバック時に学びが記録されることを確認"""
        mock_evaluator.evaluate.return_value = FeedbackResult(
            feedback_type="negative",
            confidence=0.8,
            detected_signals=["ダメ"],
            raw_response="ダメです",
        )

        # リクエストを処理
        result = orchestrator.process_request(
            task_summary="テストタスク",
        )

        # フィードバック受信
        orchestrator.receive_feedback(
            session_id=result.session_id,
            user_response="ダメです",
        )

        # TaskExecutor.record_learning が呼ばれたことを確認
        mock_task_executor.record_learning.assert_called()

    def test_receive_feedback_redo_requested(
        self, orchestrator, mock_evaluator, mock_task_executor
    ):
        """redo_requested フィードバックの処理"""
        mock_evaluator.evaluate.return_value = FeedbackResult(
            feedback_type="redo_requested",
            confidence=0.9,
            detected_signals=["もう一度"],
            raw_response="もう一度やって",
        )

        result = orchestrator.process_request(
            task_summary="テストタスク",
        )

        feedback = orchestrator.receive_feedback(
            session_id=result.session_id,
            user_response="もう一度やって",
        )

        assert feedback.feedback_type == "redo_requested"
        mock_task_executor.record_learning.assert_called()


class TestDelegateTask:
    """_delegate_task メソッドのテスト"""

    def test_delegate_task_returns_mock_result(self, orchestrator):
        """Phase 2 MVP のモック結果が返されることを確認"""
        routing_decision = RoutingDecision(
            selected_agent_id="test_agent",
            selection_reason="テスト",
            confidence=0.9,
        )

        result = orchestrator._delegate_task(
            routing_decision=routing_decision,
            task_summary="テストタスク",
        )

        assert result["agent_id"] == "test_agent"
        assert result["status"] == "completed"
        assert "Phase 2 MVP" in result["output"]


class TestShouldSleep:
    """_should_sleep メソッドのテスト"""

    def test_should_sleep_subtask_threshold(self, orchestrator):
        """サブタスク完了数による睡眠判定"""
        # デフォルトは 5 サブタスク
        orchestrator._subtask_completed_count = 4
        assert orchestrator._should_sleep() is False

        orchestrator._subtask_completed_count = 5
        assert orchestrator._should_sleep() is True

    def test_should_sleep_idle_timeout(self, orchestrator):
        """アイドル時間による睡眠判定"""
        # デフォルトは 60 分
        orchestrator._last_activity_time = datetime.now() - timedelta(minutes=59)
        assert orchestrator._should_sleep() is False

        orchestrator._last_activity_time = datetime.now() - timedelta(minutes=61)
        assert orchestrator._should_sleep() is True


class TestRunSleepPhase:
    """_run_sleep_phase メソッドのテスト"""

    def test_run_sleep_phase_calls_task_executor(
        self, orchestrator, mock_task_executor
    ):
        """TaskExecutor.run_sleep_phase が呼ばれることを確認"""
        orchestrator._subtask_completed_count = 10

        orchestrator._run_sleep_phase()

        mock_task_executor.run_sleep_phase.assert_called_once_with("orchestrator_test")

    def test_run_sleep_phase_resets_subtask_count(
        self, orchestrator, mock_task_executor
    ):
        """睡眠後にサブタスク数がリセットされることを確認"""
        orchestrator._subtask_completed_count = 10

        orchestrator._run_sleep_phase()

        assert orchestrator._subtask_completed_count == 0


class TestSessionManagement:
    """セッション管理のテスト"""

    def test_get_session(self, orchestrator):
        """get_session メソッドのテスト"""
        result = orchestrator.process_request(
            task_summary="テストタスク",
        )

        session = orchestrator.get_session(result.session_id)
        assert session is not None
        assert session.task_summary == "テストタスク"

    def test_get_session_not_found(self, orchestrator):
        """存在しないセッションの取得"""
        session = orchestrator.get_session(uuid4())
        assert session is None

    def test_clear_sessions(self, orchestrator):
        """clear_sessions メソッドのテスト"""
        orchestrator.process_request(task_summary="タスク1")
        orchestrator.process_request(task_summary="タスク2")

        assert len(orchestrator._sessions) == 2

        orchestrator.clear_sessions()

        assert len(orchestrator._sessions) == 0


class TestGetStats:
    """get_stats メソッドのテスト"""

    def test_get_stats_basic(self, orchestrator):
        """get_stats の基本テスト"""
        stats = orchestrator.get_stats()

        assert stats["agent_id"] == "orchestrator_test"
        assert stats["active_sessions"] == 0
        assert stats["subtask_completed_count"] == 0
        assert "last_activity_time" in stats
        assert "idle_minutes" in stats

    def test_get_stats_after_requests(self, orchestrator):
        """リクエスト後の統計"""
        orchestrator.process_request(task_summary="タスク1")
        orchestrator.process_request(task_summary="タスク2")

        stats = orchestrator.get_stats()

        assert stats["active_sessions"] == 2
        assert stats["subtask_completed_count"] == 2


# =============================================================================
# 統合テスト
# =============================================================================

class TestIntegration:
    """統合テスト"""

    def test_full_flow_success(
        self, orchestrator, mock_router, mock_evaluator, mock_task_executor
    ):
        """成功フローの統合テスト"""
        # 1. リクエスト処理
        result = orchestrator.process_request(
            task_summary="ユーザー認証機能を実装",
            items=["認証フロー", "セッション管理"],
        )

        assert result.is_success
        session_id = result.session_id

        # 2. フィードバック受信
        feedback = orchestrator.receive_feedback(
            session_id=session_id,
            user_response="ありがとう、良さそうです",
        )

        assert feedback.feedback_type == "positive"

        # 3. 検証
        # - Router.decide が呼ばれた
        mock_router.decide.assert_called_once()
        # - TaskExecutor.search_memories が呼ばれた
        mock_task_executor.search_memories.assert_called()
        # - Evaluator.evaluate が呼ばれた
        mock_evaluator.evaluate.assert_called_once()

    def test_full_flow_with_retry(
        self, orchestrator, mock_router, mock_evaluator, mock_task_executor
    ):
        """リトライフローの統合テスト"""
        # 1. リクエスト処理
        result1 = orchestrator.process_request(
            task_summary="テストタスク",
        )

        # 2. 否定的フィードバック
        mock_evaluator.evaluate.return_value = FeedbackResult(
            feedback_type="redo_requested",
            confidence=0.9,
            detected_signals=["もう一度"],
            raw_response="もう一度やって",
        )

        feedback = orchestrator.receive_feedback(
            session_id=result1.session_id,
            user_response="もう一度やって",
        )

        assert feedback.needs_retry

        # 3. 再リクエスト（同じセッション）
        result2 = orchestrator.process_request(
            task_summary="テストタスク（修正版）",
            session_id=result1.session_id,
        )

        assert result2.session_id == result1.session_id
        assert result2.is_success

    def test_flow_order_search_then_route_then_delegate(
        self, orchestrator, mock_router, mock_task_executor
    ):
        """処理順序の検証: 検索 → ルーティング → 委譲"""
        call_order = []

        def track_search(*args, **kwargs):
            call_order.append("search")
            return []

        def track_decide(*args, **kwargs):
            call_order.append("decide")
            return RoutingDecision(
                selected_agent_id="test_agent",
                selection_reason="テスト",
            )

        mock_task_executor.search_memories.side_effect = track_search
        mock_router.decide.side_effect = track_decide

        orchestrator.process_request(task_summary="テストタスク")

        # 呼び出し順序を検証
        assert call_order == ["search", "decide"]


# =============================================================================
# LLM統合テスト
# =============================================================================

class TestDelegateTaskWithLLM:
    """LLM統合時の_delegate_taskテスト"""

    @pytest.fixture
    def mock_llm_task_executor(self):
        """LLMTaskExecutorのモック"""
        from unittest.mock import MagicMock

        executor = MagicMock()
        # LLMTaskResultのモック
        mock_result = MagicMock()
        mock_result.content = "LLMからの応答テスト"
        mock_result.stop_reason = "end_turn"
        mock_result.tool_calls = []
        mock_result.to_dict.return_value = {
            "content": "LLMからの応答テスト",
            "tool_calls": [],
            "total_tokens": 100,
            "iterations": 1,
            "searched_memories_count": 0,
            "stop_reason": "end_turn",
            "success": True,
        }
        executor.execute_task_with_tools.return_value = mock_result
        return executor

    @pytest.fixture
    def mock_agent_definition(self):
        """AgentDefinitionのモック"""
        from unittest.mock import MagicMock

        agent = MagicMock()
        agent.agent_id = "test_agent"
        agent.system_prompt = "あなたはテスト用エージェントです。"
        agent.perspectives = ["品質", "効率", "セキュリティ"]
        return agent

    def test_delegate_task_uses_llm_when_available(
        self, mock_router, mock_evaluator, mock_task_executor,
        mock_llm_task_executor, mock_agent_definition
    ):
        """LLMTaskExecutorが設定されている場合はLLMを使用"""
        mock_router.agent_registry = MagicMock()
        mock_router.agent_registry.get_by_id.return_value = mock_agent_definition

        orchestrator = Orchestrator(
            agent_id="orchestrator_test",
            router=mock_router,
            evaluator=mock_evaluator,
            task_executor=mock_task_executor,
            llm_task_executor=mock_llm_task_executor,
        )

        routing_decision = RoutingDecision(
            selected_agent_id="test_agent",
            selection_reason="テスト",
            confidence=0.9,
        )

        result = orchestrator._delegate_task(
            routing_decision=routing_decision,
            task_summary="テストタスク",
        )

        # LLMが使用されたことを確認
        mock_llm_task_executor.execute_task_with_tools.assert_called_once()
        assert result["status"] == "completed"
        assert result["output"] == "LLMからの応答テスト"
        assert "llm_result" in result["metadata"]

    def test_delegate_task_uses_mock_when_llm_not_available(
        self, mock_router, mock_evaluator, mock_task_executor
    ):
        """LLMTaskExecutorがNoneの場合はモック動作"""
        orchestrator = Orchestrator(
            agent_id="orchestrator_test",
            router=mock_router,
            evaluator=mock_evaluator,
            task_executor=mock_task_executor,
            llm_task_executor=None,  # LLM未設定
        )

        routing_decision = RoutingDecision(
            selected_agent_id="test_agent",
            selection_reason="テスト",
            confidence=0.9,
        )

        result = orchestrator._delegate_task(
            routing_decision=routing_decision,
            task_summary="テストタスク",
        )

        # モック結果が返されることを確認
        assert result["status"] == "completed"
        assert "Phase 2 MVP" in result["output"]

    def test_delegate_task_handles_agent_not_found(
        self, mock_router, mock_evaluator, mock_task_executor,
        mock_llm_task_executor
    ):
        """エージェント定義が見つからない場合のエラーハンドリング"""
        mock_router.agent_registry = MagicMock()
        mock_router.agent_registry.get_by_id.return_value = None  # エージェント見つからず

        orchestrator = Orchestrator(
            agent_id="orchestrator_test",
            router=mock_router,
            evaluator=mock_evaluator,
            task_executor=mock_task_executor,
            llm_task_executor=mock_llm_task_executor,
        )

        routing_decision = RoutingDecision(
            selected_agent_id="unknown_agent",
            selection_reason="テスト",
            confidence=0.9,
        )

        result = orchestrator._delegate_task(
            routing_decision=routing_decision,
            task_summary="テストタスク",
        )

        assert result["status"] == "error"
        assert "見つかりません" in result["output"]

    def test_delegate_task_handles_llm_error(
        self, mock_router, mock_evaluator, mock_task_executor,
        mock_llm_task_executor, mock_agent_definition
    ):
        """LLM呼び出しエラー時のハンドリング"""
        mock_router.agent_registry = MagicMock()
        mock_router.agent_registry.get_by_id.return_value = mock_agent_definition
        mock_llm_task_executor.execute_task_with_tools.side_effect = Exception("API接続エラー")

        orchestrator = Orchestrator(
            agent_id="orchestrator_test",
            router=mock_router,
            evaluator=mock_evaluator,
            task_executor=mock_task_executor,
            llm_task_executor=mock_llm_task_executor,
        )

        routing_decision = RoutingDecision(
            selected_agent_id="test_agent",
            selection_reason="テスト",
            confidence=0.9,
        )

        result = orchestrator._delegate_task(
            routing_decision=routing_decision,
            task_summary="テストタスク",
        )

        assert result["status"] == "error"
        assert "API接続エラー" in result["output"]

    def test_delegate_task_uses_first_perspective(
        self, mock_router, mock_evaluator, mock_task_executor,
        mock_llm_task_executor, mock_agent_definition
    ):
        """エージェントの最初の観点が使用されることを確認"""
        mock_router.agent_registry = MagicMock()
        mock_router.agent_registry.get_by_id.return_value = mock_agent_definition

        orchestrator = Orchestrator(
            agent_id="orchestrator_test",
            router=mock_router,
            evaluator=mock_evaluator,
            task_executor=mock_task_executor,
            llm_task_executor=mock_llm_task_executor,
        )

        routing_decision = RoutingDecision(
            selected_agent_id="test_agent",
            selection_reason="テスト",
            confidence=0.9,
        )

        orchestrator._delegate_task(
            routing_decision=routing_decision,
            task_summary="テストタスク",
        )

        # LLM呼び出し時に最初の観点が渡されたことを確認
        call_args = mock_llm_task_executor.execute_task_with_tools.call_args
        assert call_args.kwargs["perspective"] == "品質"  # mock_agent_definitionの最初の観点

    def test_delegate_task_with_partial_result(
        self, mock_router, mock_evaluator, mock_task_executor,
        mock_llm_task_executor, mock_agent_definition
    ):
        """max_iterations到達時のpartialステータス"""
        mock_router.agent_registry = MagicMock()
        mock_router.agent_registry.get_by_id.return_value = mock_agent_definition

        # max_iterationsで終了した場合のモック
        mock_result = MagicMock()
        mock_result.content = "部分的な結果"
        mock_result.stop_reason = "max_iterations"
        mock_result.tool_calls = []
        mock_result.to_dict.return_value = {"stop_reason": "max_iterations"}
        mock_llm_task_executor.execute_task_with_tools.return_value = mock_result

        orchestrator = Orchestrator(
            agent_id="orchestrator_test",
            router=mock_router,
            evaluator=mock_evaluator,
            task_executor=mock_task_executor,
            llm_task_executor=mock_llm_task_executor,
        )

        routing_decision = RoutingDecision(
            selected_agent_id="test_agent",
            selection_reason="テスト",
            confidence=0.9,
        )

        result = orchestrator._delegate_task(
            routing_decision=routing_decision,
            task_summary="テストタスク",
        )

        assert result["status"] == "partial"
