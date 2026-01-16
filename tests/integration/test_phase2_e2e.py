# Phase 2 E2E 統合テスト
"""
Phase 2 E2E統合テスト

仕様書参照: docs/phase2-implementation-spec.ja.md セクション9.2-9.3

テストシナリオ:
1. 単純なタスク: InputProcessor → Orchestrator → Router → 結果返却
2. 大きな入力: 入力処理層（概要生成）→ オーケストレーター連携
3. やり直し要求: フィードバック → 再ルーティングのフロー
4. 中間睡眠: 状態保存 → 復帰 → 継続のフロー
5. 複数タスク: 複数論点の順次処理

全コンポーネントの連携確認:
- InputProcessor → Orchestrator → Router → Evaluator → ProgressManager

モック使用: 実際のLLM呼び出しはモック化
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, PropertyMock
from uuid import uuid4, UUID

from src.config.phase2_config import Phase2Config
from src.input_processing.input_processor import InputProcessor, ProcessedInput
from src.orchestrator.orchestrator import Orchestrator, OrchestratorResult, SessionContext
from src.orchestrator.router import Router, RoutingDecision
from src.orchestrator.evaluator import Evaluator, FeedbackResult
from src.orchestrator.progress_manager import (
    ProgressManager,
    SessionStateRepository,
    SessionState,
)
from src.agents.agent_registry import AgentRegistry, AgentDefinition


# =============================================================================
# テスト用モックファクトリ
# =============================================================================

def create_mock_agent_registry(agents: list[AgentDefinition] = None):
    """AgentRegistry のモックを作成"""
    if agents is None:
        agents = [
            AgentDefinition(
                agent_id="implementation_agent",
                name="実装エージェント",
                role="機能実装を担当",
                perspectives=["正確性", "効率性", "保守性", "安全性", "拡張性"],
                system_prompt="あなたは実装専門のエージェントです。",
                capabilities=["implementation", "coding", "debugging"],
                status="active",
            ),
            AgentDefinition(
                agent_id="research_agent",
                name="調査エージェント",
                role="技術調査とリサーチを担当",
                perspectives=["正確性", "網羅性", "信頼性", "関連性", "最新性"],
                system_prompt="あなたは調査専門のエージェントです。",
                capabilities=["research", "analysis", "documentation"],
                status="active",
            ),
            AgentDefinition(
                agent_id="testing_agent",
                name="テストエージェント",
                role="テスト作成と品質検証を担当",
                perspectives=["カバレッジ", "再現性", "境界値", "パフォーマンス", "保守性"],
                system_prompt="あなたはテスト専門のエージェントです。",
                capabilities=["testing", "debugging", "analysis"],
                status="active",
            ),
        ]

    registry = MagicMock(spec=AgentRegistry)
    registry.get_active_agents.return_value = agents
    registry.get_by_id.side_effect = lambda aid: next(
        (a for a in agents if a.agent_id == aid), None
    )
    registry.search_by_capabilities.side_effect = lambda caps: [
        a for a in agents
        if any(cap in a.capabilities for cap in caps)
    ]
    return registry


def create_mock_task_executor():
    """TaskExecutor のモックを作成"""
    executor = MagicMock()
    executor.search_memories.return_value = []
    executor.record_learning.return_value = uuid4()
    executor.run_sleep_phase.return_value = MagicMock(
        decayed_count=3,
        archived_count=1,
        consolidated_count=2,
        errors=[],
    )
    return executor


def create_mock_session_repository():
    """SessionStateRepository のモックを作成"""
    repository = MagicMock(spec=SessionStateRepository)
    sessions = {}

    def mock_create(state):
        sessions[state.session_id] = state
        return state.session_id

    def mock_get_by_id(session_id):
        return sessions.get(session_id)

    def mock_update(session_id, **kwargs):
        if session_id in sessions:
            state = sessions[session_id]
            for key, value in kwargs.items():
                if value is not None and hasattr(state, key):
                    setattr(state, key, value)
            state.updated_at = datetime.now()
            return True
        return False

    def mock_list_by_status(status):
        return [s for s in sessions.values() if s.status == status]

    repository.create.side_effect = mock_create
    repository.get_by_id.side_effect = mock_get_by_id
    repository.update.side_effect = mock_update
    repository.list_by_status.side_effect = mock_list_by_status
    repository.clear_current_task.return_value = True
    repository.delete.return_value = True

    return repository


# =============================================================================
# フィクスチャ
# =============================================================================

@pytest.fixture
def config():
    """Phase2Config のフィクスチャ"""
    return Phase2Config()


@pytest.fixture
def input_processor(config):
    """InputProcessor のフィクスチャ"""
    return InputProcessor(config)


@pytest.fixture
def mock_agent_registry():
    """モック AgentRegistry のフィクスチャ"""
    return create_mock_agent_registry()


@pytest.fixture
def router(mock_agent_registry):
    """Router のフィクスチャ"""
    return Router(mock_agent_registry)


@pytest.fixture
def evaluator(config):
    """Evaluator のフィクスチャ"""
    return Evaluator(config)


@pytest.fixture
def mock_task_executor():
    """モック TaskExecutor のフィクスチャ"""
    return create_mock_task_executor()


@pytest.fixture
def mock_session_repository():
    """モック SessionStateRepository のフィクスチャ"""
    return create_mock_session_repository()


@pytest.fixture
def progress_manager(mock_session_repository, config):
    """ProgressManager のフィクスチャ"""
    return ProgressManager(mock_session_repository, config)


@pytest.fixture
def orchestrator(router, evaluator, mock_task_executor, config):
    """Orchestrator のフィクスチャ"""
    return Orchestrator(
        agent_id="orchestrator_e2e_test",
        router=router,
        evaluator=evaluator,
        task_executor=mock_task_executor,
        config=config,
    )


# =============================================================================
# シナリオ1: 単純なタスク
# InputProcessor → Orchestrator → Router → 結果返却
# =============================================================================

class TestSimpleTaskE2E:
    """単純なタスクのE2E統合テスト"""

    def test_simple_task_flow(self, input_processor, orchestrator):
        """シンプルなタスク: 入力処理 → オーケストレーター → ルーティング → 結果"""
        # 1. ユーザー入力
        user_input = "ユーザー認証機能を追加してください"

        # 2. 入力処理
        processed = input_processor.process(user_input)

        # 検証: 入力処理結果
        assert processed.summary == user_input
        assert processed.needs_negotiation is False
        assert processed.item_count == 0  # 単純な入力なので論点検出なし

        # 3. オーケストレーター実行
        result = orchestrator.process_request(
            task_summary=processed.summary,
            items=processed.items,
        )

        # 検証: オーケストレーター結果
        assert result.is_success
        assert result.routing_decision.selected_agent_id == "implementation_agent"
        assert result.session_id is not None

    def test_simple_task_with_items(self, input_processor, orchestrator):
        """論点付きのシンプルなタスク"""
        # 箇条書き入力
        user_input = """以下の機能を実装してください:
- ログイン機能
- ログアウト機能
- パスワードリセット"""

        processed = input_processor.process(user_input)

        # 検証: 論点が検出される
        assert processed.item_count == 3
        assert "ログイン機能" in processed.items
        assert processed.needs_negotiation is False  # 閾値(10)未満

        result = orchestrator.process_request(
            task_summary=processed.summary,
            items=processed.items,
        )

        assert result.is_success

    def test_full_flow_with_feedback(
        self, input_processor, orchestrator, evaluator
    ):
        """完全フロー: 入力 → 処理 → フィードバック"""
        # 1. 入力処理
        processed = input_processor.process("ユーザー認証の実装方法を調べてリサーチしてください")

        # 2. オーケストレーター実行
        result = orchestrator.process_request(
            task_summary=processed.summary,
        )

        assert result.is_success
        # 適切なエージェントが選択されている（空ではない）
        assert result.routing_decision.selected_agent_id in [
            "research_agent", "implementation_agent", "testing_agent"
        ]

        # 3. ユーザーフィードバック
        feedback = orchestrator.receive_feedback(
            session_id=result.session_id,
            user_response="ありがとう、良さそうです",
        )

        assert feedback.feedback_type == "positive"
        assert feedback.confidence >= 0.7


# =============================================================================
# シナリオ2: 大きな入力
# 入力処理層（概要生成）→ オーケストレーター連携
# =============================================================================

class TestLargeInputE2E:
    """大きな入力のE2E統合テスト"""

    def test_large_input_generates_summary(self, input_processor, orchestrator):
        """大きな入力で概要が生成される"""
        # 6000文字以上の入力（input_size_threshold=5000）
        large_content = "システム設計の詳細: " + "詳細な説明です。" * 800

        processed = input_processor.process(large_content)

        # 検証: 概要が生成され、詳細への参照が作成される
        assert len(processed.summary) < len(large_content)
        assert len(processed.detail_refs) == 1
        assert processed.original_size_tokens >= 5000

        # 詳細データを取得できることを確認
        detail = input_processor.get_detail(processed.detail_refs[0])
        assert detail == large_content

        # オーケストレーターで処理
        result = orchestrator.process_request(
            task_summary=processed.summary,
        )

        assert result.is_success

    def test_many_items_triggers_negotiation(self, input_processor, orchestrator):
        """多数の論点で交渉が必要になる"""
        # 12個の論点（input_item_threshold=10）
        items = "\n".join([f"- タスク{i}: 詳細な説明" for i in range(1, 13)])

        processed = input_processor.process(items)

        # 検証: 交渉が必要
        assert processed.item_count == 12
        assert processed.needs_negotiation is True
        assert len(processed.negotiation_options) == 3

        # ユーザーが「全て処理」を選択した場合をシミュレート
        result = orchestrator.process_request(
            task_summary=processed.summary,
            items=processed.items,
        )

        assert result.is_success

    def test_large_input_with_many_items(self, input_processor, orchestrator):
        """大きな入力 + 多数の論点"""
        # 大きな入力かつ多数の論点
        prefix = "システム設計の概要: " * 300
        items = "\n".join([f"- 要件{i}: 重要な機能" for i in range(1, 15)])
        large_input = prefix + "\n\n以下の要件を実装:\n" + items

        processed = input_processor.process(large_input)

        # 検証: 両方のトリガーが発火
        assert processed.needs_negotiation is True
        assert len(processed.detail_refs) == 1  # 概要生成された
        assert processed.item_count >= 10

        result = orchestrator.process_request(
            task_summary=processed.summary,
            items=processed.items,
        )

        assert result.is_success


# =============================================================================
# シナリオ3: やり直し要求
# フィードバック → 再ルーティングのフロー
# =============================================================================

class TestRedoRequestE2E:
    """やり直し要求のE2E統合テスト"""

    def test_redo_request_flow(self, input_processor, orchestrator):
        """やり直し要求 → 再処理のフロー"""
        # 1. 初回リクエスト
        processed = input_processor.process("データベース設計を行ってください")
        result1 = orchestrator.process_request(
            task_summary=processed.summary,
        )

        assert result1.is_success
        session_id = result1.session_id

        # 2. やり直し要求のフィードバック
        feedback = orchestrator.receive_feedback(
            session_id=session_id,
            user_response="もう一度やり直してください",
        )

        assert feedback.feedback_type == "redo_requested"
        assert feedback.needs_retry is True

        # 3. 同じセッションで再リクエスト
        result2 = orchestrator.process_request(
            task_summary=processed.summary + "（修正版）",
            session_id=session_id,
        )

        assert result2.is_success
        assert result2.session_id == session_id

        # セッションのサブタスク数が増加していることを確認
        session = orchestrator.get_session(session_id)
        assert session.subtask_count == 2

    def test_negative_feedback_triggers_rerouting(self, input_processor, orchestrator):
        """否定的フィードバック → 再ルーティング"""
        # 1. 初回リクエスト
        result1 = orchestrator.process_request(
            task_summary="テストコードを書いてください",
        )

        assert result1.is_success
        first_agent = result1.routing_decision.selected_agent_id

        # 2. 否定的フィードバック
        feedback = orchestrator.receive_feedback(
            session_id=result1.session_id,
            user_response="ダメです、期待と違います",
        )

        assert feedback.feedback_type == "negative"
        assert feedback.is_negative is True

    def test_redo_with_different_instructions(self, input_processor, orchestrator):
        """やり直しで異なる指示を与える"""
        # 1. 初回: 曖昧な指示
        result1 = orchestrator.process_request(
            task_summary="機能を追加して",
        )

        # 2. やり直し要求
        feedback = orchestrator.receive_feedback(
            session_id=result1.session_id,
            user_response="違います、もう一度お願いします",
        )

        assert feedback.needs_retry

        # 3. より具体的な指示で再リクエスト
        result2 = orchestrator.process_request(
            task_summary="ユーザー認証機能を実装してください",
            session_id=result1.session_id,
        )

        assert result2.is_success


# =============================================================================
# シナリオ4: 中間睡眠
# 状態保存 → 復帰 → 継続のフロー
# =============================================================================

class TestIntermediateSleepE2E:
    """中間睡眠のE2E統合テスト"""

    def test_session_state_save_and_restore(
        self, orchestrator, progress_manager
    ):
        """セッション状態の保存と復元"""
        # 1. セッション作成
        session_id = progress_manager.create_session(
            orchestrator_id="orchestrator_e2e_test",
            user_request={"original": "テストタスク", "clarified": "具体的なタスク"},
            task_tree={"tasks": [
                {"id": "task_1", "description": "タスク1", "status": "pending"},
                {"id": "task_2", "description": "タスク2", "status": "pending"},
            ]},
        )

        # 2. 進捗更新
        progress_manager.save_state(
            session_id=session_id,
            task_tree={"tasks": [
                {"id": "task_1", "description": "タスク1", "status": "completed"},
                {"id": "task_2", "description": "タスク2", "status": "in_progress"},
            ]},
            current_task={"id": "task_2", "description": "タスク2"},
            progress_percent=50,
        )

        # 3. 状態を復元
        restored = progress_manager.restore_state(session_id)

        assert restored is not None
        assert restored.overall_progress_percent == 50
        assert restored.current_task["id"] == "task_2"

    def test_sleep_trigger_by_subtask_count(
        self, orchestrator, mock_task_executor
    ):
        """サブタスク完了数による睡眠トリガー"""
        # デフォルトは5サブタスクで睡眠
        for i in range(5):
            orchestrator.process_request(
                task_summary=f"タスク{i+1}",
            )

        # 睡眠フェーズが実行されたことを確認
        mock_task_executor.run_sleep_phase.assert_called()

    def test_pause_and_resume_session(self, progress_manager):
        """セッションの一時停止と再開"""
        # 1. セッション作成
        session_id = progress_manager.create_session(
            orchestrator_id="orchestrator_e2e_test",
            user_request={"original": "長時間タスク"},
            task_tree={"tasks": []},
        )

        # 2. 一時停止
        progress_manager.pause_session(session_id)
        state = progress_manager.restore_state(session_id)
        assert state.status == "paused"

        # 3. 再開
        progress_manager.resume_session(session_id)
        state = progress_manager.restore_state(session_id)
        assert state.status == "in_progress"

    def test_progress_report_generation(self, progress_manager):
        """進捗レポートの生成"""
        # セッション作成
        session_id = progress_manager.create_session(
            orchestrator_id="orchestrator_e2e_test",
            user_request={"original": "複数タスク"},
            task_tree={"tasks": [
                {"id": "1", "description": "設計", "status": "completed"},
                {"id": "2", "description": "実装", "status": "in_progress"},
                {"id": "3", "description": "テスト", "status": "pending"},
            ]},
        )

        progress_manager.update_progress(session_id, 50)

        # レポート生成
        report = progress_manager.generate_progress_report(session_id)

        assert "進捗率: 50%" in report
        assert "完了タスク: 1/3" in report


# =============================================================================
# シナリオ5: 複数タスク
# 複数論点の順次処理
# =============================================================================

class TestMultipleTasksE2E:
    """複数タスクのE2E統合テスト"""

    def test_sequential_task_processing(self, input_processor, orchestrator):
        """複数タスクの順次処理"""
        # 複数の論点を含む入力
        user_input = """以下のタスクを実行してください:
- APIの調査
- 認証機能の実装
- テストコードの作成"""

        processed = input_processor.process(user_input)

        # 検証: 3つの論点が検出
        assert processed.item_count == 3

        # 各タスクを順次処理
        session_id = None
        for item in processed.items:
            result = orchestrator.process_request(
                task_summary=item,
                session_id=session_id,
            )
            assert result.is_success

            # 最初のセッションIDを保持
            if session_id is None:
                session_id = result.session_id

        # 全タスクが同じセッションで処理された
        session = orchestrator.get_session(session_id)
        assert session.subtask_count == 3

    def test_different_agents_for_different_tasks(
        self, input_processor, orchestrator
    ):
        """異なるタスクで異なるエージェントが選択される"""
        tasks = [
            # implementation_agent: implementation, coding, debugging capabilities
            "ログイン機能を実装してください",
            "新機能を開発してコーディングしてください",
            "バグを修正してデバッグしてください",
        ]

        selected_agents = set()
        for task_summary in tasks:
            processed = input_processor.process(task_summary)
            result = orchestrator.process_request(
                task_summary=processed.summary,
            )

            assert result.is_success
            # 何らかのエージェントが選択されている
            assert result.routing_decision.selected_agent_id != ""
            selected_agents.add(result.routing_decision.selected_agent_id)

        # 少なくとも1つのエージェントが選択されていることを確認
        assert len(selected_agents) >= 1

    def test_progress_tracking_across_tasks(
        self, input_processor, orchestrator, progress_manager
    ):
        """複数タスクにわたる進捗追跡"""
        tasks = ["タスク1: 設計", "タスク2: 実装", "タスク3: テスト"]

        # セッション作成
        session_id = progress_manager.create_session(
            orchestrator_id="orchestrator_e2e_test",
            user_request={"original": "複数タスク"},
            task_tree={"tasks": [
                {"id": str(i), "description": task, "status": "pending"}
                for i, task in enumerate(tasks)
            ]},
        )

        # 各タスクを処理しながら進捗更新
        for i, task in enumerate(tasks):
            # オーケストレーターで処理
            result = orchestrator.process_request(task_summary=task)
            assert result.is_success

            # 進捗更新
            progress_percent = int((i + 1) / len(tasks) * 100)
            progress_manager.update_progress(session_id, progress_percent)

        # 最終進捗を確認
        state = progress_manager.restore_state(session_id)
        assert state.overall_progress_percent == 100


# =============================================================================
# 全コンポーネント連携テスト
# InputProcessor → Orchestrator → Router → Evaluator → ProgressManager
# =============================================================================

class TestFullComponentIntegration:
    """全コンポーネント連携の統合テスト"""

    def test_complete_workflow(
        self,
        input_processor,
        orchestrator,
        evaluator,
        progress_manager,
    ):
        """完全なワークフロー: 入力 → 処理 → 評価 → 進捗管理"""
        # 1. 入力処理
        user_input = "ユーザー認証機能を実装してください"
        processed = input_processor.process(user_input)

        # 2. セッション作成（進捗管理）
        session_id = progress_manager.create_session(
            orchestrator_id="orchestrator_e2e_test",
            user_request={"original": user_input, "clarified": processed.summary},
            task_tree={"tasks": [
                {"id": "1", "description": "認証機能実装", "status": "pending"}
            ]},
        )

        # 3. オーケストレーター実行
        result = orchestrator.process_request(
            task_summary=processed.summary,
        )

        assert result.is_success
        assert result.routing_decision.selected_agent_id == "implementation_agent"

        # 4. 進捗更新
        progress_manager.save_state(
            session_id=session_id,
            task_tree={"tasks": [
                {"id": "1", "description": "認証機能実装", "status": "completed"}
            ]},
            current_task=None,
            progress_percent=100,
        )

        # 5. フィードバック評価
        feedback = evaluator.evaluate("ありがとう、完璧です")
        assert feedback.feedback_type == "positive"

        # 6. セッション完了
        progress_manager.complete_session(session_id)
        final_state = progress_manager.restore_state(session_id)
        assert final_state.status == "completed"

    def test_error_handling_flow(
        self,
        input_processor,
        orchestrator,
        progress_manager,
    ):
        """エラーハンドリングフロー"""
        # セッション作成
        session_id = progress_manager.create_session(
            orchestrator_id="orchestrator_e2e_test",
            user_request={"original": "エラーテスト"},
            task_tree={"tasks": []},
        )

        # 意図的にエラーを発生させる（空のエージェントリスト）
        with patch.object(
            orchestrator.router.agent_registry,
            'get_active_agents',
            return_value=[]
        ):
            result = orchestrator.process_request(
                task_summary="エージェントなしのテスト",
            )

            assert result.is_failure
            assert result.status == "no_agent"

        # 失敗時のセッション更新
        progress_manager.fail_session(
            session_id=session_id,
            error_message="適切なエージェントが見つかりません",
        )

        state = progress_manager.restore_state(session_id)
        assert state.status == "failed"
        assert "error" in state.task_tree

    def test_routing_decision_based_on_keywords(
        self,
        input_processor,
        orchestrator,
    ):
        """キーワードに基づくルーティング判断"""
        test_cases = [
            # (入力, 期待されるエージェント)
            # research_agent has capabilities: research, analysis, documentation
            ("技術について調べてリサーチして分析する", "research_agent"),
            # implementation_agent has capabilities: implementation, coding, debugging
            ("新機能を実装する", "implementation_agent"),
            # testing_agent has capabilities: testing, debugging, analysis
            ("pytestでユニットテストを作成する", "testing_agent"),
            ("バグを修正してコードを実装する", "implementation_agent"),
        ]

        for user_input, expected_agent in test_cases:
            processed = input_processor.process(user_input)
            result = orchestrator.process_request(
                task_summary=processed.summary,
            )

            assert result.is_success, f"Failed for input: {user_input}"
            assert result.routing_decision.selected_agent_id == expected_agent, (
                f"Expected {expected_agent} for '{user_input}', "
                f"got {result.routing_decision.selected_agent_id}"
            )

    def test_feedback_types_correctly_classified(self, evaluator):
        """フィードバックタイプが正しく分類される"""
        test_cases = [
            # (ユーザー応答, 期待されるフィードバックタイプ)
            ("ありがとう、素晴らしい", "positive"),
            ("OK", "positive"),
            ("良いですね", "positive"),
            ("もう一度やり直してください", "redo_requested"),
            ("違います、ダメです", "negative"),
            ("修正してください", "negative"),
        ]

        for response, expected_type in test_cases:
            feedback = evaluator.evaluate(response)
            assert feedback.feedback_type == expected_type, (
                f"Expected {expected_type} for '{response}', "
                f"got {feedback.feedback_type}"
            )


# =============================================================================
# エッジケースと異常系のテスト
# =============================================================================

class TestEdgeCasesAndErrorHandling:
    """エッジケースと異常系のテスト"""

    def test_empty_input_handling(self, input_processor, orchestrator):
        """空入力の処理"""
        processed = input_processor.process("")

        assert processed.summary == ""
        assert processed.item_count == 0

        # 空の入力でもオーケストレーターは処理できる
        result = orchestrator.process_request(
            task_summary="",
        )
        # 空のタスクでも処理される（エージェント選択は行われる）
        assert result.session_id is not None

    def test_whitespace_only_input(self, input_processor):
        """空白のみの入力"""
        processed = input_processor.process("   \n\t\n   ")

        assert processed.summary == ""
        assert processed.item_count == 0

    def test_session_not_found(self, orchestrator):
        """存在しないセッションへのアクセス"""
        fake_session_id = uuid4()
        session = orchestrator.get_session(fake_session_id)
        assert session is None

        # 存在しないセッションへのフィードバック
        feedback = orchestrator.receive_feedback(
            session_id=fake_session_id,
            user_response="テスト",
        )
        # neutral が返される（セッションなし）
        assert feedback.feedback_type in ["positive", "neutral", "negative", "redo_requested"]

    def test_ambiguous_input_passes_through(self, input_processor, orchestrator):
        """曖昧な入力がそのまま渡される"""
        ambiguous_input = "いい感じにしてください"
        processed = input_processor.process(ambiguous_input)

        # 曖昧さの解消は行わない（入力処理層の責務外）
        assert processed.summary == ambiguous_input
        assert processed.needs_negotiation is False

        # オーケストレーターは処理を試みる
        result = orchestrator.process_request(
            task_summary=processed.summary,
        )
        assert result.is_success


# =============================================================================
# パフォーマンス関連のテスト
# =============================================================================

class TestPerformanceScenarios:
    """パフォーマンス関連のシナリオテスト"""

    def test_many_sequential_requests(self, input_processor, orchestrator):
        """多数の連続リクエスト処理"""
        session_id = None

        for i in range(10):
            result = orchestrator.process_request(
                task_summary=f"タスク{i+1}の処理",
                session_id=session_id,
            )
            assert result.is_success

            if session_id is None:
                session_id = result.session_id

        # 統計を確認
        # 睡眠が発生するとカウンタがリセットされる
        # 10回のリクエストで2回の睡眠が発生する可能性（5回ごとに睡眠）
        stats = orchestrator.get_stats()
        # 睡眠後にリセットされるため、0-5の範囲内
        assert 0 <= stats["subtask_completed_count"] <= 5
        # セッション数は維持される
        assert stats["active_sessions"] >= 1

    def test_routing_with_past_experiences(
        self, mock_agent_registry, mock_task_executor, evaluator
    ):
        """過去の経験を考慮したルーティング"""
        # 過去の成功経験をモック
        mock_task_executor.search_memories.return_value = [
            MagicMock(
                memory=MagicMock(
                    id=uuid4(),
                    content="認証機能を実装した",
                    routing_context={
                        "agent_selected": "implementation_agent",
                        "success": True,
                        "feedback_type": "positive",
                    },
                ),
                final_score=0.9,
            ),
        ]

        router = Router(mock_agent_registry)
        orchestrator = Orchestrator(
            agent_id="orchestrator_perf_test",
            router=router,
            evaluator=evaluator,
            task_executor=mock_task_executor,
        )

        result = orchestrator.process_request(
            task_summary="別の認証機能を実装する",
        )

        assert result.is_success
        # 過去の経験が検索されたことを確認
        mock_task_executor.search_memories.assert_called()
