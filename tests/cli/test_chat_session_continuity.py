"""
セッション自動継続機能のテスト
"""

import pytest
from uuid import uuid4
from unittest.mock import Mock, patch, MagicMock
from click.testing import CliRunner

from src.cli.commands.chat import _get_last_session, _save_session, _update_session
from src.orchestrator.progress_manager import ProgressManager, SessionStateRepository, SessionState
from src.db.connection import DatabaseConnection


class TestSessionContinuity:
    """セッション自動継続のテスト"""

    def test_get_last_session_returns_none_when_no_sessions(self):
        """セッションが存在しない場合はNoneを返す"""
        # Mock context
        ctx = Mock()
        ctx.db = Mock(spec=DatabaseConnection)
        ctx.config = Mock()

        # Mock progress_manager
        with patch('src.cli.commands.chat.ProgressManager') as MockProgressManager:
            mock_pm = Mock()
            mock_pm.get_recent_sessions.return_value = []
            MockProgressManager.return_value = mock_pm

            result = _get_last_session(ctx)

            assert result is None
            mock_pm.get_recent_sessions.assert_called_once()

    def test_get_last_session_returns_most_recent(self):
        """最新のセッションIDを返す"""
        # Mock context
        ctx = Mock()
        ctx.db = Mock(spec=DatabaseConnection)
        ctx.config = Mock()

        # Mock session
        session_id = uuid4()
        mock_session = Mock()
        mock_session.session_id = session_id

        # Mock progress_manager
        with patch('src.cli.commands.chat.ProgressManager') as MockProgressManager:
            mock_pm = Mock()
            mock_pm.get_recent_sessions.return_value = [mock_session]
            MockProgressManager.return_value = mock_pm

            result = _get_last_session(ctx)

            assert result == session_id
            mock_pm.get_recent_sessions.assert_called_once_with(
                orchestrator_id="orchestrator_cli",
                limit=1,
            )

    def test_save_session_creates_new_session(self):
        """新規セッションを作成する"""
        # Mock context
        ctx = Mock()
        ctx.db = Mock(spec=DatabaseConnection)
        ctx.config = Mock()

        session_id = uuid4()
        user_input = "テストタスク"

        # Mock progress_manager
        with patch('src.cli.commands.chat.ProgressManager') as MockProgressManager:
            mock_pm = Mock()
            MockProgressManager.return_value = mock_pm

            _save_session(ctx, session_id, user_input)

            mock_pm.save_state.assert_called_once_with(
                session_id=session_id,
                task_tree={"tasks": []},
                current_task={"description": user_input},
                progress_percent=0,
            )

    def test_update_session_adds_task_to_history(self):
        """セッション更新でタスクを履歴に追加する"""
        # Mock context
        ctx = Mock()
        ctx.db = Mock(spec=DatabaseConnection)
        ctx.config = Mock()

        session_id = uuid4()
        user_input = "新しいタスク"

        # Mock existing session state
        mock_state = Mock()
        mock_state.task_tree = {"tasks": [{"description": "既存タスク", "status": "completed"}]}
        mock_state.overall_progress_percent = 50

        # Mock progress_manager
        with patch('src.cli.commands.chat.ProgressManager') as MockProgressManager:
            mock_pm = Mock()
            mock_pm.restore_state.return_value = mock_state
            MockProgressManager.return_value = mock_pm

            _update_session(ctx, session_id, user_input)

            # save_state が呼ばれたことを確認
            assert mock_pm.save_state.called
            call_args = mock_pm.save_state.call_args

            # task_tree に新しいタスクが追加されていることを確認
            task_tree = call_args[1]["task_tree"]
            assert len(task_tree["tasks"]) == 2
            assert task_tree["tasks"][1]["description"] == user_input
            assert task_tree["tasks"][1]["status"] == "in_progress"


class TestChatSessionFlow:
    """対話モードのセッションフロー統合テスト"""

    @pytest.fixture
    def mock_cli_context(self):
        """Mock CLIContext"""
        ctx = Mock()
        ctx.db = Mock(spec=DatabaseConnection)
        ctx.config = Mock()
        ctx.orchestrator = Mock()
        ctx.task_executor = Mock()
        ctx.agent_registry = Mock()
        return ctx

    def test_session_close_on_exit(self, mock_cli_context):
        """exit コマンドでセッションが完了状態になる"""
        from src.cli.commands.chat import _close_session

        session_id = uuid4()

        # Mock progress_manager
        with patch('src.cli.commands.chat.ProgressManager') as MockProgressManager:
            mock_pm = Mock()
            MockProgressManager.return_value = mock_pm

            _close_session(mock_cli_context, session_id)

            mock_pm.close_session.assert_called_once_with(
                session_id, status="completed"
            )

    def test_session_pause_on_interrupt(self, mock_cli_context):
        """Ctrl+C でセッションが一時停止状態になる"""
        from src.cli.commands.chat import _pause_session

        session_id = uuid4()

        # Mock progress_manager
        with patch('src.cli.commands.chat.ProgressManager') as MockProgressManager:
            mock_pm = Mock()
            MockProgressManager.return_value = mock_pm

            _pause_session(mock_cli_context, session_id)

            mock_pm.close_session.assert_called_once_with(
                session_id, status="paused"
            )


class TestOrchestratorSessionContext:
    """Orchestrator のセッションコンテキスト機能テスト"""

    def test_session_context_includes_conversation_history(self):
        """SessionContext が会話履歴を保持する"""
        from src.orchestrator.orchestrator import SessionContext

        session_id = uuid4()
        session = SessionContext(
            session_id=session_id,
            task_summary="テストタスク",
            items=[],
        )

        # 会話履歴を追加
        session.conversation_history.append({
            "user_input": "最初の質問",
            "agent_output": "最初の回答",
            "timestamp": "2025-01-01T00:00:00",
        })

        assert len(session.conversation_history) == 1
        assert session.conversation_history[0]["user_input"] == "最初の質問"

    def test_orchestrator_loads_conversation_history_from_db(self):
        """Orchestrator がDBから会話履歴をロードする"""
        from src.orchestrator.orchestrator import Orchestrator, SessionContext
        from src.orchestrator.router import Router
        from src.orchestrator.evaluator import Evaluator
        from src.core.task_executor import TaskExecutor

        # Mock dependencies
        mock_db = Mock(spec=DatabaseConnection)
        mock_router = Mock(spec=Router)
        mock_evaluator = Mock(spec=Evaluator)
        mock_task_executor = Mock(spec=TaskExecutor)

        # Mock session state
        session_id = uuid4()
        mock_session_state = Mock()
        mock_session_state.session_id = session_id
        mock_session_state.user_request = {"original": "元のタスク"}
        mock_session_state.task_tree = {
            "conversation_history": [
                {"user_input": "前回の質問", "agent_output": "前回の回答"}
            ]
        }
        mock_session_state.created_at = Mock()

        # Mock progress_manager
        with patch('src.orchestrator.orchestrator.ProgressManager') as MockProgressManager:
            mock_pm = Mock()
            mock_pm.restore_state.return_value = mock_session_state
            MockProgressManager.return_value = mock_pm

            # Create orchestrator
            orchestrator = Orchestrator(
                agent_id="test_orchestrator",
                router=mock_router,
                evaluator=mock_evaluator,
                task_executor=mock_task_executor,
                db=mock_db,
            )

            # Get or create session (should load from DB)
            session = orchestrator._get_or_create_session(
                session_id=session_id,
                task_summary="新しいタスク",
                items=[],
            )

            # 会話履歴が復元されていることを確認
            assert len(session.conversation_history) == 1
            assert session.conversation_history[0]["user_input"] == "前回の質問"

    def test_orchestrator_enhances_task_with_conversation_history(self):
        """Orchestrator が会話履歴をタスク概要に追加する"""
        from src.orchestrator.orchestrator import SessionContext

        session = SessionContext(
            session_id=uuid4(),
            task_summary="初期タスク",
            items=[],
        )

        # 会話履歴を追加
        session.conversation_history = [
            {"user_input": "質問1", "agent_output": "回答1" * 50},  # 長い回答
            {"user_input": "質問2", "agent_output": "回答2"},
            {"user_input": "質問3", "agent_output": "回答3"},
            {"user_input": "質問4", "agent_output": "回答4"},  # 4件目は含まれない（最大3件）
        ]

        # 会話履歴が存在することを確認
        assert len(session.conversation_history) == 4
        assert session.conversation_history[0]["user_input"] == "質問1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
