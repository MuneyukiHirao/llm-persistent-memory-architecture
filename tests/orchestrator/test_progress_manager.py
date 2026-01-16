# ProgressManager テスト
# 実装仕様: docs/phase2-implementation-spec.ja.md セクション5.5
"""
ProgressManager クラスのユニットテスト

テスト内容:
- SessionState dataclass のバリデーションと変換
- SessionStateRepository のCRUD操作（モック）
- ProgressManager の進捗管理機能
- 進捗レポート生成
- セッション状態管理（完了、失敗、一時停止、再開）
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

from src.orchestrator.progress_manager import (
    SessionState,
    SessionStateRepository,
    ProgressManager,
)
from src.config.phase2_config import Phase2Config


class TestSessionState:
    """SessionState のテスト"""

    def test_create_session_state(self):
        """SessionState の作成"""
        session_id = uuid4()
        state = SessionState(
            session_id=session_id,
            orchestrator_id="orchestrator_01",
            user_request={"original": "タスク実行", "clarified": "具体的なタスク"},
            task_tree={"tasks": []},
        )

        assert state.session_id == session_id
        assert state.orchestrator_id == "orchestrator_01"
        assert state.user_request["original"] == "タスク実行"
        assert state.overall_progress_percent == 0
        assert state.status == "in_progress"

    def test_session_state_with_current_task(self):
        """current_task を含む SessionState"""
        state = SessionState(
            session_id=uuid4(),
            orchestrator_id="orchestrator_01",
            user_request={"original": "test"},
            task_tree={"tasks": [{"id": "task_1", "description": "タスク1"}]},
            current_task={"id": "task_1", "description": "タスク1"},
            overall_progress_percent=50,
        )

        assert state.current_task is not None
        assert state.current_task["id"] == "task_1"
        assert state.overall_progress_percent == 50

    def test_session_state_validation_progress_percent(self):
        """progress_percent のバリデーション"""
        # 負の値
        with pytest.raises(ValueError, match="overall_progress_percent must be 0-100"):
            SessionState(
                session_id=uuid4(),
                orchestrator_id="test",
                user_request={},
                task_tree={},
                overall_progress_percent=-1,
            )

        # 100超
        with pytest.raises(ValueError, match="overall_progress_percent must be 0-100"):
            SessionState(
                session_id=uuid4(),
                orchestrator_id="test",
                user_request={},
                task_tree={},
                overall_progress_percent=101,
            )

    def test_session_state_validation_status(self):
        """status のバリデーション"""
        with pytest.raises(ValueError, match="status must be one of"):
            SessionState(
                session_id=uuid4(),
                orchestrator_id="test",
                user_request={},
                task_tree={},
                status="invalid_status",
            )

    def test_to_dict(self):
        """to_dict メソッドのテスト"""
        session_id = uuid4()
        now = datetime.now()
        state = SessionState(
            session_id=session_id,
            orchestrator_id="orchestrator_01",
            user_request={"original": "test"},
            task_tree={"tasks": []},
            current_task={"id": "task_1"},
            overall_progress_percent=75,
            status="in_progress",
            created_at=now,
            updated_at=now,
            last_activity_at=now,
        )

        result = state.to_dict()

        assert result["session_id"] == str(session_id)
        assert result["orchestrator_id"] == "orchestrator_01"
        assert result["overall_progress_percent"] == 75
        assert result["status"] == "in_progress"
        assert result["current_task"]["id"] == "task_1"

    def test_from_dict(self):
        """from_dict メソッドのテスト"""
        session_id = uuid4()
        now = datetime.now()
        data = {
            "session_id": str(session_id),
            "orchestrator_id": "orchestrator_01",
            "user_request": {"original": "test"},
            "task_tree": {"tasks": []},
            "current_task": {"id": "task_1"},
            "overall_progress_percent": 75,
            "status": "in_progress",
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "last_activity_at": now.isoformat(),
        }

        state = SessionState.from_dict(data)

        assert state.session_id == session_id
        assert state.orchestrator_id == "orchestrator_01"
        assert state.overall_progress_percent == 75
        assert state.current_task["id"] == "task_1"


class MockDatabaseConnection:
    """テスト用のモックDatabaseConnection"""

    def __init__(self):
        self.cursor_mock = MagicMock()
        self.connection_mock = MagicMock()

    def get_cursor(self):
        return self._cursor_context()

    def _cursor_context(self):
        return MockCursorContext(self.cursor_mock)


class MockCursorContext:
    """モックカーソルコンテキスト"""

    def __init__(self, cursor_mock):
        self.cursor_mock = cursor_mock

    def __enter__(self):
        return self.cursor_mock

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class TestSessionStateRepository:
    """SessionStateRepository のテスト"""

    @pytest.fixture
    def mock_db(self):
        """モックDatabaseConnection"""
        return MockDatabaseConnection()

    @pytest.fixture
    def repository(self, mock_db):
        """テスト用Repository"""
        return SessionStateRepository(mock_db)

    def test_create_session(self, repository, mock_db):
        """セッション作成のテスト"""
        session_id = uuid4()
        state = SessionState(
            session_id=session_id,
            orchestrator_id="orchestrator_01",
            user_request={"original": "test"},
            task_tree={"tasks": []},
        )

        # モックの戻り値を設定
        mock_db.cursor_mock.fetchone.return_value = (str(session_id),)

        result = repository.create(state)

        assert result == session_id
        assert mock_db.cursor_mock.execute.called

    def test_get_by_id_found(self, repository, mock_db):
        """セッション取得（存在する場合）"""
        session_id = uuid4()
        now = datetime.now()

        mock_db.cursor_mock.fetchone.return_value = (
            str(session_id),
            "orchestrator_01",
            {"original": "test"},
            {"tasks": []},
            None,  # current_task
            50,
            "in_progress",
            now,
            now,
            now,
        )

        result = repository.get_by_id(session_id)

        assert result is not None
        assert result.session_id == session_id
        assert result.orchestrator_id == "orchestrator_01"
        assert result.overall_progress_percent == 50

    def test_get_by_id_not_found(self, repository, mock_db):
        """セッション取得（存在しない場合）"""
        mock_db.cursor_mock.fetchone.return_value = None

        result = repository.get_by_id(uuid4())

        assert result is None

    def test_update_session(self, repository, mock_db):
        """セッション更新のテスト"""
        session_id = uuid4()
        mock_db.cursor_mock.rowcount = 1

        result = repository.update(
            session_id=session_id,
            task_tree={"tasks": [{"id": "1", "status": "completed"}]},
            overall_progress_percent=75,
        )

        assert result is True
        assert mock_db.cursor_mock.execute.called

    def test_list_by_status(self, repository, mock_db):
        """ステータスでセッション一覧取得"""
        session_id = uuid4()
        now = datetime.now()

        mock_db.cursor_mock.fetchall.return_value = [
            (
                str(session_id),
                "orchestrator_01",
                {"original": "test"},
                {"tasks": []},
                None,
                50,
                "in_progress",
                now,
                now,
                now,
            )
        ]

        result = repository.list_by_status("in_progress")

        assert len(result) == 1
        assert result[0].session_id == session_id

    def test_delete_session(self, repository, mock_db):
        """セッション削除のテスト"""
        mock_db.cursor_mock.rowcount = 1

        result = repository.delete(uuid4())

        assert result is True


class MockSessionStateRepository:
    """テスト用のモックSessionStateRepository"""

    def __init__(self):
        self.sessions = {}

    def create(self, state: SessionState) -> UUID:
        self.sessions[state.session_id] = state
        return state.session_id

    def get_by_id(self, session_id: UUID):
        return self.sessions.get(session_id)

    def update(
        self,
        session_id: UUID,
        task_tree=None,
        current_task=None,
        overall_progress_percent=None,
        status=None,
        last_activity_at=None,
    ) -> bool:
        state = self.sessions.get(session_id)
        if not state:
            return False

        if task_tree is not None:
            state.task_tree = task_tree
        if current_task is not None:
            state.current_task = current_task
        if overall_progress_percent is not None:
            state.overall_progress_percent = overall_progress_percent
        if status is not None:
            state.status = status
        if last_activity_at is not None:
            state.last_activity_at = last_activity_at

        state.updated_at = datetime.now()
        return True

    def clear_current_task(self, session_id: UUID) -> bool:
        state = self.sessions.get(session_id)
        if state:
            state.current_task = None
            return True
        return False

    def list_by_status(self, status: str):
        return [s for s in self.sessions.values() if s.status == status]

    def list_by_orchestrator(self, orchestrator_id: str):
        return [s for s in self.sessions.values() if s.orchestrator_id == orchestrator_id]

    def delete(self, session_id: UUID) -> bool:
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False


class TestProgressManager:
    """ProgressManager のテスト"""

    @pytest.fixture
    def mock_repository(self):
        """モックRepository"""
        return MockSessionStateRepository()

    @pytest.fixture
    def manager(self, mock_repository):
        """テスト用ProgressManager"""
        return ProgressManager(
            session_repository=mock_repository,
            config=Phase2Config(),
        )

    def test_create_session(self, manager, mock_repository):
        """新規セッション作成"""
        session_id = manager.create_session(
            orchestrator_id="orchestrator_01",
            user_request={"original": "タスク実行"},
            task_tree={"tasks": []},
        )

        assert session_id is not None
        assert session_id in mock_repository.sessions

        state = mock_repository.sessions[session_id]
        assert state.orchestrator_id == "orchestrator_01"
        assert state.overall_progress_percent == 0
        assert state.status == "in_progress"

    def test_create_session_with_custom_id(self, manager, mock_repository):
        """カスタムIDでセッション作成"""
        custom_id = uuid4()
        session_id = manager.create_session(
            orchestrator_id="orchestrator_01",
            user_request={"original": "test"},
            task_tree={"tasks": []},
            session_id=custom_id,
        )

        assert session_id == custom_id

    def test_save_state(self, manager, mock_repository):
        """進捗状態の保存"""
        # セッション作成
        session_id = manager.create_session(
            orchestrator_id="orchestrator_01",
            user_request={"original": "test"},
            task_tree={"tasks": [{"id": "1", "status": "pending"}]},
        )

        # 状態保存
        result = manager.save_state(
            session_id=session_id,
            task_tree={"tasks": [{"id": "1", "status": "completed"}]},
            current_task=None,
            progress_percent=100,
        )

        assert result is True

        state = mock_repository.sessions[session_id]
        assert state.overall_progress_percent == 100
        assert state.task_tree["tasks"][0]["status"] == "completed"

    def test_restore_state(self, manager, mock_repository):
        """進捗状態の復元"""
        # セッション作成
        session_id = manager.create_session(
            orchestrator_id="orchestrator_01",
            user_request={"original": "test"},
            task_tree={"tasks": []},
        )

        # 状態復元
        state = manager.restore_state(session_id)

        assert state is not None
        assert state.session_id == session_id
        assert state.orchestrator_id == "orchestrator_01"

    def test_restore_state_not_found(self, manager):
        """存在しないセッションの復元"""
        state = manager.restore_state(uuid4())
        assert state is None

    def test_generate_progress_report(self, manager, mock_repository):
        """進捗レポート生成"""
        # セッション作成
        session_id = manager.create_session(
            orchestrator_id="orchestrator_01",
            user_request={"original": "test"},
            task_tree={
                "tasks": [
                    {"id": "1", "description": "タスク1", "status": "completed"},
                    {"id": "2", "description": "タスク2", "status": "in_progress"},
                    {"id": "3", "description": "タスク3", "status": "pending"},
                ]
            },
        )

        # 進捗更新
        manager.update_progress(session_id, 33)

        # レポート生成
        report = manager.generate_progress_report(session_id)

        assert "進捗報告" in report
        assert "33%" in report
        assert "1/3" in report
        assert "タスク1" in report
        assert "タスク2" in report
        assert "[x]" in report  # completed
        assert "[>]" in report  # in_progress
        assert "[ ]" in report  # pending

    def test_generate_progress_report_not_found(self, manager):
        """存在しないセッションのレポート"""
        report = manager.generate_progress_report(uuid4())
        assert "セッションが見つかりません" in report

    def test_list_active_sessions(self, manager, mock_repository):
        """アクティブセッション一覧"""
        # 複数セッション作成
        id1 = manager.create_session(
            orchestrator_id="orchestrator_01",
            user_request={"original": "test1"},
            task_tree={"tasks": []},
        )
        id2 = manager.create_session(
            orchestrator_id="orchestrator_01",
            user_request={"original": "test2"},
            task_tree={"tasks": []},
        )

        # 1つを一時停止
        manager.pause_session(id1)

        # アクティブセッション取得
        sessions = manager.list_active_sessions()

        assert len(sessions) == 2  # in_progress と paused 両方含む

    def test_update_progress(self, manager, mock_repository):
        """進捗率の更新"""
        session_id = manager.create_session(
            orchestrator_id="orchestrator_01",
            user_request={"original": "test"},
            task_tree={"tasks": []},
        )

        result = manager.update_progress(session_id, 50)

        assert result is True
        assert mock_repository.sessions[session_id].overall_progress_percent == 50

    def test_update_progress_invalid_percent(self, manager, mock_repository):
        """無効な進捗率"""
        session_id = manager.create_session(
            orchestrator_id="orchestrator_01",
            user_request={"original": "test"},
            task_tree={"tasks": []},
        )

        with pytest.raises(ValueError, match="progress_percent must be 0-100"):
            manager.update_progress(session_id, 150)

    def test_update_status(self, manager, mock_repository):
        """ステータス更新"""
        session_id = manager.create_session(
            orchestrator_id="orchestrator_01",
            user_request={"original": "test"},
            task_tree={"tasks": []},
        )

        result = manager.update_status(session_id, "paused")

        assert result is True
        assert mock_repository.sessions[session_id].status == "paused"

    def test_update_status_invalid(self, manager, mock_repository):
        """無効なステータス"""
        session_id = manager.create_session(
            orchestrator_id="orchestrator_01",
            user_request={"original": "test"},
            task_tree={"tasks": []},
        )

        with pytest.raises(ValueError, match="status must be one of"):
            manager.update_status(session_id, "invalid")

    def test_complete_session(self, manager, mock_repository):
        """セッション完了"""
        session_id = manager.create_session(
            orchestrator_id="orchestrator_01",
            user_request={"original": "test"},
            task_tree={"tasks": []},
        )

        # 進行中のタスクを設定
        mock_repository.sessions[session_id].current_task = {"id": "task_1"}

        result = manager.complete_session(session_id)

        assert result is True
        state = mock_repository.sessions[session_id]
        assert state.status == "completed"
        assert state.overall_progress_percent == 100
        assert state.current_task is None

    def test_fail_session(self, manager, mock_repository):
        """セッション失敗"""
        session_id = manager.create_session(
            orchestrator_id="orchestrator_01",
            user_request={"original": "test"},
            task_tree={"tasks": []},
        )

        result = manager.fail_session(session_id, "エラーが発生しました")

        assert result is True
        state = mock_repository.sessions[session_id]
        assert state.status == "failed"
        assert "error" in state.task_tree
        assert state.task_tree["error"]["message"] == "エラーが発生しました"

    def test_pause_session(self, manager, mock_repository):
        """セッション一時停止"""
        session_id = manager.create_session(
            orchestrator_id="orchestrator_01",
            user_request={"original": "test"},
            task_tree={"tasks": []},
        )

        result = manager.pause_session(session_id)

        assert result is True
        assert mock_repository.sessions[session_id].status == "paused"

    def test_resume_session(self, manager, mock_repository):
        """セッション再開"""
        session_id = manager.create_session(
            orchestrator_id="orchestrator_01",
            user_request={"original": "test"},
            task_tree={"tasks": []},
        )
        manager.pause_session(session_id)

        result = manager.resume_session(session_id)

        assert result is True
        assert mock_repository.sessions[session_id].status == "in_progress"

    def test_count_tasks(self, manager):
        """タスクカウントのテスト"""
        task_tree = {
            "tasks": [
                {"id": "1", "status": "completed"},
                {"id": "2", "status": "completed"},
                {"id": "3", "status": "in_progress"},
                {"id": "4", "status": "pending"},
            ]
        }

        completed = manager._count_completed_tasks(task_tree)
        total = manager._count_total_tasks(task_tree)

        assert completed == 2
        assert total == 4

    def test_format_task_tree(self, manager):
        """タスクツリーフォーマットのテスト"""
        task_tree = {
            "tasks": [
                {"id": "1", "description": "完了タスク", "status": "completed"},
                {"id": "2", "description": "進行中タスク", "status": "in_progress"},
                {"id": "3", "description": "保留タスク", "status": "pending"},
            ]
        }

        result = manager._format_task_tree(task_tree)

        assert "[x] 完了タスク" in result
        assert "[>] 進行中タスク" in result
        assert "[ ] 保留タスク" in result

    def test_format_task_tree_empty(self, manager):
        """空タスクツリーのフォーマット"""
        result = manager._format_task_tree({"tasks": []})
        assert "タスクなし" in result

    def test_get_status_icon(self, manager):
        """ステータスアイコンのテスト"""
        assert manager._get_status_icon("pending") == "[ ]"
        assert manager._get_status_icon("in_progress") == "[>]"
        assert manager._get_status_icon("completed") == "[x]"
        assert manager._get_status_icon("failed") == "[!]"
        assert manager._get_status_icon("skipped") == "[-]"
        assert manager._get_status_icon("unknown") == "[?]"


class TestProgressManagerIntegration:
    """ProgressManager 統合テスト（シナリオベース）"""

    @pytest.fixture
    def mock_repository(self):
        return MockSessionStateRepository()

    @pytest.fixture
    def manager(self, mock_repository):
        return ProgressManager(
            session_repository=mock_repository,
            config=Phase2Config(),
        )

    def test_full_session_lifecycle(self, manager, mock_repository):
        """セッションの完全なライフサイクル"""
        # 1. セッション作成
        session_id = manager.create_session(
            orchestrator_id="orchestrator_01",
            user_request={"original": "機能Aを実装して"},
            task_tree={
                "tasks": [
                    {"id": "1", "description": "設計", "status": "pending"},
                    {"id": "2", "description": "実装", "status": "pending"},
                    {"id": "3", "description": "テスト", "status": "pending"},
                ]
            },
        )

        # 初期状態確認
        state = manager.restore_state(session_id)
        assert state.status == "in_progress"
        assert state.overall_progress_percent == 0

        # 2. タスク1完了
        manager.save_state(
            session_id=session_id,
            task_tree={
                "tasks": [
                    {"id": "1", "description": "設計", "status": "completed"},
                    {"id": "2", "description": "実装", "status": "in_progress"},
                    {"id": "3", "description": "テスト", "status": "pending"},
                ]
            },
            current_task={"id": "2", "description": "実装"},
            progress_percent=33,
        )

        # 3. 進捗レポート確認
        report = manager.generate_progress_report(session_id)
        assert "33%" in report
        assert "1/3" in report

        # 4. 一時停止
        manager.pause_session(session_id)
        assert mock_repository.sessions[session_id].status == "paused"

        # 5. 再開
        manager.resume_session(session_id)
        assert mock_repository.sessions[session_id].status == "in_progress"

        # 6. 残りのタスク完了
        manager.save_state(
            session_id=session_id,
            task_tree={
                "tasks": [
                    {"id": "1", "description": "設計", "status": "completed"},
                    {"id": "2", "description": "実装", "status": "completed"},
                    {"id": "3", "description": "テスト", "status": "completed"},
                ]
            },
            current_task=None,
            progress_percent=100,
        )

        # 7. セッション完了
        manager.complete_session(session_id)

        final_state = manager.restore_state(session_id)
        assert final_state.status == "completed"
        assert final_state.overall_progress_percent == 100
        assert final_state.current_task is None

    def test_session_failure_scenario(self, manager, mock_repository):
        """セッション失敗シナリオ"""
        # セッション作成
        session_id = manager.create_session(
            orchestrator_id="orchestrator_01",
            user_request={"original": "test"},
            task_tree={
                "tasks": [
                    {"id": "1", "description": "タスク1", "status": "in_progress"},
                ]
            },
        )

        # 進捗更新
        manager.update_progress(session_id, 50)

        # エラー発生
        manager.fail_session(session_id, "DBエラーが発生しました")

        state = manager.restore_state(session_id)
        assert state.status == "failed"
        assert "error" in state.task_tree
        assert "DBエラー" in state.task_tree["error"]["message"]

        # アクティブセッションから除外されている
        active = manager.list_active_sessions()
        assert session_id not in [s.session_id for s in active]

    def test_multiple_sessions_management(self, manager, mock_repository):
        """複数セッションの管理"""
        # 複数セッション作成
        ids = []
        for i in range(3):
            sid = manager.create_session(
                orchestrator_id=f"orchestrator_{i}",
                user_request={"original": f"test_{i}"},
                task_tree={"tasks": []},
            )
            ids.append(sid)

        # すべてアクティブ
        active = manager.list_active_sessions()
        assert len(active) == 3

        # 1つ完了
        manager.complete_session(ids[0])

        # 1つ失敗
        manager.fail_session(ids[1], "エラー")

        # アクティブは1つだけ
        active = manager.list_active_sessions()
        assert len(active) == 1
        assert active[0].session_id == ids[2]
