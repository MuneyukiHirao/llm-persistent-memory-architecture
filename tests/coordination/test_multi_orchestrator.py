# MultiOrchestratorCoordinator テスト
"""
MultiOrchestratorCoordinator クラスの単体テスト

テスト対象:
- MultiOrchestratorCoordinator クラス
- OrchestratorInfo データクラス
- OrchestratorStatus / HealthStatus Enum

テスト方針:
- DatabaseConnection をモック化
- 各メソッドの動作を独立してテスト
- async/await パターンのテスト
- フロー整合性（ロック取得・解放順序）を検証
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, AsyncMock
from uuid import uuid4, UUID

from src.coordination.multi_orchestrator import (
    MultiOrchestratorCoordinator,
    OrchestratorInfo,
    OrchestratorStatus,
    HealthStatus,
)
from src.config.phase3_config import Phase3Config
from src.db.connection import DatabaseConnection


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_db():
    """DatabaseConnection のモック"""
    db = MagicMock(spec=DatabaseConnection)

    # get_cursor のモック
    mock_cursor = MagicMock()
    mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
    mock_cursor.__exit__ = MagicMock(return_value=None)
    mock_cursor.execute = MagicMock()
    mock_cursor.fetchone = MagicMock(return_value=None)
    mock_cursor.fetchall = MagicMock(return_value=[])
    mock_cursor.rowcount = 0

    # get_connection のモック
    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=None)
    mock_conn.cursor = MagicMock(return_value=mock_cursor)
    mock_conn.commit = MagicMock()
    mock_conn.rollback = MagicMock()

    db.get_cursor = MagicMock(return_value=mock_cursor)
    db.get_connection = MagicMock(return_value=mock_conn)

    return db


@pytest.fixture
def config():
    """Phase3Config のフィクスチャ"""
    return Phase3Config(
        orchestrator_heartbeat_interval=30,
        orchestrator_failover_timeout=90,
        session_lock_timeout=300,
    )


@pytest.fixture
def coordinator(mock_db, config):
    """MultiOrchestratorCoordinator のフィクスチャ"""
    return MultiOrchestratorCoordinator(
        orchestrator_id="orch-001",
        db_connection=mock_db,
        config=config,
    )


# =============================================================================
# OrchestratorStatus / HealthStatus テスト
# =============================================================================

class TestEnums:
    """Enum のテスト"""

    def test_orchestrator_status_values(self):
        """OrchestratorStatus の値を確認"""
        assert OrchestratorStatus.ACTIVE.value == "active"
        assert OrchestratorStatus.SLEEPING.value == "sleeping"
        assert OrchestratorStatus.TERMINATED.value == "terminated"

    def test_health_status_values(self):
        """HealthStatus の値を確認"""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"


# =============================================================================
# OrchestratorInfo テスト
# =============================================================================

class TestOrchestratorInfo:
    """OrchestratorInfo データクラスのテスト"""

    def test_create_basic(self):
        """基本的な OrchestratorInfo の作成"""
        now = datetime.now()
        session_id = uuid4()

        info = OrchestratorInfo(
            orchestrator_id="orch-001",
            status=OrchestratorStatus.ACTIVE,
            current_load=2,
            max_capacity=10,
            active_sessions=1,
            session_ids=[session_id],
            last_heartbeat=now,
            health_status=HealthStatus.HEALTHY,
            instance_info={"hostname": "localhost"},
            created_at=now,
            updated_at=now,
        )

        assert info.orchestrator_id == "orch-001"
        assert info.status == OrchestratorStatus.ACTIVE
        assert info.current_load == 2
        assert info.max_capacity == 10
        assert info.active_sessions == 1
        assert session_id in info.session_ids
        assert info.health_status == HealthStatus.HEALTHY


# =============================================================================
# MultiOrchestratorCoordinator 初期化テスト
# =============================================================================

class TestMultiOrchestratorCoordinatorInit:
    """MultiOrchestratorCoordinator 初期化のテスト"""

    def test_init_basic(self, mock_db, config):
        """基本的な初期化"""
        coordinator = MultiOrchestratorCoordinator(
            orchestrator_id="orch-001",
            db_connection=mock_db,
            config=config,
        )

        assert coordinator.orchestrator_id == "orch-001"
        assert coordinator.db == mock_db
        assert coordinator.config == config
        assert coordinator.current_load == 0
        assert coordinator.active_sessions == []

    def test_init_default_config(self, mock_db):
        """デフォルト設定での初期化"""
        coordinator = MultiOrchestratorCoordinator(
            orchestrator_id="orch-002",
            db_connection=mock_db,
        )

        assert coordinator.config is not None
        assert coordinator.config.orchestrator_heartbeat_interval == 30
        assert coordinator.config.orchestrator_failover_timeout == 90


# =============================================================================
# register テスト
# =============================================================================

class TestRegister:
    """register メソッドのテスト"""

    @pytest.mark.asyncio
    async def test_register_basic(self, coordinator, mock_db):
        """基本的なオーケストレーター登録"""
        await coordinator.register()

        # get_cursor が呼ばれたことを確認
        mock_db.get_cursor.assert_called_once()

    @pytest.mark.asyncio
    async def test_register_executes_upsert(self, coordinator, mock_db):
        """UPSERT クエリが実行されることを確認"""
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=None)
        mock_db.get_cursor.return_value = mock_cursor

        await coordinator.register()

        mock_cursor.execute.assert_called_once()
        # INSERT INTO orchestrator_state を含むことを確認
        call_args = mock_cursor.execute.call_args
        assert "INSERT INTO orchestrator_state" in call_args[0][0]
        assert "ON CONFLICT" in call_args[0][0]


# =============================================================================
# acquire_session_lock テスト
# =============================================================================

class TestAcquireSessionLock:
    """acquire_session_lock メソッドのテスト"""

    @pytest.mark.asyncio
    async def test_acquire_lock_unlocked_session(self, coordinator, mock_db):
        """ロックされていないセッションのロック取得"""
        session_id = uuid4()

        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=None)
        mock_cursor.fetchone = MagicMock(return_value=(None, None))  # locked_by, lock_acquired_at
        mock_cursor.rowcount = 1

        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=None)
        mock_conn.cursor = MagicMock(return_value=mock_cursor)
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=None)

        mock_db.get_connection.return_value = mock_conn

        result = await coordinator.acquire_session_lock(session_id)

        assert result is True
        assert session_id in coordinator.active_sessions

    @pytest.mark.asyncio
    async def test_acquire_lock_own_lock(self, coordinator, mock_db):
        """自分が既に保持しているロックの再取得"""
        session_id = uuid4()

        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=None)
        mock_cursor.fetchone = MagicMock(return_value=("orch-001", datetime.now()))

        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=None)
        mock_conn.cursor = MagicMock(return_value=mock_cursor)
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=None)

        mock_db.get_connection.return_value = mock_conn

        result = await coordinator.acquire_session_lock(session_id)

        assert result is True

    @pytest.mark.asyncio
    async def test_acquire_lock_session_not_found(self, coordinator, mock_db):
        """存在しないセッションのロック取得試行"""
        session_id = uuid4()

        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=None)
        mock_cursor.fetchone = MagicMock(return_value=None)  # セッションなし

        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=None)
        mock_conn.cursor = MagicMock(return_value=mock_cursor)
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=None)

        mock_db.get_connection.return_value = mock_conn

        result = await coordinator.acquire_session_lock(session_id)

        assert result is False
        mock_conn.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_acquire_lock_with_custom_timeout(self, coordinator, mock_db):
        """カスタムタイムアウトでのロック取得"""
        session_id = uuid4()

        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=None)
        mock_cursor.fetchone = MagicMock(return_value=(None, None))

        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=None)
        mock_conn.cursor = MagicMock(return_value=mock_cursor)
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=None)

        mock_db.get_connection.return_value = mock_conn

        result = await coordinator.acquire_session_lock(session_id, timeout=600)

        assert result is True


# =============================================================================
# release_session_lock テスト
# =============================================================================

class TestReleaseSessionLock:
    """release_session_lock メソッドのテスト"""

    @pytest.mark.asyncio
    async def test_release_lock_basic(self, coordinator, mock_db):
        """基本的なロック解放"""
        session_id = uuid4()
        coordinator._active_session_ids.append(session_id)

        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=None)
        mock_db.get_cursor.return_value = mock_cursor

        await coordinator.release_session_lock(session_id)

        mock_cursor.execute.assert_called_once()
        assert session_id not in coordinator.active_sessions

    @pytest.mark.asyncio
    async def test_release_lock_updates_db(self, coordinator, mock_db):
        """DB更新が行われることを確認"""
        session_id = uuid4()

        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=None)
        mock_db.get_cursor.return_value = mock_cursor

        await coordinator.release_session_lock(session_id)

        call_args = mock_cursor.execute.call_args
        assert "UPDATE session_state" in call_args[0][0]
        assert "locked_by = NULL" in call_args[0][0]


# =============================================================================
# send_heartbeat テスト
# =============================================================================

class TestSendHeartbeat:
    """send_heartbeat メソッドのテスト"""

    @pytest.mark.asyncio
    async def test_send_heartbeat_basic(self, coordinator, mock_db):
        """基本的なハートビート送信"""
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=None)
        mock_db.get_cursor.return_value = mock_cursor

        await coordinator.send_heartbeat()

        mock_cursor.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_heartbeat_updates_timestamp(self, coordinator, mock_db):
        """last_heartbeat が更新されることを確認"""
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=None)
        mock_db.get_cursor.return_value = mock_cursor

        await coordinator.send_heartbeat()

        call_args = mock_cursor.execute.call_args
        assert "UPDATE orchestrator_state" in call_args[0][0]
        assert "last_heartbeat = NOW()" in call_args[0][0]


# =============================================================================
# detect_failed_orchestrators テスト
# =============================================================================

class TestDetectFailedOrchestrators:
    """detect_failed_orchestrators メソッドのテスト"""

    @pytest.mark.asyncio
    async def test_detect_failed_none(self, coordinator, mock_db):
        """失敗したオーケストレーターがない場合"""
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=None)
        mock_cursor.fetchall = MagicMock(return_value=[])
        mock_db.get_cursor.return_value = mock_cursor

        result = await coordinator.detect_failed_orchestrators()

        assert result == []

    @pytest.mark.asyncio
    async def test_detect_failed_some(self, coordinator, mock_db):
        """失敗したオーケストレーターがある場合"""
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=None)
        mock_cursor.fetchall = MagicMock(return_value=[
            ("orch-002",),
            ("orch-003",),
        ])
        mock_db.get_cursor.return_value = mock_cursor

        result = await coordinator.detect_failed_orchestrators()

        assert len(result) == 2
        assert "orch-002" in result
        assert "orch-003" in result


# =============================================================================
# takeover_sessions テスト
# =============================================================================

class TestTakeoverSessions:
    """takeover_sessions メソッドのテスト"""

    @pytest.mark.asyncio
    async def test_takeover_sessions_basic(self, coordinator, mock_db):
        """基本的なセッション引き継ぎ"""
        session_id = uuid4()

        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=None)
        mock_cursor.fetchall = MagicMock(return_value=[(str(session_id),)])
        mock_cursor.rowcount = 1

        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=None)
        mock_conn.cursor = MagicMock(return_value=mock_cursor)
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=None)

        mock_db.get_connection.return_value = mock_conn

        count = await coordinator.takeover_sessions("orch-002")

        assert count == 1
        assert session_id in coordinator.active_sessions

    @pytest.mark.asyncio
    async def test_takeover_sessions_marks_failed_terminated(self, coordinator, mock_db):
        """失敗したオーケストレーターが terminated にマークされることを確認"""
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=None)
        mock_cursor.fetchall = MagicMock(return_value=[])
        mock_cursor.rowcount = 0

        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=None)
        mock_conn.cursor = MagicMock(return_value=mock_cursor)
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=None)

        mock_db.get_connection.return_value = mock_conn

        await coordinator.takeover_sessions("orch-002")

        # UPDATE orchestrator_state SET status = 'terminated' が呼ばれたことを確認
        calls = mock_cursor.execute.call_args_list
        assert any("terminated" in str(call) for call in calls)


# =============================================================================
# deregister テスト
# =============================================================================

class TestDeregister:
    """deregister メソッドのテスト"""

    @pytest.mark.asyncio
    async def test_deregister_basic(self, coordinator, mock_db):
        """基本的な登録解除"""
        session_id = uuid4()
        coordinator._active_session_ids.append(session_id)

        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=None)

        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=None)
        mock_conn.cursor = MagicMock(return_value=mock_cursor)
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=None)

        mock_db.get_connection.return_value = mock_conn

        await coordinator.deregister()

        assert len(coordinator.active_sessions) == 0

    @pytest.mark.asyncio
    async def test_deregister_releases_all_locks(self, coordinator, mock_db):
        """全てのセッションロックが解放されることを確認"""
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=None)

        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=None)
        mock_conn.cursor = MagicMock(return_value=mock_cursor)
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=None)

        mock_db.get_connection.return_value = mock_conn

        await coordinator.deregister()

        # UPDATE session_state SET locked_by = NULL が呼ばれたことを確認
        calls = mock_cursor.execute.call_args_list
        assert any("locked_by = NULL" in str(call) for call in calls)


# =============================================================================
# update_load / プロパティテスト
# =============================================================================

class TestProperties:
    """プロパティのテスト"""

    def test_update_load(self, coordinator):
        """update_load メソッドのテスト"""
        assert coordinator.current_load == 0

        coordinator.update_load(5)

        assert coordinator.current_load == 5

    def test_active_sessions_returns_copy(self, coordinator):
        """active_sessions がコピーを返すことを確認"""
        session_id = uuid4()
        coordinator._active_session_ids.append(session_id)

        sessions = coordinator.active_sessions
        sessions.append(uuid4())  # 追加してもオリジナルに影響しない

        assert len(coordinator._active_session_ids) == 1


# =============================================================================
# 統合テスト
# =============================================================================

class TestIntegration:
    """統合テスト"""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self, mock_db, config):
        """オーケストレーターのフルライフサイクル"""
        # モックのセットアップ
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=None)
        mock_cursor.fetchone = MagicMock(return_value=(None, None))
        mock_cursor.fetchall = MagicMock(return_value=[])
        mock_cursor.rowcount = 1

        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=None)
        mock_conn.cursor = MagicMock(return_value=mock_cursor)
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=None)

        mock_db.get_cursor.return_value = mock_cursor
        mock_db.get_connection.return_value = mock_conn

        coordinator = MultiOrchestratorCoordinator(
            orchestrator_id="orch-lifecycle",
            db_connection=mock_db,
            config=config,
        )

        # 1. 登録
        await coordinator.register()

        # 2. セッションロック取得
        session_id = uuid4()
        locked = await coordinator.acquire_session_lock(session_id)
        assert locked is True

        # 3. ハートビート送信
        await coordinator.send_heartbeat()

        # 4. 失敗したオーケストレーターの検出
        failed = await coordinator.detect_failed_orchestrators()
        assert failed == []

        # 5. セッションロック解放
        await coordinator.release_session_lock(session_id)

        # 6. 登録解除
        await coordinator.deregister()

        assert len(coordinator.active_sessions) == 0

    @pytest.mark.asyncio
    async def test_failover_scenario(self, mock_db, config):
        """フェイルオーバーシナリオ"""
        session_id = uuid4()

        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=None)
        mock_cursor.fetchone = MagicMock(return_value=(None, None))
        mock_cursor.fetchall = MagicMock(side_effect=[
            [("orch-failed",)],  # 失敗したオーケストレーター
            [(str(session_id),)],  # 引き継ぐセッション
        ])
        mock_cursor.rowcount = 1

        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=None)
        mock_conn.cursor = MagicMock(return_value=mock_cursor)
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=None)

        mock_db.get_cursor.return_value = mock_cursor
        mock_db.get_connection.return_value = mock_conn

        coordinator = MultiOrchestratorCoordinator(
            orchestrator_id="orch-healthy",
            db_connection=mock_db,
            config=config,
        )

        # 1. 失敗したオーケストレーターを検出
        failed = await coordinator.detect_failed_orchestrators()
        assert "orch-failed" in failed

        # 2. セッションを引き継ぐ
        count = await coordinator.takeover_sessions("orch-failed")
        assert count == 1
        assert session_id in coordinator.active_sessions
