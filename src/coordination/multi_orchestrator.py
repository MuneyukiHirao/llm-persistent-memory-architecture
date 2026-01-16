# 複数オーケストレーター協調
# 参照: docs/phase3-implementation-spec.ja.md セクション5.4
"""
複数オーケストレーター協調モジュール

複数のオーケストレーターインスタンスが協調して動作するための機能を提供。
Redis未使用、DBベースのロック機構で実装。

設計方針（タスク実行フロー観点）:
- API設計: シンプルなasyncインターフェースで呼び出し側の負担軽減
- フロー整合性: ロック取得→処理→解放の順序を厳格に管理
- エラー処理: 部分的な失敗時にもロールバック可能な設計
- 拡張性: HealthStatus, OrchestratorStatusをEnumで定義し、状態追加に対応
- テスト容易性: DBコネクションの注入により、モックでのテストが容易

注意: session_state テーブルに以下のカラムが必要：
- locked_by VARCHAR(64): ロックを保持しているオーケストレーターID
- lock_acquired_at TIMESTAMP: ロック取得時刻
"""

import asyncio
import socket
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from src.config.phase3_config import Phase3Config
from src.db.connection import DatabaseConnection


class OrchestratorStatus(str, Enum):
    """オーケストレーターの状態"""
    ACTIVE = "active"
    SLEEPING = "sleeping"
    TERMINATED = "terminated"


class HealthStatus(str, Enum):
    """ヘルスチェックの状態"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class OrchestratorInfo:
    """オーケストレーター情報"""
    orchestrator_id: str
    status: OrchestratorStatus
    current_load: int
    max_capacity: int
    active_sessions: int
    session_ids: List[UUID]
    last_heartbeat: datetime
    health_status: HealthStatus
    instance_info: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


class MultiOrchestratorCoordinator:
    """複数オーケストレーター協調クラス

    複数のオーケストレーターインスタンスが協調して動作するための機能を提供。
    セッションロック、ハートビート、フェイルオーバーを管理。

    使用例:
        coordinator = MultiOrchestratorCoordinator(
            orchestrator_id="orch-001",
            db_connection=db,
            config=Phase3Config(),
        )

        # オーケストレーターを登録
        await coordinator.register()

        # セッションロックを取得
        if await coordinator.acquire_session_lock(session_id):
            try:
                # セッションの処理
                pass
            finally:
                await coordinator.release_session_lock(session_id)

        # ハートビートを送信（定期的に呼び出す）
        await coordinator.send_heartbeat()

        # 終了時に登録解除
        await coordinator.deregister()

    Attributes:
        orchestrator_id: オーケストレーターの一意識別子
        db_connection: データベース接続
        config: Phase3設定
    """

    def __init__(
        self,
        orchestrator_id: str,
        db_connection: DatabaseConnection,
        config: Optional[Phase3Config] = None,
    ):
        """MultiOrchestratorCoordinatorを初期化

        Args:
            orchestrator_id: オーケストレーターの一意識別子
            db_connection: データベース接続
            config: Phase3設定（Noneの場合はデフォルト設定を使用）
        """
        self.orchestrator_id = orchestrator_id
        self.db = db_connection
        self.config = config or Phase3Config()
        self._current_load = 0
        self._active_session_ids: List[UUID] = []

    def _get_instance_info(self) -> Dict[str, Any]:
        """インスタンス情報を取得"""
        return {
            "hostname": socket.gethostname(),
            "pid": __import__("os").getpid(),
        }

    async def register(self) -> None:
        """オーケストレーターを登録

        orchestrator_state テーブルにオーケストレーターを登録。
        既に登録済みの場合は状態を更新。

        Raises:
            Exception: データベース操作エラー
        """
        def _register_sync():
            instance_info = self._get_instance_info()
            with self.db.get_cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO orchestrator_state (
                        orchestrator_id,
                        status,
                        current_load,
                        max_capacity,
                        active_sessions,
                        session_ids,
                        last_heartbeat,
                        health_status,
                        instance_info,
                        created_at,
                        updated_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, NOW(), %s, %s, NOW(), NOW()
                    )
                    ON CONFLICT (orchestrator_id) DO UPDATE SET
                        status = EXCLUDED.status,
                        current_load = EXCLUDED.current_load,
                        max_capacity = EXCLUDED.max_capacity,
                        active_sessions = EXCLUDED.active_sessions,
                        session_ids = EXCLUDED.session_ids,
                        last_heartbeat = NOW(),
                        health_status = EXCLUDED.health_status,
                        instance_info = EXCLUDED.instance_info,
                        updated_at = NOW()
                    """,
                    (
                        self.orchestrator_id,
                        OrchestratorStatus.ACTIVE.value,
                        self._current_load,
                        self.config.max_tasks_per_agent,
                        len(self._active_session_ids),
                        [str(sid) for sid in self._active_session_ids],
                        HealthStatus.HEALTHY.value,
                        __import__("json").dumps(instance_info),
                    ),
                )

        await asyncio.to_thread(_register_sync)

    async def acquire_session_lock(
        self,
        session_id: UUID,
        timeout: Optional[int] = None,
    ) -> bool:
        """セッションのロックを取得（DBベース）

        SELECT FOR UPDATE を使用した排他制御でセッションロックを取得。
        タイムアウト付きで、既存のロックが期限切れの場合は上書き可能。

        Args:
            session_id: ロックを取得するセッションID
            timeout: ロックのタイムアウト秒数（Noneの場合は設定値を使用）

        Returns:
            bool: ロック取得に成功した場合True

        Note:
            session_state テーブルに以下のカラムが必要：
            - locked_by VARCHAR(64)
            - lock_acquired_at TIMESTAMP
        """
        timeout = timeout or self.config.session_lock_timeout

        def _acquire_lock_sync() -> bool:
            with self.db.get_connection(auto_commit=False) as conn:
                with conn.cursor() as cur:
                    # セッションを排他ロックで取得
                    cur.execute(
                        """
                        SELECT locked_by, lock_acquired_at
                        FROM session_state
                        WHERE session_id = %s
                        FOR UPDATE NOWAIT
                        """,
                        (str(session_id),),
                    )
                    row = cur.fetchone()

                    if row is None:
                        # セッションが存在しない
                        conn.rollback()
                        return False

                    current_holder, lock_time = row

                    # ロックが取得可能かチェック
                    can_acquire = False

                    if current_holder is None:
                        # ロックされていない
                        can_acquire = True
                    elif current_holder == self.orchestrator_id:
                        # 自分が既に保持している
                        can_acquire = True
                    elif lock_time is not None:
                        # 既存ロックのタイムアウトをチェック
                        cur.execute(
                            """
                            SELECT EXTRACT(EPOCH FROM (NOW() - %s)) > %s
                            """,
                            (lock_time, timeout),
                        )
                        is_expired = cur.fetchone()[0]
                        if is_expired:
                            can_acquire = True

                    if can_acquire:
                        cur.execute(
                            """
                            UPDATE session_state
                            SET locked_by = %s,
                                lock_acquired_at = NOW(),
                                updated_at = NOW()
                            WHERE session_id = %s
                            """,
                            (self.orchestrator_id, str(session_id)),
                        )
                        conn.commit()

                        # ローカル状態を更新
                        if session_id not in self._active_session_ids:
                            self._active_session_ids.append(session_id)
                        return True
                    else:
                        conn.rollback()
                        return False

        try:
            return await asyncio.to_thread(_acquire_lock_sync)
        except Exception:
            # NOWAIT でロック取得失敗した場合など
            return False

    async def release_session_lock(self, session_id: UUID) -> None:
        """セッションのロックを解放

        自分が保持しているロックのみ解放可能。

        Args:
            session_id: ロックを解放するセッションID
        """
        def _release_lock_sync():
            with self.db.get_cursor() as cur:
                # 自分が持っているロックのみ解放
                cur.execute(
                    """
                    UPDATE session_state
                    SET locked_by = NULL,
                        lock_acquired_at = NULL,
                        updated_at = NOW()
                    WHERE session_id = %s
                      AND locked_by = %s
                    """,
                    (str(session_id), self.orchestrator_id),
                )

            # ローカル状態を更新
            if session_id in self._active_session_ids:
                self._active_session_ids.remove(session_id)

        await asyncio.to_thread(_release_lock_sync)

    async def send_heartbeat(self) -> None:
        """ハートビートを送信

        orchestrator_state テーブルの last_heartbeat を更新し、
        現在の負荷とセッション情報も同時に更新。
        """
        def _heartbeat_sync():
            with self.db.get_cursor() as cur:
                cur.execute(
                    """
                    UPDATE orchestrator_state
                    SET last_heartbeat = NOW(),
                        current_load = %s,
                        active_sessions = %s,
                        session_ids = %s,
                        updated_at = NOW()
                    WHERE orchestrator_id = %s
                    """,
                    (
                        self._current_load,
                        len(self._active_session_ids),
                        [str(sid) for sid in self._active_session_ids],
                        self.orchestrator_id,
                    ),
                )

        await asyncio.to_thread(_heartbeat_sync)

    async def detect_failed_orchestrators(self) -> List[str]:
        """失敗したオーケストレーターを検出

        last_heartbeat が orchestrator_failover_timeout を超えている
        アクティブなオーケストレーターを検出。

        Returns:
            List[str]: 失敗したオーケストレーターIDのリスト
        """
        timeout = self.config.orchestrator_failover_timeout

        def _detect_failed_sync() -> List[str]:
            with self.db.get_cursor() as cur:
                cur.execute(
                    """
                    SELECT orchestrator_id
                    FROM orchestrator_state
                    WHERE status = %s
                      AND last_heartbeat < NOW() - INTERVAL '%s seconds'
                    """,
                    (OrchestratorStatus.ACTIVE.value, timeout),
                )
                rows = cur.fetchall()
                return [row[0] for row in rows]

        return await asyncio.to_thread(_detect_failed_sync)

    async def takeover_sessions(self, failed_orchestrator_id: str) -> int:
        """失敗したオーケストレーターのセッションを引き継ぐ

        失敗したオーケストレーターが保持していたセッションのロックを
        自分に移し、オーケストレーターの状態を terminated に更新。

        Args:
            failed_orchestrator_id: 失敗したオーケストレーターID

        Returns:
            int: 引き継いだセッション数
        """
        def _takeover_sync() -> int:
            with self.db.get_connection(auto_commit=False) as conn:
                with conn.cursor() as cur:
                    # 失敗したオーケストレーターを terminated にマーク
                    cur.execute(
                        """
                        UPDATE orchestrator_state
                        SET status = %s,
                            health_status = %s,
                            updated_at = NOW()
                        WHERE orchestrator_id = %s
                          AND status = %s
                        """,
                        (
                            OrchestratorStatus.TERMINATED.value,
                            HealthStatus.UNHEALTHY.value,
                            failed_orchestrator_id,
                            OrchestratorStatus.ACTIVE.value,
                        ),
                    )

                    # 失敗したオーケストレーターが保持していたセッションを取得
                    cur.execute(
                        """
                        SELECT session_id
                        FROM session_state
                        WHERE locked_by = %s
                          AND status = 'in_progress'
                        """,
                        (failed_orchestrator_id,),
                    )
                    sessions_to_takeover = cur.fetchall()

                    # セッションロックを自分に移す
                    takeover_count = 0
                    for (session_id_str,) in sessions_to_takeover:
                        cur.execute(
                            """
                            UPDATE session_state
                            SET locked_by = %s,
                                lock_acquired_at = NOW(),
                                updated_at = NOW()
                            WHERE session_id = %s
                              AND locked_by = %s
                            """,
                            (
                                self.orchestrator_id,
                                session_id_str,
                                failed_orchestrator_id,
                            ),
                        )
                        if cur.rowcount > 0:
                            takeover_count += 1
                            session_id = UUID(session_id_str)
                            if session_id not in self._active_session_ids:
                                self._active_session_ids.append(session_id)

                    conn.commit()
                    return takeover_count

        return await asyncio.to_thread(_takeover_sync)

    async def deregister(self) -> None:
        """オーケストレーターを登録解除

        orchestrator_state テーブルのステータスを terminated に更新し、
        保持しているすべてのセッションロックを解放。
        """
        def _deregister_sync():
            with self.db.get_connection(auto_commit=False) as conn:
                with conn.cursor() as cur:
                    # 保持しているセッションロックを解放
                    cur.execute(
                        """
                        UPDATE session_state
                        SET locked_by = NULL,
                            lock_acquired_at = NULL,
                            updated_at = NOW()
                        WHERE locked_by = %s
                        """,
                        (self.orchestrator_id,),
                    )

                    # オーケストレーターを terminated にマーク
                    cur.execute(
                        """
                        UPDATE orchestrator_state
                        SET status = %s,
                            active_sessions = 0,
                            session_ids = '{}',
                            updated_at = NOW()
                        WHERE orchestrator_id = %s
                        """,
                        (
                            OrchestratorStatus.TERMINATED.value,
                            self.orchestrator_id,
                        ),
                    )

                    conn.commit()

            # ローカル状態をクリア
            self._active_session_ids.clear()

        await asyncio.to_thread(_deregister_sync)

    async def get_orchestrator_info(
        self,
        orchestrator_id: Optional[str] = None,
    ) -> Optional[OrchestratorInfo]:
        """オーケストレーター情報を取得

        Args:
            orchestrator_id: 取得するオーケストレーターID（Noneの場合は自分）

        Returns:
            OrchestratorInfo: オーケストレーター情報（存在しない場合はNone）
        """
        target_id = orchestrator_id or self.orchestrator_id

        def _get_info_sync() -> Optional[OrchestratorInfo]:
            with self.db.get_cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        orchestrator_id,
                        status,
                        current_load,
                        max_capacity,
                        active_sessions,
                        session_ids,
                        last_heartbeat,
                        health_status,
                        instance_info,
                        created_at,
                        updated_at
                    FROM orchestrator_state
                    WHERE orchestrator_id = %s
                    """,
                    (target_id,),
                )
                row = cur.fetchone()

                if row is None:
                    return None

                return OrchestratorInfo(
                    orchestrator_id=row[0],
                    status=OrchestratorStatus(row[1]),
                    current_load=row[2],
                    max_capacity=row[3],
                    active_sessions=row[4],
                    session_ids=[UUID(s) for s in (row[5] or [])],
                    last_heartbeat=row[6],
                    health_status=HealthStatus(row[7]),
                    instance_info=row[8] or {},
                    created_at=row[9],
                    updated_at=row[10],
                )

        return await asyncio.to_thread(_get_info_sync)

    async def list_active_orchestrators(self) -> List[OrchestratorInfo]:
        """アクティブなオーケストレーター一覧を取得

        Returns:
            List[OrchestratorInfo]: アクティブなオーケストレーターのリスト
        """
        def _list_active_sync() -> List[OrchestratorInfo]:
            with self.db.get_cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        orchestrator_id,
                        status,
                        current_load,
                        max_capacity,
                        active_sessions,
                        session_ids,
                        last_heartbeat,
                        health_status,
                        instance_info,
                        created_at,
                        updated_at
                    FROM orchestrator_state
                    WHERE status = %s
                    ORDER BY created_at
                    """,
                    (OrchestratorStatus.ACTIVE.value,),
                )
                rows = cur.fetchall()

                return [
                    OrchestratorInfo(
                        orchestrator_id=row[0],
                        status=OrchestratorStatus(row[1]),
                        current_load=row[2],
                        max_capacity=row[3],
                        active_sessions=row[4],
                        session_ids=[UUID(s) for s in (row[5] or [])],
                        last_heartbeat=row[6],
                        health_status=HealthStatus(row[7]),
                        instance_info=row[8] or {},
                        created_at=row[9],
                        updated_at=row[10],
                    )
                    for row in rows
                ]

        return await asyncio.to_thread(_list_active_sync)

    def update_load(self, load: int) -> None:
        """現在の負荷を更新

        Args:
            load: 新しい負荷値
        """
        self._current_load = load

    @property
    def active_sessions(self) -> List[UUID]:
        """アクティブなセッションIDのリスト"""
        return self._active_session_ids.copy()

    @property
    def current_load(self) -> int:
        """現在の負荷"""
        return self._current_load
