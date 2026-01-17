# 進捗管理モジュール
# 実装仕様: docs/phase2-implementation-spec.ja.md セクション5.5
"""
進捗管理モジュール

オーケストレーターの進捗状態を管理し、中間睡眠からの復帰を可能にする。

状態保存のタイミング:
- タスク指示時
- タスク結果受領時
- 問題発生時
- ユーザー判断受領時

設計方針（タスク実行フローエージェント観点）:
- API設計: シンプルなメソッドでCRUD+レポート生成を提供
- フロー整合性: 状態保存タイミングを明確に定義
- エラー処理: DB操作エラー時の適切なハンドリング
- 拡張性: セッション状態にメタデータを追加可能
- テスト容易性: RepositoryパターンでDB操作を分離
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import UUID, uuid4

from src.db.connection import DatabaseConnection
from src.config.phase2_config import Phase2Config


logger = logging.getLogger(__name__)


@dataclass
class SessionState:
    """セッション状態

    オーケストレーターのセッション状態を表現。
    中間睡眠からの復帰に使用。

    Attributes:
        session_id: セッション識別子
        orchestrator_id: オーケストレーターID
        user_request: ユーザーリクエスト（original, clarified）
        task_tree: タスク依存関係と完了状況
        current_task: 現在実行中のタスク（完了時はNone）
        overall_progress_percent: 進捗率（0-100）
        status: セッション状態（in_progress, paused, completed, failed）
        created_at: 作成日時
        updated_at: 更新日時
        last_activity_at: 最後のアクティビティ日時
    """

    session_id: UUID
    orchestrator_id: str
    user_request: Dict[str, Any]
    task_tree: Dict[str, Any]
    current_task: Optional[Dict[str, Any]] = None
    overall_progress_percent: int = 0
    status: str = "in_progress"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_activity_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """バリデーション"""
        if self.overall_progress_percent < 0 or self.overall_progress_percent > 100:
            raise ValueError(
                f"overall_progress_percent must be 0-100, got {self.overall_progress_percent}"
            )
        valid_statuses = ("in_progress", "paused", "completed", "failed")
        if self.status not in valid_statuses:
            raise ValueError(
                f"status must be one of {valid_statuses}, got {self.status}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換

        Returns:
            辞書形式のセッション状態
        """
        return {
            "session_id": str(self.session_id),
            "orchestrator_id": self.orchestrator_id,
            "user_request": self.user_request,
            "task_tree": self.task_tree,
            "current_task": self.current_task,
            "overall_progress_percent": self.overall_progress_percent,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_activity_at": self.last_activity_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionState":
        """辞書形式から作成

        Args:
            data: 辞書形式のセッション状態

        Returns:
            SessionState インスタンス
        """
        return cls(
            session_id=UUID(data["session_id"]) if isinstance(data["session_id"], str) else data["session_id"],
            orchestrator_id=data["orchestrator_id"],
            user_request=data["user_request"],
            task_tree=data["task_tree"],
            current_task=data.get("current_task"),
            overall_progress_percent=data.get("overall_progress_percent", 0),
            status=data.get("status", "in_progress"),
            created_at=datetime.fromisoformat(data["created_at"]) if isinstance(data.get("created_at"), str) else data.get("created_at", datetime.now()),
            updated_at=datetime.fromisoformat(data["updated_at"]) if isinstance(data.get("updated_at"), str) else data.get("updated_at", datetime.now()),
            last_activity_at=datetime.fromisoformat(data["last_activity_at"]) if isinstance(data.get("last_activity_at"), str) else data.get("last_activity_at", datetime.now()),
        )


class SessionStateRepository:
    """セッション状態リポジトリ

    session_state テーブルへのCRUD操作を提供。

    使用例:
        db = DatabaseConnection()
        repository = SessionStateRepository(db)

        # 新規セッション作成
        state = SessionState(
            session_id=uuid4(),
            orchestrator_id="orchestrator_01",
            user_request={"original": "...", "clarified": "..."},
            task_tree={"tasks": []},
        )
        repository.create(state)

        # セッション取得
        state = repository.get_by_id(session_id)

        # セッション更新
        repository.update(
            session_id=session_id,
            task_tree=updated_tree,
            current_task=current,
            overall_progress_percent=50,
        )
    """

    def __init__(self, db: DatabaseConnection):
        """SessionStateRepository を初期化

        Args:
            db: DatabaseConnection インスタンス
        """
        self.db = db

    def create(self, state: SessionState) -> UUID:
        """セッション状態を作成

        Args:
            state: SessionState インスタンス

        Returns:
            作成されたセッションID

        Raises:
            Exception: DB操作エラー
        """
        query = """
            INSERT INTO session_state (
                session_id, orchestrator_id, user_request, task_tree,
                current_task, overall_progress_percent, status,
                created_at, updated_at, last_activity_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING session_id
        """

        with self.db.get_cursor() as cur:
            cur.execute(query, (
                str(state.session_id),
                state.orchestrator_id,
                json.dumps(state.user_request),
                json.dumps(state.task_tree),
                json.dumps(state.current_task) if state.current_task else None,
                state.overall_progress_percent,
                state.status,
                state.created_at,
                state.updated_at,
                state.last_activity_at,
            ))
            result = cur.fetchone()

            logger.info(f"セッション状態を作成: session_id={state.session_id}")
            return UUID(str(result[0]))

    def get_by_id(self, session_id: UUID) -> Optional[SessionState]:
        """セッション状態をIDで取得

        Args:
            session_id: セッション識別子

        Returns:
            SessionState インスタンス、見つからない場合は None
        """
        query = """
            SELECT session_id, orchestrator_id, user_request, task_tree,
                   current_task, overall_progress_percent, status,
                   created_at, updated_at, last_activity_at
            FROM session_state
            WHERE session_id = %s
        """

        with self.db.get_cursor() as cur:
            cur.execute(query, (str(session_id),))
            row = cur.fetchone()

            if row is None:
                return None

            return self._row_to_session_state(row)

    def update(
        self,
        session_id: UUID,
        task_tree: Optional[Dict[str, Any]] = None,
        current_task: Optional[Dict[str, Any]] = None,
        overall_progress_percent: Optional[int] = None,
        status: Optional[str] = None,
        last_activity_at: Optional[datetime] = None,
    ) -> bool:
        """セッション状態を更新

        Args:
            session_id: セッション識別子
            task_tree: タスク依存関係（更新する場合）
            current_task: 現在のタスク（更新する場合）
            overall_progress_percent: 進捗率（更新する場合）
            status: ステータス（更新する場合）
            last_activity_at: 最後のアクティビティ日時（更新する場合）

        Returns:
            更新成功時 True、セッションが見つからない場合 False
        """
        updates = []
        params = []

        if task_tree is not None:
            updates.append("task_tree = %s")
            params.append(json.dumps(task_tree))

        if current_task is not None:
            updates.append("current_task = %s")
            params.append(json.dumps(current_task))
        elif current_task is None and "current_task" in []:  # 明示的にNoneを設定
            updates.append("current_task = NULL")

        if overall_progress_percent is not None:
            updates.append("overall_progress_percent = %s")
            params.append(overall_progress_percent)

        if status is not None:
            updates.append("status = %s")
            params.append(status)

        if last_activity_at is not None:
            updates.append("last_activity_at = %s")
            params.append(last_activity_at)

        # updated_at は常に更新
        updates.append("updated_at = %s")
        params.append(datetime.now())

        if not updates:
            return True

        params.append(str(session_id))

        query = f"""
            UPDATE session_state
            SET {', '.join(updates)}
            WHERE session_id = %s
        """

        with self.db.get_cursor() as cur:
            cur.execute(query, params)
            updated = cur.rowcount > 0

            if updated:
                logger.debug(f"セッション状態を更新: session_id={session_id}")

            return updated

    def clear_current_task(self, session_id: UUID) -> bool:
        """現在のタスクをクリア（NULLに設定）

        Args:
            session_id: セッション識別子

        Returns:
            更新成功時 True
        """
        query = """
            UPDATE session_state
            SET current_task = NULL, updated_at = %s
            WHERE session_id = %s
        """

        with self.db.get_cursor() as cur:
            cur.execute(query, (datetime.now(), str(session_id)))
            return cur.rowcount > 0

    def list_by_status(self, status: str) -> List[SessionState]:
        """ステータスでセッション一覧を取得

        Args:
            status: セッション状態（in_progress, paused, completed, failed）

        Returns:
            SessionState のリスト
        """
        query = """
            SELECT session_id, orchestrator_id, user_request, task_tree,
                   current_task, overall_progress_percent, status,
                   created_at, updated_at, last_activity_at
            FROM session_state
            WHERE status = %s
            ORDER BY last_activity_at DESC
        """

        with self.db.get_cursor() as cur:
            cur.execute(query, (status,))
            rows = cur.fetchall()
            return [self._row_to_session_state(row) for row in rows]

    def list_by_orchestrator(self, orchestrator_id: str) -> List[SessionState]:
        """オーケストレーターIDでセッション一覧を取得

        Args:
            orchestrator_id: オーケストレーターID

        Returns:
            SessionState のリスト
        """
        query = """
            SELECT session_id, orchestrator_id, user_request, task_tree,
                   current_task, overall_progress_percent, status,
                   created_at, updated_at, last_activity_at
            FROM session_state
            WHERE orchestrator_id = %s
            ORDER BY last_activity_at DESC
        """

        with self.db.get_cursor() as cur:
            cur.execute(query, (orchestrator_id,))
            rows = cur.fetchall()
            return [self._row_to_session_state(row) for row in rows]

    def get_by_orchestrator_and_status(
        self,
        orchestrator_id: str,
        status_filter: List[str],
        order_by: str = "last_activity_at DESC",
        limit: int = 10,
    ) -> List[SessionState]:
        """オーケストレーターIDとステータスでセッションを検索

        Args:
            orchestrator_id: オーケストレーターID
            status_filter: ステータスのリスト
            order_by: ソート順（デフォルト: last_activity_at DESC）
            limit: 取得件数の上限

        Returns:
            SessionState のリスト
        """
        placeholders = ','.join(['%s'] * len(status_filter))
        query = f"""
            SELECT session_id, orchestrator_id, user_request, task_tree,
                   current_task, overall_progress_percent, status,
                   created_at, updated_at, last_activity_at
            FROM session_state
            WHERE orchestrator_id = %s
              AND status IN ({placeholders})
            ORDER BY {order_by}
            LIMIT %s
        """

        with self.db.get_cursor() as cur:
            params = [orchestrator_id] + status_filter + [limit]
            cur.execute(query, params)
            rows = cur.fetchall()
            return [self._row_to_session_state(row) for row in rows]

    def update_status(self, session_id: UUID, status: str) -> bool:
        """セッションステータスを更新

        Args:
            session_id: セッション識別子
            status: 新しいステータス

        Returns:
            更新成功時 True
        """
        return self.update(
            session_id=session_id,
            status=status,
            last_activity_at=datetime.now(),
        )

    def delete(self, session_id: UUID) -> bool:
        """セッション状態を削除

        Args:
            session_id: セッション識別子

        Returns:
            削除成功時 True
        """
        query = "DELETE FROM session_state WHERE session_id = %s"

        with self.db.get_cursor() as cur:
            cur.execute(query, (str(session_id),))
            deleted = cur.rowcount > 0

            if deleted:
                logger.info(f"セッション状態を削除: session_id={session_id}")

            return deleted

    def _row_to_session_state(self, row: tuple) -> SessionState:
        """DBの行をSessionStateに変換

        Args:
            row: DBの行データ

        Returns:
            SessionState インスタンス
        """
        return SessionState(
            session_id=UUID(str(row[0])),
            orchestrator_id=row[1],
            user_request=row[2] if isinstance(row[2], dict) else json.loads(row[2]) if row[2] else {},
            task_tree=row[3] if isinstance(row[3], dict) else json.loads(row[3]) if row[3] else {},
            current_task=row[4] if isinstance(row[4], dict) else json.loads(row[4]) if row[4] else None,
            overall_progress_percent=row[5] or 0,
            status=row[6] or "in_progress",
            created_at=row[7],
            updated_at=row[8],
            last_activity_at=row[9],
        )


class ProgressManager:
    """進捗管理

    オーケストレーターの進捗状態を管理し、中間睡眠からの復帰を可能にする。

    使用例:
        db = DatabaseConnection()
        repository = SessionStateRepository(db)
        manager = ProgressManager(repository)

        # 新規セッション作成
        session_id = manager.create_session(
            orchestrator_id="orchestrator_01",
            user_request={"original": "タスクを実行", "clarified": "具体的なタスク内容"},
            task_tree={"tasks": [...]},
        )

        # 進捗更新
        manager.update_progress(session_id, progress_percent=50)

        # 状態保存
        manager.save_state(
            session_id=session_id,
            task_tree=updated_tree,
            current_task=current_task,
            progress_percent=75,
        )

        # 状態復元
        state = manager.restore_state(session_id)

        # 進捗レポート生成
        report = manager.generate_progress_report(session_id)

    Attributes:
        session_repository: SessionStateRepository インスタンス
        config: Phase2Config インスタンス
    """

    def __init__(
        self,
        session_repository: SessionStateRepository,
        config: Optional[Phase2Config] = None,
    ):
        """ProgressManager を初期化

        Args:
            session_repository: SessionStateRepository インスタンス
            config: Phase2Config インスタンス（省略時はデフォルト）
        """
        self.session_repository = session_repository
        self.config = config or Phase2Config()

        logger.info("ProgressManager 初期化完了")

    def create_session(
        self,
        orchestrator_id: str,
        user_request: Dict[str, Any],
        task_tree: Dict[str, Any],
        session_id: Optional[UUID] = None,
    ) -> UUID:
        """新規セッションを作成

        Args:
            orchestrator_id: オーケストレーターID
            user_request: ユーザーリクエスト（original, clarified）
            task_tree: タスク依存関係
            session_id: セッションID（省略時は自動生成）

        Returns:
            作成されたセッションID
        """
        state = SessionState(
            session_id=session_id or uuid4(),
            orchestrator_id=orchestrator_id,
            user_request=user_request,
            task_tree=task_tree,
            overall_progress_percent=0,
            status="in_progress",
        )

        created_id = self.session_repository.create(state)

        logger.info(
            f"新規セッション作成: session_id={created_id}, "
            f"orchestrator_id={orchestrator_id}"
        )

        return created_id

    def save_state(
        self,
        session_id: UUID,
        task_tree: Dict[str, Any],
        current_task: Optional[Dict[str, Any]],
        progress_percent: int,
    ) -> bool:
        """進捗状態を保存

        状態保存のタイミング:
        - タスク指示時
        - タスク結果受領時
        - 問題発生時
        - ユーザー判断受領時

        Args:
            session_id: セッション識別子
            task_tree: タスク依存関係と完了状況
            current_task: 現在実行中のタスク（完了時はNone）
            progress_percent: 進捗率（0-100）

        Returns:
            保存成功時 True
        """
        # セッションが存在するか確認
        existing_session = self.session_repository.get_by_id(session_id)

        if existing_session is None:
            # セッションが存在しない場合は新規作成
            new_session = SessionState(
                session_id=session_id,
                orchestrator_id="cli_orchestrator",  # CLI経由のセッション
                user_request={"original": "", "clarified": ""},
                task_tree=task_tree,
                current_task=current_task,
                overall_progress_percent=progress_percent,
            )
            try:
                self.session_repository.create(new_session)
                logger.debug(
                    f"新規セッションを作成: session_id={session_id}, "
                    f"progress={progress_percent}%"
                )
                return True
            except Exception as e:
                logger.error(f"セッション作成に失敗: session_id={session_id}, error={e}")
                return False

        # セッションが存在する場合は更新
        success = self.session_repository.update(
            session_id=session_id,
            task_tree=task_tree,
            current_task=current_task,
            overall_progress_percent=progress_percent,
            last_activity_at=datetime.now(),
        )

        if success:
            logger.debug(
                f"進捗状態を保存: session_id={session_id}, "
                f"progress={progress_percent}%"
            )
        else:
            logger.warning(f"進捗状態の保存に失敗: session_id={session_id}")

        return success

    def restore_state(self, session_id: UUID) -> Optional[SessionState]:
        """進捗状態を復元

        Args:
            session_id: セッション識別子

        Returns:
            SessionState インスタンス、見つからない場合は None
        """
        state = self.session_repository.get_by_id(session_id)

        if state:
            logger.debug(f"進捗状態を復元: session_id={session_id}")
        else:
            logger.warning(f"セッションが見つかりません: session_id={session_id}")

        return state

    def generate_progress_report(self, session_id: UUID) -> str:
        """進捗レポートを生成

        Args:
            session_id: セッション識別子

        Returns:
            進捗レポート文字列
        """
        state = self.restore_state(session_id)
        if not state:
            return "セッションが見つかりません"

        completed = self._count_completed_tasks(state.task_tree)
        total = self._count_total_tasks(state.task_tree)

        current_task_desc = "なし"
        if state.current_task:
            current_task_desc = state.current_task.get("description", "なし")

        report = f"""【進捗報告】
進捗率: {state.overall_progress_percent}%
ステータス: {state.status}

完了タスク: {completed}/{total}
{self._format_task_tree(state.task_tree)}

現在のタスク:
{current_task_desc}

最終更新: {state.last_activity_at.strftime('%Y-%m-%d %H:%M:%S')}
"""
        return report

    def list_active_sessions(self) -> List[SessionState]:
        """アクティブなセッション一覧を取得

        Returns:
            in_progress または paused 状態のセッションリスト
        """
        in_progress = self.session_repository.list_by_status("in_progress")
        paused = self.session_repository.list_by_status("paused")

        sessions = in_progress + paused
        sessions.sort(key=lambda s: s.last_activity_at, reverse=True)

        logger.debug(f"アクティブなセッション: {len(sessions)}件")

        return sessions

    def update_progress(
        self,
        session_id: UUID,
        progress_percent: int,
    ) -> bool:
        """進捗率を更新

        Args:
            session_id: セッション識別子
            progress_percent: 進捗率（0-100）

        Returns:
            更新成功時 True
        """
        if progress_percent < 0 or progress_percent > 100:
            raise ValueError(
                f"progress_percent must be 0-100, got {progress_percent}"
            )

        success = self.session_repository.update(
            session_id=session_id,
            overall_progress_percent=progress_percent,
            last_activity_at=datetime.now(),
        )

        if success:
            logger.debug(
                f"進捗率を更新: session_id={session_id}, "
                f"progress={progress_percent}%"
            )

        return success

    def update_status(
        self,
        session_id: UUID,
        status: str,
    ) -> bool:
        """セッションステータスを更新

        Args:
            session_id: セッション識別子
            status: 新しいステータス（in_progress, paused, completed, failed）

        Returns:
            更新成功時 True
        """
        valid_statuses = ("in_progress", "paused", "completed", "failed")
        if status not in valid_statuses:
            raise ValueError(f"status must be one of {valid_statuses}, got {status}")

        success = self.session_repository.update(
            session_id=session_id,
            status=status,
            last_activity_at=datetime.now(),
        )

        if success:
            logger.info(
                f"セッションステータスを更新: session_id={session_id}, "
                f"status={status}"
            )

        return success

    def complete_session(self, session_id: UUID) -> bool:
        """セッションを完了状態にする

        Args:
            session_id: セッション識別子

        Returns:
            更新成功時 True
        """
        # current_task をクリア
        self.session_repository.clear_current_task(session_id)

        # ステータスと進捗を更新
        success = self.session_repository.update(
            session_id=session_id,
            overall_progress_percent=100,
            status="completed",
            last_activity_at=datetime.now(),
        )

        if success:
            logger.info(f"セッション完了: session_id={session_id}")

        return success

    def fail_session(
        self,
        session_id: UUID,
        error_message: Optional[str] = None,
    ) -> bool:
        """セッションを失敗状態にする

        Args:
            session_id: セッション識別子
            error_message: エラーメッセージ（オプション）

        Returns:
            更新成功時 True
        """
        state = self.restore_state(session_id)
        if state and error_message:
            # task_tree にエラー情報を追加
            task_tree = state.task_tree.copy()
            task_tree["error"] = {
                "message": error_message,
                "occurred_at": datetime.now().isoformat(),
            }
            self.session_repository.update(
                session_id=session_id,
                task_tree=task_tree,
            )

        success = self.session_repository.update(
            session_id=session_id,
            status="failed",
            last_activity_at=datetime.now(),
        )

        if success:
            logger.warning(
                f"セッション失敗: session_id={session_id}, "
                f"error={error_message}"
            )

        return success

    def pause_session(self, session_id: UUID) -> bool:
        """セッションを一時停止状態にする

        Args:
            session_id: セッション識別子

        Returns:
            更新成功時 True
        """
        return self.update_status(session_id, "paused")

    def resume_session(self, session_id: UUID) -> bool:
        """セッションを再開する

        Args:
            session_id: セッション識別子

        Returns:
            更新成功時 True
        """
        return self.update_status(session_id, "in_progress")

    def get_recent_sessions(
        self,
        orchestrator_id: str,
        limit: int = 10,
        status_filter: Optional[List[str]] = None,
    ) -> List[SessionState]:
        """最近のセッションを取得

        Args:
            orchestrator_id: オーケストレーターID
            limit: 取得件数の上限（デフォルト: 10）
            status_filter: ステータスフィルタ（デフォルト: ["in_progress", "paused"]）

        Returns:
            SessionState のリスト
        """
        if status_filter is None:
            status_filter = ["in_progress", "paused"]

        return self.session_repository.get_by_orchestrator_and_status(
            orchestrator_id=orchestrator_id,
            status_filter=status_filter,
            order_by="last_activity_at DESC",
            limit=limit,
        )

    def close_session(self, session_id: UUID, status: str = "completed") -> bool:
        """セッションを閉じる

        Args:
            session_id: セッション識別子
            status: 終了時のステータス（デフォルト: completed）

        Returns:
            更新成功時 True
        """
        if status == "completed":
            return self.complete_session(session_id)
        elif status == "failed":
            return self.fail_session(session_id)
        else:
            return self.session_repository.update_status(session_id, status)

    def _count_completed_tasks(self, task_tree: Dict[str, Any]) -> int:
        """完了タスク数をカウント

        Args:
            task_tree: タスク依存関係

        Returns:
            完了タスク数
        """
        tasks = task_tree.get("tasks", [])
        return sum(1 for task in tasks if task.get("status") == "completed")

    def _count_total_tasks(self, task_tree: Dict[str, Any]) -> int:
        """全タスク数をカウント

        Args:
            task_tree: タスク依存関係

        Returns:
            全タスク数
        """
        return len(task_tree.get("tasks", []))

    def _format_task_tree(self, task_tree: Dict[str, Any]) -> str:
        """タスクツリーをフォーマット

        Args:
            task_tree: タスク依存関係

        Returns:
            フォーマットされた文字列
        """
        tasks = task_tree.get("tasks", [])
        if not tasks:
            return "  タスクなし"

        lines = []
        for task in tasks:
            status_icon = self._get_status_icon(task.get("status", "pending"))
            description = task.get("description", "不明")
            lines.append(f"  {status_icon} {description}")

        return "\n".join(lines)

    def _get_status_icon(self, status: str) -> str:
        """ステータスアイコンを取得

        Args:
            status: タスクステータス

        Returns:
            ステータスアイコン
        """
        icons = {
            "pending": "[ ]",
            "in_progress": "[>]",
            "completed": "[x]",
            "failed": "[!]",
            "skipped": "[-]",
        }
        return icons.get(status, "[?]")
