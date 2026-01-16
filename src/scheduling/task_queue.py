# src/scheduling/task_queue.py
"""タスクキューモジュール

Phase 3 MVP のタスクキュー管理。
優先度付きキューでタスクを管理し、非同期でエージェントに配分する。

Redis依存なしのインメモリ実装だが、将来的なRedis + Celery移行を想定した
インターフェース設計となっている。

実装仕様: docs/phase3-implementation-spec.ja.md セクション5.2
設定参照: src/config/phase3_config.py (task_queue_enabled, max_queue_size, task_timeout_seconds)

処理フロー:
    タスク登録 (enqueue)
        ↓
    優先度付きキュー（heapq）
    ├── priority: 1-10（1が最優先）
    ├── created_at: タイムスタンプ
    └── task_payload: 辞書
        ↓
    ワーカーがタスク取得 (dequeue)
        ↓
    タスク実行 → 完了/失敗記録 (complete/fail)

設計方針（タスク実行フローエージェント観点）:
    - API設計: シンプルなenqueue/dequeue/complete/failインターフェース
    - フロー整合性: タスク状態遷移を厳密に管理（pending→processing→completed/failed）
    - エラー処理: リトライ機能でタスク失敗に対応
    - 拡張性: Redis移行を想定したインターフェース（async対応準備）
    - テスト容易性: 依存性注入でconfig差し替え可能

使用例:
    from src.scheduling.task_queue import TaskQueue, TaskStatus
    from src.config.phase3_config import Phase3Config

    # タスクキューの初期化
    config = Phase3Config()
    queue = TaskQueue(config)

    # タスクをエンキュー
    task_id = queue.enqueue(
        task_type="routing",
        task_payload={"query": "検索クエリ", "agent_id": "agent_01"},
        priority=3,
    )

    # ワーカーがタスクを取得
    task = queue.dequeue(worker_id="worker_01")
    if task:
        try:
            # タスクを実行...
            result = {"status": "success", "data": {...}}
            queue.complete(task.id, result)
        except Exception as e:
            # 失敗時はリトライ判定
            should_retry = queue.fail(task.id, str(e))
            if should_retry:
                print("タスクは再キューされました")
"""

import heapq
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

from src.config.phase3_config import Phase3Config


logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """タスクの状態

    DBスキーマ（task_queue テーブル）の status カラムに対応。

    状態遷移:
        PENDING → PROCESSING → COMPLETED
                            └→ FAILED → (リトライ) → PENDING
    """
    PENDING = "pending"        # 待機中（キュー内）
    PROCESSING = "processing"  # 処理中（ワーカーが取得）
    COMPLETED = "completed"    # 完了
    FAILED = "failed"          # 失敗


@dataclass
class TaskItem:
    """タスクアイテム

    DBスキーマ（task_queue テーブル）に対応するデータ構造。
    優先度付きキューで管理されるタスクの情報を保持する。

    Attributes:
        id: タスクの一意識別子（UUID）
        session_id: 関連するセッションID（Optional）
        task_type: タスクの種類（"routing", "execution", "evaluation"）
        task_payload: タスクのペイロード（任意の辞書）
        priority: 優先度（1-10、1が最優先）
        status: タスクの状態
        assigned_worker: 処理中のワーカーID
        created_at: 作成日時
        started_at: 処理開始日時
        completed_at: 完了日時
        retry_count: リトライ回数
        max_retries: 最大リトライ回数
        error_message: エラーメッセージ（失敗時）
        result: タスク結果（完了時）

    Note:
        - priority は 1-10 の範囲で、1 が最優先
        - heapq で優先度ソートするため、(priority, created_at) で比較
    """
    id: UUID
    task_type: str
    task_payload: Dict[str, Any]
    session_id: Optional[UUID] = None
    priority: int = 5
    status: TaskStatus = TaskStatus.PENDING
    assigned_worker: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    error_message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換

        REST API レスポンスや JSON シリアライズ用。

        Returns:
            タスク情報の辞書
        """
        return {
            "id": str(self.id),
            "session_id": str(self.session_id) if self.session_id else None,
            "task_type": self.task_type,
            "task_payload": self.task_payload,
            "priority": self.priority,
            "status": self.status.value,
            "assigned_worker": self.assigned_worker,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "error_message": self.error_message,
            "result": self.result,
        }


# heapq用のラッパークラス（比較演算子を定義）
@dataclass(order=True)
class _PriorityEntry:
    """heapq用の優先度エントリ

    heapq は最小ヒープなので、priority が小さいほど優先される。
    同じ priority の場合は created_at が早いほうが優先される。

    Attributes:
        priority: 優先度（1-10）
        timestamp: 作成時のタイムスタンプ（秒）
        task_id: タスクID（比較には使用しない）
    """
    priority: int
    timestamp: float
    task_id: UUID = field(compare=False)


class TaskQueue:
    """タスクキュー管理クラス

    heapqベースのインメモリ優先度付きキュー実装。
    スレッドセーフで、将来的なRedis移行を想定したインターフェース。

    処理フロー:
        enqueue() → 優先度付きキューに追加
            ↓
        dequeue() → 最優先タスクを取得（PENDING → PROCESSING）
            ↓
        complete() / fail() → タスク完了/失敗を記録

    スレッドセーフ:
        - threading.Lock を使用して並行アクセスを保護
        - 複数ワーカーからの同時アクセスに対応

    Attributes:
        config: Phase3Config インスタンス
        _heap: 優先度付きキュー（heapq）
        _tasks: タスクID → TaskItem のマッピング
        _lock: スレッドセーフ用ロック
    """

    def __init__(self, config: Optional[Phase3Config] = None):
        """TaskQueue を初期化

        Args:
            config: Phase3Config インスタンス（省略時はデフォルト設定）
        """
        self.config = config or Phase3Config()

        # 優先度付きキュー（heapq）
        self._heap: List[_PriorityEntry] = []

        # タスクID → TaskItem のマッピング
        self._tasks: Dict[UUID, TaskItem] = {}

        # スレッドセーフ用ロック
        self._lock = Lock()

        logger.info(
            f"TaskQueue 初期化完了: "
            f"max_queue_size={self.config.max_queue_size}, "
            f"task_timeout_seconds={self.config.task_timeout_seconds}"
        )

    def enqueue(
        self,
        task_type: str,
        task_payload: Dict[str, Any],
        priority: int = 5,
        session_id: Optional[UUID] = None,
        max_retries: int = 3,
    ) -> UUID:
        """タスクをキューに追加

        Args:
            task_type: タスクの種類（"routing", "execution", "evaluation"）
            task_payload: タスクのペイロード
            priority: 優先度（1-10、1が最優先）
            session_id: 関連するセッションID
            max_retries: 最大リトライ回数

        Returns:
            作成されたタスクのUUID

        Raises:
            ValueError: priority が範囲外、または キューが満杯の場合
        """
        # priority のバリデーション
        if not 1 <= priority <= 10:
            raise ValueError(f"priority は 1-10 の範囲で指定してください: {priority}")

        with self._lock:
            # キューサイズのチェック
            pending_count = sum(
                1 for t in self._tasks.values()
                if t.status == TaskStatus.PENDING
            )
            if pending_count >= self.config.max_queue_size:
                raise ValueError(
                    f"キューが満杯です: {pending_count}/{self.config.max_queue_size}"
                )

            # タスクを作成
            task_id = uuid4()
            now = datetime.now()

            task = TaskItem(
                id=task_id,
                task_type=task_type,
                task_payload=task_payload,
                session_id=session_id,
                priority=priority,
                status=TaskStatus.PENDING,
                created_at=now,
                max_retries=max_retries,
            )

            # マッピングに追加
            self._tasks[task_id] = task

            # ヒープに追加
            entry = _PriorityEntry(
                priority=priority,
                timestamp=now.timestamp(),
                task_id=task_id,
            )
            heapq.heappush(self._heap, entry)

            logger.info(
                f"タスクをエンキュー: "
                f"task_id={task_id}, "
                f"task_type={task_type}, "
                f"priority={priority}, "
                f"queue_size={len(self._heap)}"
            )

            return task_id

    def dequeue(self, worker_id: str) -> Optional[TaskItem]:
        """タスクをキューから取得（アトミック操作）

        最優先のPENDINGタスクを取得し、PROCESSINGに状態遷移させる。
        取得したタスクは worker_id に割り当てられる。

        Args:
            worker_id: ワーカーの識別子

        Returns:
            取得したタスク（キューが空の場合は None）
        """
        with self._lock:
            # PENDINGタスクを探す
            while self._heap:
                entry = heapq.heappop(self._heap)
                task = self._tasks.get(entry.task_id)

                # タスクが存在し、かつPENDINGの場合のみ取得
                if task and task.status == TaskStatus.PENDING:
                    # 状態を PROCESSING に遷移
                    task.status = TaskStatus.PROCESSING
                    task.assigned_worker = worker_id
                    task.started_at = datetime.now()

                    logger.info(
                        f"タスクをデキュー: "
                        f"task_id={task.id}, "
                        f"worker_id={worker_id}, "
                        f"task_type={task.task_type}"
                    )

                    return task

                # PENDING でないタスクはスキップ（既に処理済み等）
                logger.debug(
                    f"スキップ: task_id={entry.task_id}, "
                    f"status={task.status if task else 'not found'}"
                )

            # キューが空
            logger.debug(f"キューが空です: worker_id={worker_id}")
            return None

    def complete(
        self,
        task_id: UUID,
        result: Optional[Dict[str, Any]] = None,
    ) -> None:
        """タスク完了を記録

        PROCESSINGのタスクをCOMPLETEDに遷移させる。

        Args:
            task_id: タスクID
            result: タスクの結果（Optional）

        Raises:
            ValueError: タスクが存在しない、またはPROCESSINGでない場合
        """
        with self._lock:
            task = self._tasks.get(task_id)

            if not task:
                raise ValueError(f"タスクが存在しません: {task_id}")

            if task.status != TaskStatus.PROCESSING:
                raise ValueError(
                    f"タスクはPROCESSINGではありません: "
                    f"task_id={task_id}, status={task.status.value}"
                )

            # 状態を COMPLETED に遷移
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.result = result

            logger.info(
                f"タスク完了: "
                f"task_id={task_id}, "
                f"duration={(task.completed_at - task.started_at).total_seconds():.3f}s"
            )

    def fail(
        self,
        task_id: UUID,
        error_message: str,
    ) -> bool:
        """タスク失敗を記録、リトライ判定

        PROCESSINGのタスクをFAILEDに遷移させ、リトライ可能であれば
        PENDINGに戻してキューに再追加する。

        Args:
            task_id: タスクID
            error_message: エラーメッセージ

        Returns:
            リトライされた場合 True、そうでなければ False

        Raises:
            ValueError: タスクが存在しない、またはPROCESSINGでない場合
        """
        with self._lock:
            task = self._tasks.get(task_id)

            if not task:
                raise ValueError(f"タスクが存在しません: {task_id}")

            if task.status != TaskStatus.PROCESSING:
                raise ValueError(
                    f"タスクはPROCESSINGではありません: "
                    f"task_id={task_id}, status={task.status.value}"
                )

            task.retry_count += 1
            task.error_message = error_message

            # リトライ判定
            if task.retry_count < task.max_retries:
                # リトライ: PENDING に戻す
                task.status = TaskStatus.PENDING
                task.assigned_worker = None
                task.started_at = None

                # ヒープに再追加
                entry = _PriorityEntry(
                    priority=task.priority,
                    timestamp=datetime.now().timestamp(),
                    task_id=task.id,
                )
                heapq.heappush(self._heap, entry)

                logger.warning(
                    f"タスク失敗（リトライ）: "
                    f"task_id={task_id}, "
                    f"retry_count={task.retry_count}/{task.max_retries}, "
                    f"error={error_message}"
                )
                return True
            else:
                # リトライ上限: FAILED に遷移
                task.status = TaskStatus.FAILED
                task.completed_at = datetime.now()

                logger.error(
                    f"タスク失敗（リトライ上限）: "
                    f"task_id={task_id}, "
                    f"retry_count={task.retry_count}/{task.max_retries}, "
                    f"error={error_message}"
                )
                return False

    def get_task(self, task_id: UUID) -> Optional[TaskItem]:
        """タスクを取得

        Args:
            task_id: タスクID

        Returns:
            タスク（存在しない場合は None）
        """
        with self._lock:
            return self._tasks.get(task_id)

    def get_queue_size(self) -> int:
        """キュー内のPENDINGタスク数を取得

        Returns:
            PENDINGタスクの数
        """
        with self._lock:
            return sum(
                1 for t in self._tasks.values()
                if t.status == TaskStatus.PENDING
            )

    def get_processing_count(self) -> int:
        """処理中のタスク数を取得

        Returns:
            PROCESSINGタスクの数
        """
        with self._lock:
            return sum(
                1 for t in self._tasks.values()
                if t.status == TaskStatus.PROCESSING
            )

    def get_stats(self) -> Dict[str, int]:
        """キューの統計情報を取得

        Returns:
            各状態のタスク数を含む辞書
        """
        with self._lock:
            stats = {
                "pending": 0,
                "processing": 0,
                "completed": 0,
                "failed": 0,
                "total": len(self._tasks),
            }

            for task in self._tasks.values():
                stats[task.status.value] += 1

            return stats

    def get_timed_out_tasks(self) -> List[TaskItem]:
        """タイムアウトしたタスクを取得

        config.task_timeout_seconds を超えて PROCESSING のままの
        タスクを検出する。

        Returns:
            タイムアウトしたタスクのリスト
        """
        with self._lock:
            now = datetime.now()
            timeout_seconds = self.config.task_timeout_seconds
            timed_out = []

            for task in self._tasks.values():
                if task.status == TaskStatus.PROCESSING and task.started_at:
                    elapsed = (now - task.started_at).total_seconds()
                    if elapsed > timeout_seconds:
                        timed_out.append(task)

            return timed_out

    def cleanup_completed(self, max_age_seconds: int = 3600) -> int:
        """完了済みタスクをクリーンアップ

        指定時間以上経過した COMPLETED/FAILED タスクを削除する。

        Args:
            max_age_seconds: 保持する最大時間（秒）

        Returns:
            削除されたタスク数
        """
        with self._lock:
            now = datetime.now()
            to_remove = []

            for task_id, task in self._tasks.items():
                if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                    if task.completed_at:
                        age = (now - task.completed_at).total_seconds()
                        if age > max_age_seconds:
                            to_remove.append(task_id)

            for task_id in to_remove:
                del self._tasks[task_id]

            if to_remove:
                logger.info(
                    f"完了済みタスクをクリーンアップ: {len(to_remove)} 件削除"
                )

            return len(to_remove)

    def cancel(self, task_id: UUID) -> bool:
        """タスクをキャンセル

        PENDINGのタスクのみキャンセル可能。

        Args:
            task_id: タスクID

        Returns:
            キャンセル成功時 True、失敗時 False
        """
        with self._lock:
            task = self._tasks.get(task_id)

            if not task:
                logger.warning(f"キャンセル失敗: タスクが存在しません: {task_id}")
                return False

            if task.status != TaskStatus.PENDING:
                logger.warning(
                    f"キャンセル失敗: タスクはPENDINGではありません: "
                    f"task_id={task_id}, status={task.status.value}"
                )
                return False

            # タスクを FAILED に遷移（キャンセル扱い）
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()
            task.error_message = "Cancelled"

            logger.info(f"タスクをキャンセル: task_id={task_id}")
            return True

    def clear(self) -> int:
        """全タスクをクリア（テスト用）

        Returns:
            削除されたタスク数
        """
        with self._lock:
            count = len(self._tasks)
            self._tasks.clear()
            self._heap.clear()

            logger.info(f"キューをクリア: {count} 件削除")
            return count

    # === 将来のRedis移行を想定した非同期インターフェース ===
    # Phase 3 MVP では同期実装だが、インターフェースを揃えておく

    async def enqueue_async(
        self,
        task_type: str,
        task_payload: Dict[str, Any],
        priority: int = 5,
        session_id: Optional[UUID] = None,
        max_retries: int = 3,
    ) -> UUID:
        """非同期版 enqueue（将来のRedis移行用）

        現在は同期実装を呼び出す。
        """
        return self.enqueue(
            task_type=task_type,
            task_payload=task_payload,
            priority=priority,
            session_id=session_id,
            max_retries=max_retries,
        )

    async def dequeue_async(self, worker_id: str) -> Optional[TaskItem]:
        """非同期版 dequeue（将来のRedis移行用）

        現在は同期実装を呼び出す。
        """
        return self.dequeue(worker_id)

    async def complete_async(
        self,
        task_id: UUID,
        result: Optional[Dict[str, Any]] = None,
    ) -> None:
        """非同期版 complete（将来のRedis移行用）

        現在は同期実装を呼び出す。
        """
        self.complete(task_id, result)

    async def fail_async(
        self,
        task_id: UUID,
        error_message: str,
    ) -> bool:
        """非同期版 fail（将来のRedis移行用）

        現在は同期実装を呼び出す。
        """
        return self.fail(task_id, error_message)
