# tests/scheduling/test_task_queue.py
"""タスクキューの単体テスト

TaskQueue クラスのエンキュー、デキュー、完了、失敗、リトライ機能をテスト。
スレッドセーフ性も検証する。

実行方法:
    pytest tests/scheduling/test_task_queue.py -v
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from unittest.mock import patch
from uuid import uuid4

import pytest

from src.config.phase3_config import Phase3Config
from src.scheduling.task_queue import TaskItem, TaskQueue, TaskStatus


class TestTaskQueue:
    """TaskQueue クラスのテスト"""

    @pytest.fixture
    def config(self) -> Phase3Config:
        """テスト用の設定"""
        config = Phase3Config()
        config.max_queue_size = 100
        config.task_timeout_seconds = 60
        return config

    @pytest.fixture
    def queue(self, config: Phase3Config) -> TaskQueue:
        """テスト用のタスクキュー"""
        return TaskQueue(config)

    # === 基本動作テスト ===

    def test_enqueue_returns_uuid(self, queue: TaskQueue):
        """enqueue がUUIDを返すこと"""
        task_id = queue.enqueue(
            task_type="routing",
            task_payload={"query": "test query"},
        )

        assert task_id is not None
        assert isinstance(task_id, type(uuid4()))

    def test_enqueue_creates_pending_task(self, queue: TaskQueue):
        """enqueue がPENDINGタスクを作成すること"""
        task_id = queue.enqueue(
            task_type="routing",
            task_payload={"query": "test query"},
            priority=3,
        )

        task = queue.get_task(task_id)
        assert task is not None
        assert task.status == TaskStatus.PENDING
        assert task.task_type == "routing"
        assert task.priority == 3
        assert task.task_payload == {"query": "test query"}

    def test_dequeue_returns_task(self, queue: TaskQueue):
        """dequeue がタスクを返すこと"""
        task_id = queue.enqueue(
            task_type="routing",
            task_payload={"query": "test query"},
        )

        task = queue.dequeue(worker_id="worker_01")

        assert task is not None
        assert task.id == task_id
        assert task.status == TaskStatus.PROCESSING
        assert task.assigned_worker == "worker_01"
        assert task.started_at is not None

    def test_dequeue_returns_none_when_empty(self, queue: TaskQueue):
        """キューが空の場合、dequeue が None を返すこと"""
        task = queue.dequeue(worker_id="worker_01")
        assert task is None

    def test_complete_marks_task_completed(self, queue: TaskQueue):
        """complete がタスクをCOMPLETEDにすること"""
        task_id = queue.enqueue(
            task_type="routing",
            task_payload={"query": "test query"},
        )
        queue.dequeue(worker_id="worker_01")

        result = {"status": "success", "data": [1, 2, 3]}
        queue.complete(task_id, result)

        task = queue.get_task(task_id)
        assert task.status == TaskStatus.COMPLETED
        assert task.result == result
        assert task.completed_at is not None

    def test_fail_marks_task_failed_after_max_retries(self, queue: TaskQueue):
        """最大リトライ後、fail がタスクをFAILEDにすること"""
        task_id = queue.enqueue(
            task_type="routing",
            task_payload={"query": "test query"},
            max_retries=2,
        )

        # 1回目の失敗（リトライ）
        queue.dequeue(worker_id="worker_01")
        should_retry = queue.fail(task_id, "Error 1")
        assert should_retry is True

        # 2回目の失敗（リトライ上限）
        queue.dequeue(worker_id="worker_01")
        should_retry = queue.fail(task_id, "Error 2")
        assert should_retry is False

        task = queue.get_task(task_id)
        assert task.status == TaskStatus.FAILED
        assert task.retry_count == 2
        assert task.error_message == "Error 2"

    # === 優先度テスト ===

    def test_priority_ordering(self, queue: TaskQueue):
        """優先度順でデキューされること"""
        # 優先度 5, 1, 10 の順でエンキュー
        queue.enqueue(
            task_type="routing",
            task_payload={"name": "medium"},
            priority=5,
        )
        queue.enqueue(
            task_type="routing",
            task_payload={"name": "highest"},
            priority=1,
        )
        queue.enqueue(
            task_type="routing",
            task_payload={"name": "lowest"},
            priority=10,
        )

        # デキュー順序: 1 → 5 → 10
        task1 = queue.dequeue(worker_id="worker_01")
        task2 = queue.dequeue(worker_id="worker_01")
        task3 = queue.dequeue(worker_id="worker_01")

        assert task1.task_payload["name"] == "highest"
        assert task2.task_payload["name"] == "medium"
        assert task3.task_payload["name"] == "lowest"

    def test_same_priority_fifo(self, queue: TaskQueue):
        """同じ優先度の場合、FIFOでデキューされること"""
        # 同じ優先度でエンキュー
        queue.enqueue(
            task_type="routing",
            task_payload={"order": 1},
            priority=5,
        )
        time.sleep(0.01)  # タイムスタンプを少しずらす
        queue.enqueue(
            task_type="routing",
            task_payload={"order": 2},
            priority=5,
        )
        time.sleep(0.01)
        queue.enqueue(
            task_type="routing",
            task_payload={"order": 3},
            priority=5,
        )

        # デキュー順序: 1 → 2 → 3
        task1 = queue.dequeue(worker_id="worker_01")
        task2 = queue.dequeue(worker_id="worker_01")
        task3 = queue.dequeue(worker_id="worker_01")

        assert task1.task_payload["order"] == 1
        assert task2.task_payload["order"] == 2
        assert task3.task_payload["order"] == 3

    def test_priority_validation(self, queue: TaskQueue):
        """優先度のバリデーション"""
        # 範囲外の優先度でエラー
        with pytest.raises(ValueError, match="1-10"):
            queue.enqueue(
                task_type="routing",
                task_payload={},
                priority=0,
            )

        with pytest.raises(ValueError, match="1-10"):
            queue.enqueue(
                task_type="routing",
                task_payload={},
                priority=11,
            )

    # === リトライテスト ===

    def test_retry_requeues_task(self, queue: TaskQueue):
        """fail がタスクを再キューすること"""
        task_id = queue.enqueue(
            task_type="routing",
            task_payload={"query": "test query"},
            max_retries=3,
        )

        # 1回目の失敗
        queue.dequeue(worker_id="worker_01")
        should_retry = queue.fail(task_id, "Error 1")
        assert should_retry is True

        # タスクが再キューされている
        task = queue.get_task(task_id)
        assert task.status == TaskStatus.PENDING
        assert task.retry_count == 1

        # 再度デキュー可能
        task2 = queue.dequeue(worker_id="worker_02")
        assert task2 is not None
        assert task2.id == task_id

    def test_custom_max_retries(self, queue: TaskQueue):
        """カスタムmax_retriesが機能すること"""
        task_id = queue.enqueue(
            task_type="routing",
            task_payload={},
            max_retries=1,
        )

        # 1回目の失敗でリトライ上限
        queue.dequeue(worker_id="worker_01")
        should_retry = queue.fail(task_id, "Error")
        assert should_retry is False

        task = queue.get_task(task_id)
        assert task.status == TaskStatus.FAILED

    # === 状態管理テスト ===

    def test_get_queue_size(self, queue: TaskQueue):
        """get_queue_size がPENDINGタスク数を返すこと"""
        # 3タスクエンキュー
        queue.enqueue(task_type="routing", task_payload={})
        queue.enqueue(task_type="routing", task_payload={})
        queue.enqueue(task_type="routing", task_payload={})

        assert queue.get_queue_size() == 3

        # 1タスクデキュー
        queue.dequeue(worker_id="worker_01")

        assert queue.get_queue_size() == 2

    def test_get_processing_count(self, queue: TaskQueue):
        """get_processing_count がPROCESSINGタスク数を返すこと"""
        queue.enqueue(task_type="routing", task_payload={})
        queue.enqueue(task_type="routing", task_payload={})

        assert queue.get_processing_count() == 0

        queue.dequeue(worker_id="worker_01")

        assert queue.get_processing_count() == 1

    def test_get_stats(self, queue: TaskQueue):
        """get_stats が統計情報を返すこと"""
        # 初期状態
        stats = queue.get_stats()
        assert stats["pending"] == 0
        assert stats["processing"] == 0
        assert stats["completed"] == 0
        assert stats["failed"] == 0
        assert stats["total"] == 0

        # タスク追加
        queue.enqueue(task_type="routing", task_payload={})
        queue.enqueue(task_type="routing", task_payload={})
        queue.enqueue(task_type="routing", task_payload={}, max_retries=1)

        # 1つ完了、1つ失敗（デキュー順にタスクを処理）
        task1 = queue.dequeue(worker_id="worker_01")
        queue.complete(task1.id, {"result": "ok"})

        task2 = queue.dequeue(worker_id="worker_01")
        queue.fail(task2.id, "Error")  # max_retries=3 なのでリトライ→PENDING

        # task2 はリトライされてPENDINGに戻るため、再度デキューして失敗させる
        # ただし、max_retries=3（デフォルト）なので、複数回失敗が必要
        # テストを簡略化: 3回目のタスクをmax_retries=1で作成済みなのでそれを使う
        task3 = queue.dequeue(worker_id="worker_01")
        queue.fail(task3.id, "Error")  # max_retries=1 でFAILED

        stats = queue.get_stats()
        assert stats["pending"] == 1  # task2がリトライでPENDINGに戻っている
        assert stats["processing"] == 0
        assert stats["completed"] == 1
        assert stats["failed"] == 1
        assert stats["total"] == 3

    # === バリデーションテスト ===

    def test_complete_non_processing_task_raises_error(self, queue: TaskQueue):
        """PROCESSING以外のタスクにcompleteするとエラー"""
        task_id = queue.enqueue(task_type="routing", task_payload={})

        with pytest.raises(ValueError, match="PROCESSING"):
            queue.complete(task_id, {})

    def test_fail_non_processing_task_raises_error(self, queue: TaskQueue):
        """PROCESSING以外のタスクにfailするとエラー"""
        task_id = queue.enqueue(task_type="routing", task_payload={})

        with pytest.raises(ValueError, match="PROCESSING"):
            queue.fail(task_id, "Error")

    def test_complete_nonexistent_task_raises_error(self, queue: TaskQueue):
        """存在しないタスクにcompleteするとエラー"""
        with pytest.raises(ValueError, match="存在しません"):
            queue.complete(uuid4(), {})

    def test_fail_nonexistent_task_raises_error(self, queue: TaskQueue):
        """存在しないタスクにfailするとエラー"""
        with pytest.raises(ValueError, match="存在しません"):
            queue.fail(uuid4(), "Error")

    # === キューサイズ制限テスト ===

    def test_queue_size_limit(self, queue: TaskQueue):
        """キューサイズ制限が機能すること"""
        queue.config.max_queue_size = 3

        queue.enqueue(task_type="routing", task_payload={})
        queue.enqueue(task_type="routing", task_payload={})
        queue.enqueue(task_type="routing", task_payload={})

        with pytest.raises(ValueError, match="満杯"):
            queue.enqueue(task_type="routing", task_payload={})

    def test_queue_size_limit_considers_only_pending(self, queue: TaskQueue):
        """キューサイズ制限がPENDINGのみを考慮すること"""
        queue.config.max_queue_size = 2

        task_id = queue.enqueue(task_type="routing", task_payload={})
        queue.enqueue(task_type="routing", task_payload={})

        # 1つデキュー（PROCESSING）
        queue.dequeue(worker_id="worker_01")

        # 新しいタスクをエンキュー可能（PENDINGは1つ）
        queue.enqueue(task_type="routing", task_payload={})

        assert queue.get_queue_size() == 2

    # === タイムアウトテスト ===

    def test_get_timed_out_tasks(self, queue: TaskQueue):
        """タイムアウトしたタスクを検出すること"""
        queue.config.task_timeout_seconds = 1

        task_id = queue.enqueue(task_type="routing", task_payload={})
        queue.dequeue(worker_id="worker_01")

        # 即座にはタイムアウトしない
        timed_out = queue.get_timed_out_tasks()
        assert len(timed_out) == 0

        # タイムアウト後
        time.sleep(1.1)
        timed_out = queue.get_timed_out_tasks()
        assert len(timed_out) == 1
        assert timed_out[0].id == task_id

    # === クリーンアップテスト ===

    def test_cleanup_completed(self, queue: TaskQueue):
        """cleanup_completed が古いタスクを削除すること"""
        task_id = queue.enqueue(task_type="routing", task_payload={})
        queue.dequeue(worker_id="worker_01")
        queue.complete(task_id, {})

        # 直後はクリーンアップされない
        deleted = queue.cleanup_completed(max_age_seconds=1)
        assert deleted == 0

        # 時間経過後にクリーンアップ
        time.sleep(1.1)
        deleted = queue.cleanup_completed(max_age_seconds=1)
        assert deleted == 1

        assert queue.get_task(task_id) is None

    # === キャンセルテスト ===

    def test_cancel_pending_task(self, queue: TaskQueue):
        """PENDINGタスクをキャンセルできること"""
        task_id = queue.enqueue(task_type="routing", task_payload={})

        success = queue.cancel(task_id)
        assert success is True

        task = queue.get_task(task_id)
        assert task.status == TaskStatus.FAILED
        assert task.error_message == "Cancelled"

    def test_cancel_processing_task_fails(self, queue: TaskQueue):
        """PROCESSINGタスクはキャンセルできないこと"""
        task_id = queue.enqueue(task_type="routing", task_payload={})
        queue.dequeue(worker_id="worker_01")

        success = queue.cancel(task_id)
        assert success is False

        task = queue.get_task(task_id)
        assert task.status == TaskStatus.PROCESSING

    def test_cancel_nonexistent_task_fails(self, queue: TaskQueue):
        """存在しないタスクのキャンセルは失敗すること"""
        success = queue.cancel(uuid4())
        assert success is False

    # === クリアテスト ===

    def test_clear(self, queue: TaskQueue):
        """clear がすべてのタスクを削除すること"""
        queue.enqueue(task_type="routing", task_payload={})
        queue.enqueue(task_type="routing", task_payload={})

        deleted = queue.clear()
        assert deleted == 2
        assert queue.get_queue_size() == 0
        assert queue.get_stats()["total"] == 0

    # === セッションIDテスト ===

    def test_session_id(self, queue: TaskQueue):
        """session_id が保存されること"""
        session_id = uuid4()
        task_id = queue.enqueue(
            task_type="routing",
            task_payload={},
            session_id=session_id,
        )

        task = queue.get_task(task_id)
        assert task.session_id == session_id

    # === to_dict テスト ===

    def test_task_item_to_dict(self, queue: TaskQueue):
        """TaskItem.to_dict が正しい辞書を返すこと"""
        session_id = uuid4()
        task_id = queue.enqueue(
            task_type="execution",
            task_payload={"key": "value"},
            priority=3,
            session_id=session_id,
        )

        task = queue.get_task(task_id)
        d = task.to_dict()

        assert d["id"] == str(task_id)
        assert d["session_id"] == str(session_id)
        assert d["task_type"] == "execution"
        assert d["task_payload"] == {"key": "value"}
        assert d["priority"] == 3
        assert d["status"] == "pending"
        assert d["created_at"] is not None


class TestTaskQueueThreadSafety:
    """TaskQueue のスレッドセーフ性テスト"""

    @pytest.fixture
    def queue(self) -> TaskQueue:
        """テスト用のタスクキュー"""
        config = Phase3Config()
        config.max_queue_size = 1000
        return TaskQueue(config)

    def test_concurrent_enqueue(self, queue: TaskQueue):
        """並行enqueueが安全に動作すること"""
        num_tasks = 100
        task_ids = []
        lock = threading.Lock()

        def enqueue_task(i: int):
            task_id = queue.enqueue(
                task_type="routing",
                task_payload={"index": i},
            )
            with lock:
                task_ids.append(task_id)

        with ThreadPoolExecutor(max_workers=10) as executor:
            list(executor.map(enqueue_task, range(num_tasks)))

        assert len(task_ids) == num_tasks
        assert len(set(task_ids)) == num_tasks  # 重複なし
        assert queue.get_queue_size() == num_tasks

    def test_concurrent_dequeue(self, queue: TaskQueue):
        """並行dequeueが安全に動作すること（重複取得なし）"""
        num_tasks = 50

        for i in range(num_tasks):
            queue.enqueue(
                task_type="routing",
                task_payload={"index": i},
            )

        dequeued_ids = []
        lock = threading.Lock()

        def dequeue_task(worker_id: str):
            task = queue.dequeue(worker_id=worker_id)
            if task:
                with lock:
                    dequeued_ids.append(task.id)

        with ThreadPoolExecutor(max_workers=10) as executor:
            worker_ids = [f"worker_{i}" for i in range(num_tasks)]
            list(executor.map(dequeue_task, worker_ids))

        # 重複なく全タスクが取得される
        assert len(dequeued_ids) == num_tasks
        assert len(set(dequeued_ids)) == num_tasks

    def test_concurrent_enqueue_dequeue(self, queue: TaskQueue):
        """並行enqueue/dequeueが安全に動作すること"""
        num_producers = 5
        num_consumers = 5
        tasks_per_producer = 20

        enqueued = []
        dequeued = []
        enqueue_lock = threading.Lock()
        dequeue_lock = threading.Lock()

        def producer(producer_id: int):
            for i in range(tasks_per_producer):
                task_id = queue.enqueue(
                    task_type="routing",
                    task_payload={"producer": producer_id, "index": i},
                )
                with enqueue_lock:
                    enqueued.append(task_id)

        def consumer(consumer_id: int):
            while True:
                task = queue.dequeue(worker_id=f"consumer_{consumer_id}")
                if task:
                    with dequeue_lock:
                        dequeued.append(task.id)
                    queue.complete(task.id, {"consumer": consumer_id})
                else:
                    # キューが空の場合は少し待機してリトライ
                    time.sleep(0.01)

                # 十分な数を処理したら終了
                with dequeue_lock:
                    if len(dequeued) >= num_producers * tasks_per_producer:
                        break

        with ThreadPoolExecutor(max_workers=num_producers + num_consumers) as executor:
            # プロデューサーを起動
            producer_futures = [
                executor.submit(producer, i) for i in range(num_producers)
            ]
            # コンシューマーを起動
            consumer_futures = [
                executor.submit(consumer, i) for i in range(num_consumers)
            ]

            # プロデューサー完了を待機
            for f in producer_futures:
                f.result()

            # コンシューマー完了を待機（タイムアウト付き）
            for f in consumer_futures:
                f.result(timeout=10)

        # 全タスクが処理された
        assert len(enqueued) == num_producers * tasks_per_producer
        assert len(dequeued) == num_producers * tasks_per_producer
        assert set(enqueued) == set(dequeued)


class TestTaskQueueAsyncInterface:
    """TaskQueue の非同期インターフェーステスト"""

    @pytest.fixture
    def queue(self) -> TaskQueue:
        """テスト用のタスクキュー"""
        return TaskQueue()

    @pytest.mark.asyncio
    async def test_enqueue_async(self, queue: TaskQueue):
        """非同期enqueueが動作すること"""
        task_id = await queue.enqueue_async(
            task_type="routing",
            task_payload={"query": "test"},
        )

        assert task_id is not None
        task = queue.get_task(task_id)
        assert task.status == TaskStatus.PENDING

    @pytest.mark.asyncio
    async def test_dequeue_async(self, queue: TaskQueue):
        """非同期dequeueが動作すること"""
        task_id = await queue.enqueue_async(
            task_type="routing",
            task_payload={},
        )

        task = await queue.dequeue_async(worker_id="worker_01")
        assert task is not None
        assert task.id == task_id
        assert task.status == TaskStatus.PROCESSING

    @pytest.mark.asyncio
    async def test_complete_async(self, queue: TaskQueue):
        """非同期completeが動作すること"""
        task_id = await queue.enqueue_async(
            task_type="routing",
            task_payload={},
        )
        await queue.dequeue_async(worker_id="worker_01")

        await queue.complete_async(task_id, {"result": "ok"})

        task = queue.get_task(task_id)
        assert task.status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_fail_async(self, queue: TaskQueue):
        """非同期failが動作すること"""
        task_id = await queue.enqueue_async(
            task_type="routing",
            task_payload={},
            max_retries=1,
        )
        await queue.dequeue_async(worker_id="worker_01")

        should_retry = await queue.fail_async(task_id, "Error")
        assert should_retry is False

        task = queue.get_task(task_id)
        assert task.status == TaskStatus.FAILED
