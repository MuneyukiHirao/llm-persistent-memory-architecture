# tests/monitoring/test_metrics_collector.py
"""MetricsCollector のユニットテスト

テスト観点:
- Counter: インクリメント、取得、ラベル管理
- Gauge: 設定、インクリメント、デクリメント
- Histogram: 観測、バケットカウント、合計・回数
- MetricsCollector: 便利メソッド、Prometheusフォーマット出力
- スレッドセーフ性
"""

import pytest
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from src.monitoring.metrics_collector import (
    Counter,
    Gauge,
    Histogram,
    MetricsCollector,
    MetricType,
    get_metrics_collector,
    reset_global_collector,
)


class TestCounter:
    """Counter クラスのテスト"""

    def test_basic_increment(self):
        """基本的なインクリメント"""
        counter = Counter("test_counter", "Test counter")
        assert counter.get() == 0.0

        counter.inc()
        assert counter.get() == 1.0

        counter.inc(value=5.0)
        assert counter.get() == 6.0

    def test_increment_with_labels(self):
        """ラベル付きインクリメント"""
        counter = Counter("test_counter", "Test counter", ["method", "status"])

        counter.inc({"method": "GET", "status": "200"})
        counter.inc({"method": "GET", "status": "200"})
        counter.inc({"method": "POST", "status": "201"})

        assert counter.get({"method": "GET", "status": "200"}) == 2.0
        assert counter.get({"method": "POST", "status": "201"}) == 1.0
        assert counter.get({"method": "DELETE", "status": "404"}) == 0.0

    def test_negative_increment_raises_error(self):
        """負の値でのインクリメントはエラー"""
        counter = Counter("test_counter", "Test counter")

        with pytest.raises(ValueError, match="Counter can only be incremented"):
            counter.inc(value=-1.0)

    def test_collect(self):
        """collect メソッドのテスト"""
        counter = Counter("test_counter", "Test counter", ["status"])

        counter.inc({"status": "success"}, 3.0)
        counter.inc({"status": "failure"}, 1.0)

        collected = counter.collect()
        assert len(collected) == 2

        # 結果を辞書に変換して検証
        results = {tuple(sorted(labels.items())): value for labels, value in collected}
        assert results[(("status", "success"),)] == 3.0
        assert results[(("status", "failure"),)] == 1.0


class TestGauge:
    """Gauge クラスのテスト"""

    def test_set_and_get(self):
        """設定と取得"""
        gauge = Gauge("test_gauge", "Test gauge")
        assert gauge.get() == 0.0

        gauge.set(value=42.0)
        assert gauge.get() == 42.0

        gauge.set(value=10.5)
        assert gauge.get() == 10.5

    def test_increment_and_decrement(self):
        """インクリメントとデクリメント"""
        gauge = Gauge("test_gauge", "Test gauge")

        gauge.inc(value=5.0)
        assert gauge.get() == 5.0

        gauge.dec(value=2.0)
        assert gauge.get() == 3.0

        gauge.inc()  # デフォルト +1
        assert gauge.get() == 4.0

        gauge.dec()  # デフォルト -1
        assert gauge.get() == 3.0

    def test_with_labels(self):
        """ラベル付きゲージ"""
        gauge = Gauge("test_gauge", "Test gauge", ["queue_name"])

        gauge.set({"queue_name": "tasks"}, 100)
        gauge.set({"queue_name": "events"}, 50)

        assert gauge.get({"queue_name": "tasks"}) == 100
        assert gauge.get({"queue_name": "events"}) == 50

    def test_collect(self):
        """collect メソッドのテスト"""
        gauge = Gauge("test_gauge", "Test gauge", ["region"])

        gauge.set({"region": "us-east"}, 10)
        gauge.set({"region": "us-west"}, 20)

        collected = gauge.collect()
        assert len(collected) == 2

        results = {tuple(sorted(labels.items())): value for labels, value in collected}
        assert results[(("region", "us-east"),)] == 10
        assert results[(("region", "us-west"),)] == 20


class TestHistogram:
    """Histogram クラスのテスト"""

    def test_basic_observe(self):
        """基本的な観測"""
        histogram = Histogram(
            "test_histogram",
            "Test histogram",
            buckets=(0.1, 0.5, 1.0, 5.0),
        )

        histogram.observe(value=0.05)  # <= 0.1
        histogram.observe(value=0.3)   # <= 0.5
        histogram.observe(value=0.8)   # <= 1.0
        histogram.observe(value=2.0)   # <= 5.0

        assert histogram.get_count() == 4
        assert histogram.get_sum() == pytest.approx(3.15)

        bucket_counts = histogram.get_bucket_counts()
        assert bucket_counts[0.1] == 1   # 0.05 のみ
        assert bucket_counts[0.5] == 2   # 0.05, 0.3
        assert bucket_counts[1.0] == 3   # 0.05, 0.3, 0.8
        assert bucket_counts[5.0] == 4   # 全て

    def test_with_labels(self):
        """ラベル付きヒストグラム"""
        histogram = Histogram(
            "test_histogram",
            "Test histogram",
            labels=["endpoint"],
            buckets=(0.1, 0.5, 1.0),
        )

        histogram.observe({"endpoint": "/api/search"}, 0.05)
        histogram.observe({"endpoint": "/api/search"}, 0.3)
        histogram.observe({"endpoint": "/api/update"}, 0.8)

        assert histogram.get_count({"endpoint": "/api/search"}) == 2
        assert histogram.get_count({"endpoint": "/api/update"}) == 1
        assert histogram.get_sum({"endpoint": "/api/search"}) == pytest.approx(0.35)

    def test_default_buckets(self):
        """デフォルトバケット使用"""
        histogram = Histogram("test_histogram", "Test histogram")
        assert histogram.buckets == Histogram.DEFAULT_BUCKETS

    def test_collect(self):
        """collect メソッドのテスト"""
        histogram = Histogram(
            "test_histogram",
            "Test histogram",
            buckets=(0.1, 0.5, 1.0),
        )

        histogram.observe(value=0.05)
        histogram.observe(value=0.3)

        collected = histogram.collect()
        assert len(collected) == 1

        labels, data = collected[0]
        assert labels == {}
        assert data["count"] == 2.0
        assert data["sum"] == pytest.approx(0.35)
        assert data["bucket_0.1"] == 1.0
        assert data["bucket_0.5"] == 2.0
        assert data["bucket_1.0"] == 2.0


class TestMetricsCollector:
    """MetricsCollector クラスのテスト"""

    @pytest.fixture
    def collector(self):
        """テスト用コレクター"""
        return MetricsCollector(prefix="test")

    def test_register_and_get_metrics(self, collector):
        """メトリクスの登録と取得"""
        # カウンター
        counter = collector.register_counter("my_counter", "My counter", ["label1"])
        assert collector.get_counter("my_counter") is counter

        # ゲージ
        gauge = collector.register_gauge("my_gauge", "My gauge")
        assert collector.get_gauge("my_gauge") is gauge

        # ヒストグラム
        histogram = collector.register_histogram(
            "my_histogram", "My histogram",
            buckets=(0.1, 0.5, 1.0),
        )
        assert collector.get_histogram("my_histogram") is histogram

        # 存在しないメトリクス
        assert collector.get_counter("nonexistent") is None
        assert collector.get_gauge("nonexistent") is None
        assert collector.get_histogram("nonexistent") is None

    def test_predefined_metrics_exist(self, collector):
        """定義済みメトリクスが存在すること"""
        # スケーラビリティ指標
        assert collector.get_counter("tasks_completed_total") is not None
        assert collector.get_counter("tasks_started_total") is not None
        assert collector.get_histogram("task_wait_seconds") is not None
        assert collector.get_gauge("queue_length") is not None
        assert collector.get_gauge("agent_utilization") is not None

        # 可用性指標
        assert collector.get_gauge("orchestrator_healthy") is not None
        assert collector.get_gauge("orchestrator_total") is not None
        assert collector.get_histogram("failover_duration_seconds") is not None

        # ニューラルスコアラー指標
        assert collector.get_counter("routing_decisions_total") is not None
        assert collector.get_histogram("inference_latency_seconds") is not None

    def test_record_task_completion(self, collector):
        """タスク完了記録"""
        collector.record_task_started("routing")
        collector.record_task_completion("routing", success=True, wait_time_seconds=0.5)
        collector.record_task_completion("routing", success=False, wait_time_seconds=1.0)

        started = collector.get_counter("tasks_started_total")
        assert started.get({"task_type": "routing"}) == 1.0

        completed = collector.get_counter("tasks_completed_total")
        assert completed.get({"task_type": "routing", "status": "success"}) == 1.0
        assert completed.get({"task_type": "routing", "status": "failure"}) == 1.0

        wait_histogram = collector.get_histogram("task_wait_seconds")
        assert wait_histogram.get_count({"task_type": "routing"}) == 2

    def test_record_queue_length(self, collector):
        """キュー長記録"""
        collector.record_queue_length("tasks", 42)
        collector.record_queue_length("events", 10)

        gauge = collector.get_gauge("queue_length")
        assert gauge.get({"queue_name": "tasks"}) == 42
        assert gauge.get({"queue_name": "events"}) == 10

    def test_record_agent_utilization(self, collector):
        """エージェント利用率記録"""
        collector.record_agent_utilization("agent_001", 0.75)
        collector.record_agent_utilization("agent_002", 0.50)

        gauge = collector.get_gauge("agent_utilization")
        assert gauge.get({"agent_id": "agent_001"}) == 0.75
        assert gauge.get({"agent_id": "agent_002"}) == 0.50

    def test_record_scale_event(self, collector):
        """スケールイベント記録"""
        collector.record_scale_event("up")
        collector.record_scale_event("up")
        collector.record_scale_event("down")

        counter = collector.get_counter("scale_events_total")
        assert counter.get({"direction": "up"}) == 2.0
        assert counter.get({"direction": "down"}) == 1.0

    def test_record_orchestrator_health(self, collector):
        """オーケストレーター可用性記録"""
        collector.record_orchestrator_health(healthy=3, total=4)

        healthy = collector.get_gauge("orchestrator_healthy")
        total = collector.get_gauge("orchestrator_total")

        assert healthy.get() == 3.0
        assert total.get() == 4.0

    def test_record_failover_duration(self, collector):
        """フェイルオーバー時間記録"""
        collector.record_failover_duration(15.5)
        collector.record_failover_duration(30.0)

        histogram = collector.get_histogram("failover_duration_seconds")
        assert histogram.get_count() == 2
        assert histogram.get_sum() == pytest.approx(45.5)

    def test_record_session_outcome(self, collector):
        """セッション結果記録"""
        collector.record_session_outcome(recovered=True)
        collector.record_session_outcome(recovered=True)
        collector.record_session_outcome(recovered=False)

        counter = collector.get_counter("sessions_total")
        assert counter.get({"outcome": "recovered"}) == 2.0
        assert counter.get({"outcome": "lost"}) == 1.0

    def test_record_websocket_connections(self, collector):
        """WebSocket接続記録"""
        collector.record_websocket_connections(active=50, peak=100)
        collector.record_websocket_connections(active=60, peak=90)  # peak更新されない

        active = collector.get_gauge("websocket_connections_active")
        peak = collector.get_gauge("websocket_connections_peak")

        assert active.get() == 60.0
        assert peak.get() == 100.0  # 最大値維持

    def test_record_routing_decision(self, collector):
        """ルーティング判断記録"""
        collector.record_routing_decision("neural", correct=True)
        collector.record_routing_decision("neural", correct=True)
        collector.record_routing_decision("neural", correct=False)
        collector.record_routing_decision("rule", correct=True)

        counter = collector.get_counter("routing_decisions_total")
        assert counter.get({"method": "neural", "result": "correct"}) == 2.0
        assert counter.get({"method": "neural", "result": "incorrect"}) == 1.0
        assert counter.get({"method": "rule", "result": "correct"}) == 1.0

    def test_record_inference_latency(self, collector):
        """推論レイテンシ記録"""
        collector.record_inference_latency("neural_scorer", 0.025)
        collector.record_inference_latency("neural_scorer", 0.035)
        collector.record_inference_latency("embedding", 0.100)

        histogram = collector.get_histogram("inference_latency_seconds")
        assert histogram.get_count({"model_type": "neural_scorer"}) == 2
        assert histogram.get_count({"model_type": "embedding"}) == 1

    def test_record_model_update(self, collector):
        """モデル更新記録"""
        collector.record_model_update("neural_scorer")
        collector.record_model_update("neural_scorer")

        counter = collector.get_counter("model_updates_total")
        assert counter.get({"model_type": "neural_scorer"}) == 2.0


class TestPrometheusExport:
    """Prometheusフォーマットエクスポートのテスト"""

    @pytest.fixture
    def collector(self):
        """テスト用コレクター"""
        return MetricsCollector(prefix="test")

    def test_export_counter(self, collector):
        """カウンターのエクスポート"""
        counter = collector.get_counter("tasks_completed_total")
        counter.inc({"task_type": "routing", "status": "success"}, 5.0)

        output = collector.export_prometheus_format()

        assert "# HELP test_tasks_completed_total" in output
        assert "# TYPE test_tasks_completed_total counter" in output
        assert 'test_tasks_completed_total{status="success",task_type="routing"} 5.0' in output

    def test_export_gauge(self, collector):
        """ゲージのエクスポート"""
        collector.record_queue_length("tasks", 42)

        output = collector.export_prometheus_format()

        assert "# HELP test_queue_length" in output
        assert "# TYPE test_queue_length gauge" in output
        assert 'test_queue_length{queue_name="tasks"} 42.0' in output

    def test_export_histogram(self, collector):
        """ヒストグラムのエクスポート"""
        collector.record_inference_latency("neural_scorer", 0.025)
        collector.record_inference_latency("neural_scorer", 0.075)

        output = collector.export_prometheus_format()

        assert "# HELP test_inference_latency_seconds" in output
        assert "# TYPE test_inference_latency_seconds histogram" in output
        assert "test_inference_latency_seconds_bucket" in output
        assert "test_inference_latency_seconds_sum" in output
        assert "test_inference_latency_seconds_count" in output

    def test_export_dict(self, collector):
        """辞書形式エクスポート"""
        collector.record_task_started("test")

        result = collector.export_dict()

        assert "counters" in result
        assert "gauges" in result
        assert "histograms" in result
        assert "exported_at" in result
        assert "tasks_started_total" in result["counters"]

    def test_label_escaping(self, collector):
        """ラベル値のエスケープ"""
        counter = collector.register_counter(
            "custom_counter", "Custom counter", ["message"]
        )
        counter.inc({"message": 'test "quoted" value'})

        output = collector.export_prometheus_format()
        assert r'message="test \"quoted\" value"' in output


class TestThreadSafety:
    """スレッドセーフ性のテスト"""

    def test_counter_thread_safety(self):
        """Counter のスレッドセーフ性"""
        counter = Counter("test_counter", "Test counter")
        iterations = 1000
        threads = 10

        def increment():
            for _ in range(iterations):
                counter.inc()

        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = [executor.submit(increment) for _ in range(threads)]
            for future in futures:
                future.result()

        assert counter.get() == iterations * threads

    def test_gauge_thread_safety(self):
        """Gauge のスレッドセーフ性"""
        gauge = Gauge("test_gauge", "Test gauge")
        iterations = 1000
        threads = 10

        def modify():
            for _ in range(iterations):
                gauge.inc()
                gauge.dec()

        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = [executor.submit(modify) for _ in range(threads)]
            for future in futures:
                future.result()

        # inc と dec が同数なので 0 になるはず
        assert gauge.get() == 0.0

    def test_histogram_thread_safety(self):
        """Histogram のスレッドセーフ性"""
        histogram = Histogram(
            "test_histogram",
            "Test histogram",
            buckets=(0.1, 0.5, 1.0),
        )
        iterations = 1000
        threads = 10
        test_value = 0.3

        def observe():
            for _ in range(iterations):
                histogram.observe(value=test_value)

        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = [executor.submit(observe) for _ in range(threads)]
            for future in futures:
                future.result()

        expected_count = iterations * threads
        assert histogram.get_count() == expected_count
        assert histogram.get_sum() == pytest.approx(expected_count * test_value)

    def test_collector_thread_safety(self):
        """MetricsCollector のスレッドセーフ性"""
        collector = MetricsCollector(prefix="test")
        iterations = 100
        threads = 10

        def operate():
            for i in range(iterations):
                collector.record_task_started("test")
                collector.record_task_completion("test", success=True)
                collector.record_queue_length("test", i)
                collector.record_inference_latency("test", 0.01)

        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = [executor.submit(operate) for _ in range(threads)]
            for future in futures:
                future.result()

        started = collector.get_counter("tasks_started_total")
        completed = collector.get_counter("tasks_completed_total")

        assert started.get({"task_type": "test"}) == iterations * threads
        assert completed.get({"task_type": "test", "status": "success"}) == iterations * threads


class TestGlobalCollector:
    """グローバルコレクターのテスト"""

    def setup_method(self):
        """各テストの前にグローバルコレクターをリセット"""
        reset_global_collector()

    def teardown_method(self):
        """各テストの後にグローバルコレクターをリセット"""
        reset_global_collector()

    def test_singleton_behavior(self):
        """シングルトン動作確認"""
        collector1 = get_metrics_collector()
        collector2 = get_metrics_collector()

        assert collector1 is collector2

    def test_reset_global_collector(self):
        """グローバルコレクターのリセット"""
        collector1 = get_metrics_collector()
        collector1.record_task_started("test")

        reset_global_collector()

        collector2 = get_metrics_collector()
        assert collector1 is not collector2

        # 新しいコレクターには古いデータがない
        started = collector2.get_counter("tasks_started_total")
        assert started.get({"task_type": "test"}) == 0.0


class TestMetricsCollectorReset:
    """MetricsCollector.reset() のテスト"""

    def test_reset_clears_all_data(self):
        """reset() が全データをクリアすること"""
        collector = MetricsCollector(prefix="test")

        # データを追加
        collector.record_task_started("test")
        collector.record_queue_length("test", 100)
        collector.record_inference_latency("test", 0.05)

        # リセット
        collector.reset()

        # 定義済みメトリクスは再作成されているが、データはクリア
        started = collector.get_counter("tasks_started_total")
        queue = collector.get_gauge("queue_length")
        latency = collector.get_histogram("inference_latency_seconds")

        assert started is not None
        assert started.get({"task_type": "test"}) == 0.0
        assert queue.get({"queue_name": "test"}) == 0.0
        assert latency.get_count({"model_type": "test"}) == 0
