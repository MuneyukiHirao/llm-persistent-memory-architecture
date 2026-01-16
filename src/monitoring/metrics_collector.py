# src/monitoring/metrics_collector.py
"""メトリクス収集モジュール

Phase 3 MVP のメトリクス収集基盤。
Prometheus依存なしの簡易実装だが、将来的なprometheus_client移行を想定した
インターフェース設計となっている。

実装仕様: docs/phase3-implementation-spec.ja.md セクション7

収集するメトリクス:
- スケーラビリティ指標: タスクスループット、平均待ち時間、キュー長、エージェント利用率
- 可用性指標: オーケストレーター可用性、セッション継続率
- ニューラルスコアラー指標: ルーティング精度、推論レイテンシ
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from threading import Lock
from typing import Dict, List, Optional, Tuple
import time


class MetricType(Enum):
    """メトリクスの種類"""
    COUNTER = "counter"      # 単調増加（例: リクエスト数）
    GAUGE = "gauge"          # 上下する値（例: 現在のキュー長）
    HISTOGRAM = "histogram"  # 分布（例: レイテンシ）


@dataclass
class MetricMetadata:
    """メトリクスのメタデータ"""
    name: str
    metric_type: MetricType
    description: str
    labels: List[str] = field(default_factory=list)


class Counter:
    """カウンターメトリクス（単調増加）

    Prometheusのカウンターと互換性のあるインターフェース。
    スレッドセーフな実装。

    使用例:
        counter = Counter("requests_total", "Total requests", ["method", "status"])
        counter.inc({"method": "GET", "status": "200"})
        counter.inc({"method": "POST", "status": "201"}, 2.0)
    """

    def __init__(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
    ):
        self.name = name
        self.description = description
        self.labels = labels or []
        self._values: Dict[Tuple[str, ...], float] = {}
        self._lock = Lock()
        self._created_at = time.time()

    def inc(
        self,
        labels: Optional[Dict[str, str]] = None,
        value: float = 1.0,
    ) -> None:
        """カウンターをインクリメント

        Args:
            labels: ラベル値の辞書
            value: 増加量（デフォルト1.0、負の値は不可）

        Raises:
            ValueError: 負の値が指定された場合
        """
        if value < 0:
            raise ValueError("Counter can only be incremented (value must be >= 0)")

        label_key = self._get_label_key(labels)

        with self._lock:
            if label_key not in self._values:
                self._values[label_key] = 0.0
            self._values[label_key] += value

    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        """現在の値を取得"""
        label_key = self._get_label_key(labels)
        with self._lock:
            return self._values.get(label_key, 0.0)

    def _get_label_key(self, labels: Optional[Dict[str, str]]) -> Tuple[str, ...]:
        """ラベル値からキーを生成"""
        if not self.labels:
            return ()
        if labels is None:
            labels = {}
        return tuple(labels.get(label, "") for label in self.labels)

    def collect(self) -> List[Tuple[Dict[str, str], float]]:
        """全ての値を収集"""
        with self._lock:
            results = []
            for label_key, value in self._values.items():
                label_dict = dict(zip(self.labels, label_key)) if self.labels else {}
                results.append((label_dict, value))
            return results


class Gauge:
    """ゲージメトリクス（上下する値）

    Prometheusのゲージと互換性のあるインターフェース。
    スレッドセーフな実装。

    使用例:
        gauge = Gauge("queue_length", "Current queue length", ["queue_name"])
        gauge.set({"queue_name": "tasks"}, 42)
        gauge.inc({"queue_name": "tasks"})
        gauge.dec({"queue_name": "tasks"}, 5)
    """

    def __init__(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
    ):
        self.name = name
        self.description = description
        self.labels = labels or []
        self._values: Dict[Tuple[str, ...], float] = {}
        self._lock = Lock()

    def set(
        self,
        labels: Optional[Dict[str, str]] = None,
        value: float = 0.0,
    ) -> None:
        """値を設定"""
        label_key = self._get_label_key(labels)
        with self._lock:
            self._values[label_key] = value

    def inc(
        self,
        labels: Optional[Dict[str, str]] = None,
        value: float = 1.0,
    ) -> None:
        """値をインクリメント"""
        label_key = self._get_label_key(labels)
        with self._lock:
            if label_key not in self._values:
                self._values[label_key] = 0.0
            self._values[label_key] += value

    def dec(
        self,
        labels: Optional[Dict[str, str]] = None,
        value: float = 1.0,
    ) -> None:
        """値をデクリメント"""
        label_key = self._get_label_key(labels)
        with self._lock:
            if label_key not in self._values:
                self._values[label_key] = 0.0
            self._values[label_key] -= value

    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        """現在の値を取得"""
        label_key = self._get_label_key(labels)
        with self._lock:
            return self._values.get(label_key, 0.0)

    def _get_label_key(self, labels: Optional[Dict[str, str]]) -> Tuple[str, ...]:
        """ラベル値からキーを生成"""
        if not self.labels:
            return ()
        if labels is None:
            labels = {}
        return tuple(labels.get(label, "") for label in self.labels)

    def collect(self) -> List[Tuple[Dict[str, str], float]]:
        """全ての値を収集"""
        with self._lock:
            results = []
            for label_key, value in self._values.items():
                label_dict = dict(zip(self.labels, label_key)) if self.labels else {}
                results.append((label_dict, value))
            return results


class Histogram:
    """ヒストグラムメトリクス（分布）

    Prometheusのヒストグラムと互換性のあるインターフェース。
    バケット境界を指定して値の分布を追跡。
    スレッドセーフな実装。

    使用例:
        histogram = Histogram(
            "request_latency_seconds",
            "Request latency",
            ["endpoint"],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
        )
        histogram.observe({"endpoint": "/api/search"}, 0.123)
    """

    # デフォルトのバケット境界（Prometheus標準）
    DEFAULT_BUCKETS = (
        0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0
    )

    def __init__(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
        buckets: Optional[Tuple[float, ...]] = None,
    ):
        self.name = name
        self.description = description
        self.labels = labels or []
        self.buckets = tuple(sorted(buckets)) if buckets else self.DEFAULT_BUCKETS

        # バケットカウント（label_key -> bucket -> count）
        self._bucket_counts: Dict[Tuple[str, ...], Dict[float, int]] = {}
        # 合計値（label_key -> sum）
        self._sums: Dict[Tuple[str, ...], float] = {}
        # 観測回数（label_key -> count）
        self._counts: Dict[Tuple[str, ...], int] = {}
        self._lock = Lock()

    def observe(
        self,
        labels: Optional[Dict[str, str]] = None,
        value: float = 0.0,
    ) -> None:
        """値を観測"""
        label_key = self._get_label_key(labels)

        with self._lock:
            # バケット初期化
            if label_key not in self._bucket_counts:
                self._bucket_counts[label_key] = {b: 0 for b in self.buckets}
                self._sums[label_key] = 0.0
                self._counts[label_key] = 0

            # バケットカウント更新
            for bucket in self.buckets:
                if value <= bucket:
                    self._bucket_counts[label_key][bucket] += 1

            # 合計と回数を更新
            self._sums[label_key] += value
            self._counts[label_key] += 1

    def get_bucket_counts(
        self,
        labels: Optional[Dict[str, str]] = None,
    ) -> Dict[float, int]:
        """バケットカウントを取得"""
        label_key = self._get_label_key(labels)
        with self._lock:
            if label_key not in self._bucket_counts:
                return {b: 0 for b in self.buckets}
            return dict(self._bucket_counts[label_key])

    def get_sum(self, labels: Optional[Dict[str, str]] = None) -> float:
        """合計値を取得"""
        label_key = self._get_label_key(labels)
        with self._lock:
            return self._sums.get(label_key, 0.0)

    def get_count(self, labels: Optional[Dict[str, str]] = None) -> int:
        """観測回数を取得"""
        label_key = self._get_label_key(labels)
        with self._lock:
            return self._counts.get(label_key, 0)

    def _get_label_key(self, labels: Optional[Dict[str, str]]) -> Tuple[str, ...]:
        """ラベル値からキーを生成"""
        if not self.labels:
            return ()
        if labels is None:
            labels = {}
        return tuple(labels.get(label, "") for label in self.labels)

    def collect(self) -> List[Tuple[Dict[str, str], Dict[str, float]]]:
        """全ての値を収集

        Returns:
            List of (labels, {"bucket_X": count, ..., "sum": sum, "count": count})
        """
        with self._lock:
            results = []
            for label_key in self._bucket_counts.keys():
                label_dict = dict(zip(self.labels, label_key)) if self.labels else {}
                data = {}

                # バケットカウント
                for bucket, count in self._bucket_counts[label_key].items():
                    data[f"bucket_{bucket}"] = float(count)

                # +Inf バケット（全観測数）
                data["bucket_+Inf"] = float(self._counts[label_key])
                data["sum"] = self._sums[label_key]
                data["count"] = float(self._counts[label_key])

                results.append((label_dict, data))
            return results


class MetricsCollector:
    """メトリクス収集・管理クラス

    Phase 3 MVP のメトリクス収集基盤。
    カウンター、ゲージ、ヒストグラムを統合管理し、
    Prometheusフォーマットでのエクスポート機能を提供する。

    使用例:
        collector = MetricsCollector()

        # 定義済みメトリクスを使用
        collector.record_task_completion("routing", True)
        collector.record_queue_length("tasks", 42)
        collector.record_latency("inference", 0.05)

        # Prometheusフォーマットで出力
        metrics_text = collector.export_prometheus_format()
    """

    def __init__(self, prefix: str = "llm_memory"):
        """
        Args:
            prefix: メトリクス名のプレフィックス
        """
        self.prefix = prefix
        self._counters: Dict[str, Counter] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._histograms: Dict[str, Histogram] = {}
        self._lock = Lock()

        # 定義済みメトリクスを初期化
        self._init_predefined_metrics()

    def _init_predefined_metrics(self) -> None:
        """仕様書セクション7で定義されたメトリクスを初期化"""

        # === スケーラビリティ指標 ===

        # タスクスループット: tasks_completed / time_window
        self.register_counter(
            "tasks_completed_total",
            "Total number of completed tasks",
            ["task_type", "status"],
        )

        # タスク開始数（スループット計算用）
        self.register_counter(
            "tasks_started_total",
            "Total number of started tasks",
            ["task_type"],
        )

        # 平均待ち時間: avg(started_at - created_at)
        self.register_histogram(
            "task_wait_seconds",
            "Task wait time in queue (seconds)",
            ["task_type"],
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0),
        )

        # キュー長: current_queue_size
        self.register_gauge(
            "queue_length",
            "Current queue length",
            ["queue_name"],
        )

        # エージェント利用率: active_tasks / max_capacity
        self.register_gauge(
            "agent_utilization",
            "Agent utilization ratio (0.0-1.0)",
            ["agent_id"],
        )

        # スケールイベント数
        self.register_counter(
            "scale_events_total",
            "Total number of scale events",
            ["direction"],  # "up" or "down"
        )

        # === 可用性指標 ===

        # オーケストレーター可用性: healthy_orchestrators / total
        self.register_gauge(
            "orchestrator_healthy",
            "Number of healthy orchestrators",
        )

        self.register_gauge(
            "orchestrator_total",
            "Total number of orchestrators",
        )

        # フェイルオーバー時間
        self.register_histogram(
            "failover_duration_seconds",
            "Failover duration in seconds",
            buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 90.0, 120.0),
        )

        # セッション継続率
        self.register_counter(
            "sessions_total",
            "Total number of sessions",
            ["outcome"],  # "recovered", "lost"
        )

        # WebSocket接続維持率
        self.register_gauge(
            "websocket_connections_active",
            "Number of active WebSocket connections",
        )

        self.register_gauge(
            "websocket_connections_peak",
            "Peak number of WebSocket connections",
        )

        # === ニューラルスコアラー指標 ===

        # ルーティング精度: correct_routing / total_routing
        self.register_counter(
            "routing_decisions_total",
            "Total number of routing decisions",
            ["method", "result"],  # method: "neural", "rule", result: "correct", "incorrect"
        )

        # 推論レイテンシ
        self.register_histogram(
            "inference_latency_seconds",
            "Inference latency in seconds",
            ["model_type"],  # "neural_scorer", "embedding"
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
        )

        # モデル更新回数
        self.register_counter(
            "model_updates_total",
            "Total number of model updates",
            ["model_type"],
        )

    def register_counter(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
    ) -> Counter:
        """カウンターを登録"""
        full_name = f"{self.prefix}_{name}"
        counter = Counter(full_name, description, labels)

        with self._lock:
            self._counters[name] = counter

        return counter

    def register_gauge(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
    ) -> Gauge:
        """ゲージを登録"""
        full_name = f"{self.prefix}_{name}"
        gauge = Gauge(full_name, description, labels)

        with self._lock:
            self._gauges[name] = gauge

        return gauge

    def register_histogram(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
        buckets: Optional[Tuple[float, ...]] = None,
    ) -> Histogram:
        """ヒストグラムを登録"""
        full_name = f"{self.prefix}_{name}"
        histogram = Histogram(full_name, description, labels, buckets)

        with self._lock:
            self._histograms[name] = histogram

        return histogram

    def get_counter(self, name: str) -> Optional[Counter]:
        """カウンターを取得"""
        with self._lock:
            return self._counters.get(name)

    def get_gauge(self, name: str) -> Optional[Gauge]:
        """ゲージを取得"""
        with self._lock:
            return self._gauges.get(name)

    def get_histogram(self, name: str) -> Optional[Histogram]:
        """ヒストグラムを取得"""
        with self._lock:
            return self._histograms.get(name)

    # === 便利メソッド（定義済みメトリクスへの操作）===

    def record_task_started(self, task_type: str) -> None:
        """タスク開始を記録"""
        counter = self.get_counter("tasks_started_total")
        if counter:
            counter.inc({"task_type": task_type})

    def record_task_completion(
        self,
        task_type: str,
        success: bool,
        wait_time_seconds: Optional[float] = None,
    ) -> None:
        """タスク完了を記録"""
        status = "success" if success else "failure"

        counter = self.get_counter("tasks_completed_total")
        if counter:
            counter.inc({"task_type": task_type, "status": status})

        if wait_time_seconds is not None:
            histogram = self.get_histogram("task_wait_seconds")
            if histogram:
                histogram.observe({"task_type": task_type}, wait_time_seconds)

    def record_queue_length(self, queue_name: str, length: int) -> None:
        """キュー長を記録"""
        gauge = self.get_gauge("queue_length")
        if gauge:
            gauge.set({"queue_name": queue_name}, float(length))

    def record_agent_utilization(self, agent_id: str, utilization: float) -> None:
        """エージェント利用率を記録（0.0-1.0）"""
        gauge = self.get_gauge("agent_utilization")
        if gauge:
            gauge.set({"agent_id": agent_id}, utilization)

    def record_scale_event(self, direction: str) -> None:
        """スケールイベントを記録

        Args:
            direction: "up" or "down"
        """
        counter = self.get_counter("scale_events_total")
        if counter:
            counter.inc({"direction": direction})

    def record_orchestrator_health(self, healthy: int, total: int) -> None:
        """オーケストレーター可用性を記録"""
        healthy_gauge = self.get_gauge("orchestrator_healthy")
        total_gauge = self.get_gauge("orchestrator_total")

        if healthy_gauge:
            healthy_gauge.set(value=float(healthy))
        if total_gauge:
            total_gauge.set(value=float(total))

    def record_failover_duration(self, duration_seconds: float) -> None:
        """フェイルオーバー時間を記録"""
        histogram = self.get_histogram("failover_duration_seconds")
        if histogram:
            histogram.observe(value=duration_seconds)

    def record_session_outcome(self, recovered: bool) -> None:
        """セッション結果を記録"""
        counter = self.get_counter("sessions_total")
        outcome = "recovered" if recovered else "lost"
        if counter:
            counter.inc({"outcome": outcome})

    def record_websocket_connections(self, active: int, peak: Optional[int] = None) -> None:
        """WebSocket接続数を記録"""
        active_gauge = self.get_gauge("websocket_connections_active")
        if active_gauge:
            active_gauge.set(value=float(active))

        if peak is not None:
            peak_gauge = self.get_gauge("websocket_connections_peak")
            if peak_gauge:
                # ピークは最大値のみ更新
                current_peak = peak_gauge.get()
                if peak > current_peak:
                    peak_gauge.set(value=float(peak))

    def record_routing_decision(
        self,
        method: str,
        correct: bool,
    ) -> None:
        """ルーティング判断を記録

        Args:
            method: "neural" or "rule"
            correct: 正解かどうか
        """
        counter = self.get_counter("routing_decisions_total")
        result = "correct" if correct else "incorrect"
        if counter:
            counter.inc({"method": method, "result": result})

    def record_inference_latency(
        self,
        model_type: str,
        latency_seconds: float,
    ) -> None:
        """推論レイテンシを記録

        Args:
            model_type: "neural_scorer", "embedding" など
            latency_seconds: レイテンシ（秒）
        """
        histogram = self.get_histogram("inference_latency_seconds")
        if histogram:
            histogram.observe({"model_type": model_type}, latency_seconds)

    def record_model_update(self, model_type: str) -> None:
        """モデル更新を記録"""
        counter = self.get_counter("model_updates_total")
        if counter:
            counter.inc({"model_type": model_type})

    def export_prometheus_format(self) -> str:
        """Prometheusテキストフォーマットでエクスポート

        Returns:
            Prometheus exposition format の文字列
        """
        lines = []
        timestamp_ms = int(time.time() * 1000)

        with self._lock:
            # カウンター
            for name, counter in self._counters.items():
                lines.append(f"# HELP {counter.name} {counter.description}")
                lines.append(f"# TYPE {counter.name} counter")

                for labels, value in counter.collect():
                    label_str = self._format_labels(labels)
                    lines.append(f"{counter.name}{label_str} {value}")
                lines.append("")

            # ゲージ
            for name, gauge in self._gauges.items():
                lines.append(f"# HELP {gauge.name} {gauge.description}")
                lines.append(f"# TYPE {gauge.name} gauge")

                for labels, value in gauge.collect():
                    label_str = self._format_labels(labels)
                    lines.append(f"{gauge.name}{label_str} {value}")
                lines.append("")

            # ヒストグラム
            for name, histogram in self._histograms.items():
                lines.append(f"# HELP {histogram.name} {histogram.description}")
                lines.append(f"# TYPE {histogram.name} histogram")

                for labels, data in histogram.collect():
                    base_label_str = self._format_labels(labels)

                    # バケット
                    for bucket in histogram.buckets:
                        bucket_labels = {**labels, "le": str(bucket)}
                        label_str = self._format_labels(bucket_labels)
                        count = data.get(f"bucket_{bucket}", 0)
                        lines.append(f"{histogram.name}_bucket{label_str} {count}")

                    # +Inf バケット
                    inf_labels = {**labels, "le": "+Inf"}
                    inf_label_str = self._format_labels(inf_labels)
                    lines.append(f"{histogram.name}_bucket{inf_label_str} {data.get('count', 0)}")

                    # sum と count
                    lines.append(f"{histogram.name}_sum{base_label_str} {data.get('sum', 0)}")
                    lines.append(f"{histogram.name}_count{base_label_str} {data.get('count', 0)}")
                lines.append("")

        return "\n".join(lines)

    def _format_labels(self, labels: Dict[str, str]) -> str:
        """ラベルをPrometheusフォーマットに変換"""
        if not labels:
            return ""

        parts = []
        for key, value in sorted(labels.items()):
            # 値をエスケープ
            escaped_value = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
            parts.append(f'{key}="{escaped_value}"')

        return "{" + ",".join(parts) + "}"

    def export_dict(self) -> Dict[str, any]:
        """辞書形式でエクスポート（デバッグ・内部利用）

        Returns:
            全メトリクスの辞書
        """
        result = {
            "counters": {},
            "gauges": {},
            "histograms": {},
            "exported_at": datetime.now().isoformat(),
        }

        with self._lock:
            for name, counter in self._counters.items():
                result["counters"][name] = {
                    "description": counter.description,
                    "values": [
                        {"labels": labels, "value": value}
                        for labels, value in counter.collect()
                    ],
                }

            for name, gauge in self._gauges.items():
                result["gauges"][name] = {
                    "description": gauge.description,
                    "values": [
                        {"labels": labels, "value": value}
                        for labels, value in gauge.collect()
                    ],
                }

            for name, histogram in self._histograms.items():
                result["histograms"][name] = {
                    "description": histogram.description,
                    "buckets": histogram.buckets,
                    "values": [
                        {"labels": labels, "data": data}
                        for labels, data in histogram.collect()
                    ],
                }

        return result

    def reset(self) -> None:
        """全メトリクスをリセット（テスト用）"""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()

        # 定義済みメトリクスを再初期化
        self._init_predefined_metrics()


# グローバルなメトリクスコレクターインスタンス（シングルトンパターン）
_global_collector: Optional[MetricsCollector] = None
_global_lock = Lock()


def get_metrics_collector(prefix: str = "llm_memory") -> MetricsCollector:
    """グローバルなメトリクスコレクターを取得

    シングルトンパターンでグローバルインスタンスを管理。

    Args:
        prefix: メトリクス名のプレフィックス（初回呼び出し時のみ有効）

    Returns:
        MetricsCollector インスタンス
    """
    global _global_collector

    with _global_lock:
        if _global_collector is None:
            _global_collector = MetricsCollector(prefix)
        return _global_collector


def reset_global_collector() -> None:
    """グローバルなメトリクスコレクターをリセット（テスト用）"""
    global _global_collector

    with _global_lock:
        _global_collector = None
