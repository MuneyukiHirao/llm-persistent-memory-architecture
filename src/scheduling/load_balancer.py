# 負荷分散（LoadBalancer）
# 実装仕様: docs/phase3-implementation-spec.ja.md セクション5.3
"""
負荷分散モジュール

複数のエージェントインスタンスに対してタスクを効率的に分散する。

設計方針（タスク実行フローエージェント観点）:
- API設計: シンプルなselect_instance()呼び出しでインスタンス選択
- フロー整合性: 負荷更新とレスポンス時間記録を呼び出し側が行う設計
- エラー処理: 健全なインスタンスがない場合はNone返却
- 拡張性: アルゴリズムを設定で切り替え可能
- テスト容易性: 状態を持つがthreading.Lockで保護

対応アルゴリズム:
- round_robin: シンプルなラウンドロビン
- weighted_round_robin: 重み付きラウンドロビン（成功率考慮）
- least_connections: 最小接続数優先
- adaptive: 適応型（レスポンス時間考慮）
"""

import random
import threading
from collections import defaultdict
from typing import Dict, List, Optional

from src.agents.agent_registry import AgentRegistry
from src.config.phase3_config import (
    Phase3Config,
    LOAD_BALANCER_ALGORITHMS,
    SCALING_CONFIG,
    HEALTH_CHECK_CONFIG,
)


class LoadBalancer:
    """負荷分散

    複数のエージェントインスタンスに対してタスクを効率的に分散する。

    使用例:
        registry = AgentRegistry(db)
        config = Phase3Config()
        balancer = LoadBalancer(registry, config)

        # インスタンス選択
        instance_id = balancer.select_instance("research_agent")

        # 負荷追跡
        balancer.update_load(instance_id, +1)  # タスク開始
        # ... タスク実行 ...
        balancer.update_load(instance_id, -1)  # タスク完了
        balancer.record_response_time(instance_id, 1.5)  # レスポンス時間記録

    Attributes:
        agent_registry: AgentRegistryインスタンス
        config: Phase3Configインスタンス
    """

    def __init__(
        self,
        agent_registry: AgentRegistry,
        config: Phase3Config,
    ):
        """LoadBalancerを初期化

        Args:
            agent_registry: AgentRegistryインスタンス
            config: Phase3Configインスタンス
        """
        self.agent_registry = agent_registry
        self.config = config

        # 内部状態（スレッドセーフに管理）
        self._lock = threading.Lock()
        self._agent_loads: Dict[str, int] = defaultdict(int)
        self._agent_response_times: Dict[str, List[float]] = defaultdict(list)
        self._agent_success_counts: Dict[str, int] = defaultdict(int)
        self._agent_total_counts: Dict[str, int] = defaultdict(int)
        self._round_robin_indices: Dict[str, int] = defaultdict(int)

        # 設定値
        self._max_response_time_samples = 100  # 保持するレスポンス時間の最大サンプル数
        self._default_success_rate = 0.8  # デフォルトの成功率
        self._default_response_time = 1.0  # デフォルトのレスポンス時間（秒）

    def select_instance(
        self,
        agent_id: str,
        algorithm: Optional[str] = None,
    ) -> Optional[str]:
        """エージェントインスタンスを選択

        指定されたアルゴリズムに基づいて、最適なインスタンスを選択する。

        Args:
            agent_id: エージェントID
            algorithm: 負荷分散アルゴリズム（省略時はconfig設定を使用）
                - "round_robin": シンプルなラウンドロビン
                - "weighted_round_robin": 重み付きラウンドロビン
                - "least_connections": 最小接続数優先
                - "adaptive": 適応型

        Returns:
            選択されたインスタンスID、またはNone（利用可能なインスタンスがない場合）
        """
        algorithm = algorithm or self.config.load_balancer_algorithm
        instances = self._get_healthy_instances(agent_id)

        if not instances:
            return None

        if algorithm == "round_robin":
            return self._round_robin(instances, agent_id)

        elif algorithm == "weighted_round_robin":
            return self._weighted_round_robin(instances)

        elif algorithm == "least_connections":
            return self._least_connections(instances)

        elif algorithm == "adaptive":
            return self._adaptive(instances)

        # 未知のアルゴリズムの場合は最初のインスタンスを返す
        return instances[0]

    def _round_robin(self, instances: List[str], agent_id: str) -> str:
        """シンプルなラウンドロビン

        インスタンスを順番に選択する。

        Args:
            instances: 健全なインスタンスのリスト
            agent_id: エージェントID（インデックス管理用）

        Returns:
            選択されたインスタンスID
        """
        with self._lock:
            index = self._round_robin_indices[agent_id]
            selected = instances[index % len(instances)]
            self._round_robin_indices[agent_id] = (index + 1) % len(instances)
            return selected

    def _weighted_round_robin(self, instances: List[str]) -> str:
        """重み付きラウンドロビン

        成功率を重みとして使用し、成功率の高いインスタンスが選ばれやすくなる。

        Args:
            instances: 健全なインスタンスのリスト

        Returns:
            選択されたインスタンスID
        """
        weights = []
        for instance in instances:
            # 成功率を重みとして使用
            success_rate = self._get_success_rate(instance)
            weights.append(success_rate)

        # 全ての重みが0の場合は均等に選択
        if sum(weights) == 0:
            return random.choice(instances)

        # 重み付き選択
        return random.choices(instances, weights=weights)[0]

    def _least_connections(self, instances: List[str]) -> str:
        """最小接続数優先

        現在の負荷（接続数）が最も少ないインスタンスを選択する。

        Args:
            instances: 健全なインスタンスのリスト

        Returns:
            選択されたインスタンスID
        """
        with self._lock:
            loads = [(i, self._agent_loads.get(i, 0)) for i in instances]
        return min(loads, key=lambda x: x[1])[0]

    def _adaptive(self, instances: List[str]) -> str:
        """適応型（レスポンス時間考慮）

        負荷と平均レスポンス時間を考慮してスコアを計算し、
        最もスコアの低い（効率的な）インスタンスを選択する。

        スコア = 負荷 × 平均レスポンス時間

        Args:
            instances: 健全なインスタンスのリスト

        Returns:
            選択されたインスタンスID
        """
        scores = []
        with self._lock:
            for instance in instances:
                load = self._agent_loads.get(instance, 0)
                avg_response = self._get_avg_response_time(instance)
                # スコア = 負荷 × 平均レスポンス時間（低いほど良い）
                # 負荷が0の場合は0.1として計算（ゼロ除算回避）
                score = max(load, 0.1) * avg_response
                scores.append((instance, score))

        return min(scores, key=lambda x: x[1])[0]

    def _get_healthy_instances(self, agent_id: str) -> List[str]:
        """健全なインスタンス一覧を取得

        Phase 3 MVPでは簡易実装として、アクティブなエージェントがあれば
        そのagent_idをインスタンスIDとして返す。
        将来のコンテナオーケストレーション対応を想定した設計。

        Args:
            agent_id: エージェントID

        Returns:
            健全なインスタンスIDのリスト
        """
        # Phase 3 MVP: AgentRegistryからエージェントを取得
        # 将来的にはインスタンス管理サービスから取得
        agent = self.agent_registry.get_by_id(agent_id)

        if agent is None or agent.status != "active":
            return []

        # Phase 3 MVP: 単一インスタンスとして扱う
        # 将来的には複数インスタンスを返す
        # フォーマット: {agent_id}_{instance_number}
        instance_id = f"{agent_id}_0"

        # 負荷が上限に達していないか確認
        with self._lock:
            current_load = self._agent_loads.get(instance_id, 0)

        if current_load >= self.config.max_tasks_per_agent:
            return []

        return [instance_id]

    def _get_success_rate(self, instance: str) -> float:
        """インスタンスの成功率を取得

        Args:
            instance: インスタンスID

        Returns:
            成功率（0.0-1.0）
        """
        with self._lock:
            total = self._agent_total_counts.get(instance, 0)
            if total == 0:
                return self._default_success_rate

            success = self._agent_success_counts.get(instance, 0)
            return success / total

    def _get_avg_response_time(self, instance: str) -> float:
        """平均レスポンス時間を取得

        Args:
            instance: インスタンスID

        Returns:
            平均レスポンス時間（秒）
        """
        # ロックは呼び出し元で取得済みの想定だが、安全のため再取得
        response_times = self._agent_response_times.get(instance, [])
        if not response_times:
            return self._default_response_time

        return sum(response_times) / len(response_times)

    def update_load(self, instance_id: str, delta: int) -> None:
        """負荷カウントを更新

        タスク開始時に+1、完了時に-1を渡す。

        Args:
            instance_id: インスタンスID
            delta: 負荷の変化量（+1 または -1）
        """
        with self._lock:
            self._agent_loads[instance_id] = max(
                0, self._agent_loads.get(instance_id, 0) + delta
            )

    def record_response_time(self, instance_id: str, duration: float) -> None:
        """レスポンス時間を記録

        Args:
            instance_id: インスタンスID
            duration: レスポンス時間（秒）
        """
        with self._lock:
            times = self._agent_response_times[instance_id]
            times.append(duration)

            # サンプル数を制限（古いものを削除）
            if len(times) > self._max_response_time_samples:
                self._agent_response_times[instance_id] = times[
                    -self._max_response_time_samples :
                ]

    def record_result(self, instance_id: str, success: bool) -> None:
        """タスク結果を記録

        成功率計算のために呼び出す。

        Args:
            instance_id: インスタンスID
            success: タスクが成功したかどうか
        """
        with self._lock:
            self._agent_total_counts[instance_id] += 1
            if success:
                self._agent_success_counts[instance_id] += 1

    def get_instance_stats(self, instance_id: str) -> Dict:
        """インスタンスの統計情報を取得

        Args:
            instance_id: インスタンスID

        Returns:
            統計情報の辞書
        """
        with self._lock:
            total = self._agent_total_counts.get(instance_id, 0)
            success = self._agent_success_counts.get(instance_id, 0)
            response_times = self._agent_response_times.get(instance_id, [])

            return {
                "instance_id": instance_id,
                "current_load": self._agent_loads.get(instance_id, 0),
                "total_tasks": total,
                "successful_tasks": success,
                "success_rate": success / total if total > 0 else None,
                "avg_response_time": (
                    sum(response_times) / len(response_times)
                    if response_times
                    else None
                ),
                "response_time_samples": len(response_times),
            }

    def get_all_stats(self) -> Dict[str, Dict]:
        """全インスタンスの統計情報を取得

        Returns:
            インスタンスIDをキーとした統計情報の辞書
        """
        with self._lock:
            all_instances = set(self._agent_loads.keys())
            all_instances.update(self._agent_total_counts.keys())
            all_instances.update(self._agent_response_times.keys())

        return {
            instance_id: self.get_instance_stats(instance_id)
            for instance_id in all_instances
        }

    def reset_stats(self, instance_id: Optional[str] = None) -> None:
        """統計情報をリセット

        Args:
            instance_id: リセット対象のインスタンスID（省略時は全てリセット）
        """
        with self._lock:
            if instance_id:
                self._agent_loads.pop(instance_id, None)
                self._agent_response_times.pop(instance_id, None)
                self._agent_success_counts.pop(instance_id, None)
                self._agent_total_counts.pop(instance_id, None)
                self._round_robin_indices.pop(instance_id, None)
            else:
                self._agent_loads.clear()
                self._agent_response_times.clear()
                self._agent_success_counts.clear()
                self._agent_total_counts.clear()
                self._round_robin_indices.clear()

    def should_scale_up(self, agent_id: str) -> bool:
        """スケールアップが必要か判定

        Args:
            agent_id: エージェントID

        Returns:
            スケールアップが必要な場合True
        """
        instances = self._get_healthy_instances(agent_id)
        if not instances:
            return True

        with self._lock:
            total_load = sum(
                self._agent_loads.get(i, 0) for i in instances
            )
            max_capacity = len(instances) * self.config.max_tasks_per_agent
            load_ratio = total_load / max_capacity if max_capacity > 0 else 1.0

        return load_ratio >= SCALING_CONFIG["scale_up_threshold"]

    def should_scale_down(self, agent_id: str) -> bool:
        """スケールダウンが必要か判定

        Args:
            agent_id: エージェントID

        Returns:
            スケールダウンが必要な場合True
        """
        instances = self._get_healthy_instances(agent_id)
        if len(instances) <= SCALING_CONFIG["min_instances"]:
            return False

        with self._lock:
            total_load = sum(
                self._agent_loads.get(i, 0) for i in instances
            )
            max_capacity = len(instances) * self.config.max_tasks_per_agent
            load_ratio = total_load / max_capacity if max_capacity > 0 else 0.0

        return load_ratio <= SCALING_CONFIG["scale_down_threshold"]
