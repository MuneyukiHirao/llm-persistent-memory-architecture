# tests/scheduling/test_load_balancer.py
"""LoadBalancer の単体テスト

LoadBalancer クラスのインスタンス選択、負荷分散アルゴリズム、
統計管理、スケーリング判定機能をテスト。
スレッドセーフ性も検証する。

実行方法:
    pytest tests/scheduling/test_load_balancer.py -v
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

from src.agents.agent_registry import AgentDefinition
from src.config.phase3_config import Phase3Config, SCALING_CONFIG
from src.scheduling.load_balancer import LoadBalancer


class TestLoadBalancerInit:
    """LoadBalancer 初期化テスト"""

    @pytest.fixture
    def mock_registry(self) -> MagicMock:
        """モックAgentRegistry"""
        return MagicMock()

    @pytest.fixture
    def config(self) -> Phase3Config:
        """テスト用設定"""
        config = Phase3Config()
        config.max_tasks_per_agent = 5
        config.load_balancer_algorithm = "round_robin"
        return config

    def test_init_sets_dependencies(self, mock_registry: MagicMock, config: Phase3Config):
        """初期化で依存関係が設定されること"""
        balancer = LoadBalancer(mock_registry, config)

        assert balancer.agent_registry == mock_registry
        assert balancer.config == config

    def test_init_creates_empty_state(self, mock_registry: MagicMock, config: Phase3Config):
        """初期化で空の内部状態が作成されること"""
        balancer = LoadBalancer(mock_registry, config)

        assert len(balancer._agent_loads) == 0
        assert len(balancer._agent_response_times) == 0
        assert len(balancer._agent_success_counts) == 0
        assert len(balancer._agent_total_counts) == 0
        assert len(balancer._round_robin_indices) == 0

    def test_init_sets_default_values(self, mock_registry: MagicMock, config: Phase3Config):
        """初期化でデフォルト値が設定されること"""
        balancer = LoadBalancer(mock_registry, config)

        assert balancer._max_response_time_samples == 100
        assert balancer._default_success_rate == 0.8
        assert balancer._default_response_time == 1.0


class TestSelectInstance:
    """select_instance メソッドのテスト"""

    @pytest.fixture
    def mock_registry(self) -> MagicMock:
        """アクティブなエージェントを返すモック"""
        registry = MagicMock()
        agent = MagicMock(spec=AgentDefinition)
        agent.status = "active"
        registry.get_by_id.return_value = agent
        return registry

    @pytest.fixture
    def config(self) -> Phase3Config:
        """テスト用設定"""
        config = Phase3Config()
        config.max_tasks_per_agent = 5
        config.load_balancer_algorithm = "round_robin"
        return config

    @pytest.fixture
    def balancer(self, mock_registry: MagicMock, config: Phase3Config) -> LoadBalancer:
        """テスト用LoadBalancer"""
        return LoadBalancer(mock_registry, config)

    def test_select_instance_returns_instance_id(self, balancer: LoadBalancer):
        """select_instance がインスタンスIDを返すこと"""
        instance_id = balancer.select_instance("test_agent")

        assert instance_id == "test_agent_0"

    def test_select_instance_uses_config_algorithm(self, balancer: LoadBalancer):
        """select_instance がconfig設定のアルゴリズムを使用すること"""
        # round_robinがデフォルト
        instance_id = balancer.select_instance("test_agent")
        assert instance_id == "test_agent_0"

    def test_select_instance_uses_specified_algorithm(self, balancer: LoadBalancer):
        """select_instance が指定されたアルゴリズムを使用すること"""
        instance_id = balancer.select_instance("test_agent", algorithm="least_connections")
        assert instance_id == "test_agent_0"

    def test_select_instance_returns_none_for_inactive_agent(
        self, mock_registry: MagicMock, config: Phase3Config
    ):
        """非アクティブエージェントの場合はNoneを返すこと"""
        agent = MagicMock(spec=AgentDefinition)
        agent.status = "disabled"
        mock_registry.get_by_id.return_value = agent

        balancer = LoadBalancer(mock_registry, config)
        instance_id = balancer.select_instance("disabled_agent")

        assert instance_id is None

    def test_select_instance_returns_none_for_nonexistent_agent(
        self, mock_registry: MagicMock, config: Phase3Config
    ):
        """存在しないエージェントの場合はNoneを返すこと"""
        mock_registry.get_by_id.return_value = None

        balancer = LoadBalancer(mock_registry, config)
        instance_id = balancer.select_instance("nonexistent_agent")

        assert instance_id is None

    def test_select_instance_returns_none_when_at_capacity(
        self, balancer: LoadBalancer
    ):
        """負荷上限に達している場合はNoneを返すこと"""
        # 負荷を上限に設定
        balancer._agent_loads["test_agent_0"] = 5

        instance_id = balancer.select_instance("test_agent")
        assert instance_id is None

    def test_select_instance_unknown_algorithm_returns_first(
        self, balancer: LoadBalancer
    ):
        """未知のアルゴリズムの場合は最初のインスタンスを返すこと"""
        instance_id = balancer.select_instance("test_agent", algorithm="unknown")
        assert instance_id == "test_agent_0"


class TestRoundRobin:
    """_round_robin メソッドのテスト"""

    @pytest.fixture
    def balancer(self) -> LoadBalancer:
        """テスト用LoadBalancer"""
        mock_registry = MagicMock()
        config = Phase3Config()
        return LoadBalancer(mock_registry, config)

    def test_round_robin_cycles_through_instances(self, balancer: LoadBalancer):
        """ラウンドロビンがインスタンスを順番に選択すること"""
        instances = ["instance_0", "instance_1", "instance_2"]

        results = [
            balancer._round_robin(instances, "agent_1")
            for _ in range(6)
        ]

        assert results == [
            "instance_0", "instance_1", "instance_2",
            "instance_0", "instance_1", "instance_2",
        ]

    def test_round_robin_maintains_separate_indices_per_agent(self, balancer: LoadBalancer):
        """異なるエージェントで別々のインデックスを維持すること"""
        instances = ["instance_0", "instance_1"]

        # agent_1 を2回選択
        balancer._round_robin(instances, "agent_1")
        balancer._round_robin(instances, "agent_1")

        # agent_2 を1回選択（0から開始）
        result = balancer._round_robin(instances, "agent_2")
        assert result == "instance_0"

        # agent_1 の次の選択（続きから）
        result = balancer._round_robin(instances, "agent_1")
        assert result == "instance_0"

    def test_round_robin_single_instance(self, balancer: LoadBalancer):
        """単一インスタンスの場合は常にそれを返すこと"""
        instances = ["only_instance"]

        results = [
            balancer._round_robin(instances, "agent_1")
            for _ in range(3)
        ]

        assert results == ["only_instance", "only_instance", "only_instance"]


class TestWeightedRoundRobin:
    """_weighted_round_robin メソッドのテスト"""

    @pytest.fixture
    def balancer(self) -> LoadBalancer:
        """テスト用LoadBalancer"""
        mock_registry = MagicMock()
        config = Phase3Config()
        return LoadBalancer(mock_registry, config)

    def test_weighted_round_robin_prefers_higher_success_rate(self, balancer: LoadBalancer):
        """成功率の高いインスタンスが選ばれやすいこと"""
        instances = ["high_success", "low_success"]

        # 成功率を設定
        balancer._agent_total_counts["high_success"] = 100
        balancer._agent_success_counts["high_success"] = 90  # 90%
        balancer._agent_total_counts["low_success"] = 100
        balancer._agent_success_counts["low_success"] = 10  # 10%

        # 多数回実行して傾向を確認
        counts = {"high_success": 0, "low_success": 0}
        for _ in range(100):
            selected = balancer._weighted_round_robin(instances)
            counts[selected] += 1

        # 高成功率のインスタンスがより多く選ばれる（確率的なので緩い条件）
        assert counts["high_success"] > counts["low_success"]

    def test_weighted_round_robin_uses_default_rate_for_new_instances(
        self, balancer: LoadBalancer
    ):
        """新しいインスタンスにはデフォルト成功率が使用されること"""
        instances = ["new_instance"]

        # 実行でエラーが出ないこと
        result = balancer._weighted_round_robin(instances)
        assert result == "new_instance"

    def test_weighted_round_robin_all_zero_weights_random_choice(
        self, balancer: LoadBalancer
    ):
        """全ての重みが0の場合はランダムに選択されること"""
        instances = ["instance_0", "instance_1"]

        # 成功率を0に設定
        balancer._agent_total_counts["instance_0"] = 10
        balancer._agent_success_counts["instance_0"] = 0
        balancer._agent_total_counts["instance_1"] = 10
        balancer._agent_success_counts["instance_1"] = 0

        # 実行でエラーが出ないこと
        result = balancer._weighted_round_robin(instances)
        assert result in instances


class TestLeastConnections:
    """_least_connections メソッドのテスト"""

    @pytest.fixture
    def balancer(self) -> LoadBalancer:
        """テスト用LoadBalancer"""
        mock_registry = MagicMock()
        config = Phase3Config()
        return LoadBalancer(mock_registry, config)

    def test_least_connections_selects_minimum_load(self, balancer: LoadBalancer):
        """最小接続数のインスタンスが選択されること"""
        instances = ["instance_0", "instance_1", "instance_2"]

        balancer._agent_loads["instance_0"] = 5
        balancer._agent_loads["instance_1"] = 2
        balancer._agent_loads["instance_2"] = 8

        result = balancer._least_connections(instances)
        assert result == "instance_1"

    def test_least_connections_new_instances_have_zero_load(self, balancer: LoadBalancer):
        """新しいインスタンスは負荷0として扱われること"""
        instances = ["existing", "new"]

        balancer._agent_loads["existing"] = 3

        result = balancer._least_connections(instances)
        assert result == "new"

    def test_least_connections_equal_loads_returns_first(self, balancer: LoadBalancer):
        """同じ負荷の場合は最初のインスタンスを返すこと"""
        instances = ["first", "second", "third"]

        # 全て同じ負荷
        balancer._agent_loads["first"] = 2
        balancer._agent_loads["second"] = 2
        balancer._agent_loads["third"] = 2

        result = balancer._least_connections(instances)
        assert result == "first"


class TestAdaptive:
    """_adaptive メソッドのテスト"""

    @pytest.fixture
    def balancer(self) -> LoadBalancer:
        """テスト用LoadBalancer"""
        mock_registry = MagicMock()
        config = Phase3Config()
        return LoadBalancer(mock_registry, config)

    def test_adaptive_considers_load_and_response_time(self, balancer: LoadBalancer):
        """負荷とレスポンス時間の両方を考慮すること"""
        instances = ["fast_low", "slow_high"]

        # fast_low: 負荷低い、レスポンス速い → スコア低い（良い）
        balancer._agent_loads["fast_low"] = 1
        balancer._agent_response_times["fast_low"] = [0.5, 0.6, 0.4]

        # slow_high: 負荷高い、レスポンス遅い → スコア高い（悪い）
        balancer._agent_loads["slow_high"] = 5
        balancer._agent_response_times["slow_high"] = [2.0, 2.5, 3.0]

        result = balancer._adaptive(instances)
        assert result == "fast_low"

    def test_adaptive_uses_default_response_time_for_new(self, balancer: LoadBalancer):
        """新しいインスタンスにはデフォルトレスポンス時間が使用されること"""
        instances = ["new_instance"]

        result = balancer._adaptive(instances)
        assert result == "new_instance"

    def test_adaptive_zero_load_uses_minimum_value(self, balancer: LoadBalancer):
        """負荷0の場合は最小値0.1を使用すること"""
        instances = ["zero_load", "some_load"]

        balancer._agent_loads["zero_load"] = 0
        balancer._agent_response_times["zero_load"] = [1.0]

        balancer._agent_loads["some_load"] = 2
        balancer._agent_response_times["some_load"] = [1.0]

        # zero_load: 0.1 * 1.0 = 0.1
        # some_load: 2 * 1.0 = 2.0
        result = balancer._adaptive(instances)
        assert result == "zero_load"


class TestGetHealthyInstances:
    """_get_healthy_instances メソッドのテスト"""

    @pytest.fixture
    def mock_registry(self) -> MagicMock:
        """モックAgentRegistry"""
        return MagicMock()

    @pytest.fixture
    def config(self) -> Phase3Config:
        """テスト用設定"""
        config = Phase3Config()
        config.max_tasks_per_agent = 5
        return config

    def test_get_healthy_instances_active_agent(
        self, mock_registry: MagicMock, config: Phase3Config
    ):
        """アクティブなエージェントの場合はインスタンスを返すこと"""
        agent = MagicMock(spec=AgentDefinition)
        agent.status = "active"
        mock_registry.get_by_id.return_value = agent

        balancer = LoadBalancer(mock_registry, config)
        instances = balancer._get_healthy_instances("test_agent")

        assert instances == ["test_agent_0"]

    def test_get_healthy_instances_disabled_agent(
        self, mock_registry: MagicMock, config: Phase3Config
    ):
        """無効なエージェントの場合は空リストを返すこと"""
        agent = MagicMock(spec=AgentDefinition)
        agent.status = "disabled"
        mock_registry.get_by_id.return_value = agent

        balancer = LoadBalancer(mock_registry, config)
        instances = balancer._get_healthy_instances("test_agent")

        assert instances == []

    def test_get_healthy_instances_nonexistent_agent(
        self, mock_registry: MagicMock, config: Phase3Config
    ):
        """存在しないエージェントの場合は空リストを返すこと"""
        mock_registry.get_by_id.return_value = None

        balancer = LoadBalancer(mock_registry, config)
        instances = balancer._get_healthy_instances("nonexistent")

        assert instances == []

    def test_get_healthy_instances_at_capacity(
        self, mock_registry: MagicMock, config: Phase3Config
    ):
        """負荷上限に達している場合は空リストを返すこと"""
        agent = MagicMock(spec=AgentDefinition)
        agent.status = "active"
        mock_registry.get_by_id.return_value = agent

        balancer = LoadBalancer(mock_registry, config)
        balancer._agent_loads["test_agent_0"] = 5  # 上限

        instances = balancer._get_healthy_instances("test_agent")
        assert instances == []


class TestGetSuccessRate:
    """_get_success_rate メソッドのテスト"""

    @pytest.fixture
    def balancer(self) -> LoadBalancer:
        """テスト用LoadBalancer"""
        mock_registry = MagicMock()
        config = Phase3Config()
        return LoadBalancer(mock_registry, config)

    def test_get_success_rate_calculates_correctly(self, balancer: LoadBalancer):
        """成功率が正しく計算されること"""
        balancer._agent_total_counts["instance_1"] = 100
        balancer._agent_success_counts["instance_1"] = 75

        rate = balancer._get_success_rate("instance_1")
        assert rate == 0.75

    def test_get_success_rate_default_for_new_instance(self, balancer: LoadBalancer):
        """新しいインスタンスにはデフォルト値が返されること"""
        rate = balancer._get_success_rate("new_instance")
        assert rate == 0.8  # デフォルト値

    def test_get_success_rate_zero_total(self, balancer: LoadBalancer):
        """total=0の場合はデフォルト値が返されること"""
        balancer._agent_total_counts["instance_1"] = 0

        rate = balancer._get_success_rate("instance_1")
        assert rate == 0.8


class TestGetAvgResponseTime:
    """_get_avg_response_time メソッドのテスト"""

    @pytest.fixture
    def balancer(self) -> LoadBalancer:
        """テスト用LoadBalancer"""
        mock_registry = MagicMock()
        config = Phase3Config()
        return LoadBalancer(mock_registry, config)

    def test_get_avg_response_time_calculates_correctly(self, balancer: LoadBalancer):
        """平均レスポンス時間が正しく計算されること"""
        balancer._agent_response_times["instance_1"] = [1.0, 2.0, 3.0]

        avg = balancer._get_avg_response_time("instance_1")
        assert avg == 2.0

    def test_get_avg_response_time_default_for_new_instance(self, balancer: LoadBalancer):
        """新しいインスタンスにはデフォルト値が返されること"""
        avg = balancer._get_avg_response_time("new_instance")
        assert avg == 1.0  # デフォルト値

    def test_get_avg_response_time_empty_list(self, balancer: LoadBalancer):
        """空リストの場合はデフォルト値が返されること"""
        balancer._agent_response_times["instance_1"] = []

        avg = balancer._get_avg_response_time("instance_1")
        assert avg == 1.0


class TestUpdateLoad:
    """update_load メソッドのテスト"""

    @pytest.fixture
    def balancer(self) -> LoadBalancer:
        """テスト用LoadBalancer"""
        mock_registry = MagicMock()
        config = Phase3Config()
        return LoadBalancer(mock_registry, config)

    def test_update_load_increment(self, balancer: LoadBalancer):
        """負荷をインクリメントできること"""
        balancer.update_load("instance_1", +1)
        assert balancer._agent_loads["instance_1"] == 1

        balancer.update_load("instance_1", +1)
        assert balancer._agent_loads["instance_1"] == 2

    def test_update_load_decrement(self, balancer: LoadBalancer):
        """負荷をデクリメントできること"""
        balancer._agent_loads["instance_1"] = 5

        balancer.update_load("instance_1", -1)
        assert balancer._agent_loads["instance_1"] == 4

    def test_update_load_not_negative(self, balancer: LoadBalancer):
        """負荷が負の値にならないこと"""
        balancer._agent_loads["instance_1"] = 1

        balancer.update_load("instance_1", -5)
        assert balancer._agent_loads["instance_1"] == 0

    def test_update_load_new_instance(self, balancer: LoadBalancer):
        """新しいインスタンスの負荷を更新できること"""
        balancer.update_load("new_instance", +3)
        assert balancer._agent_loads["new_instance"] == 3


class TestRecordResponseTime:
    """record_response_time メソッドのテスト"""

    @pytest.fixture
    def balancer(self) -> LoadBalancer:
        """テスト用LoadBalancer"""
        mock_registry = MagicMock()
        config = Phase3Config()
        return LoadBalancer(mock_registry, config)

    def test_record_response_time_adds_to_list(self, balancer: LoadBalancer):
        """レスポンス時間がリストに追加されること"""
        balancer.record_response_time("instance_1", 1.5)
        balancer.record_response_time("instance_1", 2.0)

        assert balancer._agent_response_times["instance_1"] == [1.5, 2.0]

    def test_record_response_time_limits_samples(self, balancer: LoadBalancer):
        """サンプル数が制限されること"""
        balancer._max_response_time_samples = 5

        for i in range(10):
            balancer.record_response_time("instance_1", float(i))

        # 最新の5つのみ保持
        assert len(balancer._agent_response_times["instance_1"]) == 5
        assert balancer._agent_response_times["instance_1"] == [5.0, 6.0, 7.0, 8.0, 9.0]


class TestRecordResult:
    """record_result メソッドのテスト"""

    @pytest.fixture
    def balancer(self) -> LoadBalancer:
        """テスト用LoadBalancer"""
        mock_registry = MagicMock()
        config = Phase3Config()
        return LoadBalancer(mock_registry, config)

    def test_record_result_success(self, balancer: LoadBalancer):
        """成功結果が記録されること"""
        balancer.record_result("instance_1", True)
        balancer.record_result("instance_1", True)

        assert balancer._agent_total_counts["instance_1"] == 2
        assert balancer._agent_success_counts["instance_1"] == 2

    def test_record_result_failure(self, balancer: LoadBalancer):
        """失敗結果が記録されること"""
        balancer.record_result("instance_1", False)
        balancer.record_result("instance_1", False)

        assert balancer._agent_total_counts["instance_1"] == 2
        assert balancer._agent_success_counts["instance_1"] == 0

    def test_record_result_mixed(self, balancer: LoadBalancer):
        """成功と失敗の混合が正しく記録されること"""
        balancer.record_result("instance_1", True)
        balancer.record_result("instance_1", False)
        balancer.record_result("instance_1", True)

        assert balancer._agent_total_counts["instance_1"] == 3
        assert balancer._agent_success_counts["instance_1"] == 2


class TestGetInstanceStats:
    """get_instance_stats メソッドのテスト"""

    @pytest.fixture
    def balancer(self) -> LoadBalancer:
        """テスト用LoadBalancer"""
        mock_registry = MagicMock()
        config = Phase3Config()
        return LoadBalancer(mock_registry, config)

    def test_get_instance_stats_full_data(self, balancer: LoadBalancer):
        """すべてのデータがある場合の統計情報"""
        balancer._agent_loads["instance_1"] = 3
        balancer._agent_total_counts["instance_1"] = 100
        balancer._agent_success_counts["instance_1"] = 80
        balancer._agent_response_times["instance_1"] = [1.0, 2.0, 3.0]

        stats = balancer.get_instance_stats("instance_1")

        assert stats["instance_id"] == "instance_1"
        assert stats["current_load"] == 3
        assert stats["total_tasks"] == 100
        assert stats["successful_tasks"] == 80
        assert stats["success_rate"] == 0.8
        assert stats["avg_response_time"] == 2.0
        assert stats["response_time_samples"] == 3

    def test_get_instance_stats_new_instance(self, balancer: LoadBalancer):
        """新しいインスタンスの統計情報"""
        stats = balancer.get_instance_stats("new_instance")

        assert stats["instance_id"] == "new_instance"
        assert stats["current_load"] == 0
        assert stats["total_tasks"] == 0
        assert stats["successful_tasks"] == 0
        assert stats["success_rate"] is None
        assert stats["avg_response_time"] is None
        assert stats["response_time_samples"] == 0


class TestGetAllStats:
    """get_all_stats メソッドのテスト"""

    @pytest.fixture
    def balancer(self) -> LoadBalancer:
        """テスト用LoadBalancer"""
        mock_registry = MagicMock()
        config = Phase3Config()
        return LoadBalancer(mock_registry, config)

    def test_get_all_stats_empty(self, balancer: LoadBalancer):
        """空の場合は空辞書を返すこと"""
        stats = balancer.get_all_stats()
        assert stats == {}

    def test_get_all_stats_multiple_instances(self, balancer: LoadBalancer):
        """複数インスタンスの統計情報を返すこと"""
        balancer._agent_loads["instance_1"] = 2
        balancer._agent_loads["instance_2"] = 3
        balancer._agent_total_counts["instance_3"] = 10

        stats = balancer.get_all_stats()

        assert "instance_1" in stats
        assert "instance_2" in stats
        assert "instance_3" in stats
        assert len(stats) == 3


class TestResetStats:
    """reset_stats メソッドのテスト"""

    @pytest.fixture
    def balancer(self) -> LoadBalancer:
        """テスト用LoadBalancer"""
        mock_registry = MagicMock()
        config = Phase3Config()
        return LoadBalancer(mock_registry, config)

    def test_reset_stats_single_instance(self, balancer: LoadBalancer):
        """特定インスタンスの統計をリセットできること"""
        balancer._agent_loads["instance_1"] = 5
        balancer._agent_loads["instance_2"] = 3
        balancer._agent_response_times["instance_1"] = [1.0, 2.0]
        balancer._agent_success_counts["instance_1"] = 10
        balancer._agent_total_counts["instance_1"] = 20

        balancer.reset_stats("instance_1")

        assert "instance_1" not in balancer._agent_loads
        assert "instance_1" not in balancer._agent_response_times
        assert "instance_1" not in balancer._agent_success_counts
        assert "instance_1" not in balancer._agent_total_counts
        # instance_2 は残っている
        assert balancer._agent_loads["instance_2"] == 3

    def test_reset_stats_all(self, balancer: LoadBalancer):
        """全ての統計をリセットできること"""
        balancer._agent_loads["instance_1"] = 5
        balancer._agent_loads["instance_2"] = 3
        balancer._agent_response_times["instance_1"] = [1.0]
        balancer._round_robin_indices["agent_1"] = 5

        balancer.reset_stats()

        assert len(balancer._agent_loads) == 0
        assert len(balancer._agent_response_times) == 0
        assert len(balancer._agent_success_counts) == 0
        assert len(balancer._agent_total_counts) == 0
        assert len(balancer._round_robin_indices) == 0


class TestShouldScaleUp:
    """should_scale_up メソッドのテスト"""

    @pytest.fixture
    def mock_registry(self) -> MagicMock:
        """アクティブなエージェントを返すモック"""
        registry = MagicMock()
        agent = MagicMock(spec=AgentDefinition)
        agent.status = "active"
        registry.get_by_id.return_value = agent
        return registry

    @pytest.fixture
    def config(self) -> Phase3Config:
        """テスト用設定"""
        config = Phase3Config()
        config.max_tasks_per_agent = 10
        return config

    @pytest.fixture
    def balancer(self, mock_registry: MagicMock, config: Phase3Config) -> LoadBalancer:
        """テスト用LoadBalancer"""
        return LoadBalancer(mock_registry, config)

    def test_should_scale_up_no_instances(
        self, mock_registry: MagicMock, config: Phase3Config
    ):
        """インスタンスがない場合はTrueを返すこと"""
        mock_registry.get_by_id.return_value = None
        balancer = LoadBalancer(mock_registry, config)

        assert balancer.should_scale_up("nonexistent_agent") is True

    def test_should_scale_up_high_load(self, balancer: LoadBalancer):
        """負荷が閾値以上の場合はTrueを返すこと"""
        # 80%以上の負荷
        balancer._agent_loads["test_agent_0"] = 8  # 8/10 = 80%

        assert balancer.should_scale_up("test_agent") is True

    def test_should_scale_up_low_load(self, balancer: LoadBalancer):
        """負荷が閾値未満の場合はFalseを返すこと"""
        balancer._agent_loads["test_agent_0"] = 5  # 5/10 = 50%

        assert balancer.should_scale_up("test_agent") is False


class TestShouldScaleDown:
    """should_scale_down メソッドのテスト"""

    @pytest.fixture
    def mock_registry(self) -> MagicMock:
        """アクティブなエージェントを返すモック"""
        registry = MagicMock()
        agent = MagicMock(spec=AgentDefinition)
        agent.status = "active"
        registry.get_by_id.return_value = agent
        return registry

    @pytest.fixture
    def config(self) -> Phase3Config:
        """テスト用設定"""
        config = Phase3Config()
        config.max_tasks_per_agent = 10
        return config

    @pytest.fixture
    def balancer(self, mock_registry: MagicMock, config: Phase3Config) -> LoadBalancer:
        """テスト用LoadBalancer"""
        return LoadBalancer(mock_registry, config)

    def test_should_scale_down_at_min_instances(self, balancer: LoadBalancer):
        """最小インスタンス数の場合はFalseを返すこと"""
        # Phase 3 MVPでは1インスタンスのみなので常にFalse
        balancer._agent_loads["test_agent_0"] = 0

        assert balancer.should_scale_down("test_agent") is False

    def test_should_scale_down_no_instances(
        self, mock_registry: MagicMock, config: Phase3Config
    ):
        """インスタンスがない場合はFalseを返すこと"""
        mock_registry.get_by_id.return_value = None
        balancer = LoadBalancer(mock_registry, config)

        assert balancer.should_scale_down("nonexistent_agent") is False


class TestLoadBalancerThreadSafety:
    """LoadBalancer のスレッドセーフ性テスト"""

    @pytest.fixture
    def balancer(self) -> LoadBalancer:
        """テスト用LoadBalancer"""
        mock_registry = MagicMock()
        config = Phase3Config()
        return LoadBalancer(mock_registry, config)

    def test_concurrent_update_load(self, balancer: LoadBalancer):
        """並行update_loadが安全に動作すること"""
        num_threads = 10
        increments_per_thread = 100

        def increment_load():
            for _ in range(increments_per_thread):
                balancer.update_load("instance_1", +1)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(increment_load) for _ in range(num_threads)]
            for f in futures:
                f.result()

        assert balancer._agent_loads["instance_1"] == num_threads * increments_per_thread

    def test_concurrent_record_response_time(self, balancer: LoadBalancer):
        """並行record_response_timeが安全に動作すること"""
        num_threads = 10
        records_per_thread = 100
        balancer._max_response_time_samples = 1000

        def record_times():
            for i in range(records_per_thread):
                balancer.record_response_time("instance_1", float(i))

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(record_times) for _ in range(num_threads)]
            for f in futures:
                f.result()

        # 全ての記録が追加されている
        assert len(balancer._agent_response_times["instance_1"]) == min(
            num_threads * records_per_thread, balancer._max_response_time_samples
        )

    def test_concurrent_record_result(self, balancer: LoadBalancer):
        """並行record_resultが安全に動作すること"""
        num_threads = 10
        records_per_thread = 100

        def record_results():
            for _ in range(records_per_thread):
                balancer.record_result("instance_1", True)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(record_results) for _ in range(num_threads)]
            for f in futures:
                f.result()

        assert balancer._agent_total_counts["instance_1"] == num_threads * records_per_thread
        assert balancer._agent_success_counts["instance_1"] == num_threads * records_per_thread

    def test_concurrent_round_robin(self, balancer: LoadBalancer):
        """並行round_robinが安全に動作すること"""
        instances = ["i0", "i1", "i2"]
        num_threads = 10
        selections_per_thread = 100

        results = []
        lock = threading.Lock()

        def select_instance():
            for _ in range(selections_per_thread):
                selected = balancer._round_robin(instances, "agent_1")
                with lock:
                    results.append(selected)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(select_instance) for _ in range(num_threads)]
            for f in futures:
                f.result()

        # 全ての選択が完了している
        assert len(results) == num_threads * selections_per_thread

        # 各インスタンスが均等に選択されている（誤差許容）
        counts = {i: results.count(i) for i in instances}
        expected = num_threads * selections_per_thread // len(instances)
        for count in counts.values():
            assert abs(count - expected) < expected * 0.2  # 20%の誤差許容

    def test_concurrent_mixed_operations(self, balancer: LoadBalancer):
        """複数種類の操作が並行して安全に動作すること"""
        num_threads = 5
        ops_per_thread = 50

        def mixed_operations(thread_id: int):
            for i in range(ops_per_thread):
                instance_id = f"instance_{thread_id}"
                balancer.update_load(instance_id, +1)
                balancer.record_response_time(instance_id, float(i))
                balancer.record_result(instance_id, i % 2 == 0)
                balancer.get_instance_stats(instance_id)
                balancer.update_load(instance_id, -1)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(mixed_operations, i) for i in range(num_threads)]
            for f in futures:
                f.result()

        # 各インスタンスの負荷が0に戻っている
        for i in range(num_threads):
            assert balancer._agent_loads[f"instance_{i}"] == 0


class TestLoadBalancerEdgeCases:
    """LoadBalancer 境界値・エッジケーステスト"""

    @pytest.fixture
    def mock_registry(self) -> MagicMock:
        """モックAgentRegistry"""
        return MagicMock()

    @pytest.fixture
    def config(self) -> Phase3Config:
        """テスト用設定"""
        return Phase3Config()

    def test_empty_instances_list(self, mock_registry: MagicMock, config: Phase3Config):
        """空のインスタンスリストの処理"""
        mock_registry.get_by_id.return_value = None
        balancer = LoadBalancer(mock_registry, config)

        result = balancer.select_instance("nonexistent")
        assert result is None

    def test_very_long_response_time_list(
        self, mock_registry: MagicMock, config: Phase3Config
    ):
        """非常に長いレスポンス時間リストの処理"""
        balancer = LoadBalancer(mock_registry, config)
        balancer._max_response_time_samples = 10

        # 1000件記録
        for i in range(1000):
            balancer.record_response_time("instance_1", float(i))

        # 最新の10件のみ保持
        assert len(balancer._agent_response_times["instance_1"]) == 10
        assert balancer._agent_response_times["instance_1"][0] == 990.0

    def test_zero_max_tasks_per_agent(
        self, mock_registry: MagicMock, config: Phase3Config
    ):
        """max_tasks_per_agent=0の場合"""
        config.max_tasks_per_agent = 0
        agent = MagicMock(spec=AgentDefinition)
        agent.status = "active"
        mock_registry.get_by_id.return_value = agent

        balancer = LoadBalancer(mock_registry, config)

        # 負荷0でも上限に達している
        result = balancer.select_instance("test_agent")
        assert result is None

    def test_negative_response_time(
        self, mock_registry: MagicMock, config: Phase3Config
    ):
        """負のレスポンス時間の記録"""
        balancer = LoadBalancer(mock_registry, config)

        # 負の値も記録される（呼び出し側の責任で適切な値を渡す想定）
        balancer.record_response_time("instance_1", -1.0)
        assert balancer._agent_response_times["instance_1"] == [-1.0]

    def test_very_high_success_rate_weight(
        self, mock_registry: MagicMock, config: Phase3Config
    ):
        """非常に高い成功率の重み付け選択"""
        balancer = LoadBalancer(mock_registry, config)

        instances = ["perfect", "terrible"]

        # 100%成功率
        balancer._agent_total_counts["perfect"] = 1000
        balancer._agent_success_counts["perfect"] = 1000

        # 0%成功率
        balancer._agent_total_counts["terrible"] = 1000
        balancer._agent_success_counts["terrible"] = 0

        # 100回選択
        selections = [balancer._weighted_round_robin(instances) for _ in range(100)]

        # perfectがほぼ全て選ばれる
        perfect_count = selections.count("perfect")
        assert perfect_count == 100  # 0%の重みは0なので全てperfect

    def test_special_characters_in_agent_id(
        self, mock_registry: MagicMock, config: Phase3Config
    ):
        """特殊文字を含むagent_idの処理"""
        agent = MagicMock(spec=AgentDefinition)
        agent.status = "active"
        mock_registry.get_by_id.return_value = agent
        config.max_tasks_per_agent = 5

        balancer = LoadBalancer(mock_registry, config)

        # 特殊文字を含むagent_id
        result = balancer.select_instance("agent-with-special_chars.v1")
        assert result == "agent-with-special_chars.v1_0"

    def test_reset_stats_nonexistent_instance(
        self, mock_registry: MagicMock, config: Phase3Config
    ):
        """存在しないインスタンスのリセット"""
        balancer = LoadBalancer(mock_registry, config)

        # エラーが発生しないこと
        balancer.reset_stats("nonexistent_instance")

    def test_get_stats_concurrent_modification(
        self, mock_registry: MagicMock, config: Phase3Config
    ):
        """統計取得中の並行変更"""
        balancer = LoadBalancer(mock_registry, config)
        balancer._agent_loads["instance_1"] = 5

        def modify_stats():
            for _ in range(100):
                balancer.update_load("instance_1", +1)
                balancer.update_load("instance_1", -1)

        def get_stats():
            for _ in range(100):
                balancer.get_instance_stats("instance_1")

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(modify_stats),
                executor.submit(modify_stats),
                executor.submit(get_stats),
                executor.submit(get_stats),
            ]
            for f in futures:
                f.result()

        # 最終的に元の値に戻っている
        assert balancer._agent_loads["instance_1"] == 5
