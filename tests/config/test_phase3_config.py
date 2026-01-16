# Phase 3 設定クラスの単体テスト
# 実装仕様: docs/phase3-implementation-spec.ja.md セクション4

import pytest

from src.config.phase1_config import Phase1Config
from src.config.phase2_config import Phase2Config
from src.config.phase3_config import (
    Phase3Config,
    phase3_config,
    NEURAL_SCORER_CONFIG,
    TRAINING_CONFIG,
    TASK_FEATURES,
    AGENT_FEATURES,
    LOAD_BALANCER_ALGORITHMS,
    SCALING_CONFIG,
    HEALTH_CHECK_CONFIG,
    WEBSOCKET_EVENT_TYPES,
    WEBSOCKET_CONFIG,
)


class TestPhase3ConfigInheritance:
    """Phase3Configの継承関係のテスト"""

    def test_inherits_from_phase2_config(self):
        """Phase2Configを継承していることを確認"""
        assert issubclass(Phase3Config, Phase2Config)

    def test_inherits_from_phase1_config(self):
        """Phase1Configを継承していることを確認（Phase2経由）"""
        assert issubclass(Phase3Config, Phase1Config)

    def test_phase2_attributes_accessible(self):
        """Phase2の属性にアクセスできることを確認"""
        config = Phase3Config()
        # Phase2固有の属性
        assert hasattr(config, "orchestrator_model")
        assert hasattr(config, "routing_method")
        assert hasattr(config, "feedback_detection_method")

    def test_phase1_attributes_accessible(self):
        """Phase1の属性にアクセスできることを確認"""
        config = Phase3Config()
        # Phase1固有の属性
        assert hasattr(config, "initial_strength")
        assert hasattr(config, "similarity_threshold")
        assert hasattr(config, "embedding_model")

    def test_phase1_methods_accessible(self):
        """Phase1のメソッドにアクセスできることを確認"""
        config = Phase3Config()
        # Phase1のメソッド
        assert hasattr(config, "get_decay_rate")
        assert hasattr(config, "get_consolidation_level")
        # 実際に動作することを確認
        assert config.get_consolidation_level(10) == 1


class TestNeuralScorerSettings:
    """ニューラルスコアラー設定のテスト"""

    def test_neural_scorer_enabled_default(self):
        """neural_scorer_enabledのデフォルト値（False）を確認"""
        config = Phase3Config()
        assert config.neural_scorer_enabled is False

    def test_neural_scorer_model_path_default(self):
        """neural_scorer_model_pathのデフォルト値を確認"""
        config = Phase3Config()
        assert config.neural_scorer_model_path == "models/routing_scorer.pt"

    def test_min_training_samples_default(self):
        """min_training_samplesのデフォルト値（1000）を確認"""
        config = Phase3Config()
        assert config.min_training_samples == 1000

    def test_neural_scorer_threshold_default(self):
        """neural_scorer_thresholdのデフォルト値（0.7）を確認"""
        config = Phase3Config()
        assert config.neural_scorer_threshold == 0.7

    def test_min_training_samples_overrides_phase1(self):
        """Phase1のmin_training_samples（100）をオーバーライドしていることを確認"""
        phase1_config = Phase1Config()
        phase3_config_inst = Phase3Config()
        assert phase1_config.min_training_samples == 100
        assert phase3_config_inst.min_training_samples == 1000


class TestTaskQueueSettings:
    """タスクキュー設定のテスト"""

    def test_task_queue_enabled_default(self):
        """task_queue_enabledのデフォルト値（True）を確認"""
        config = Phase3Config()
        assert config.task_queue_enabled is True

    def test_redis_url_default(self):
        """redis_urlのデフォルト値を確認"""
        config = Phase3Config()
        assert config.redis_url == "redis://localhost:6379/0"

    def test_max_queue_size_default(self):
        """max_queue_sizeのデフォルト値（1000）を確認"""
        config = Phase3Config()
        assert config.max_queue_size == 1000

    def test_task_timeout_seconds_default(self):
        """task_timeout_secondsのデフォルト値（600）を確認"""
        config = Phase3Config()
        assert config.task_timeout_seconds == 600


class TestLoadBalancerSettings:
    """負荷分散設定のテスト"""

    def test_load_balancer_algorithm_default(self):
        """load_balancer_algorithmのデフォルト値を確認"""
        config = Phase3Config()
        assert config.load_balancer_algorithm == "weighted_round_robin"

    def test_max_tasks_per_agent_default(self):
        """max_tasks_per_agentのデフォルト値（5）を確認"""
        config = Phase3Config()
        assert config.max_tasks_per_agent == 5

    def test_agent_scale_threshold_default(self):
        """agent_scale_thresholdのデフォルト値（0.8）を確認"""
        config = Phase3Config()
        assert config.agent_scale_threshold == 0.8

    def test_min_agent_instances_default(self):
        """min_agent_instancesのデフォルト値（1）を確認"""
        config = Phase3Config()
        assert config.min_agent_instances == 1

    def test_max_agent_instances_default(self):
        """max_agent_instancesのデフォルト値（10）を確認"""
        config = Phase3Config()
        assert config.max_agent_instances == 10


class TestMultiOrchestratorSettings:
    """複数オーケストレーター設定のテスト"""

    def test_multi_orchestrator_enabled_default(self):
        """multi_orchestrator_enabledのデフォルト値（False）を確認"""
        config = Phase3Config()
        assert config.multi_orchestrator_enabled is False

    def test_orchestrator_heartbeat_interval_default(self):
        """orchestrator_heartbeat_intervalのデフォルト値（30秒）を確認"""
        config = Phase3Config()
        assert config.orchestrator_heartbeat_interval == 30

    def test_orchestrator_failover_timeout_default(self):
        """orchestrator_failover_timeoutのデフォルト値（90秒）を確認"""
        config = Phase3Config()
        assert config.orchestrator_failover_timeout == 90

    def test_session_lock_timeout_default(self):
        """session_lock_timeoutのデフォルト値（300秒）を確認"""
        config = Phase3Config()
        assert config.session_lock_timeout == 300


class TestWebSocketSettings:
    """WebSocket設定のテスト"""

    def test_websocket_enabled_default(self):
        """websocket_enabledのデフォルト値（True）を確認"""
        config = Phase3Config()
        assert config.websocket_enabled is True

    def test_websocket_ping_interval_default(self):
        """websocket_ping_intervalのデフォルト値（30秒）を確認"""
        config = Phase3Config()
        assert config.websocket_ping_interval == 30

    def test_websocket_max_connections_default(self):
        """websocket_max_connectionsのデフォルト値（100）を確認"""
        config = Phase3Config()
        assert config.websocket_max_connections == 100


class TestMetricsSettings:
    """メトリクス設定のテスト"""

    def test_metrics_enabled_default(self):
        """metrics_enabledのデフォルト値（True）を確認"""
        config = Phase3Config()
        assert config.metrics_enabled is True

    def test_metrics_port_default(self):
        """metrics_portのデフォルト値（9090）を確認"""
        config = Phase3Config()
        assert config.metrics_port == 9090

    def test_metrics_collection_interval_default(self):
        """metrics_collection_intervalのデフォルト値（15秒）を確認"""
        config = Phase3Config()
        assert config.metrics_collection_interval == 15


class TestABTestingSettings:
    """A/Bテスト設定のテスト"""

    def test_ab_testing_enabled_default(self):
        """ab_testing_enabledのデフォルト値（False）を確認"""
        config = Phase3Config()
        assert config.ab_testing_enabled is False

    def test_default_experiment_duration_days_default(self):
        """default_experiment_duration_daysのデフォルト値（14日）を確認"""
        config = Phase3Config()
        assert config.default_experiment_duration_days == 14

    def test_min_samples_per_variant_default(self):
        """min_samples_per_variantのデフォルト値（100）を確認"""
        config = Phase3Config()
        assert config.min_samples_per_variant == 100

    def test_significance_threshold_default(self):
        """significance_thresholdのデフォルト値（0.95）を確認"""
        config = Phase3Config()
        assert config.significance_threshold == 0.95


class TestDefaultInstance:
    """デフォルトインスタンスのテスト"""

    def test_phase3_config_instance_exists(self):
        """phase3_configインスタンスが存在することを確認"""
        assert phase3_config is not None

    def test_phase3_config_is_phase3_config(self):
        """phase3_configがPhase3Configインスタンスであることを確認"""
        assert isinstance(phase3_config, Phase3Config)

    def test_phase3_config_is_singleton_like(self):
        """モジュールレベルのインスタンスとして共有されることを確認"""
        from src.config.phase3_config import phase3_config as config2
        assert phase3_config is config2


class TestNeuralScorerConstants:
    """ニューラルスコアラー定数のテスト"""

    def test_neural_scorer_config_keys(self):
        """NEURAL_SCORER_CONFIGに必要なキーが存在することを確認"""
        assert "input_dim" in NEURAL_SCORER_CONFIG
        assert "hidden_dims" in NEURAL_SCORER_CONFIG
        assert "output_dim" in NEURAL_SCORER_CONFIG
        assert "dropout" in NEURAL_SCORER_CONFIG
        assert "activation" in NEURAL_SCORER_CONFIG

    def test_neural_scorer_config_values(self):
        """NEURAL_SCORER_CONFIGの値を確認"""
        assert NEURAL_SCORER_CONFIG["input_dim"] == 1536 + 64
        assert NEURAL_SCORER_CONFIG["hidden_dims"] == [256, 128, 64]
        assert NEURAL_SCORER_CONFIG["output_dim"] == 1
        assert NEURAL_SCORER_CONFIG["dropout"] == 0.2
        assert NEURAL_SCORER_CONFIG["activation"] == "relu"

    def test_training_config_keys(self):
        """TRAINING_CONFIGに必要なキーが存在することを確認"""
        assert "batch_size" in TRAINING_CONFIG
        assert "learning_rate" in TRAINING_CONFIG
        assert "epochs" in TRAINING_CONFIG
        assert "early_stopping_patience" in TRAINING_CONFIG
        assert "validation_split" in TRAINING_CONFIG

    def test_task_features_list(self):
        """TASK_FEATURESが正しいリストであることを確認"""
        assert len(TASK_FEATURES) == 6
        assert "task_length" in TASK_FEATURES
        assert "complexity_score" in TASK_FEATURES

    def test_agent_features_list(self):
        """AGENT_FEATURESが正しいリストであることを確認"""
        assert len(AGENT_FEATURES) == 5
        assert "capability_count" in AGENT_FEATURES
        assert "past_success_rate" in AGENT_FEATURES


class TestLoadBalancerConstants:
    """負荷分散定数のテスト"""

    def test_load_balancer_algorithms_keys(self):
        """LOAD_BALANCER_ALGORITHMSに必要なキーが存在することを確認"""
        assert "round_robin" in LOAD_BALANCER_ALGORITHMS
        assert "weighted_round_robin" in LOAD_BALANCER_ALGORITHMS
        assert "least_connections" in LOAD_BALANCER_ALGORITHMS
        assert "adaptive" in LOAD_BALANCER_ALGORITHMS

    def test_scaling_config_keys(self):
        """SCALING_CONFIGに必要なキーが存在することを確認"""
        assert "scale_up_threshold" in SCALING_CONFIG
        assert "scale_down_threshold" in SCALING_CONFIG
        assert "scale_up_cooldown" in SCALING_CONFIG
        assert "scale_down_cooldown" in SCALING_CONFIG

    def test_health_check_config_keys(self):
        """HEALTH_CHECK_CONFIGに必要なキーが存在することを確認"""
        assert "interval" in HEALTH_CHECK_CONFIG
        assert "timeout" in HEALTH_CHECK_CONFIG
        assert "unhealthy_threshold" in HEALTH_CHECK_CONFIG
        assert "healthy_threshold" in HEALTH_CHECK_CONFIG


class TestWebSocketConstants:
    """WebSocket定数のテスト"""

    def test_websocket_event_types_keys(self):
        """WEBSOCKET_EVENT_TYPESに必要なキーが存在することを確認"""
        assert "progress_update" in WEBSOCKET_EVENT_TYPES
        assert "task_started" in WEBSOCKET_EVENT_TYPES
        assert "task_completed" in WEBSOCKET_EVENT_TYPES
        assert "task_failed" in WEBSOCKET_EVENT_TYPES
        assert "agent_assigned" in WEBSOCKET_EVENT_TYPES
        assert "alert" in WEBSOCKET_EVENT_TYPES

    def test_websocket_config_keys(self):
        """WEBSOCKET_CONFIGに必要なキーが存在することを確認"""
        assert "ping_interval" in WEBSOCKET_CONFIG
        assert "ping_timeout" in WEBSOCKET_CONFIG
        assert "max_message_size" in WEBSOCKET_CONFIG
        assert "max_connections_per_user" in WEBSOCKET_CONFIG


class TestCustomConfiguration:
    """カスタム設定のテスト"""

    def test_override_neural_scorer_settings(self):
        """ニューラルスコアラー設定をオーバーライドできることを確認"""
        config = Phase3Config(
            neural_scorer_enabled=True,
            min_training_samples=500,
        )
        assert config.neural_scorer_enabled is True
        assert config.min_training_samples == 500

    def test_override_task_queue_settings(self):
        """タスクキュー設定をオーバーライドできることを確認"""
        config = Phase3Config(
            redis_url="redis://custom-host:6380/1",
            max_queue_size=2000,
        )
        assert config.redis_url == "redis://custom-host:6380/1"
        assert config.max_queue_size == 2000

    def test_override_load_balancer_settings(self):
        """負荷分散設定をオーバーライドできることを確認"""
        config = Phase3Config(
            load_balancer_algorithm="adaptive",
            max_agent_instances=20,
        )
        assert config.load_balancer_algorithm == "adaptive"
        assert config.max_agent_instances == 20

    def test_override_inherited_settings(self):
        """継承元の設定もオーバーライドできることを確認"""
        config = Phase3Config(
            # Phase1の設定
            initial_strength=0.8,
            # Phase2の設定
            routing_method="similarity",
            # Phase3の設定
            neural_scorer_enabled=True,
        )
        assert config.initial_strength == 0.8
        assert config.routing_method == "similarity"
        assert config.neural_scorer_enabled is True
