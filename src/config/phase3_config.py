# Phase 3 MVP パラメータ設定
# 実装仕様: docs/phase3-implementation-spec.ja.md セクション4

from dataclasses import dataclass, field
from typing import Dict, List

from src.config.phase2_config import Phase2Config


@dataclass
class Phase3Config(Phase2Config):
    """Phase 3 MVP パラメータ設定（Phase 2 設定を継承）

    仕様書参照:
    - セクション4.1: Phase 3 設定クラス
    - セクション4.2: ニューラルスコアラーパラメータ
    - セクション4.3: 負荷分散パラメータ
    - セクション4.4: WebSocketパラメータ
    """

    # === ニューラルスコアラー ===
    neural_scorer_enabled: bool = False
    """学習データが十分になるまでFalse"""

    neural_scorer_model_path: str = "models/routing_scorer.pt"
    """ニューラルスコアラーモデルのパス"""

    min_training_samples: int = 1000
    """学習開始に必要な最小サンプル数"""

    neural_scorer_threshold: float = 0.7
    """ニューラルスコアラーを採用する閾値"""

    # === タスクキュー ===
    task_queue_enabled: bool = True
    """タスクキューを有効にするか"""

    redis_url: str = "redis://localhost:6379/0"
    """Redis接続URL"""

    max_queue_size: int = 1000
    """タスクキューの最大サイズ"""

    task_timeout_seconds: int = 600
    """タスクタイムアウト（10分）"""

    # === 負荷分散 ===
    load_balancer_algorithm: str = "weighted_round_robin"
    """負荷分散アルゴリズム: "round_robin" | "weighted_round_robin" | "least_connections" | "adaptive" """

    max_tasks_per_agent: int = 5
    """エージェントあたり最大同時タスク"""

    agent_scale_threshold: float = 0.8
    """スケールアウト閾値（80%負荷）"""

    min_agent_instances: int = 1
    """最小エージェントインスタンス数"""

    max_agent_instances: int = 10
    """最大エージェントインスタンス数"""

    # === 複数オーケストレーター ===
    multi_orchestrator_enabled: bool = False
    """単一オーケストレーターで検証後有効化"""

    orchestrator_heartbeat_interval: int = 30
    """ハートビート間隔（秒）"""

    orchestrator_failover_timeout: int = 90
    """フェイルオーバータイムアウト（秒）"""

    session_lock_timeout: int = 300
    """セッションロックタイムアウト（秒）"""

    # === WebSocket ===
    websocket_enabled: bool = True
    """WebSocketを有効にするか"""

    websocket_ping_interval: int = 30
    """WebSocket ping間隔（秒）"""

    websocket_max_connections: int = 100
    """WebSocket最大接続数"""

    # === メトリクス ===
    metrics_enabled: bool = True
    """メトリクス収集を有効にするか"""

    metrics_port: int = 9090
    """Prometheusメトリクスポート"""

    metrics_collection_interval: int = 15
    """メトリクス収集間隔（秒）"""

    # === A/Bテスト ===
    ab_testing_enabled: bool = False
    """安定稼働確認後に有効化"""

    default_experiment_duration_days: int = 14
    """デフォルト実験期間（日）"""

    min_samples_per_variant: int = 100
    """バリアントあたり最小サンプル数"""

    significance_threshold: float = 0.95
    """統計的有意性の閾値"""


# === ニューラルスコアラー モデルアーキテクチャ（セクション4.2）===
NEURAL_SCORER_CONFIG: Dict[str, any] = {
    "input_dim": 1536 + 64,          # エンベディング + 特徴量
    "hidden_dims": [256, 128, 64],   # 隠れ層の次元
    "output_dim": 1,                 # スコア（0-1）
    "dropout": 0.2,
    "activation": "relu",
}

# === ニューラルスコアラー 学習パラメータ ===
TRAINING_CONFIG: Dict[str, any] = {
    "batch_size": 64,
    "learning_rate": 0.001,
    "epochs": 100,
    "early_stopping_patience": 10,
    "validation_split": 0.2,
    "optimizer": "adam",
    "loss_function": "binary_cross_entropy",
}

# === ニューラルスコアラー 特徴量 ===
TASK_FEATURES: List[str] = [
    "task_length",                   # タスク文字数
    "item_count",                    # 論点数
    "has_code_keywords",             # コード関連キーワードの有無
    "has_research_keywords",         # 調査関連キーワードの有無
    "has_test_keywords",             # テスト関連キーワードの有無
    "complexity_score",              # 複雑度スコア
]

AGENT_FEATURES: List[str] = [
    "capability_count",              # 能力タグ数
    "perspective_count",             # 観点数
    "past_success_rate",             # 過去の成功率
    "recent_task_count",             # 最近のタスク数
    "avg_task_duration",             # 平均タスク処理時間
]

# === 負荷分散アルゴリズム（セクション4.3）===
LOAD_BALANCER_ALGORITHMS: Dict[str, str] = {
    "round_robin": "シンプルなラウンドロビン",
    "weighted_round_robin": "重み付きラウンドロビン（成功率考慮）",
    "least_connections": "最小接続数優先",
    "adaptive": "適応型（レスポンス時間考慮）",
}

# === スケーリング設定 ===
SCALING_CONFIG: Dict[str, any] = {
    "scale_up_threshold": 0.8,       # 80%負荷でスケールアップ
    "scale_down_threshold": 0.3,     # 30%負荷でスケールダウン
    "scale_up_cooldown": 300,        # スケールアップ後のクールダウン（秒）
    "scale_down_cooldown": 600,      # スケールダウン後のクールダウン（秒）
    "min_instances": 1,
    "max_instances": 10,
}

# === ヘルスチェック設定 ===
HEALTH_CHECK_CONFIG: Dict[str, any] = {
    "interval": 30,                  # ヘルスチェック間隔（秒）
    "timeout": 10,                   # ヘルスチェックタイムアウト（秒）
    "unhealthy_threshold": 3,        # 不健全判定の連続失敗回数
    "healthy_threshold": 2,          # 健全判定の連続成功回数
}

# === WebSocketイベントタイプ（セクション4.4）===
WEBSOCKET_EVENT_TYPES: Dict[str, str] = {
    "progress_update": "進捗更新",
    "task_started": "タスク開始",
    "task_completed": "タスク完了",
    "task_failed": "タスク失敗",
    "agent_assigned": "エージェント割り当て",
    "feedback_received": "フィードバック受信",
    "alert": "アラート",
}

# === WebSocket詳細設定 ===
WEBSOCKET_CONFIG: Dict[str, any] = {
    "ping_interval": 30,
    "ping_timeout": 10,
    "max_message_size": 65536,       # 64KB
    "max_connections_per_user": 5,
}


# デフォルト設定のインスタンス
phase3_config = Phase3Config()
