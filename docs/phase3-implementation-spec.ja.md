# Phase 3 実装仕様書: スケーラビリティと実運用

## 概要

本ドキュメントは、[LLMエージェントの永続的メモリアーキテクチャ設計](./architecture.ja.md)に基づくPhase 3 MVP（最小検証可能プロダクト）の実装仕様を定義する。

**検証目標**: スケーラビリティと実運用（大規模環境での安定動作と運用効率化）

---

## 1. MVPの範囲

### 1.1 フェーズ分割（再掲）

| フェーズ | 検証対象 | 実装範囲 |
|---------|---------|---------|
| Phase 1 | 強度管理と減衰が「個性」を生むか | 単一エージェント + 外部メモリ（**完了**） |
| Phase 2 | オーケストレーションが機能するか | 複数エージェント + ルーティング（**完了**） |
| **Phase 3** | **スケーラビリティと実運用** | **負荷分散 + 監視 + 最適化** |

### 1.2 Phase 3 で検証する核心機能

1. **学習可能なルーティングモデル**
   - ルールベース → ニューラルネットベースのルーティング
   - ルーティング履歴からの学習
   - 動的な重み調整

2. **高度な負荷分散**
   - タスクキュー方式の導入
   - 動的スケジューリング
   - エージェントインスタンスの水平スケーリング

3. **複数オーケストレーターの協調**
   - オーケストレーター間の状態共有
   - 分散セッション管理
   - コンフリクト解決

4. **リアルタイム進捗監視**
   - WebSocketによるリアルタイム通知
   - ダッシュボード用API
   - アラート機能

5. **A/Bテストによるパラメータ最適化**
   - パラメータのバリアント管理
   - 統計的有意性の評価
   - 自動パラメータ調整

### 1.3 Phase 3 で実装しないもの

- 完全自動のエージェント生成（教育プロセスで対応）
- クロスリージョン分散（単一クラスタで開始）
- 複数LLMプロバイダの同時使用（Anthropic API に固定）
- ファインチューニング（推論のみ）

---

## 2. 技術スタック

### 2.1 Phase 2 からの追加・変更

| コンポーネント | Phase 2 | Phase 3 追加 |
|---------------|---------|-------------|
| タスクキュー | - | Redis + Celery |
| WebSocket | - | FastAPI WebSocket |
| ニューラルスコアラー | - | PyTorch (軽量モデル) |
| 監視 | ログのみ | Prometheus + Grafana |
| キャッシュ | インメモリ | Redis |
| セッション管理 | インメモリ | Redis + DB |

### 2.2 ディレクトリ構成（追加分）

```
src/
├── routing/
│   ├── neural_scorer.py      # ニューラルネットスコアラー
│   ├── training_pipeline.py  # 学習パイプライン
│   └── feature_extractor.py  # 特徴量抽出
├── scheduling/
│   ├── task_queue.py         # タスクキュー管理
│   ├── scheduler.py          # 動的スケジューラー
│   └── load_balancer.py      # 負荷分散
├── coordination/
│   ├── multi_orchestrator.py # 複数オーケストレーター協調
│   ├── state_sync.py         # 状態同期
│   └── conflict_resolver.py  # コンフリクト解決
├── monitoring/
│   ├── websocket_server.py   # WebSocketサーバー
│   ├── metrics_collector.py  # メトリクス収集
│   └── alert_manager.py      # アラート管理
├── ab_testing/
│   ├── experiment_manager.py # 実験管理
│   ├── variant_selector.py   # バリアント選択
│   └── stats_analyzer.py     # 統計分析
└── config/
    └── phase3_config.py      # Phase 3 設定パラメータ
```

---

## 3. データスキーマ

### 3.1 追加テーブル: routing_training_data

ニューラルスコアラーの学習データ

```sql
CREATE TABLE routing_training_data (
    -- 識別子
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- 入力特徴量
    task_embedding vector(1536),         -- タスク概要のエンベディング
    task_features JSONB NOT NULL,        -- 抽出された特徴量
    agent_features JSONB NOT NULL,       -- エージェントの特徴量

    -- ルーティング情報
    selected_agent_id VARCHAR(64) NOT NULL,
    candidate_scores JSONB NOT NULL,     -- 候補エージェントとスコア

    -- 結果（ラベル）
    user_feedback VARCHAR(32),           -- positive / neutral / negative
    result_status VARCHAR(16),           -- success / failure
    actual_score FLOAT,                  -- 0.0-1.0 の正解スコア

    -- メタデータ
    created_at TIMESTAMP DEFAULT NOW(),
    used_for_training BOOLEAN DEFAULT FALSE
);

-- インデックス
CREATE INDEX idx_training_data_feedback ON routing_training_data(user_feedback);
CREATE INDEX idx_training_data_created ON routing_training_data(created_at);
CREATE INDEX idx_training_data_used ON routing_training_data(used_for_training);
```

### 3.2 追加テーブル: task_queue

タスクキュー

```sql
CREATE TABLE task_queue (
    -- 識別子
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL,

    -- タスク情報
    task_type VARCHAR(32) NOT NULL,      -- routing / execution / evaluation
    task_payload JSONB NOT NULL,
    priority INT DEFAULT 5,              -- 1-10（1が最優先）

    -- 状態
    status VARCHAR(16) DEFAULT 'pending',  -- pending / processing / completed / failed
    assigned_worker VARCHAR(64),           -- 処理中のワーカーID

    -- タイムスタンプ
    created_at TIMESTAMP DEFAULT NOW(),
    started_at TIMESTAMP,
    completed_at TIMESTAMP,

    -- リトライ
    retry_count INT DEFAULT 0,
    max_retries INT DEFAULT 3,
    error_message TEXT
);

-- インデックス
CREATE INDEX idx_task_queue_status ON task_queue(status);
CREATE INDEX idx_task_queue_priority ON task_queue(priority);
CREATE INDEX idx_task_queue_session ON task_queue(session_id);
```

### 3.3 追加テーブル: orchestrator_state

複数オーケストレーター間の状態共有

```sql
CREATE TABLE orchestrator_state (
    -- 識別子
    orchestrator_id VARCHAR(64) PRIMARY KEY,

    -- 状態
    status VARCHAR(16) DEFAULT 'active',   -- active / sleeping / terminated
    current_load INT DEFAULT 0,            -- 現在の負荷（タスク数）
    max_capacity INT DEFAULT 10,           -- 最大キャパシティ

    -- セッション管理
    active_sessions INT DEFAULT 0,
    session_ids UUID[] DEFAULT '{}',

    -- ヘルスチェック
    last_heartbeat TIMESTAMP DEFAULT NOW(),
    health_status VARCHAR(16) DEFAULT 'healthy',  -- healthy / degraded / unhealthy

    -- メタデータ
    instance_info JSONB,                   -- ホスト名、IP等
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- インデックス
CREATE INDEX idx_orchestrator_state_status ON orchestrator_state(status);
CREATE INDEX idx_orchestrator_state_heartbeat ON orchestrator_state(last_heartbeat);
```

### 3.4 追加テーブル: ab_experiments

A/Bテスト実験管理

```sql
CREATE TABLE ab_experiments (
    -- 識別子
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(128) NOT NULL,

    -- 実験設定
    parameter_name VARCHAR(64) NOT NULL,   -- 対象パラメータ名
    variants JSONB NOT NULL,               -- バリアント定義
    traffic_split JSONB NOT NULL,          -- トラフィック配分

    -- 状態
    status VARCHAR(16) DEFAULT 'draft',    -- draft / running / paused / completed

    -- 結果
    results JSONB,                         -- 集計結果
    winner_variant VARCHAR(64),            -- 勝者バリアント
    statistical_significance FLOAT,        -- 統計的有意性

    -- タイムスタンプ
    created_at TIMESTAMP DEFAULT NOW(),
    started_at TIMESTAMP,
    ended_at TIMESTAMP
);

-- 実験ログ
CREATE TABLE ab_experiment_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_id UUID NOT NULL REFERENCES ab_experiments(id),
    variant_id VARCHAR(64) NOT NULL,
    session_id UUID NOT NULL,

    -- メトリクス
    metric_name VARCHAR(64) NOT NULL,
    metric_value FLOAT NOT NULL,

    created_at TIMESTAMP DEFAULT NOW()
);

-- インデックス
CREATE INDEX idx_ab_experiments_status ON ab_experiments(status);
CREATE INDEX idx_ab_experiment_logs_experiment ON ab_experiment_logs(experiment_id);
CREATE INDEX idx_ab_experiment_logs_variant ON ab_experiment_logs(variant_id);
```

---

## 4. パラメータ初期値

### 4.1 Phase 3 設定クラス

```python
# config/phase3_config.py

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from src.config.phase2_config import Phase2Config


@dataclass
class Phase3Config(Phase2Config):
    """Phase 3 MVP パラメータ設定（Phase 2 設定を継承）"""

    # === ニューラルスコアラー ===
    neural_scorer_enabled: bool = False  # 学習データが十分になるまでFalse
    neural_scorer_model_path: str = "models/routing_scorer.pt"
    min_training_samples: int = 1000     # 学習開始に必要な最小サンプル数
    neural_scorer_threshold: float = 0.7  # ニューラルスコアラーを採用する閾値

    # === タスクキュー ===
    task_queue_enabled: bool = True
    redis_url: str = "redis://localhost:6379/0"
    max_queue_size: int = 1000
    task_timeout_seconds: int = 600      # タスクタイムアウト（10分）

    # === 負荷分散 ===
    load_balancer_algorithm: str = "weighted_round_robin"
    max_tasks_per_agent: int = 5         # エージェントあたり最大同時タスク
    agent_scale_threshold: float = 0.8   # スケールアウト閾値（80%負荷）
    min_agent_instances: int = 1
    max_agent_instances: int = 10

    # === 複数オーケストレーター ===
    multi_orchestrator_enabled: bool = False  # 単一オーケストレーターで検証後有効化
    orchestrator_heartbeat_interval: int = 30  # ハートビート間隔（秒）
    orchestrator_failover_timeout: int = 90    # フェイルオーバータイムアウト（秒）
    session_lock_timeout: int = 300            # セッションロックタイムアウト（秒）

    # === WebSocket ===
    websocket_enabled: bool = True
    websocket_ping_interval: int = 30
    websocket_max_connections: int = 100

    # === メトリクス ===
    metrics_enabled: bool = True
    metrics_port: int = 9090
    metrics_collection_interval: int = 15  # メトリクス収集間隔（秒）

    # === A/Bテスト ===
    ab_testing_enabled: bool = False  # 安定稼働確認後に有効化
    default_experiment_duration_days: int = 14
    min_samples_per_variant: int = 100
    significance_threshold: float = 0.95  # 統計的有意性の閾値


# デフォルト設定のインスタンス
phase3_config = Phase3Config()
```

### 4.2 ニューラルスコアラーパラメータ

```python
# === モデルアーキテクチャ ===
NEURAL_SCORER_CONFIG = {
    "input_dim": 1536 + 64,          # エンベディング + 特徴量
    "hidden_dims": [256, 128, 64],   # 隠れ層の次元
    "output_dim": 1,                 # スコア（0-1）
    "dropout": 0.2,
    "activation": "relu",
}

# === 学習パラメータ ===
TRAINING_CONFIG = {
    "batch_size": 64,
    "learning_rate": 0.001,
    "epochs": 100,
    "early_stopping_patience": 10,
    "validation_split": 0.2,
    "optimizer": "adam",
    "loss_function": "binary_cross_entropy",
}

# === 特徴量 ===
TASK_FEATURES = [
    "task_length",                   # タスク文字数
    "item_count",                    # 論点数
    "has_code_keywords",             # コード関連キーワードの有無
    "has_research_keywords",         # 調査関連キーワードの有無
    "has_test_keywords",             # テスト関連キーワードの有無
    "complexity_score",              # 複雑度スコア
]

AGENT_FEATURES = [
    "capability_count",              # 能力タグ数
    "perspective_count",             # 観点数
    "past_success_rate",             # 過去の成功率
    "recent_task_count",             # 最近のタスク数
    "avg_task_duration",             # 平均タスク処理時間
]
```

### 4.3 負荷分散パラメータ

```python
# === 負荷分散アルゴリズム ===
LOAD_BALANCER_ALGORITHMS = {
    "round_robin": "シンプルなラウンドロビン",
    "weighted_round_robin": "重み付きラウンドロビン（成功率考慮）",
    "least_connections": "最小接続数優先",
    "adaptive": "適応型（レスポンス時間考慮）",
}

# === スケーリング ===
SCALING_CONFIG = {
    "scale_up_threshold": 0.8,       # 80%負荷でスケールアップ
    "scale_down_threshold": 0.3,     # 30%負荷でスケールダウン
    "scale_up_cooldown": 300,        # スケールアップ後のクールダウン（秒）
    "scale_down_cooldown": 600,      # スケールダウン後のクールダウン（秒）
    "min_instances": 1,
    "max_instances": 10,
}

# === ヘルスチェック ===
HEALTH_CHECK_CONFIG = {
    "interval": 30,                  # ヘルスチェック間隔（秒）
    "timeout": 10,                   # ヘルスチェックタイムアウト（秒）
    "unhealthy_threshold": 3,        # 不健全判定の連続失敗回数
    "healthy_threshold": 2,          # 健全判定の連続成功回数
}
```

### 4.4 WebSocket パラメータ

```python
# === WebSocketイベントタイプ ===
WEBSOCKET_EVENT_TYPES = {
    "progress_update": "進捗更新",
    "task_started": "タスク開始",
    "task_completed": "タスク完了",
    "task_failed": "タスク失敗",
    "agent_assigned": "エージェント割り当て",
    "feedback_received": "フィードバック受信",
    "alert": "アラート",
}

# === WebSocket設定 ===
WEBSOCKET_CONFIG = {
    "ping_interval": 30,
    "ping_timeout": 10,
    "max_message_size": 65536,       # 64KB
    "max_connections_per_user": 5,
}
```

---

## 5. コンポーネント設計

### 5.1 ニューラルスコアラー（NeuralScorer）

#### 役割
ルーティング判断をニューラルネットで行い、ルールベースよりも高精度なエージェント選択を実現する。

#### モデルアーキテクチャ

```
入力層
├── タスクエンベディング (1536次元)
├── タスク特徴量 (6次元)
└── エージェント特徴量 (5次元)
    ↓
隠れ層1 (256ユニット, ReLU, Dropout 0.2)
    ↓
隠れ層2 (128ユニット, ReLU, Dropout 0.2)
    ↓
隠れ層3 (64ユニット, ReLU, Dropout 0.2)
    ↓
出力層 (1ユニット, Sigmoid)
    ↓
適性スコア (0.0-1.0)
```

#### クラス設計

```python
class NeuralScorer:
    """ニューラルネットベースのルーティングスコアラー"""

    def __init__(
        self,
        model_path: str,
        embedding_client: EmbeddingClient,
        config: Phase3Config,
    ):
        self.model = self._load_model(model_path)
        self.embedding_client = embedding_client
        self.config = config
        self.feature_extractor = FeatureExtractor()

    def score(
        self,
        task_summary: str,
        agent: AgentDefinition,
        past_experiences: List[Dict],
    ) -> float:
        """エージェントの適性スコアを計算"""

        # 1. タスクエンベディングを取得
        task_embedding = self.embedding_client.create_embedding(task_summary)

        # 2. 特徴量を抽出
        task_features = self.feature_extractor.extract_task_features(task_summary)
        agent_features = self.feature_extractor.extract_agent_features(
            agent, past_experiences
        )

        # 3. 入力テンソルを構築
        input_tensor = self._build_input_tensor(
            task_embedding, task_features, agent_features
        )

        # 4. 推論
        with torch.no_grad():
            score = self.model(input_tensor).item()

        return score

    def train(self, training_data: List[TrainingExample]) -> TrainingResult:
        """モデルを学習"""
        # バッチ処理、検証、早期終了を含む学習パイプライン
        pass
```

#### 学習パイプライン

```
ルーティング履歴（routing_history）
    ↓
ラベル付けデータ生成
├── user_feedback == "positive" → label = 1.0
├── user_feedback == "negative" → label = 0.0
└── result_status == "success" → label に 0.5 加算
    ↓
特徴量抽出
├── タスクエンベディング
├── タスク特徴量
└── エージェント特徴量
    ↓
学習データセット（routing_training_data）
    ↓
モデル学習（バッチ処理）
    ↓
評価（検証データセット）
    ↓
モデル保存
```

### 5.2 タスクキュー（TaskQueue）

#### 役割
タスクを優先度付きキューで管理し、非同期でエージェントに配分する。

#### 処理フロー

```
タスク登録
    ↓
┌─────────────────────────────────────┐
│  優先度付きキュー（Redis Sorted Set）│
│  ├── priority: 1-10                 │
│  ├── created_at: タイムスタンプ      │
│  └── task_payload: JSON             │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  ワーカープール                       │
│  ├── ワーカー1 → タスク取得 → 実行   │
│  ├── ワーカー2 → タスク取得 → 実行   │
│  └── ワーカーN → タスク取得 → 実行   │
└─────────────────────────────────────┘
    ↓
結果を返す（コールバック or WebSocket通知）
```

#### クラス設計

```python
class TaskQueue:
    """タスクキュー管理"""

    def __init__(self, redis_client: Redis, config: Phase3Config):
        self.redis = redis_client
        self.config = config

    async def enqueue(
        self,
        task_type: str,
        task_payload: Dict,
        priority: int = 5,
        session_id: Optional[UUID] = None,
    ) -> UUID:
        """タスクをキューに追加"""
        task_id = uuid4()

        task = {
            "id": str(task_id),
            "type": task_type,
            "payload": task_payload,
            "session_id": str(session_id) if session_id else None,
            "created_at": datetime.now().isoformat(),
        }

        # 優先度スコア（小さいほど優先）
        score = priority * 1e10 + time.time()

        await self.redis.zadd("task_queue", {json.dumps(task): score})

        return task_id

    async def dequeue(self, worker_id: str) -> Optional[Dict]:
        """タスクをキューから取得（アトミック操作）"""
        # ZPOPMIN でアトミックに取得
        result = await self.redis.zpopmin("task_queue", count=1)

        if not result:
            return None

        task_json, _ = result[0]
        task = json.loads(task_json)

        # 処理中としてマーク
        await self._mark_processing(task["id"], worker_id)

        return task

    async def complete(self, task_id: UUID, result: Dict) -> None:
        """タスク完了を記録"""
        pass

    async def fail(self, task_id: UUID, error: str) -> bool:
        """タスク失敗を記録、リトライ判定"""
        pass
```

### 5.3 負荷分散（LoadBalancer）

#### 役割
複数のエージェントインスタンスに対してタスクを効率的に分散する。

#### アルゴリズム

```python
class LoadBalancer:
    """負荷分散"""

    def __init__(
        self,
        agent_registry: AgentRegistry,
        config: Phase3Config,
    ):
        self.agent_registry = agent_registry
        self.config = config
        self._agent_loads: Dict[str, int] = {}
        self._agent_response_times: Dict[str, List[float]] = {}

    def select_instance(
        self,
        agent_id: str,
        algorithm: Optional[str] = None,
    ) -> Optional[str]:
        """エージェントインスタンスを選択"""

        algorithm = algorithm or self.config.load_balancer_algorithm
        instances = self._get_healthy_instances(agent_id)

        if not instances:
            return None

        if algorithm == "round_robin":
            return self._round_robin(instances)

        elif algorithm == "weighted_round_robin":
            return self._weighted_round_robin(instances)

        elif algorithm == "least_connections":
            return self._least_connections(instances)

        elif algorithm == "adaptive":
            return self._adaptive(instances)

        return instances[0]

    def _weighted_round_robin(self, instances: List[str]) -> str:
        """重み付きラウンドロビン"""
        weights = []
        for instance in instances:
            # 成功率を重みとして使用
            success_rate = self._get_success_rate(instance)
            weights.append(success_rate)

        # 重み付き選択
        return random.choices(instances, weights=weights)[0]

    def _least_connections(self, instances: List[str]) -> str:
        """最小接続数優先"""
        loads = [(i, self._agent_loads.get(i, 0)) for i in instances]
        return min(loads, key=lambda x: x[1])[0]

    def _adaptive(self, instances: List[str]) -> str:
        """適応型（レスポンス時間考慮）"""
        scores = []
        for instance in instances:
            load = self._agent_loads.get(instance, 0)
            avg_response = self._get_avg_response_time(instance)
            # スコア = 負荷 × 平均レスポンス時間（低いほど良い）
            score = load * avg_response
            scores.append((instance, score))

        return min(scores, key=lambda x: x[1])[0]
```

### 5.4 複数オーケストレーター協調（MultiOrchestratorCoordinator）

#### 役割
複数のオーケストレーターインスタンスが協調して動作できるようにする。

#### 状態同期フロー

```
オーケストレーター1                オーケストレーター2
       │                                 │
       ├── セッション開始 ─────────────────┤
       │         │                       │
       │   Redis Lock 取得               │
       │   session_state に書き込み       │
       │   Lock 解放                      │
       │         │                       │
       │   ← 変更通知（Pub/Sub）→          │
       │         │                       │
       ├── セッション参照 ─────────────────┤
       │   session_state から読み取り      │
       │         │                       │
       └─────────┴───────────────────────┘
```

#### クラス設計

```python
class MultiOrchestratorCoordinator:
    """複数オーケストレーター協調"""

    def __init__(
        self,
        orchestrator_id: str,
        redis_client: Redis,
        db_connection: DatabaseConnection,
        config: Phase3Config,
    ):
        self.orchestrator_id = orchestrator_id
        self.redis = redis_client
        self.db = db_connection
        self.config = config

    async def acquire_session_lock(
        self,
        session_id: UUID,
        timeout: Optional[int] = None,
    ) -> bool:
        """セッションのロックを取得"""
        timeout = timeout or self.config.session_lock_timeout
        lock_key = f"session_lock:{session_id}"

        # Redisでアトミックにロック取得
        acquired = await self.redis.set(
            lock_key,
            self.orchestrator_id,
            nx=True,  # 存在しない場合のみ
            ex=timeout,  # TTL
        )

        return acquired

    async def release_session_lock(self, session_id: UUID) -> None:
        """セッションのロックを解放"""
        lock_key = f"session_lock:{session_id}"

        # 自分が持っているロックのみ解放
        current_holder = await self.redis.get(lock_key)
        if current_holder == self.orchestrator_id:
            await self.redis.delete(lock_key)

    async def send_heartbeat(self) -> None:
        """ハートビートを送信"""
        await self.db.execute(
            """
            UPDATE orchestrator_state
            SET last_heartbeat = NOW(),
                current_load = $2,
                active_sessions = $3
            WHERE orchestrator_id = $1
            """,
            self.orchestrator_id,
            self._get_current_load(),
            self._get_active_session_count(),
        )

    async def detect_failed_orchestrators(self) -> List[str]:
        """失敗したオーケストレーターを検出"""
        timeout = self.config.orchestrator_failover_timeout

        failed = await self.db.fetch_all(
            """
            SELECT orchestrator_id
            FROM orchestrator_state
            WHERE status = 'active'
              AND last_heartbeat < NOW() - INTERVAL '$1 seconds'
            """,
            timeout,
        )

        return [r["orchestrator_id"] for r in failed]

    async def takeover_sessions(self, failed_orchestrator_id: str) -> int:
        """失敗したオーケストレーターのセッションを引き継ぐ"""
        # セッションを自分に割り当て
        pass
```

### 5.5 WebSocket サーバー（WebSocketServer）

#### 役割
リアルタイムで進捗やイベントをクライアントに通知する。

#### イベントフロー

```
クライアント                        サーバー
    │                                 │
    ├── WebSocket 接続 ────────────────┤
    │                                 │
    ├── セッション購読 ─────────────────┤
    │   {"action": "subscribe",       │
    │    "session_id": "xxx"}         │
    │                                 │
    │   ← イベント通知 ────────────────┤
    │   {"event": "task_started",     │
    │    "session_id": "xxx",         │
    │    "data": {...}}               │
    │                                 │
    │   ← 進捗更新 ────────────────────┤
    │   {"event": "progress_update",  │
    │    "session_id": "xxx",         │
    │    "progress": 45}              │
    │                                 │
    └─────────────────────────────────┘
```

#### クラス設計

```python
class WebSocketServer:
    """WebSocketサーバー"""

    def __init__(self, config: Phase3Config):
        self.config = config
        self._connections: Dict[str, Set[WebSocket]] = {}  # session_id -> connections
        self._user_connections: Dict[str, Set[WebSocket]] = {}  # user_id -> connections

    async def handle_connection(
        self,
        websocket: WebSocket,
        user_id: str,
    ):
        """WebSocket接続を処理"""
        await websocket.accept()

        # ユーザー接続を記録
        if user_id not in self._user_connections:
            self._user_connections[user_id] = set()
        self._user_connections[user_id].add(websocket)

        try:
            while True:
                message = await websocket.receive_json()
                await self._handle_message(websocket, user_id, message)

        except WebSocketDisconnect:
            self._cleanup_connection(websocket, user_id)

    async def _handle_message(
        self,
        websocket: WebSocket,
        user_id: str,
        message: Dict,
    ):
        """メッセージを処理"""
        action = message.get("action")

        if action == "subscribe":
            session_id = message.get("session_id")
            await self._subscribe(websocket, session_id)

        elif action == "unsubscribe":
            session_id = message.get("session_id")
            await self._unsubscribe(websocket, session_id)

    async def broadcast(
        self,
        session_id: str,
        event_type: str,
        data: Dict,
    ):
        """セッション購読者にイベントをブロードキャスト"""
        message = {
            "event": event_type,
            "session_id": session_id,
            "data": data,
            "timestamp": datetime.now().isoformat(),
        }

        connections = self._connections.get(session_id, set())

        for ws in connections:
            try:
                await ws.send_json(message)
            except Exception:
                # 接続エラーは無視（クリーンアップで処理）
                pass
```

### 5.6 A/Bテスト管理（ExperimentManager）

#### 役割
パラメータの最適化のためのA/Bテストを実行・評価する。

#### 実験フロー

```
実験定義
├── parameter: "similarity_threshold"
├── variants:
│   ├── control: 0.3
│   ├── variant_a: 0.25
│   └── variant_b: 0.35
└── traffic_split: [0.34, 0.33, 0.33]
    ↓
実験開始
    ↓
┌─────────────────────────────────────┐
│  セッションごとにバリアントを割り当て  │
│  ├── session_1 → control            │
│  ├── session_2 → variant_a          │
│  └── session_3 → variant_b          │
└─────────────────────────────────────┘
    ↓
メトリクス収集
├── ルーティング成功率
├── やり直し率
└── タスク完了時間
    ↓
統計分析
├── t検定 / Mann-Whitney U検定
└── 有意水準: 0.05
    ↓
結果判定
├── 有意差あり → 勝者バリアントを採用
└── 有意差なし → 実験延長 or 終了
```

#### クラス設計

```python
class ExperimentManager:
    """A/Bテスト実験管理"""

    def __init__(
        self,
        db_connection: DatabaseConnection,
        config: Phase3Config,
    ):
        self.db = db_connection
        self.config = config

    async def create_experiment(
        self,
        name: str,
        parameter_name: str,
        variants: Dict[str, Any],
        traffic_split: Optional[List[float]] = None,
    ) -> UUID:
        """実験を作成"""
        # デフォルトは均等分割
        if traffic_split is None:
            n = len(variants)
            traffic_split = [1.0 / n] * n

        experiment_id = uuid4()

        await self.db.execute(
            """
            INSERT INTO ab_experiments
            (id, name, parameter_name, variants, traffic_split, status)
            VALUES ($1, $2, $3, $4, $5, 'draft')
            """,
            experiment_id,
            name,
            parameter_name,
            json.dumps(variants),
            json.dumps(traffic_split),
        )

        return experiment_id

    async def get_variant(
        self,
        experiment_id: UUID,
        session_id: UUID,
    ) -> Tuple[str, Any]:
        """セッションに割り当てるバリアントを取得"""
        experiment = await self._get_experiment(experiment_id)

        if experiment["status"] != "running":
            # 実験が実行中でない場合はコントロール
            return "control", experiment["variants"]["control"]

        # 決定論的にバリアントを選択（session_idベース）
        variant_id = self._select_variant(
            session_id,
            experiment["variants"],
            experiment["traffic_split"],
        )

        return variant_id, experiment["variants"][variant_id]

    async def record_metric(
        self,
        experiment_id: UUID,
        session_id: UUID,
        variant_id: str,
        metric_name: str,
        metric_value: float,
    ) -> None:
        """メトリクスを記録"""
        await self.db.execute(
            """
            INSERT INTO ab_experiment_logs
            (experiment_id, variant_id, session_id, metric_name, metric_value)
            VALUES ($1, $2, $3, $4, $5)
            """,
            experiment_id,
            variant_id,
            session_id,
            metric_name,
            metric_value,
        )

    async def analyze_results(
        self,
        experiment_id: UUID,
    ) -> ExperimentResult:
        """実験結果を分析"""
        # 各バリアントのメトリクスを集計
        # 統計的有意性を計算
        # 勝者を判定
        pass
```

---

## 6. 実装順序

### 6.1 Phase 3 実装順序

安定性を重視し、以下の順序で実装する。

| 優先度 | コンポーネント | 依存 | 検証ポイント |
|--------|---------------|------|-------------|
| 1 | Phase3Config | Phase2Config | 設定パラメータの追加 |
| 2 | メトリクス収集 | - | 基本的な観測基盤 |
| 3 | タスクキュー | Redis | 非同期タスク処理 |
| 4 | 負荷分散 | タスクキュー | タスク分散 |
| 5 | WebSocketサーバー | - | リアルタイム通知 |
| 6 | 学習データ収集 | routing_history | ニューラルスコアラー用データ |
| 7 | ニューラルスコアラー | 学習データ | ルーティング精度向上 |
| 8 | オーケストレーター状態共有 | Redis | 複数インスタンス対応 |
| 9 | 複数オーケストレーター協調 | 状態共有 | フェイルオーバー |
| 10 | A/Bテスト基盤 | メトリクス | パラメータ最適化 |
| 11 | 統合テスト | 全コンポーネント | E2Eフロー検証 |

### 6.2 マイルストーン

| マイルストーン | 達成条件 |
|---------------|---------|
| M1: 基盤完了 | Phase3Config, メトリクス収集が動作 |
| M2: 非同期処理完了 | タスクキュー、負荷分散が動作 |
| M3: リアルタイム通知完了 | WebSocketで進捗通知が可能 |
| M4: 学習基盤完了 | 学習データ収集、ニューラルスコアラーが動作 |
| M5: 協調動作完了 | 複数オーケストレーターが協調して動作 |
| M6: 最適化基盤完了 | A/Bテストでパラメータ最適化が可能 |
| M7: 統合完了 | E2Eテストが成功 |

---

## 7. 観測指標

### 7.1 スケーラビリティ指標

| 指標 | 計算方法 | 正常範囲 |
|------|---------|---------|
| タスクスループット | tasks_completed / time_window | 目標値以上 |
| 平均待ち時間 | avg(started_at - created_at) | < 30秒 |
| キュー長 | current_queue_size | < max_queue_size * 0.8 |
| エージェント利用率 | active_tasks / max_capacity | 0.5-0.8 |
| スケールイベント数 | scale_up_count + scale_down_count | 適度 |

### 7.2 可用性指標

| 指標 | 計算方法 | 正常範囲 |
|------|---------|---------|
| オーケストレーター可用性 | healthy_orchestrators / total | > 0.99 |
| フェイルオーバー時間 | recovery_time | < 90秒 |
| セッション継続率 | sessions_recovered / sessions_affected | > 0.95 |
| WebSocket接続維持率 | active_connections / peak_connections | > 0.9 |

### 7.3 ニューラルスコアラー指標

| 指標 | 計算方法 | 正常範囲 |
|------|---------|---------|
| ルーティング精度 | correct_routing / total_routing | > 0.75 |
| ニューラル vs ルールベース | neural_success_rate / rule_success_rate | > 1.0 |
| 推論レイテンシ | avg(inference_time) | < 50ms |
| モデル更新頻度 | model_updates / time_window | 週1-2回 |

### 7.4 A/Bテスト指標

| 指標 | 計算方法 | 目標 |
|------|---------|------|
| 実験あたりサンプル数 | samples / experiment | > min_samples |
| 統計的有意性達成率 | significant_experiments / total | > 0.5 |
| パラメータ改善率 | improved_params / total_tested | > 0.3 |

---

## 8. Phase 2 モジュールとの連携

### 8.1 再利用するコンポーネント

| Phase 2 コンポーネント | Phase 3 での使用方法 |
|----------------------|---------------------|
| Orchestrator | 基本フローは維持、拡張ポイントを追加 |
| Router | ニューラルスコアラーとのハイブリッド使用 |
| Evaluator | 学習データ収集のフック追加 |
| ProgressManager | WebSocket通知の統合 |
| AgentRegistry | 負荷分散との連携 |

### 8.2 拡張ポイント

```python
# Router の拡張: ニューラルスコアラーとのハイブリッド
class HybridRouter(Router):
    """ハイブリッドルーター（ルールベース + ニューラル）"""

    def __init__(
        self,
        agent_registry: AgentRegistry,
        neural_scorer: Optional[NeuralScorer] = None,
        config: Phase3Config = None,
    ):
        super().__init__(agent_registry)
        self.neural_scorer = neural_scorer
        self.config = config or Phase3Config()

    def decide(
        self,
        task_summary: str,
        items: Optional[List[str]] = None,
        past_experiences: Optional[List[Dict]] = None,
    ) -> RoutingDecision:
        """ルーティング判断（ハイブリッド）"""

        # ルールベーススコアを計算
        rule_decision = super().decide(task_summary, items, past_experiences)

        # ニューラルスコアラーが有効で、十分な学習データがある場合
        if (
            self.config.neural_scorer_enabled
            and self.neural_scorer
            and self.neural_scorer.is_ready()
        ):
            neural_decision = self._neural_decide(
                task_summary, items, past_experiences
            )

            # 確信度が高い方を採用
            if neural_decision.confidence > rule_decision.confidence:
                return neural_decision

        return rule_decision
```

---

## 9. テスト戦略

### 9.1 単体テスト

| コンポーネント | テスト内容 |
|---------------|-----------|
| NeuralScorer | モデル推論の精度、特徴量抽出 |
| TaskQueue | エンキュー/デキューの整合性 |
| LoadBalancer | アルゴリズム別の分散均等性 |
| MultiOrchestratorCoordinator | ロック取得/解放、ハートビート |
| WebSocketServer | 接続管理、ブロードキャスト |
| ExperimentManager | バリアント割り当て、統計分析 |

### 9.2 統合テスト

| シナリオ | 検証内容 |
|---------|---------|
| 高負荷時のスケーリング | タスク増加時のスケールアウト動作 |
| オーケストレーターフェイルオーバー | 障害時のセッション引き継ぎ |
| ニューラルスコアラー学習 | データ収集から推論までの一連の流れ |
| リアルタイム通知 | WebSocketでの進捗更新 |
| A/Bテスト実行 | 実験作成から結果分析まで |

### 9.3 負荷テスト

```python
# 負荷テストシナリオ
LOAD_TEST_SCENARIOS = {
    "baseline": {
        "concurrent_sessions": 10,
        "tasks_per_second": 1,
        "duration_minutes": 10,
    },
    "moderate": {
        "concurrent_sessions": 50,
        "tasks_per_second": 5,
        "duration_minutes": 30,
    },
    "stress": {
        "concurrent_sessions": 100,
        "tasks_per_second": 10,
        "duration_minutes": 60,
    },
    "spike": {
        "concurrent_sessions": 200,
        "tasks_per_second": 20,
        "duration_minutes": 5,
    },
}
```

---

## 10. 次のステップ

### 10.1 Phase 3 完了後の展望

Phase 3 完了後、システムは以下の状態になる：

1. **スケーラブル**: 負荷に応じて自動スケーリング
2. **高可用性**: フェイルオーバーによる継続稼働
3. **最適化可能**: A/Bテストによる継続的改善
4. **監視可能**: リアルタイムメトリクスとアラート

### 10.2 将来の拡張候補（Phase 4以降）

- クロスリージョン分散（グローバル展開）
- 複数LLMプロバイダのサポート（OpenAI, Gemini等）
- エージェントの自動生成（メタエージェント）
- ファインチューニングによる専門性強化
- マルチテナント対応
- コンプライアンス・監査機能

### 10.3 Phase 2 からの移行条件

Phase 3 の実装を開始する前に、以下の条件を確認：

- [ ] Phase 2 の核心機能が安定動作（E2Eテスト成功）
- [ ] ルーティング正解率が 70% 以上
- [ ] やり直し率が 15% 以下
- [ ] 中間睡眠からの復帰が正常動作
- [ ] ルーティング履歴が 1000件以上蓄積（ニューラルスコアラー学習用）

---

*本ドキュメントは [architecture.ja.md](./architecture.ja.md)、[phase1-implementation-spec.ja.md](./phase1-implementation-spec.ja.md)、および [phase2-implementation-spec.ja.md](./phase2-implementation-spec.ja.md) に基づいて作成された Phase 3 MVP の実装仕様書である。*

*作成日: 2026年1月15日*
