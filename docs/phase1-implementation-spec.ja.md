# Phase 1 実装仕様書: 永続的メモリアーキテクチャ MVP

## 概要

本ドキュメントは、[LLMエージェントの永続的メモリアーキテクチャ設計](./architecture.ja.md)に基づくPhase 1 MVP（最小検証可能プロダクト）の実装仕様を定義する。

**検証目標**: 強度管理と減衰が「個性」を生むかを検証する

---

## 1. MVPの範囲

### 1.1 フェーズ分割

| フェーズ | 検証対象 | 実装範囲 |
|---------|---------|---------|
| **Phase 1** | 強度管理と減衰が「個性」を生むか | 単一エージェント + 外部メモリ |
| Phase 2 | オーケストレーションが機能するか | 複数エージェント + ルーティング |

### 1.2 Phase 1 で検証する核心機能

1. **2段階強化**
   - 検索候補になった（candidate_count++）と実際に使用された（access_count++, strength++）の分離
   - ノイズの強化を防ぎ、本当に有用な情報のみを残す

2. **観点別強度**
   - 同じ情報でも観点によって異なる重要度を持つ
   - `strength_by_perspective` による観点ごとの強度管理

3. **睡眠フェーズ**
   - タスク完了時の減衰処理
   - 定着レベルに応じた減衰率の差別化
   - アーカイブ判定

### 1.3 Phase 1 で実装しないもの

- オーケストレーター（Phase 2）
- 複数エージェント間のルーティング（Phase 2）
- 入力処理層（Phase 2）
- 学習可能なニューラルスコアラー（線形スコアで開始）
- 重みベースの容量管理（件数制限で開始）
- エージェント定義のDB化（コード内定義で開始）

---

## 2. 技術スタック

| コンポーネント | 技術選定 | 備考 |
|---------------|---------|------|
| ベクトルDB + メタデータ | Azure Database for PostgreSQL + pgvector | 単一DBでベクトルとメタデータを管理 |
| エンベディング | text-embedding-3-small (1536次元) | コスト効率と性能のバランス |
| LLM | Claude Sonnet 4 | タスク実行・学び抽出に使用 |
| 実行環境 | ローカル (Docker Compose) | 開発・検証用 |
| 言語 | Python (素のSDK) | anthropic SDK, psycopg2, openai (embedding用) |

### 2.1 Docker Compose 構成

```yaml
services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: agent_memory
      POSTGRES_USER: agent
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  app:
    build: .
    environment:
      DATABASE_URL: postgresql://agent:${POSTGRES_PASSWORD}@postgres:5432/agent_memory
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY}
      OPENAI_API_KEY: ${OPENAI_API_KEY}
    depends_on:
      - postgres
```

---

## 3. データスキーマ

### 3.1 メインテーブル: agent_memory

```sql
-- PostgreSQL + pgvector
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE agent_memory (
    -- 識別子
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id VARCHAR(64) NOT NULL,

    -- コンテンツ
    content TEXT NOT NULL,
    embedding vector(1536),
    tags TEXT[] DEFAULT '{}',

    -- スコープ（知識の階層管理）
    scope_level VARCHAR(16) DEFAULT 'project',  -- universal / domain / project
    scope_domain VARCHAR(64),                    -- ドメイン名（domain レベルの場合）
    scope_project VARCHAR(64),                   -- プロジェクトID（project レベルの場合）

    -- 強度管理
    strength FLOAT DEFAULT 1.0,
    strength_by_perspective JSONB,  -- {"コスト": 1.2, "納期": 0.8, ...}

    -- 使用追跡
    access_count INT DEFAULT 0,
    candidate_count INT DEFAULT 0,
    last_accessed_at TIMESTAMP,

    -- インパクト
    impact_score FLOAT DEFAULT 0.0,

    -- 定着管理
    consolidation_level INT DEFAULT 0,  -- 0-5

    -- 学び（観点別）
    learnings JSONB,  -- {"コスト": "...", "納期": "...", ...}

    -- 状態
    status VARCHAR(16) DEFAULT 'active',  -- active / archived
    source VARCHAR(32),  -- education / task / manual

    -- タイムスタンプ
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    last_decay_at TIMESTAMP  -- 睡眠フェーズの処理追跡
);

-- インデックス
CREATE INDEX idx_agent_memory_agent_id ON agent_memory(agent_id);
CREATE INDEX idx_agent_memory_status ON agent_memory(status);
CREATE INDEX idx_agent_memory_tags ON agent_memory USING GIN(tags);
CREATE INDEX idx_agent_memory_strength ON agent_memory(strength);
CREATE INDEX idx_agent_memory_scope ON agent_memory(scope_level, scope_domain, scope_project);

-- Phase 1: ベクトルインデックスなし（1万件未満想定）
-- 1万件超えたら以下を追加:
-- CREATE INDEX idx_agent_memory_embedding ON agent_memory
--     USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

### 3.2 観点別データの保存（JSONB非正規化）

```sql
-- strength_by_perspective の例
'{
    "コスト": 1.2,
    "納期": 0.8,
    "サプライヤー": 1.5,
    "品質": 0.3,
    "代替": 1.2
}'::JSONB

-- learnings の例
'{
    "コスト": "緊急調達で15%コスト増",
    "納期": "2週間バッファが必要",
    "サプライヤー": "サプライヤーYは単一拠点リスクあり"
}'::JSONB
```

### 3.3 スコープによる知識の階層管理

エージェントをプロジェクトをまたいで「育てていく」ために、知識のスコープを管理する。

**3つのスコープレベル**：

| レベル | scope_level | scope_domain | scope_project | 用途 |
|--------|-------------|--------------|---------------|------|
| 汎用 | `universal` | NULL | NULL | どのプロジェクトでも有効な原則 |
| ドメイン | `domain` | 値あり | NULL | 特定技術領域の知識 |
| プロジェクト | `project` | NULL | 値あり | プロジェクト固有の決定事項 |

**スコープ設定の例**：

```sql
-- 汎用知識
INSERT INTO agent_memory (agent_id, content, scope_level)
VALUES ('agent_01', 'トランザクション整合性を常に確保する', 'universal');

-- ドメイン知識
INSERT INTO agent_memory (agent_id, content, scope_level, scope_domain)
VALUES ('agent_01', 'pgvectorのIVFflatは1万件超で有効', 'domain', 'vector-database');

-- プロジェクト固有
INSERT INTO agent_memory (agent_id, content, scope_level, scope_project)
VALUES ('agent_01', 'similarity_threshold=0.3が適切', 'project', 'llm-persistent-memory-phase1');
```

**検索時のスコープフィルタリング**：

```sql
-- 現在のプロジェクトで有効な知識を検索
SELECT * FROM agent_memory
WHERE agent_id = 'agent_01'
  AND status = 'active'
  AND (
    scope_level = 'universal'
    OR (scope_level = 'domain' AND scope_domain = ANY(ARRAY['vector-database', 'postgresql']))
    OR (scope_level = 'project' AND scope_project = 'llm-persistent-memory-phase1')
  )
ORDER BY strength DESC;
```

**Phase 1 での運用**：

Phase 1 は単一プロジェクトのため、scope_level は主に `project` を使用。
ただし、将来の拡張を見据えてスキーマに含めておく。

### 3.4 エージェント定義（Phase 1: コード内定義）

```python
# config/agents.py

AGENTS = {
    "procurement_agent_01": {
        "agent_id": "procurement_agent_01",
        "role": "調達エージェント",
        "perspectives": ["コスト", "納期", "サプライヤー", "品質", "代替"],
        "system_prompt": """あなたは調達専門のエージェントです。
サプライヤー選定、コスト分析、納期管理、品質評価、代替調達の観点から判断を行います。
過去の経験と学びを活用して、最適な調達判断を支援してください。""",
    }
}

def get_initial_strength_by_perspective(agent_id: str) -> dict:
    """エージェントの観点に基づいて初期strength_by_perspectiveを生成"""
    agent = AGENTS.get(agent_id)
    if not agent:
        return {}
    return {p: 1.0 for p in agent["perspectives"]}
```

### 3.5 タイムスタンプ設計

| カラム | 用途 |
|-------|------|
| `created_at` | 記憶の作成日時 |
| `updated_at` | 最終更新日時（強度変更等） |
| `last_accessed_at` | 最後に使用された日時（recency計算用） |
| `last_decay_at` | 最後に減衰処理が適用された日時（睡眠フェーズ追跡） |

---

## 4. パラメータ初期値

### 4.1 強度管理パラメータ

```python
# === 初期強度 ===
INITIAL_STRENGTH = 1.0              # 新規記憶の初期強度
INITIAL_STRENGTH_EDUCATION = 0.5    # 教育プロセスで読んだだけの記憶

# === 強化量 ===
STRENGTH_INCREMENT_ON_USE = 0.1           # 使用時の強化量
PERSPECTIVE_STRENGTH_INCREMENT = 0.15     # 観点別強度の強化量

# === 閾値 ===
ARCHIVE_THRESHOLD = 0.1             # これ以下でアーカイブ
REACTIVATION_STRENGTH = 0.5         # 再活性化時の初期強度
```

**設計根拠**:
- 初期強度1.0は中立的な出発点
- 5%/日減衰で約20日で閾値0.1に到達
- 強化量0.1は1回の使用で1日分の減衰を補う設計（0.95 × 1.1 ≒ 1.05）

### 4.2 定着レベルと減衰率

```python
# === 想定タスク数 ===
EXPECTED_TASKS_PER_DAY = 10

# === 定着レベル閾値（access_count） ===
CONSOLIDATION_THRESHOLDS = [0, 5, 15, 30, 60, 100]

# === 日次減衰目標 ===
DAILY_DECAY_TARGETS = {
    0: 0.95,    # 未定着: 5%/日
    1: 0.97,    # レベル1: 3%/日
    2: 0.98,    # レベル2: 2%/日
    3: 0.99,    # レベル3: 1%/日
    4: 0.995,   # レベル4: 0.5%/日
    5: 0.998,   # 完全定着: 0.2%/日
}

# === タスク単位の減衰率（自動計算） ===
def calculate_decay_rates(daily_targets: dict, tasks_per_day: int) -> dict:
    """日次減衰目標からタスク単位の減衰率を計算"""
    return {
        level: target ** (1 / tasks_per_day)
        for level, target in daily_targets.items()
    }

# 結果:
# Level 0: 0.9949 (≒ 0.995)
# Level 1: 0.9970
# Level 2: 0.9980
# Level 3: 0.9990
# Level 4: 0.9995
# Level 5: 0.9998
```

### 4.3 検索パラメータ

```python
# === Stage 1: 関連性フィルタ ===
SIMILARITY_THRESHOLD = 0.3          # 類似度の最低閾値
CANDIDATE_LIMIT = 50                # Stage 1の最大候補数

# === Stage 2: 優先度ランキング（線形スコア） ===
SCORE_WEIGHTS = {
    "similarity": 0.50,             # 類似度の重み
    "strength": 0.30,               # 強度の重み
    "recency": 0.20,                # 新鮮さの重み
}

# === 最終結果 ===
TOP_K_RESULTS = 10                  # コンテキストに渡す件数

# === 学習可能スコアラー移行閾値 ===
MIN_TRAINING_SAMPLES = 100          # これ以上溜まったらニューラルネット移行検討
```

**設計根拠**:
- 類似度0.3は「明らかに無関係」を除外するゆるめの閾値
- 類似度を最重視（0.50）し、強度は補助的役割（0.30）
- Phase 1では線形スコアで十分

### 4.4 インパクトスコア

```python
# === インパクト加算量 ===
IMPACT_USER_POSITIVE = 2.0          # ユーザーから肯定的フィードバック
IMPACT_TASK_SUCCESS = 1.5           # タスク成功に貢献
IMPACT_PREVENTED_ERROR = 2.0        # エラー防止に貢献

# === 強度へのインパクト反映率 ===
IMPACT_TO_STRENGTH_RATIO = 0.2      # impact × 0.2 を strength に加算
```

### 4.5 容量管理

```python
# === Phase 1: シンプルな件数制限 ===
MAX_ACTIVE_MEMORIES = 5000          # アクティブ記憶の最大件数

# === Phase 2 以降で検討 ===
# MAX_TOTAL_WEIGHT = 10000
# CONSOLIDATION_WEIGHTS = {0: 1, 1: 2, 2: 4, 3: 8, 4: 16, 5: 32}
```

### 4.6 使用判定パラメータ

```python
# === 使用判定（identify_used_memories） ===
USE_DETECTION_METHOD = "keyword"                # "keyword" | "similarity" | "llm"
USE_DETECTION_SIMILARITY_THRESHOLD = 0.3        # similarity方式での閾値
```

**設計根拠**:
- Phase 1はキーワードマッチングで開始（低コスト）
- 精度が不足なら similarity → llm に段階的に移行

### 4.7 クエリ拡張

```python
# === クエリ拡張 ===
ENABLE_QUERY_EXPANSION = True

# === 観点別キーワード（エージェントごとに定義） ===
PERSPECTIVE_KEYWORDS = {
    "procurement_agent_01": {
        "コスト": ["価格", "費用", "予算", "TCO", "コスト削減", "単価"],
        "納期": ["リードタイム", "遅延", "スケジュール", "期限", "納品"],
        "サプライヤー": ["取引先", "仕入先", "ベンダー", "調達先", "供給元"],
        "品質": ["不良", "検査", "規格", "品質基準", "歩留まり"],
        "代替": ["代替品", "代替調達", "セカンドソース", "冗長化", "バックアップ"],
    }
}
```

### 4.8 エンベディング設定

```python
# === OpenAI Embedding ===
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
```

---

## 5. 設定ファイル統合版

```python
# config/phase1_config.py

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Phase1Config:
    """Phase 1 MVP パラメータ設定"""

    # === 強度管理 ===
    initial_strength: float = 1.0
    initial_strength_education: float = 0.5
    strength_increment_on_use: float = 0.1
    perspective_strength_increment: float = 0.15
    archive_threshold: float = 0.1
    reactivation_strength: float = 0.5

    # === 減衰 ===
    expected_tasks_per_day: int = 10
    consolidation_thresholds: List[int] = field(
        default_factory=lambda: [0, 5, 15, 30, 60, 100]
    )
    daily_decay_targets: Dict[int, float] = field(
        default_factory=lambda: {
            0: 0.95,
            1: 0.97,
            2: 0.98,
            3: 0.99,
            4: 0.995,
            5: 0.998,
        }
    )

    # === 検索 ===
    similarity_threshold: float = 0.3
    candidate_limit: int = 50
    top_k_results: int = 10
    score_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "similarity": 0.50,
            "strength": 0.30,
            "recency": 0.20,
        }
    )

    # === インパクト ===
    impact_user_positive: float = 2.0
    impact_task_success: float = 1.5
    impact_prevented_error: float = 2.0
    impact_to_strength_ratio: float = 0.2

    # === 容量 ===
    max_active_memories: int = 5000

    # === 使用判定 ===
    use_detection_method: str = "keyword"  # "keyword" | "similarity" | "llm"
    use_detection_similarity_threshold: float = 0.3

    # === クエリ拡張 ===
    enable_query_expansion: bool = True

    # === エンベディング ===
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536

    # === スコープ（プロジェクト横断の知識管理） ===
    current_project_id: str = "llm-persistent-memory-phase1"
    related_domains: List[str] = field(
        default_factory=lambda: ["vector-database", "postgresql", "llm-applications"]
    )
    default_scope_level: str = "project"  # 新規記憶のデフォルトスコープ

    def get_decay_rate(self, consolidation_level: int) -> float:
        """定着レベルに応じたタスク単位の減衰率を取得"""
        daily_target = self.daily_decay_targets.get(consolidation_level, 0.95)
        return daily_target ** (1 / self.expected_tasks_per_day)

    def get_consolidation_level(self, access_count: int) -> int:
        """access_countから定着レベルを計算"""
        level = 0
        for i, threshold in enumerate(self.consolidation_thresholds):
            if access_count >= threshold:
                level = i
        return level


# デフォルト設定のインスタンス
config = Phase1Config()
```

---

## 6. 観測指標とパラメータ調整ガイド

### 6.1 観測すべき指標

| 指標 | 計算方法 | 正常範囲 |
|------|---------|---------|
| アーカイブ率 | archived / total | 10-30%/月 |
| 平均定着レベル | avg(consolidation_level) | 1.0-2.0 |
| 使用率 | avg(access_count / candidate_count) | 0.1-0.3 |
| 検索ヒット率 | tasks_with_hits / total_tasks | 0.7-0.9 |

### 6.2 パラメータ調整の方向

| 観測結果 | 調整パラメータ | 方向 |
|---------|---------------|------|
| アーカイブ率が高すぎる | `archive_threshold` | 下げる |
| アーカイブ率が低すぎる | `archive_threshold` | 上げる |
| 定着が遅すぎる | `strength_increment_on_use` | 上げる |
| 定着が速すぎる | `strength_increment_on_use` | 下げる |
| 検索漏れが多い | `similarity_threshold` | 下げる |
| ノイズが多い | `similarity_threshold` | 上げる |
| コンテキスト溢れ | `top_k_results` | 減らす |
| 情報不足 | `top_k_results` | 増やす |

### 6.3 EXPECTED_TASKS_PER_DAYの調整

実運用開始後、実際のタスク頻度を1週間程度観測し、`expected_tasks_per_day`を調整する。

```python
# 観測例
# 1週間で70タスク実行 → 10タスク/日 → そのまま
# 1週間で140タスク実行 → 20タスク/日 → expected_tasks_per_day = 20 に変更
```

---

## 7. 次のステップ

### 7.1 Phase 1 実装順序

1. **基盤構築**
   - Docker Compose環境構築
   - PostgreSQL + pgvector セットアップ
   - テーブル作成

2. **コア機能実装**
   - エンベディング生成
   - ベクトル検索（Stage 1）
   - スコア合成（Stage 2）
   - 強度更新（2段階強化）

3. **睡眠フェーズ実装**
   - 定着レベル更新
   - 減衰処理
   - アーカイブ処理

4. **タスク実行フロー実装**
   - 検索 → タスク実行 → 使用判定 → 強化 → 学び抽出 → 睡眠

5. **検証・調整**
   - パラメータ調整
   - 「個性」形成の検証

### 7.2 Phase 2 への移行条件

- Phase 1 の核心機能（2段階強化、観点別強度、睡眠フェーズ）が安定動作
- パラメータの適正値が概ね決定
- 単一エージェントで「個性」の形成が確認できた

---

*本ドキュメントは [architecture.ja.md](./architecture.ja.md) に基づいて作成された Phase 1 MVP の実装仕様書である。*

*作成日: 2025年1月12日*
