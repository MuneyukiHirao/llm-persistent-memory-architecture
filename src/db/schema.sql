-- agent_memory テーブルスキーマ
-- このファイルは記録用。DBには既に適用済み。
-- 参照: docs/phase1-implementation-spec.ja.md セクション3.1

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

    -- 学び（例外的なイベントのみ記録）
    learning TEXT,  -- NULL許容、記録すべき学びがある場合のみ設定

    -- 状態
    status VARCHAR(16) DEFAULT 'active',  -- active / archived
    source VARCHAR(32),  -- education / task / manual

    -- タイムスタンプ
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    last_decay_at TIMESTAMP,  -- 睡眠フェーズの処理追跡

    -- Spaced Repetition（間隔反復学習）
    next_review_at TIMESTAMP,    -- 次回復習予定日時
    review_count INT DEFAULT 0   -- 復習回数（正解回数を追跡）
);

-- インデックス
CREATE INDEX idx_agent_memory_agent_id ON agent_memory(agent_id);
CREATE INDEX idx_agent_memory_status ON agent_memory(status);
CREATE INDEX idx_agent_memory_tags ON agent_memory USING GIN(tags);
CREATE INDEX idx_agent_memory_strength ON agent_memory(strength);
CREATE INDEX idx_agent_memory_scope ON agent_memory(scope_level, scope_domain, scope_project);

-- Spaced Repetition用インデックス（復習対象メモリの高速検索）
CREATE INDEX idx_agent_memory_next_review ON agent_memory(next_review_at)
    WHERE next_review_at IS NOT NULL AND status = 'active';

-- Phase 1: ベクトルインデックスなし（1万件未満想定）
-- 1万件超えたら以下を追加:
-- CREATE INDEX idx_agent_memory_embedding ON agent_memory
--     USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);


-- ============================================================================
-- Phase 2: オーケストレーション層のスキーマ
-- 参照: docs/phase2-implementation-spec.ja.md セクション3
-- ============================================================================

-- ----------------------------------------------------------------------------
-- 3.1 agent_definitions テーブル
-- エージェント定義のDB管理（Phase 1ではコード内定義だったものをDB管理に移行）
-- ----------------------------------------------------------------------------
CREATE TABLE agent_definitions (
    -- 識別子
    agent_id VARCHAR(64) PRIMARY KEY,

    -- 基本情報
    name VARCHAR(128) NOT NULL,
    role VARCHAR(256) NOT NULL,
    perspectives TEXT[] NOT NULL,  -- 観点（5つ程度）

    -- プロンプト
    system_prompt TEXT NOT NULL,

    -- ツール定義（JSON Schema形式）
    tools JSONB DEFAULT '[]',

    -- 能力タグ（ルーティング判断に使用）
    capabilities TEXT[] DEFAULT '{}',

    -- 状態
    status VARCHAR(16) DEFAULT 'active',  -- active / disabled

    -- タイムスタンプ
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- インデックス
CREATE INDEX idx_agent_definitions_status ON agent_definitions(status);
CREATE INDEX idx_agent_definitions_capabilities ON agent_definitions USING GIN(capabilities);

-- ----------------------------------------------------------------------------
-- 3.2 routing_history テーブル
-- ルーティング履歴（評価・学習用）
-- ----------------------------------------------------------------------------
CREATE TABLE routing_history (
    -- 識別子
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- セッション情報
    session_id UUID NOT NULL,
    orchestrator_id VARCHAR(64) NOT NULL,

    -- タスク情報
    task_summary TEXT NOT NULL,

    -- エージェント選択情報
    selected_agent_id VARCHAR(64) NOT NULL,
    selection_reason TEXT,
    candidate_agents JSONB,  -- [{"agent_id": "...", "score": 0.8, "reason": "..."}, ...]

    -- 結果
    result_status VARCHAR(16),  -- success / failure / timeout / cancelled
    result_summary TEXT,
    user_feedback VARCHAR(32),  -- positive / negative / neutral / null

    -- タイムスタンプ
    started_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP,

    -- 外部キー制約
    CONSTRAINT fk_selected_agent FOREIGN KEY (selected_agent_id)
        REFERENCES agent_definitions(agent_id)
);

-- インデックス
CREATE INDEX idx_routing_history_session ON routing_history(session_id);
CREATE INDEX idx_routing_history_orchestrator ON routing_history(orchestrator_id);
CREATE INDEX idx_routing_history_agent ON routing_history(selected_agent_id);
CREATE INDEX idx_routing_history_result ON routing_history(result_status);

-- ----------------------------------------------------------------------------
-- 3.3 session_state テーブル
-- オーケストレーターのセッション状態（中間睡眠からの復帰用）
-- ----------------------------------------------------------------------------
CREATE TABLE session_state (
    -- 識別子
    session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    orchestrator_id VARCHAR(64) NOT NULL,

    -- 元のリクエスト
    user_request JSONB NOT NULL,  -- {original: "...", clarified: "..."}

    -- 進捗状態
    task_tree JSONB NOT NULL,            -- タスク依存関係と完了状況
    current_task JSONB,                  -- 現在実行中のタスク（完了時はNULL）
    overall_progress_percent INT DEFAULT 0 CHECK (overall_progress_percent >= 0 AND overall_progress_percent <= 100),

    -- 状態
    status VARCHAR(16) DEFAULT 'in_progress' CHECK (status IN ('in_progress', 'paused', 'completed', 'failed')),

    -- タイムスタンプ
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    last_activity_at TIMESTAMP DEFAULT NOW(),

    -- Phase 3: マルチオーケストレーター連携用（排他制御）
    locked_by VARCHAR(64),            -- ロック保持オーケストレーターID
    lock_acquired_at TIMESTAMP        -- ロック取得時刻
);

-- インデックス
CREATE INDEX idx_session_state_orchestrator ON session_state(orchestrator_id);
CREATE INDEX idx_session_state_status ON session_state(status);
CREATE INDEX idx_session_state_locked_by ON session_state(locked_by);

-- routing_history テーブルへの外部キー制約追加（session_state 作成後）
ALTER TABLE routing_history
    ADD CONSTRAINT fk_routing_history_session
    FOREIGN KEY (session_id) REFERENCES session_state(session_id);


-- ============================================================================
-- Phase 3: スケーラビリティと実運用のスキーマ
-- 参照: docs/phase3-implementation-spec.ja.md セクション3
-- ============================================================================

-- ----------------------------------------------------------------------------
-- 3.1 routing_training_data テーブル
-- ニューラルスコアラーの学習データ
-- ----------------------------------------------------------------------------
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

-- ----------------------------------------------------------------------------
-- 3.3 orchestrator_state テーブル
-- 複数オーケストレーター間の状態共有
-- ----------------------------------------------------------------------------
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

-- ----------------------------------------------------------------------------
-- 3.4 ab_experiments テーブル
-- A/Bテスト実験管理
-- ----------------------------------------------------------------------------
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
