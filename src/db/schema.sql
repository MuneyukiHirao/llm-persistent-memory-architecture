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
