-- Migration: Add Spaced Repetition columns
-- Date: 2026-01-15
-- Task ID: edu_002a_schema
--
-- 変更内容:
--   - next_review_at TIMESTAMP カラムを追加（次回復習予定日時）
--   - review_count INT カラムを追加（復習回数）
--
-- 設計意図:
--   - Spaced Repetition（間隔反復学習）機能のサポート
--   - SpacedRepetitionScheduler が次回復習タイミングを管理
--   - review_count は正解回数を追跡し、復習間隔の計算に使用

-- ============================================================================
-- UP Migration (適用)
-- ============================================================================

BEGIN;

-- 1. next_review_at カラムを追加（次回復習予定日時）
ALTER TABLE agent_memory ADD COLUMN next_review_at TIMESTAMP;

-- 2. review_count カラムを追加（復習回数、デフォルト0）
ALTER TABLE agent_memory ADD COLUMN review_count INT DEFAULT 0;

-- 3. 復習対象メモリの効率的な検索のためのインデックス
-- 「現在時刻より前の next_review_at を持つアクティブなメモリ」を高速に検索
CREATE INDEX idx_agent_memory_next_review ON agent_memory(next_review_at)
    WHERE next_review_at IS NOT NULL AND status = 'active';

COMMIT;

-- ============================================================================
-- DOWN Migration (ロールバック)
-- ============================================================================
-- 以下は手動でロールバックが必要な場合に使用
--
-- BEGIN;
--
-- -- 1. インデックスを削除
-- DROP INDEX IF EXISTS idx_agent_memory_next_review;
--
-- -- 2. カラムを削除
-- ALTER TABLE agent_memory DROP COLUMN review_count;
-- ALTER TABLE agent_memory DROP COLUMN next_review_at;
--
-- COMMIT;
