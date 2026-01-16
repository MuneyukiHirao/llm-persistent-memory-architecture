-- Migration: learnings JSONB → learning TEXT
-- Date: 2026-01-15
-- Task ID: learn_001
--
-- 変更内容:
--   - learnings JSONB カラムを削除
--   - learning TEXT カラムを追加（NULL許容）
--
-- 設計意図:
--   - 学びは観点別ではなく、1つのオプショナルなテキストとして記録
--   - 学びがない場合は NULL（強制的に記録しない）
--   - 学びは例外的なイベントのみ（エラー解決、予想外の挙動、効率的な方法の発見）

-- ============================================================================
-- UP Migration (適用)
-- ============================================================================

BEGIN;

-- 1. 新しい learning TEXT カラムを追加
ALTER TABLE agent_memory ADD COLUMN learning TEXT;

-- 2. 既存の learnings JSONB データを移行
--    JSONB の各キー:値を改行区切りで結合
--    例: {"コスト": "学び1", "納期": "学び2"} → "コスト: 学び1\n納期: 学び2"
UPDATE agent_memory
SET learning = (
    SELECT string_agg(key || ': ' || value, E'\n')
    FROM jsonb_each_text(learnings)
)
WHERE learnings IS NOT NULL
  AND learnings != '{}'::jsonb
  AND jsonb_typeof(learnings) = 'object';

-- 3. 古い learnings カラムを削除
ALTER TABLE agent_memory DROP COLUMN learnings;

COMMIT;

-- ============================================================================
-- DOWN Migration (ロールバック)
-- ============================================================================
-- 以下は手動でロールバックが必要な場合に使用
--
-- BEGIN;
--
-- -- 1. learnings JSONB カラムを復元
-- ALTER TABLE agent_memory ADD COLUMN learnings JSONB;
--
-- -- 2. learning TEXT データを JSONB に変換（シンプルに {"content": "..."} 形式）
-- UPDATE agent_memory
-- SET learnings = jsonb_build_object('content', learning)
-- WHERE learning IS NOT NULL;
--
-- -- 3. learning TEXT カラムを削除
-- ALTER TABLE agent_memory DROP COLUMN learning;
--
-- COMMIT;
