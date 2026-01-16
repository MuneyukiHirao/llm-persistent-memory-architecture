# エージェント管理モジュール
# 実装仕様: docs/phase2-implementation-spec.ja.md セクション3.1, 5.3
"""
エージェント管理モジュール

エージェント定義の登録・検索・管理機能を提供。

設計方針（メモリ管理エージェント観点）:
- スキーマ設計: agent_definitionsテーブルとの1:1マッピング
- データ整合性: DBの制約と整合した検証ロジック
- パフォーマンス: GINインデックスを活用したcapabilities検索
"""

from src.agents.agent_registry import AgentDefinition, AgentRegistry

__all__ = ["AgentDefinition", "AgentRegistry"]
