# 複数オーケストレーター協調モジュール
# 参照: docs/phase3-implementation-spec.ja.md セクション5.4
"""
複数オーケストレーター協調モジュール

複数のオーケストレーターインスタンスが協調して動作するための機能を提供。

主な機能:
- セッションロックの取得・解放（DBベース）
- ハートビート送信による生存確認
- 失敗したオーケストレーターの検出
- セッション引き継ぎ（フェイルオーバー）
"""

from src.coordination.multi_orchestrator import (
    MultiOrchestratorCoordinator,
    OrchestratorInfo,
    OrchestratorStatus,
    HealthStatus,
)

__all__ = [
    "MultiOrchestratorCoordinator",
    "OrchestratorInfo",
    "OrchestratorStatus",
    "HealthStatus",
]
