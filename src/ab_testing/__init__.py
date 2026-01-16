# A/B Testing Module
# 仕様: docs/phase3-implementation-spec.ja.md セクション5.6
"""
A/Bテスト実験管理モジュール

パラメータの最適化のためのA/Bテストを実行・評価する。

設計方針:
- 実験の作成・開始・一時停止・完了のライフサイクル管理
- セッションIDベースの決定論的バリアント割り当て
- scipy.stats による統計的有意性分析（t検定）
"""

from src.ab_testing.experiment_manager import (
    ExperimentManager,
    ExperimentResult,
    VariantStats,
    ExperimentStatus,
    ExperimentNotFoundError,
    ExperimentStateError,
)

__all__ = [
    "ExperimentManager",
    "ExperimentResult",
    "VariantStats",
    "ExperimentStatus",
    "ExperimentNotFoundError",
    "ExperimentStateError",
]
