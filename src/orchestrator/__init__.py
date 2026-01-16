# オーケストレーターモジュール
# 実装仕様: docs/phase2-implementation-spec.ja.md セクション5.2, 5.3, 5.4, 5.5
"""
オーケストレーターモジュール

オーケストレーションに関するコンポーネントを提供。

コンポーネント:
- Orchestrator: オーケストレーター本体（Phase 2 中心コンポーネント）
- OrchestratorResult: オーケストレーション実行結果
- SessionContext: セッションコンテキスト
- Router: ルーティングロジック（エージェント選択）
- RoutingDecision: ルーティング判断結果
- Evaluator: 評価フロー（フィードバック検出）
- FeedbackResult: フィードバック評価結果
- ProgressManager: 進捗管理（中間睡眠からの復帰）
- SessionState: セッション状態
- SessionStateRepository: セッション状態リポジトリ
"""

from src.orchestrator.router import Router, RoutingDecision
from src.orchestrator.evaluator import Evaluator, FeedbackResult
from src.orchestrator.orchestrator import (
    Orchestrator,
    OrchestratorResult,
    SessionContext,
)
from src.orchestrator.progress_manager import (
    ProgressManager,
    SessionState,
    SessionStateRepository,
)

__all__ = [
    "Orchestrator",
    "OrchestratorResult",
    "SessionContext",
    "Router",
    "RoutingDecision",
    "Evaluator",
    "FeedbackResult",
    "ProgressManager",
    "SessionState",
    "SessionStateRepository",
]
