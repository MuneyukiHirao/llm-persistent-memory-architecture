# Phase 2 MVP パラメータ設定
# 実装仕様: docs/phase2-implementation-spec.ja.md セクション4

from dataclasses import dataclass, field
from typing import Dict, List

from src.config.phase1_config import Phase1Config


@dataclass
class Phase2Config(Phase1Config):
    """Phase 2 MVP パラメータ設定（Phase 1 設定を継承）

    仕様書参照:
    - セクション4.1: Phase 2 設定クラス
    - セクション4.2: ルーティングパラメータ
    - セクション4.3: エージェント定義パラメータ
    """

    # === 入力処理層 ===
    input_item_threshold: int = 10
    """これ以上の論点数で警告"""

    input_size_threshold: int = 5000
    """トークン数、これ以上で概要生成"""

    summary_max_tokens: int = 1000
    """概要の最大トークン数"""

    # === オーケストレーター ===
    orchestrator_model: str = "claude-sonnet-4-20250514"
    """オーケストレーター用LLMモデル"""

    input_processor_model: str = "claude-3-5-haiku-20241022"
    """入力処理層用LLMモデル（軽量・高速）"""

    orchestrator_context_threshold: float = 0.7
    """コンテキスト使用率70%で中間睡眠"""

    orchestrator_idle_timeout_minutes: int = 60
    """アイドル1時間で睡眠"""

    orchestrator_subtask_batch_size: int = 5
    """5サブタスク完了ごとに睡眠検討"""

    # === ルーティング ===
    routing_method: str = "rule_based"
    """ルーティング方式: "rule_based" | "similarity" | "llm" """

    routing_similarity_threshold: float = 0.5
    """similarity方式での適性閾値"""

    max_routing_candidates: int = 3
    """候補エージェントの最大数"""

    # === 評価 ===
    feedback_detection_method: str = "keyword"
    """フィードバック検出方式: "keyword" | "similarity" | "llm" """

    implicit_feedback_enabled: bool = True
    """暗黙的フィードバック検出を有効にするか"""

    # === エージェント管理 ===
    agent_timeout_seconds: int = 300
    """エージェント実行タイムアウト（5分）"""

    max_retry_count: int = 2
    """失敗時の最大リトライ回数"""

    # === 進捗管理 ===
    progress_check_interval_seconds: int = 30
    """進捗チェック間隔（秒）"""

    progress_state_file: str = "memory/progress_state.json"
    """進捗状態ファイルパス"""


# === ルーティングスコア重み（rule_based方式）===
ROUTING_SCORE_WEIGHTS: Dict[str, float] = {
    "capability_match": 0.40,    # 能力タグのマッチ度
    "past_success_rate": 0.30,   # 過去の成功率
    "recent_activity": 0.20,     # 最近のアクティビティ（負荷考慮）
    "perspective_match": 0.10,   # 観点のマッチ度
}

# === フィードバック判定 ===
FEEDBACK_SIGNALS: Dict[str, List[str]] = {
    "positive": ["ありがとう", "良い", "完璧", "OK", "了解"],
    "negative": ["やり直し", "違う", "ダメ", "修正して"],
    "redo_requested": ["もう一度", "再度", "別のエージェント"],
}

# === タスクサイズ上限 ===
MAX_TASK_CONTEXT_TOKENS: int = 50000
"""エージェントに渡すタスクの最大トークン数"""

TASK_SIZE_MARGIN: float = 0.7
"""コンテキストウィンドウの70%以内に収める"""

# === オーケストレーターの観点 ===
ORCHESTRATOR_PERSPECTIVES: List[str] = [
    "ユーザー意図",       # ユーザーが本当に求めていることは何か
    "エージェント適性",   # どの専門エージェントに任せるべきか
    "タスク依存関係",     # どの順序で進めるべきか
    "タスク結果評価",     # エージェントの結果は期待を満たしているか
    "ユーザー満足度",     # ユーザーは満足しているか
]

# === 入力処理層の観点（軽量、判断のみ）===
INPUT_PROCESSOR_PERSPECTIVES: List[str] = [
    "論点数",            # 項目がいくつあるか
    "入力サイズ",        # 入力が大きすぎないか
    "曖昧さ",            # 曖昧な指示かどうか（判断のみ、解消はしない）
]


# デフォルト設定のインスタンス
phase2_config = Phase2Config()
