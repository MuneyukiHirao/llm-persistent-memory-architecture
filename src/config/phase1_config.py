# Phase 1 MVP パラメータ設定
# 実装仕様: docs/phase1-implementation-spec.ja.md セクション5

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Phase1Config:
    """Phase 1 MVP パラメータ設定

    仕様書参照:
    - セクション4.1: 強度管理パラメータ
    - セクション4.2: 定着レベルと減衰率
    - セクション4.3: 検索パラメータ
    - セクション4.4: インパクトスコア
    - セクション4.5: 容量管理
    - セクション4.6: 使用判定パラメータ
    - セクション4.7: クエリ拡張
    - セクション4.8: エンベディング設定
    - セクション5: スコープ設定
    """

    # === 強度管理（セクション4.1） ===
    initial_strength: float = 1.0
    """新規記憶の初期強度"""

    initial_strength_education: float = 0.5
    """教育プロセスで読んだだけの記憶の初期強度"""

    strength_increment_on_use: float = 0.1
    """使用時の強化量"""

    perspective_strength_increment: float = 0.15
    """観点別強度の強化量"""

    archive_threshold: float = 0.1
    """これ以下でアーカイブ"""

    reactivation_strength: float = 0.5
    """再活性化時の初期強度"""

    # === 減衰（セクション4.2） ===
    expected_tasks_per_day: int = 10
    """想定タスク数/日"""

    consolidation_thresholds: List[int] = field(
        default_factory=lambda: [0, 5, 15, 30, 60, 100]
    )
    """定着レベル閾値（access_count）
    Level 0: 0回以上
    Level 1: 5回以上
    Level 2: 15回以上
    Level 3: 30回以上
    Level 4: 60回以上
    Level 5: 100回以上
    """

    daily_decay_targets: Dict[int, float] = field(
        default_factory=lambda: {
            0: 0.95,   # 未定着: 5%/日
            1: 0.97,   # レベル1: 3%/日
            2: 0.98,   # レベル2: 2%/日
            3: 0.99,   # レベル3: 1%/日
            4: 0.995,  # レベル4: 0.5%/日
            5: 0.998,  # 完全定着: 0.2%/日
        }
    )
    """日次減衰目標（定着レベル -> 1日後の残存率）"""

    # === 検索（セクション4.3） ===
    similarity_threshold: float = 0.3
    """類似度の最低閾値（Stage 1: 関連性フィルタ）"""

    candidate_limit: int = 50
    """Stage 1の最大候補数"""

    top_k_results: int = 10
    """コンテキストに渡す件数"""

    score_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "similarity": 0.50,  # 類似度の重み
            "strength": 0.30,   # 強度の重み
            "recency": 0.20,    # 新鮮さの重み
        }
    )
    """Stage 2: 優先度ランキングの線形スコア重み"""

    min_training_samples: int = 100
    """学習可能スコアラー移行閾値"""

    # === インパクト（セクション4.4） ===
    impact_user_positive: float = 2.0
    """ユーザーから肯定的フィードバック時の加算量"""

    impact_task_success: float = 1.5
    """タスク成功に貢献時の加算量"""

    impact_prevented_error: float = 2.0
    """エラー防止に貢献時の加算量"""

    impact_to_strength_ratio: float = 0.2
    """強度へのインパクト反映率（impact × 0.2 を strength に加算）"""

    # === 容量（セクション4.5） ===
    max_active_memories: int = 5000
    """アクティブ記憶の最大件数"""

    # === 使用判定（セクション4.6） ===
    use_detection_method: str = "keyword"
    """使用判定方式: "keyword" | "similarity" | "llm" """

    use_detection_similarity_threshold: float = 0.3
    """similarity方式での閾値"""

    # === クエリ拡張（セクション4.7） ===
    enable_query_expansion: bool = True
    """クエリ拡張を有効にするか"""

    # === エンベディング（セクション4.8） ===
    embedding_model: str = "text-embedding-3-small"
    """OpenAI Embeddingモデル名"""

    embedding_dimension: int = 1536
    """エンベディング次元数"""

    # === スコープ（セクション5） ===
    current_project_id: str = "llm-persistent-memory-phase1"
    """現在のプロジェクトID"""

    related_domains: List[str] = field(
        default_factory=lambda: ["vector-database", "postgresql", "llm-applications"]
    )
    """関連ドメイン（ドメインレベルの知識検索に使用）"""

    default_scope_level: str = "project"
    """新規記憶のデフォルトスコープ: "universal" | "domain" | "project" """

    def get_decay_rate(self, consolidation_level: int) -> float:
        """定着レベルに応じたタスク単位の減衰率を取得

        Args:
            consolidation_level: 定着レベル (0-5)

        Returns:
            タスク単位の減衰率（例: 0.9949 は 1タスクで0.51%減衰）

        計算式:
            decay_rate = daily_target ** (1 / expected_tasks_per_day)

        例（expected_tasks_per_day=10の場合）:
            Level 0: 0.95^0.1 ≒ 0.9949 (5%/日 → 0.51%/タスク)
            Level 5: 0.998^0.1 ≒ 0.9998 (0.2%/日 → 0.02%/タスク)
        """
        daily_target = self.daily_decay_targets.get(consolidation_level, 0.95)
        return daily_target ** (1 / self.expected_tasks_per_day)

    def get_consolidation_level(self, access_count: int) -> int:
        """access_countから定着レベルを計算

        Args:
            access_count: 実際に使用された回数

        Returns:
            定着レベル (0-5)

        閾値:
            Level 0: access_count >= 0
            Level 1: access_count >= 5
            Level 2: access_count >= 15
            Level 3: access_count >= 30
            Level 4: access_count >= 60
            Level 5: access_count >= 100
        """
        level = 0
        for i, threshold in enumerate(self.consolidation_thresholds):
            if access_count >= threshold:
                level = i
        return level


# デフォルト設定のインスタンス
config = Phase1Config()
