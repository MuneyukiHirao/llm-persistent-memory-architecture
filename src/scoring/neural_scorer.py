# ニューラルネットベースのルーティングスコアラー
# 実装仕様: docs/phase3-implementation-spec.ja.md セクション5.1
"""
NeuralScorer モジュール

ニューラルネットワークを使用してエージェントの適性スコアを計算する。
モデル未学習時はルールベースにフォールバック。

設計方針（検索エンジンエージェント観点）:
- 検索精度: ルールベースより高精度なスコア計算を目指す
- レスポンス性能: 推論は軽量モデルで高速化
- スケーラビリティ: バッチ推論対応（将来拡張）
- API連携: EmbeddingClientとの連携を安定化
- フォールバック: モデル未学習時のルールベースフォールバック
"""

import logging
import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from src.agents.agent_registry import AgentDefinition
from src.config.phase3_config import (
    AGENT_FEATURES,
    NEURAL_SCORER_CONFIG,
    TASK_FEATURES,
    Phase3Config,
)
from src.embedding.azure_client import AzureEmbeddingClient
from src.scoring.feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)


# === 定数 ===
EMBEDDING_DIM = 1536
TASK_FEATURE_DIM = len(TASK_FEATURES)      # 6
AGENT_FEATURE_DIM = len(AGENT_FEATURES)    # 5
TOTAL_FEATURE_DIM = TASK_FEATURE_DIM + AGENT_FEATURE_DIM  # 11
INPUT_DIM = EMBEDDING_DIM + TOTAL_FEATURE_DIM  # 1547


class RoutingScorerModel(nn.Module):
    """ルーティングスコア予測モデル

    入力:
        - タスクエンベディング (1536次元)
        - タスク特徴量 (6次元)
        - エージェント特徴量 (5次元)
        計: 1547次元

    出力:
        - 適性スコア (0.0-1.0)

    アーキテクチャ:
        入力層 → 隠れ層1 (256, ReLU, Dropout)
               → 隠れ層2 (128, ReLU, Dropout)
               → 隠れ層3 (64, ReLU, Dropout)
               → 出力層 (1, Sigmoid)
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
    ):
        """モデルを初期化

        Args:
            input_dim: 入力次元（デフォルト: 1547）
            hidden_dims: 隠れ層の次元リスト（デフォルト: [256, 128, 64]）
            dropout: ドロップアウト率（デフォルト: 0.2）
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = NEURAL_SCORER_CONFIG.get("hidden_dims", [256, 128, 64])

        layers = []
        prev_dim = input_dim

        # 隠れ層を構築
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        # 出力層
        layers.extend([
            nn.Linear(prev_dim, 1),
            nn.Sigmoid(),
        ])

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """順伝播

        Args:
            x: 入力テンソル [batch_size, input_dim]

        Returns:
            スコアテンソル [batch_size, 1]
        """
        return self.network(x)


class NeuralScorer:
    """ニューラルネットベースのルーティングスコアラー

    タスクとエージェントの適性スコアをニューラルネットで計算する。
    モデルがロードされていない場合はルールベースにフォールバック。

    使用例:
        config = Phase3Config()
        embedding_client = AzureEmbeddingClient()
        scorer = NeuralScorer(
            model_path="models/routing_scorer.pt",
            embedding_client=embedding_client,
            config=config,
        )

        score = scorer.score(
            task_summary="APIエンドポイントを実装",
            agent=agent_definition,
            past_experiences=[{"success": True, "duration_seconds": 300}],
        )
        # => 0.85

    Attributes:
        model: RoutingScorerModel インスタンス（または None）
        embedding_client: AzureEmbeddingClient インスタンス
        config: Phase3Config インスタンス
        feature_extractor: FeatureExtractor インスタンス
    """

    def __init__(
        self,
        model_path: Optional[str],
        embedding_client: AzureEmbeddingClient,
        config: Phase3Config,
    ):
        """NeuralScorer を初期化

        Args:
            model_path: モデルファイルのパス（Noneまたは存在しない場合はフォールバック）
            embedding_client: AzureEmbeddingClient インスタンス
            config: Phase3Config インスタンス
        """
        self.embedding_client = embedding_client
        self.config = config
        self.feature_extractor = FeatureExtractor(config)

        # モデルのロード
        self.model: Optional[RoutingScorerModel] = None
        self._model_path = model_path

        if model_path and config.neural_scorer_enabled:
            self._load_model(model_path)

    def _load_model(self, model_path: str) -> None:
        """モデルをロード

        Args:
            model_path: モデルファイルのパス
        """
        if not os.path.exists(model_path):
            logger.warning(
                f"モデルファイルが見つかりません: {model_path}。"
                "フォールバックスコアを使用します。"
            )
            return

        try:
            self.model = RoutingScorerModel()
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            logger.info(f"NeuralScorer モデルをロードしました: {model_path}")
        except Exception as e:
            logger.error(f"モデルのロードに失敗: {e}")
            self.model = None

    def is_model_loaded(self) -> bool:
        """モデルがロードされているか

        Returns:
            True: モデルがロードされている
            False: モデルがロードされていない
        """
        return self.model is not None

    def score(
        self,
        task_summary: str,
        agent: AgentDefinition,
        past_experiences: Optional[List[Dict[str, Any]]] = None,
    ) -> float:
        """エージェントの適性スコアを計算

        Args:
            task_summary: タスク概要の文字列
            agent: AgentDefinition インスタンス
            past_experiences: 過去のタスク実行結果のリスト

        Returns:
            適性スコア（0.0-1.0）
        """
        # 特徴量を抽出
        task_features = self.feature_extractor.extract_task_features(task_summary)
        agent_features = self.feature_extractor.extract_agent_features(
            agent, past_experiences
        )

        # モデルがロードされていない、または neural_scorer_enabled が False の場合
        if not self.is_model_loaded() or not self.config.neural_scorer_enabled:
            return self._fallback_score(task_features, agent_features)

        # 1. タスクエンベディングを取得
        try:
            task_embedding = self.embedding_client.get_embedding(task_summary)
        except Exception as e:
            logger.warning(f"エンベディング取得に失敗、フォールバックを使用: {e}")
            return self._fallback_score(task_features, agent_features)

        # 2. 入力テンソルを構築
        input_tensor = self._build_input_tensor(
            task_embedding, task_features, agent_features
        )

        # 3. 推論
        try:
            with torch.no_grad():
                score_tensor = self.model(input_tensor)
                score = score_tensor.item()
        except Exception as e:
            logger.warning(f"推論に失敗、フォールバックを使用: {e}")
            return self._fallback_score(task_features, agent_features)

        return score

    def _build_input_tensor(
        self,
        task_embedding: List[float],
        task_features: Dict[str, float],
        agent_features: Dict[str, float],
    ) -> torch.Tensor:
        """入力テンソルを構築

        タスクエンベディング、タスク特徴量、エージェント特徴量を
        連結して入力テンソルを作成。

        Args:
            task_embedding: タスクエンベディング（1536次元）
            task_features: タスク特徴量の辞書
            agent_features: エージェント特徴量の辞書

        Returns:
            入力テンソル [1, INPUT_DIM]
        """
        # タスク特徴量を正規化・ベクトル化
        task_feat_vector = self._normalize_task_features(task_features)

        # エージェント特徴量を正規化・ベクトル化
        agent_feat_vector = self._normalize_agent_features(agent_features)

        # 連結
        input_vector = task_embedding + task_feat_vector + agent_feat_vector

        # テンソルに変換（バッチ次元を追加）
        return torch.tensor([input_vector], dtype=torch.float32)

    def _normalize_task_features(self, features: Dict[str, float]) -> List[float]:
        """タスク特徴量を正規化

        各特徴量を0-1の範囲に正規化。

        Args:
            features: タスク特徴量の辞書

        Returns:
            正規化された特徴量のリスト
        """
        # 正規化パラメータ（経験的な値）
        max_task_length = 2000.0
        max_item_count = 20.0

        return [
            min(features.get("task_length", 0.0) / max_task_length, 1.0),
            min(features.get("item_count", 0.0) / max_item_count, 1.0),
            features.get("has_code_keywords", 0.0),
            features.get("has_research_keywords", 0.0),
            features.get("has_test_keywords", 0.0),
            features.get("complexity_score", 0.0),
        ]

    def _normalize_agent_features(self, features: Dict[str, float]) -> List[float]:
        """エージェント特徴量を正規化

        各特徴量を0-1の範囲に正規化。

        Args:
            features: エージェント特徴量の辞書

        Returns:
            正規化された特徴量のリスト
        """
        # 正規化パラメータ（経験的な値）
        max_capability_count = 10.0
        max_perspective_count = 10.0
        max_recent_task_count = 50.0
        max_avg_duration = 3600.0  # 1時間

        return [
            min(features.get("capability_count", 0.0) / max_capability_count, 1.0),
            min(features.get("perspective_count", 0.0) / max_perspective_count, 1.0),
            features.get("past_success_rate", 0.5),  # すでに0-1
            min(features.get("recent_task_count", 0.0) / max_recent_task_count, 1.0),
            min(features.get("avg_task_duration", 0.0) / max_avg_duration, 1.0),
        ]

    def _fallback_score(
        self,
        task_features: Dict[str, float],
        agent_features: Dict[str, float],
    ) -> float:
        """モデル未学習時のルールベーススコア

        Phase2のRouterと同様のスコア計算を特徴量ベースで実施。

        重み付け:
        - capability_match相当: 0.40（キーワード特徴量から推定）
        - past_success_rate: 0.30
        - recent_activity: 0.20（タスク数から負荷を推定）
        - perspective_match相当: 0.10（capability_countから推定）

        Args:
            task_features: タスク特徴量の辞書
            agent_features: エージェント特徴量の辞書

        Returns:
            ルールベースの適性スコア（0.0-1.0）
        """
        score = 0.0

        # 1. capability_match 相当（0.40）
        # タスクのキーワード特徴量とエージェントの能力数から推定
        keyword_count = (
            task_features.get("has_code_keywords", 0.0) +
            task_features.get("has_research_keywords", 0.0) +
            task_features.get("has_test_keywords", 0.0)
        )
        # 能力数が多いほどマッチしやすいと仮定
        capability_count = agent_features.get("capability_count", 0.0)
        capability_score = min(capability_count / 5.0, 1.0) * (0.5 + keyword_count * 0.167)
        score += capability_score * 0.40

        # 2. past_success_rate（0.30）
        success_rate = agent_features.get("past_success_rate", 0.5)
        score += success_rate * 0.30

        # 3. recent_activity（0.20）
        # タスク数が少ないほど高スコア（負荷分散）
        recent_task_count = agent_features.get("recent_task_count", 0.0)
        activity_score = max(0.0, 1.0 - (recent_task_count / 10.0))
        score += activity_score * 0.20

        # 4. perspective_match 相当（0.10）
        # 観点数が多いほどカバー範囲が広いと仮定
        perspective_count = agent_features.get("perspective_count", 0.0)
        perspective_score = min(perspective_count / 5.0, 1.0)
        score += perspective_score * 0.10

        return min(max(score, 0.0), 1.0)

    def score_batch(
        self,
        task_summary: str,
        agents: List[AgentDefinition],
        past_experiences_map: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> Dict[str, float]:
        """複数エージェントの適性スコアをバッチ計算

        Args:
            task_summary: タスク概要の文字列
            agents: AgentDefinition のリスト
            past_experiences_map: エージェントIDをキーとした過去経験のマップ

        Returns:
            エージェントIDをキー、スコアを値とする辞書
        """
        if past_experiences_map is None:
            past_experiences_map = {}

        scores = {}
        for agent in agents:
            past_exp = past_experiences_map.get(agent.agent_id)
            scores[agent.agent_id] = self.score(task_summary, agent, past_exp)

        return scores

    def save_model(self, path: str) -> None:
        """モデルを保存

        Args:
            path: 保存先のパス
        """
        if self.model is None:
            raise ValueError("保存するモデルがありません")

        # ディレクトリがなければ作成
        os.makedirs(os.path.dirname(path), exist_ok=True)

        torch.save(self.model.state_dict(), path)
        logger.info(f"モデルを保存しました: {path}")

    def get_model_info(self) -> Dict[str, Any]:
        """モデル情報を取得

        Returns:
            モデル情報の辞書
        """
        return {
            "model_loaded": self.is_model_loaded(),
            "model_path": self._model_path,
            "neural_scorer_enabled": self.config.neural_scorer_enabled,
            "input_dim": INPUT_DIM,
            "embedding_dim": EMBEDDING_DIM,
            "task_feature_dim": TASK_FEATURE_DIM,
            "agent_feature_dim": AGENT_FEATURE_DIM,
            "hidden_dims": NEURAL_SCORER_CONFIG.get("hidden_dims", [256, 128, 64]),
        }
