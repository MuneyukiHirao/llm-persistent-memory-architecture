# NeuralScorer テスト
# 実装仕様: docs/phase3-implementation-spec.ja.md セクション5.1
"""
NeuralScorer クラスのユニットテスト

テスト観点:
- テストカバレッジ: 全メソッド（score, is_model_loaded, fallback等）
- フォールバック動作: モデル未学習時のルールベーススコア
- 入力テンソル構築: 特徴量の正規化と連結
- エラーハンドリング: API障害時のフォールバック
- モデル構造: RoutingScorerModel の層構成

テスト対象:
1. NeuralScorer - メインクラス
2. RoutingScorerModel - PyTorchモデル
"""

import os
import tempfile
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.agents.agent_registry import AgentDefinition
from src.config.phase3_config import Phase3Config
from src.scoring.neural_scorer import (
    AGENT_FEATURE_DIM,
    EMBEDDING_DIM,
    INPUT_DIM,
    TASK_FEATURE_DIM,
    NeuralScorer,
    RoutingScorerModel,
)


# === フィクスチャ ===


@pytest.fixture
def config() -> Phase3Config:
    """テスト用 Phase3Config"""
    return Phase3Config()


@pytest.fixture
def config_with_neural_enabled() -> Phase3Config:
    """ニューラルスコアラー有効の設定"""
    config = Phase3Config()
    config.neural_scorer_enabled = True
    return config


@pytest.fixture
def mock_embedding_client() -> MagicMock:
    """モックEmbeddingClient"""
    client = MagicMock()
    # 1536次元のダミーエンベディングを返す
    client.get_embedding.return_value = [0.1] * EMBEDDING_DIM
    return client


@pytest.fixture
def sample_agent() -> AgentDefinition:
    """テスト用エージェント定義"""
    return AgentDefinition(
        agent_id="test_agent",
        name="テストエージェント",
        role="テスト作成と品質検証を担当",
        perspectives=["正確性", "網羅性", "効率性", "再現性", "保守性"],
        system_prompt="あなたはテスト専門のエージェントです",
        capabilities=["testing", "debugging", "analysis"],
        status="active",
    )


@pytest.fixture
def agent_with_high_success() -> AgentDefinition:
    """成功率の高いエージェント"""
    return AgentDefinition(
        agent_id="high_success_agent",
        name="高成功率エージェント",
        role="高成功率タスク実行",
        perspectives=["効率性", "正確性", "品質", "速度", "安定性"],
        system_prompt="",
        capabilities=["implementation", "coding", "testing", "debugging", "design"],
        status="active",
    )


@pytest.fixture
def agent_with_low_capabilities() -> AgentDefinition:
    """能力が少ないエージェント"""
    return AgentDefinition(
        agent_id="low_cap_agent",
        name="最小エージェント",
        role="最小構成",
        perspectives=["観点1"],
        system_prompt="",
        capabilities=["basic"],
        status="active",
    )


@pytest.fixture
def past_experiences_success() -> List[Dict[str, Any]]:
    """成功履歴を含む過去経験"""
    return [
        {"success": True, "duration_seconds": 120.0},
        {"success": True, "duration_seconds": 180.0},
        {"success": True, "duration_seconds": 150.0},
    ]


@pytest.fixture
def past_experiences_mixed() -> List[Dict[str, Any]]:
    """成功と失敗が混在する過去経験"""
    return [
        {"success": True, "duration_seconds": 100.0},
        {"success": False, "duration_seconds": 200.0},
        {"success": True, "duration_seconds": 150.0},
        {"success": False, "duration_seconds": 50.0},
    ]


@pytest.fixture
def past_experiences_many() -> List[Dict[str, Any]]:
    """多数のタスク履歴"""
    return [{"success": True, "duration_seconds": 100.0 + i * 10} for i in range(20)]


# === RoutingScorerModel テスト ===


class TestRoutingScorerModel:
    """RoutingScorerModel のテスト"""

    def test_model_init_default(self):
        """デフォルトパラメータでモデル初期化"""
        # Act
        model = RoutingScorerModel()

        # Assert
        assert model is not None

    def test_model_init_custom_input_dim(self):
        """カスタム入力次元でモデル初期化"""
        # Arrange
        custom_input_dim = 100

        # Act
        model = RoutingScorerModel(input_dim=custom_input_dim)

        # Assert
        assert model is not None

    def test_model_init_custom_hidden_dims(self):
        """カスタム隠れ層でモデル初期化"""
        # Arrange
        custom_hidden_dims = [128, 64, 32]

        # Act
        model = RoutingScorerModel(hidden_dims=custom_hidden_dims)

        # Assert
        assert model is not None

    def test_model_forward_shape(self):
        """forward の出力形状"""
        # Arrange
        model = RoutingScorerModel()
        batch_size = 4
        input_tensor = torch.randn(batch_size, INPUT_DIM)

        # Act
        output = model(input_tensor)

        # Assert
        assert output.shape == (batch_size, 1)

    def test_model_forward_output_range(self):
        """forward の出力範囲（Sigmoid なので 0-1）"""
        # Arrange
        model = RoutingScorerModel()
        input_tensor = torch.randn(10, INPUT_DIM)

        # Act
        output = model(input_tensor)

        # Assert
        assert (output >= 0.0).all()
        assert (output <= 1.0).all()

    def test_model_eval_mode(self):
        """eval モードでの推論"""
        # Arrange
        model = RoutingScorerModel()
        model.eval()
        input_tensor = torch.randn(1, INPUT_DIM)

        # Act
        with torch.no_grad():
            output = model(input_tensor)

        # Assert
        assert output.shape == (1, 1)

    def test_model_parameter_count(self):
        """モデルパラメータ数の確認"""
        # Arrange
        model = RoutingScorerModel()

        # Act
        param_count = sum(p.numel() for p in model.parameters())

        # Assert
        # 入力層: INPUT_DIM * 256 + 256
        # 隠れ層1: 256 * 128 + 128
        # 隠れ層2: 128 * 64 + 64
        # 出力層: 64 * 1 + 1
        # Total: 約43万パラメータ
        assert param_count > 0


# === NeuralScorer 初期化テスト ===


class TestNeuralScorerInit:
    """NeuralScorer 初期化のテスト"""

    def test_init_without_model_path(
        self,
        mock_embedding_client: MagicMock,
        config: Phase3Config,
    ):
        """model_path=None での初期化"""
        # Act
        scorer = NeuralScorer(
            model_path=None,
            embedding_client=mock_embedding_client,
            config=config,
        )

        # Assert
        assert scorer.model is None
        assert not scorer.is_model_loaded()

    def test_init_with_nonexistent_model_path(
        self,
        mock_embedding_client: MagicMock,
        config_with_neural_enabled: Phase3Config,
    ):
        """存在しないモデルパスでの初期化"""
        # Act
        scorer = NeuralScorer(
            model_path="/nonexistent/path/model.pt",
            embedding_client=mock_embedding_client,
            config=config_with_neural_enabled,
        )

        # Assert
        assert scorer.model is None
        assert not scorer.is_model_loaded()

    def test_init_with_neural_disabled(
        self,
        mock_embedding_client: MagicMock,
        config: Phase3Config,
    ):
        """neural_scorer_enabled=False での初期化"""
        # Arrange
        config.neural_scorer_enabled = False

        # Act
        scorer = NeuralScorer(
            model_path="models/routing_scorer.pt",
            embedding_client=mock_embedding_client,
            config=config,
        )

        # Assert
        # neural_scorer_enabled=False の場合はモデルをロードしない
        assert scorer.model is None

    def test_init_loads_valid_model(
        self,
        mock_embedding_client: MagicMock,
        config_with_neural_enabled: Phase3Config,
    ):
        """有効なモデルファイルのロード"""
        # Arrange: 一時ファイルにモデルを保存
        model = RoutingScorerModel()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(model.state_dict(), f.name)
            temp_path = f.name

        try:
            # Act
            scorer = NeuralScorer(
                model_path=temp_path,
                embedding_client=mock_embedding_client,
                config=config_with_neural_enabled,
            )

            # Assert
            assert scorer.is_model_loaded()
            assert scorer.model is not None
        finally:
            os.unlink(temp_path)


# === NeuralScorer.is_model_loaded テスト ===


class TestIsModelLoaded:
    """is_model_loaded() メソッドのテスト"""

    def test_returns_false_when_no_model(
        self,
        mock_embedding_client: MagicMock,
        config: Phase3Config,
    ):
        """モデルがない場合 False"""
        # Arrange
        scorer = NeuralScorer(
            model_path=None,
            embedding_client=mock_embedding_client,
            config=config,
        )

        # Act & Assert
        assert not scorer.is_model_loaded()

    def test_returns_true_when_model_loaded(
        self,
        mock_embedding_client: MagicMock,
        config_with_neural_enabled: Phase3Config,
    ):
        """モデルがある場合 True"""
        # Arrange
        model = RoutingScorerModel()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(model.state_dict(), f.name)
            temp_path = f.name

        try:
            scorer = NeuralScorer(
                model_path=temp_path,
                embedding_client=mock_embedding_client,
                config=config_with_neural_enabled,
            )

            # Act & Assert
            assert scorer.is_model_loaded()
        finally:
            os.unlink(temp_path)


# === NeuralScorer.score テスト ===


class TestScoreFallback:
    """score() メソッドのフォールバック動作テスト"""

    def test_uses_fallback_when_no_model(
        self,
        mock_embedding_client: MagicMock,
        config: Phase3Config,
        sample_agent: AgentDefinition,
    ):
        """モデルがない場合フォールバックを使用"""
        # Arrange
        scorer = NeuralScorer(
            model_path=None,
            embedding_client=mock_embedding_client,
            config=config,
        )

        # Act
        score = scorer.score("APIを実装してテストを書く", sample_agent)

        # Assert
        assert 0.0 <= score <= 1.0
        # エンベディングは呼ばれない（フォールバック使用）
        mock_embedding_client.get_embedding.assert_not_called()

    def test_uses_fallback_when_neural_disabled(
        self,
        mock_embedding_client: MagicMock,
        config: Phase3Config,
        sample_agent: AgentDefinition,
    ):
        """neural_scorer_enabled=False でフォールバックを使用"""
        # Arrange
        config.neural_scorer_enabled = False
        scorer = NeuralScorer(
            model_path=None,
            embedding_client=mock_embedding_client,
            config=config,
        )

        # Act
        score = scorer.score("APIを実装", sample_agent)

        # Assert
        assert 0.0 <= score <= 1.0

    def test_fallback_score_higher_for_capable_agent(
        self,
        mock_embedding_client: MagicMock,
        config: Phase3Config,
        agent_with_high_success: AgentDefinition,
        agent_with_low_capabilities: AgentDefinition,
    ):
        """能力が多いエージェントはフォールバックスコアが高い"""
        # Arrange
        scorer = NeuralScorer(
            model_path=None,
            embedding_client=mock_embedding_client,
            config=config,
        )
        task = "コードを実装してテストを書く"

        # Act
        high_score = scorer.score(task, agent_with_high_success)
        low_score = scorer.score(task, agent_with_low_capabilities)

        # Assert
        assert high_score > low_score

    def test_fallback_considers_past_success_rate(
        self,
        mock_embedding_client: MagicMock,
        config: Phase3Config,
        sample_agent: AgentDefinition,
        past_experiences_success: List[Dict[str, Any]],
        past_experiences_mixed: List[Dict[str, Any]],
    ):
        """フォールバックが過去の成功率を考慮"""
        # Arrange
        scorer = NeuralScorer(
            model_path=None,
            embedding_client=mock_embedding_client,
            config=config,
        )
        task = "テストを実行"

        # Act
        score_high_success = scorer.score(task, sample_agent, past_experiences_success)
        score_mixed = scorer.score(task, sample_agent, past_experiences_mixed)

        # Assert
        assert score_high_success > score_mixed

    def test_fallback_considers_recent_activity(
        self,
        mock_embedding_client: MagicMock,
        config: Phase3Config,
        sample_agent: AgentDefinition,
        past_experiences_many: List[Dict[str, Any]],
    ):
        """フォールバックが最近のアクティビティ（負荷）を考慮"""
        # Arrange
        scorer = NeuralScorer(
            model_path=None,
            embedding_client=mock_embedding_client,
            config=config,
        )
        task = "タスク実行"

        # Act
        score_no_activity = scorer.score(task, sample_agent, None)
        score_high_activity = scorer.score(task, sample_agent, past_experiences_many)

        # Assert
        # 高アクティビティ = 負荷が高い = スコアが低い
        assert score_no_activity > score_high_activity


class TestScoreWithModel:
    """score() メソッドのモデル使用テスト"""

    def test_score_uses_model_when_loaded(
        self,
        mock_embedding_client: MagicMock,
        config_with_neural_enabled: Phase3Config,
        sample_agent: AgentDefinition,
    ):
        """モデルがロードされている場合は推論を実行"""
        # Arrange
        model = RoutingScorerModel()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(model.state_dict(), f.name)
            temp_path = f.name

        try:
            scorer = NeuralScorer(
                model_path=temp_path,
                embedding_client=mock_embedding_client,
                config=config_with_neural_enabled,
            )

            # Act
            score = scorer.score("APIを実装", sample_agent)

            # Assert
            assert 0.0 <= score <= 1.0
            # エンベディングが呼ばれる
            mock_embedding_client.get_embedding.assert_called_once()
        finally:
            os.unlink(temp_path)

    def test_score_falls_back_on_embedding_error(
        self,
        config_with_neural_enabled: Phase3Config,
        sample_agent: AgentDefinition,
    ):
        """エンベディング取得エラー時はフォールバック"""
        # Arrange
        error_client = MagicMock()
        error_client.get_embedding.side_effect = Exception("API Error")

        model = RoutingScorerModel()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(model.state_dict(), f.name)
            temp_path = f.name

        try:
            scorer = NeuralScorer(
                model_path=temp_path,
                embedding_client=error_client,
                config=config_with_neural_enabled,
            )

            # Act
            score = scorer.score("APIを実装", sample_agent)

            # Assert
            assert 0.0 <= score <= 1.0  # フォールバックスコアが返る
        finally:
            os.unlink(temp_path)


# === NeuralScorer._build_input_tensor テスト ===


class TestBuildInputTensor:
    """_build_input_tensor() メソッドのテスト"""

    def test_tensor_shape(
        self,
        mock_embedding_client: MagicMock,
        config: Phase3Config,
    ):
        """出力テンソルの形状"""
        # Arrange
        scorer = NeuralScorer(
            model_path=None,
            embedding_client=mock_embedding_client,
            config=config,
        )
        task_embedding = [0.1] * EMBEDDING_DIM
        task_features = {
            "task_length": 100.0,
            "item_count": 3.0,
            "has_code_keywords": 1.0,
            "has_research_keywords": 0.0,
            "has_test_keywords": 1.0,
            "complexity_score": 0.5,
        }
        agent_features = {
            "capability_count": 5.0,
            "perspective_count": 5.0,
            "past_success_rate": 0.8,
            "recent_task_count": 10.0,
            "avg_task_duration": 300.0,
        }

        # Act
        tensor = scorer._build_input_tensor(task_embedding, task_features, agent_features)

        # Assert
        assert tensor.shape == (1, INPUT_DIM)

    def test_tensor_dtype(
        self,
        mock_embedding_client: MagicMock,
        config: Phase3Config,
    ):
        """出力テンソルのデータ型"""
        # Arrange
        scorer = NeuralScorer(
            model_path=None,
            embedding_client=mock_embedding_client,
            config=config,
        )
        task_embedding = [0.1] * EMBEDDING_DIM
        task_features = {
            "task_length": 100.0,
            "item_count": 3.0,
            "has_code_keywords": 1.0,
            "has_research_keywords": 0.0,
            "has_test_keywords": 1.0,
            "complexity_score": 0.5,
        }
        agent_features = {
            "capability_count": 5.0,
            "perspective_count": 5.0,
            "past_success_rate": 0.8,
            "recent_task_count": 10.0,
            "avg_task_duration": 300.0,
        }

        # Act
        tensor = scorer._build_input_tensor(task_embedding, task_features, agent_features)

        # Assert
        assert tensor.dtype == torch.float32


# === NeuralScorer._normalize_task_features テスト ===


class TestNormalizeTaskFeatures:
    """_normalize_task_features() メソッドのテスト"""

    def test_output_length(
        self,
        mock_embedding_client: MagicMock,
        config: Phase3Config,
    ):
        """出力リストの長さ"""
        # Arrange
        scorer = NeuralScorer(
            model_path=None,
            embedding_client=mock_embedding_client,
            config=config,
        )
        features = {
            "task_length": 100.0,
            "item_count": 3.0,
            "has_code_keywords": 1.0,
            "has_research_keywords": 0.0,
            "has_test_keywords": 1.0,
            "complexity_score": 0.5,
        }

        # Act
        normalized = scorer._normalize_task_features(features)

        # Assert
        assert len(normalized) == TASK_FEATURE_DIM

    def test_values_in_range(
        self,
        mock_embedding_client: MagicMock,
        config: Phase3Config,
    ):
        """正規化後の値が 0-1 の範囲内"""
        # Arrange
        scorer = NeuralScorer(
            model_path=None,
            embedding_client=mock_embedding_client,
            config=config,
        )
        features = {
            "task_length": 5000.0,  # 最大値を超える
            "item_count": 100.0,  # 最大値を超える
            "has_code_keywords": 1.0,
            "has_research_keywords": 0.0,
            "has_test_keywords": 1.0,
            "complexity_score": 0.5,
        }

        # Act
        normalized = scorer._normalize_task_features(features)

        # Assert
        for val in normalized:
            assert 0.0 <= val <= 1.0

    def test_handles_missing_keys(
        self,
        mock_embedding_client: MagicMock,
        config: Phase3Config,
    ):
        """キーがない場合はデフォルト値"""
        # Arrange
        scorer = NeuralScorer(
            model_path=None,
            embedding_client=mock_embedding_client,
            config=config,
        )
        features = {}  # 空辞書

        # Act
        normalized = scorer._normalize_task_features(features)

        # Assert
        assert len(normalized) == TASK_FEATURE_DIM
        assert all(val == 0.0 for val in normalized)


# === NeuralScorer._normalize_agent_features テスト ===


class TestNormalizeAgentFeatures:
    """_normalize_agent_features() メソッドのテスト"""

    def test_output_length(
        self,
        mock_embedding_client: MagicMock,
        config: Phase3Config,
    ):
        """出力リストの長さ"""
        # Arrange
        scorer = NeuralScorer(
            model_path=None,
            embedding_client=mock_embedding_client,
            config=config,
        )
        features = {
            "capability_count": 5.0,
            "perspective_count": 5.0,
            "past_success_rate": 0.8,
            "recent_task_count": 10.0,
            "avg_task_duration": 300.0,
        }

        # Act
        normalized = scorer._normalize_agent_features(features)

        # Assert
        assert len(normalized) == AGENT_FEATURE_DIM

    def test_success_rate_preserved(
        self,
        mock_embedding_client: MagicMock,
        config: Phase3Config,
    ):
        """past_success_rate はそのまま保持"""
        # Arrange
        scorer = NeuralScorer(
            model_path=None,
            embedding_client=mock_embedding_client,
            config=config,
        )
        features = {
            "capability_count": 5.0,
            "perspective_count": 5.0,
            "past_success_rate": 0.75,
            "recent_task_count": 10.0,
            "avg_task_duration": 300.0,
        }

        # Act
        normalized = scorer._normalize_agent_features(features)

        # Assert
        # past_success_rate は index 2
        assert normalized[2] == 0.75


# === NeuralScorer._fallback_score テスト ===


class TestFallbackScore:
    """_fallback_score() メソッドのテスト"""

    def test_score_range(
        self,
        mock_embedding_client: MagicMock,
        config: Phase3Config,
    ):
        """フォールバックスコアが 0-1 の範囲内"""
        # Arrange
        scorer = NeuralScorer(
            model_path=None,
            embedding_client=mock_embedding_client,
            config=config,
        )
        task_features = {
            "task_length": 100.0,
            "item_count": 3.0,
            "has_code_keywords": 1.0,
            "has_research_keywords": 0.0,
            "has_test_keywords": 1.0,
            "complexity_score": 0.5,
        }
        agent_features = {
            "capability_count": 5.0,
            "perspective_count": 5.0,
            "past_success_rate": 0.8,
            "recent_task_count": 10.0,
            "avg_task_duration": 300.0,
        }

        # Act
        score = scorer._fallback_score(task_features, agent_features)

        # Assert
        assert 0.0 <= score <= 1.0

    def test_high_capability_high_score(
        self,
        mock_embedding_client: MagicMock,
        config: Phase3Config,
    ):
        """能力数が多いと高スコア"""
        # Arrange
        scorer = NeuralScorer(
            model_path=None,
            embedding_client=mock_embedding_client,
            config=config,
        )
        task_features = {
            "task_length": 100.0,
            "item_count": 1.0,
            "has_code_keywords": 1.0,
            "has_research_keywords": 0.0,
            "has_test_keywords": 0.0,
            "complexity_score": 0.3,
        }
        high_cap_features = {
            "capability_count": 10.0,
            "perspective_count": 5.0,
            "past_success_rate": 0.5,
            "recent_task_count": 0.0,
            "avg_task_duration": 0.0,
        }
        low_cap_features = {
            "capability_count": 1.0,
            "perspective_count": 1.0,
            "past_success_rate": 0.5,
            "recent_task_count": 0.0,
            "avg_task_duration": 0.0,
        }

        # Act
        high_score = scorer._fallback_score(task_features, high_cap_features)
        low_score = scorer._fallback_score(task_features, low_cap_features)

        # Assert
        assert high_score > low_score

    def test_high_success_rate_high_score(
        self,
        mock_embedding_client: MagicMock,
        config: Phase3Config,
    ):
        """成功率が高いと高スコア"""
        # Arrange
        scorer = NeuralScorer(
            model_path=None,
            embedding_client=mock_embedding_client,
            config=config,
        )
        task_features = {
            "task_length": 100.0,
            "item_count": 1.0,
            "has_code_keywords": 0.0,
            "has_research_keywords": 0.0,
            "has_test_keywords": 0.0,
            "complexity_score": 0.3,
        }
        high_success_features = {
            "capability_count": 5.0,
            "perspective_count": 5.0,
            "past_success_rate": 1.0,
            "recent_task_count": 5.0,
            "avg_task_duration": 300.0,
        }
        low_success_features = {
            "capability_count": 5.0,
            "perspective_count": 5.0,
            "past_success_rate": 0.0,
            "recent_task_count": 5.0,
            "avg_task_duration": 300.0,
        }

        # Act
        high_score = scorer._fallback_score(task_features, high_success_features)
        low_score = scorer._fallback_score(task_features, low_success_features)

        # Assert
        assert high_score > low_score

    def test_low_activity_high_score(
        self,
        mock_embedding_client: MagicMock,
        config: Phase3Config,
    ):
        """アクティビティが低いと高スコア（負荷分散）"""
        # Arrange
        scorer = NeuralScorer(
            model_path=None,
            embedding_client=mock_embedding_client,
            config=config,
        )
        task_features = {
            "task_length": 100.0,
            "item_count": 1.0,
            "has_code_keywords": 0.0,
            "has_research_keywords": 0.0,
            "has_test_keywords": 0.0,
            "complexity_score": 0.3,
        }
        low_activity_features = {
            "capability_count": 5.0,
            "perspective_count": 5.0,
            "past_success_rate": 0.5,
            "recent_task_count": 0.0,
            "avg_task_duration": 300.0,
        }
        high_activity_features = {
            "capability_count": 5.0,
            "perspective_count": 5.0,
            "past_success_rate": 0.5,
            "recent_task_count": 20.0,
            "avg_task_duration": 300.0,
        }

        # Act
        low_activity_score = scorer._fallback_score(task_features, low_activity_features)
        high_activity_score = scorer._fallback_score(task_features, high_activity_features)

        # Assert
        assert low_activity_score > high_activity_score


# === NeuralScorer.score_batch テスト ===


class TestScoreBatch:
    """score_batch() メソッドのテスト"""

    def test_returns_dict_with_agent_ids(
        self,
        mock_embedding_client: MagicMock,
        config: Phase3Config,
        sample_agent: AgentDefinition,
        agent_with_high_success: AgentDefinition,
    ):
        """エージェントIDをキーとした辞書を返す"""
        # Arrange
        scorer = NeuralScorer(
            model_path=None,
            embedding_client=mock_embedding_client,
            config=config,
        )
        agents = [sample_agent, agent_with_high_success]

        # Act
        scores = scorer.score_batch("タスク実行", agents)

        # Assert
        assert sample_agent.agent_id in scores
        assert agent_with_high_success.agent_id in scores
        assert 0.0 <= scores[sample_agent.agent_id] <= 1.0
        assert 0.0 <= scores[agent_with_high_success.agent_id] <= 1.0

    def test_uses_past_experiences_map(
        self,
        mock_embedding_client: MagicMock,
        config: Phase3Config,
        sample_agent: AgentDefinition,
        past_experiences_success: List[Dict[str, Any]],
    ):
        """past_experiences_map を使用"""
        # Arrange
        scorer = NeuralScorer(
            model_path=None,
            embedding_client=mock_embedding_client,
            config=config,
        )
        agents = [sample_agent]
        exp_map = {sample_agent.agent_id: past_experiences_success}

        # Act
        scores_with_exp = scorer.score_batch("タスク", agents, exp_map)
        scores_without_exp = scorer.score_batch("タスク", agents, None)

        # Assert
        # 成功履歴があるとスコアが高い
        assert scores_with_exp[sample_agent.agent_id] > scores_without_exp[sample_agent.agent_id]


# === NeuralScorer.save_model テスト ===


class TestSaveModel:
    """save_model() メソッドのテスト"""

    def test_save_model_creates_file(
        self,
        mock_embedding_client: MagicMock,
        config_with_neural_enabled: Phase3Config,
    ):
        """モデル保存でファイルが作成される"""
        # Arrange
        model = RoutingScorerModel()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(model.state_dict(), f.name)
            temp_path = f.name

        scorer = NeuralScorer(
            model_path=temp_path,
            embedding_client=mock_embedding_client,
            config=config_with_neural_enabled,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "new_model.pt")

            # Act
            scorer.save_model(save_path)

            # Assert
            assert os.path.exists(save_path)

        os.unlink(temp_path)

    def test_save_model_raises_when_no_model(
        self,
        mock_embedding_client: MagicMock,
        config: Phase3Config,
    ):
        """モデルがない場合はエラー"""
        # Arrange
        scorer = NeuralScorer(
            model_path=None,
            embedding_client=mock_embedding_client,
            config=config,
        )

        # Act & Assert
        with pytest.raises(ValueError):
            scorer.save_model("/tmp/test.pt")


# === NeuralScorer.get_model_info テスト ===


class TestGetModelInfo:
    """get_model_info() メソッドのテスト"""

    def test_returns_expected_keys(
        self,
        mock_embedding_client: MagicMock,
        config: Phase3Config,
    ):
        """期待されるキーを含む辞書を返す"""
        # Arrange
        scorer = NeuralScorer(
            model_path=None,
            embedding_client=mock_embedding_client,
            config=config,
        )

        # Act
        info = scorer.get_model_info()

        # Assert
        expected_keys = [
            "model_loaded",
            "model_path",
            "neural_scorer_enabled",
            "input_dim",
            "embedding_dim",
            "task_feature_dim",
            "agent_feature_dim",
            "hidden_dims",
        ]
        for key in expected_keys:
            assert key in info

    def test_model_loaded_reflects_state(
        self,
        mock_embedding_client: MagicMock,
        config: Phase3Config,
    ):
        """model_loaded が実際の状態を反映"""
        # Arrange
        scorer = NeuralScorer(
            model_path=None,
            embedding_client=mock_embedding_client,
            config=config,
        )

        # Act
        info = scorer.get_model_info()

        # Assert
        assert info["model_loaded"] == scorer.is_model_loaded()

    def test_dimensions_are_correct(
        self,
        mock_embedding_client: MagicMock,
        config: Phase3Config,
    ):
        """次元情報が正しい"""
        # Arrange
        scorer = NeuralScorer(
            model_path=None,
            embedding_client=mock_embedding_client,
            config=config,
        )

        # Act
        info = scorer.get_model_info()

        # Assert
        assert info["input_dim"] == INPUT_DIM
        assert info["embedding_dim"] == EMBEDDING_DIM
        assert info["task_feature_dim"] == TASK_FEATURE_DIM
        assert info["agent_feature_dim"] == AGENT_FEATURE_DIM


# === 定数テスト ===


class TestConstants:
    """定数のテスト"""

    def test_embedding_dim(self):
        """EMBEDDING_DIM が正しい"""
        assert EMBEDDING_DIM == 1536

    def test_task_feature_dim(self):
        """TASK_FEATURE_DIM が正しい"""
        assert TASK_FEATURE_DIM == 6

    def test_agent_feature_dim(self):
        """AGENT_FEATURE_DIM が正しい"""
        assert AGENT_FEATURE_DIM == 5

    def test_input_dim(self):
        """INPUT_DIM が正しく計算されている"""
        assert INPUT_DIM == EMBEDDING_DIM + TASK_FEATURE_DIM + AGENT_FEATURE_DIM


# === 再現性テスト ===


class TestReproducibility:
    """再現性のテスト"""

    def test_same_input_same_fallback_score(
        self,
        mock_embedding_client: MagicMock,
        config: Phase3Config,
        sample_agent: AgentDefinition,
    ):
        """同一入力に対して常に同じフォールバックスコア"""
        # Arrange
        scorer = NeuralScorer(
            model_path=None,
            embedding_client=mock_embedding_client,
            config=config,
        )
        task = "APIを実装してテストを書く"

        # Act
        scores = [scorer.score(task, sample_agent) for _ in range(5)]

        # Assert
        assert all(s == scores[0] for s in scores)
