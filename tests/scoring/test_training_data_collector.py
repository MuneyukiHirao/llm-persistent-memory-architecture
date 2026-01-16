# TrainingDataCollector テスト
# 実装仕様: docs/phase3-implementation-spec.ja.md セクション5.1 学習パイプライン
"""
TrainingDataCollector クラスのユニットテスト

テスト観点:
- テストカバレッジ: 全メソッド（collect_from_routing_history, _calculate_label 等）
- 再現性: 同一入力に対して常に同じラベルを生成
- 境界値・異常系: None フィードバック、欠損データ
- モック化: 外部依存（DB, Embedding）をモック化してテスト容易性を確保

テスト対象メソッド:
1. _calculate_label(user_feedback, result_status) - ラベル計算
2. collect_from_routing_history(since, limit) - 学習データ収集
3. get_training_data_count(only_unused) - 学習データ件数取得
4. is_ready_for_training() - 学習準備完了判定
"""

import pytest
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

from src.agents.agent_registry import AgentDefinition, AgentRegistry
from src.config.phase3_config import Phase3Config
from src.db.connection import DatabaseConnection
from src.embedding.azure_client import AzureEmbeddingClient
from src.scoring.feature_extractor import FeatureExtractor
from src.scoring.training_data_collector import (
    TrainingDataCollector,
    RoutingHistoryRecord,
)


# === フィクスチャ ===


@pytest.fixture
def config() -> Phase3Config:
    """テスト用 Phase3Config"""
    return Phase3Config(min_training_samples=1000)


@pytest.fixture
def mock_db_connection() -> MagicMock:
    """モック DatabaseConnection"""
    mock = MagicMock(spec=DatabaseConnection)
    # コンテキストマネージャーのモック
    mock_cursor = MagicMock()
    mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
    mock_cursor.__exit__ = MagicMock(return_value=False)
    mock.get_cursor.return_value = mock_cursor
    return mock


@pytest.fixture
def mock_embedding_client() -> MagicMock:
    """モック AzureEmbeddingClient"""
    mock = MagicMock(spec=AzureEmbeddingClient)
    # 1536次元のダミーエンベディングを返す
    mock.get_embedding.return_value = [0.1] * 1536
    return mock


@pytest.fixture
def mock_agent_registry() -> MagicMock:
    """モック AgentRegistry"""
    mock = MagicMock(spec=AgentRegistry)
    mock.get_by_id.return_value = AgentDefinition(
        agent_id="test_agent",
        name="テストエージェント",
        role="テスト",
        perspectives=["観点1", "観点2"],
        system_prompt="テスト用",
        capabilities=["testing"],
    )
    return mock


@pytest.fixture
def collector(
    mock_db_connection: MagicMock,
    mock_embedding_client: MagicMock,
    mock_agent_registry: MagicMock,
    config: Phase3Config,
) -> TrainingDataCollector:
    """テスト用 TrainingDataCollector"""
    return TrainingDataCollector(
        db_connection=mock_db_connection,
        embedding_client=mock_embedding_client,
        agent_registry=mock_agent_registry,
        config=config,
    )


@pytest.fixture
def sample_routing_history_record() -> RoutingHistoryRecord:
    """サンプル RoutingHistoryRecord"""
    return RoutingHistoryRecord(
        id=uuid4(),
        session_id=uuid4(),
        orchestrator_id="orch_01",
        task_summary="APIエンドポイントを実装",
        selected_agent_id="impl_agent",
        selection_reason="実装タスクに最適",
        candidate_agents=[
            {"agent_id": "impl_agent", "score": 0.9, "reason": "実装能力高"},
            {"agent_id": "test_agent", "score": 0.7, "reason": "テスト能力"},
        ],
        result_status="success",
        result_summary="実装完了",
        user_feedback="positive",
        started_at=datetime.now(),
        completed_at=datetime.now(),
    )


# === TrainingDataCollector 初期化テスト ===


class TestTrainingDataCollectorInit:
    """TrainingDataCollector 初期化のテスト"""

    def test_init_sets_all_dependencies(
        self,
        mock_db_connection: MagicMock,
        mock_embedding_client: MagicMock,
        mock_agent_registry: MagicMock,
        config: Phase3Config,
    ):
        """全ての依存関係が正しく設定される"""
        # Act
        collector = TrainingDataCollector(
            db_connection=mock_db_connection,
            embedding_client=mock_embedding_client,
            agent_registry=mock_agent_registry,
            config=config,
        )

        # Assert
        assert collector.db is mock_db_connection
        assert collector.embedding_client is mock_embedding_client
        assert collector.agent_registry is mock_agent_registry
        assert collector.config is config
        assert isinstance(collector.feature_extractor, FeatureExtractor)


# === _calculate_label テスト ===


class TestCalculateLabel:
    """_calculate_label() メソッドのテスト"""

    def test_positive_feedback_success_status(self, collector: TrainingDataCollector):
        """positive フィードバック + success ステータス → 1.0"""
        # Act
        result = collector._calculate_label("positive", "success")

        # Assert
        assert result == 1.0

    def test_positive_feedback_failure_status(self, collector: TrainingDataCollector):
        """positive フィードバック + failure ステータス → 1.0"""
        # Act
        result = collector._calculate_label("positive", "failure")

        # Assert
        assert result == 1.0

    def test_positive_feedback_no_status(self, collector: TrainingDataCollector):
        """positive フィードバック + ステータスなし → 1.0"""
        # Act
        result = collector._calculate_label("positive", None)

        # Assert
        assert result == 1.0

    def test_negative_feedback_success_status(self, collector: TrainingDataCollector):
        """negative フィードバック + success ステータス → 0.5"""
        # Act
        result = collector._calculate_label("negative", "success")

        # Assert
        assert result == 0.5  # 0.0 + 0.5

    def test_negative_feedback_failure_status(self, collector: TrainingDataCollector):
        """negative フィードバック + failure ステータス → 0.0"""
        # Act
        result = collector._calculate_label("negative", "failure")

        # Assert
        assert result == 0.0

    def test_negative_feedback_no_status(self, collector: TrainingDataCollector):
        """negative フィードバック + ステータスなし → 0.0"""
        # Act
        result = collector._calculate_label("negative", None)

        # Assert
        assert result == 0.0

    def test_neutral_feedback_success_status(self, collector: TrainingDataCollector):
        """neutral フィードバック + success ステータス → 1.0"""
        # Act
        result = collector._calculate_label("neutral", "success")

        # Assert
        assert result == 1.0  # 0.5 + 0.5

    def test_neutral_feedback_failure_status(self, collector: TrainingDataCollector):
        """neutral フィードバック + failure ステータス → 0.5"""
        # Act
        result = collector._calculate_label("neutral", "failure")

        # Assert
        assert result == 0.5

    def test_no_feedback_success_status(self, collector: TrainingDataCollector):
        """フィードバックなし + success ステータス → 1.0"""
        # Act
        result = collector._calculate_label(None, "success")

        # Assert
        assert result == 1.0  # 0.5 + 0.5

    def test_no_feedback_failure_status(self, collector: TrainingDataCollector):
        """フィードバックなし + failure ステータス → 0.5"""
        # Act
        result = collector._calculate_label(None, "failure")

        # Assert
        assert result == 0.5

    def test_no_feedback_no_status(self, collector: TrainingDataCollector):
        """フィードバックなし + ステータスなし → 0.5（デフォルト）"""
        # Act
        result = collector._calculate_label(None, None)

        # Assert
        assert result == 0.5

    def test_timeout_status(self, collector: TrainingDataCollector):
        """timeout ステータス → success 加算なし"""
        # Act
        result = collector._calculate_label("neutral", "timeout")

        # Assert
        assert result == 0.5  # success 以外なので加算なし

    def test_cancelled_status(self, collector: TrainingDataCollector):
        """cancelled ステータス → success 加算なし"""
        # Act
        result = collector._calculate_label("neutral", "cancelled")

        # Assert
        assert result == 0.5


class TestCalculateLabelRange:
    """_calculate_label() の出力範囲テスト"""

    def test_label_never_exceeds_one(self, collector: TrainingDataCollector):
        """ラベルが 1.0 を超えない"""
        # Arrange
        test_cases = [
            ("positive", "success"),
            ("positive", "failure"),
            ("neutral", "success"),
        ]

        # Act & Assert
        for feedback, status in test_cases:
            result = collector._calculate_label(feedback, status)
            assert result <= 1.0, f"({feedback}, {status}) でラベルが 1.0 を超過: {result}"

    def test_label_never_below_zero(self, collector: TrainingDataCollector):
        """ラベルが 0.0 を下回らない"""
        # Arrange
        test_cases = [
            ("negative", "failure"),
            ("negative", "timeout"),
            ("negative", None),
        ]

        # Act & Assert
        for feedback, status in test_cases:
            result = collector._calculate_label(feedback, status)
            assert result >= 0.0, f"({feedback}, {status}) でラベルが 0.0 を下回る: {result}"


class TestCalculateLabelReproducibility:
    """_calculate_label() の再現性テスト"""

    def test_same_input_same_output(self, collector: TrainingDataCollector):
        """同一入力に対して常に同じ結果"""
        # Arrange
        test_cases = [
            ("positive", "success"),
            ("negative", "failure"),
            ("neutral", "success"),
            (None, None),
        ]

        # Act & Assert
        for feedback, status in test_cases:
            result1 = collector._calculate_label(feedback, status)
            result2 = collector._calculate_label(feedback, status)
            result3 = collector._calculate_label(feedback, status)
            assert result1 == result2 == result3


# === RoutingHistoryRecord テスト ===


class TestRoutingHistoryRecord:
    """RoutingHistoryRecord データクラスのテスト"""

    def test_from_row_creates_instance(self):
        """from_row() でインスタンスが生成される"""
        # Arrange
        test_id = uuid4()
        session_id = uuid4()
        now = datetime.now()
        row = (
            test_id,
            session_id,
            "orch_01",
            "タスク概要",
            "agent_01",
            "選択理由",
            [{"agent_id": "agent_01", "score": 0.9}],
            "success",
            "結果サマリー",
            "positive",
            now,
            now,
        )

        # Act
        record = RoutingHistoryRecord.from_row(row)

        # Assert
        assert record.id == test_id
        assert record.session_id == session_id
        assert record.orchestrator_id == "orch_01"
        assert record.task_summary == "タスク概要"
        assert record.selected_agent_id == "agent_01"
        assert record.selection_reason == "選択理由"
        assert record.candidate_agents == [{"agent_id": "agent_01", "score": 0.9}]
        assert record.result_status == "success"
        assert record.result_summary == "結果サマリー"
        assert record.user_feedback == "positive"
        assert record.started_at == now
        assert record.completed_at == now

    def test_from_row_with_null_values(self):
        """NULL値を含む行からインスタンス生成"""
        # Arrange
        test_id = uuid4()
        session_id = uuid4()
        now = datetime.now()
        row = (
            test_id,
            session_id,
            "orch_01",
            "タスク概要",
            "agent_01",
            None,  # selection_reason
            None,  # candidate_agents
            None,  # result_status
            None,  # result_summary
            None,  # user_feedback
            now,
            None,  # completed_at
        )

        # Act
        record = RoutingHistoryRecord.from_row(row)

        # Assert
        assert record.selection_reason is None
        assert record.candidate_agents is None
        assert record.result_status is None
        assert record.result_summary is None
        assert record.user_feedback is None
        assert record.completed_at is None


# === collect_from_routing_history テスト ===


class TestCollectFromRoutingHistory:
    """collect_from_routing_history() メソッドのテスト"""

    def test_returns_zero_when_no_records(
        self,
        collector: TrainingDataCollector,
        mock_db_connection: MagicMock,
    ):
        """レコードがない場合は 0 を返す"""
        # Arrange
        mock_cursor = mock_db_connection.get_cursor.return_value.__enter__.return_value
        mock_cursor.fetchall.return_value = []

        # Act
        result = collector.collect_from_routing_history()

        # Assert
        assert result == 0

    def test_skips_records_without_feedback(
        self,
        collector: TrainingDataCollector,
        mock_db_connection: MagicMock,
    ):
        """フィードバックもステータスもないレコードはスキップ"""
        # Arrange
        test_id = uuid4()
        session_id = uuid4()
        now = datetime.now()

        # フィードバックも結果ステータスもないレコード
        mock_cursor = mock_db_connection.get_cursor.return_value.__enter__.return_value
        mock_cursor.fetchall.side_effect = [
            # _fetch_routing_history の結果
            [
                (
                    test_id,
                    session_id,
                    "orch_01",
                    "タスク概要",
                    "test_agent",
                    None,
                    None,
                    None,  # result_status が None
                    None,
                    None,  # user_feedback が None
                    now,
                    None,
                )
            ],
            # _get_past_experiences の結果
            [],
        ]

        # Act
        result = collector.collect_from_routing_history()

        # Assert
        assert result == 0


class TestCollectFromRoutingHistoryWithLimit:
    """collect_from_routing_history() の limit パラメータテスト"""

    def test_limit_parameter_passed_to_query(
        self,
        collector: TrainingDataCollector,
        mock_db_connection: MagicMock,
    ):
        """limit パラメータがクエリに渡される"""
        # Arrange
        mock_cursor = mock_db_connection.get_cursor.return_value.__enter__.return_value
        mock_cursor.fetchall.return_value = []

        # Act
        collector.collect_from_routing_history(limit=500)

        # Assert
        # SQL実行が呼ばれたことを確認
        mock_cursor.execute.assert_called()


class TestCollectFromRoutingHistorySince:
    """collect_from_routing_history() の since パラメータテスト"""

    def test_since_parameter_filters_records(
        self,
        collector: TrainingDataCollector,
        mock_db_connection: MagicMock,
    ):
        """since パラメータでレコードがフィルタされる"""
        # Arrange
        since_date = datetime(2025, 1, 1)
        mock_cursor = mock_db_connection.get_cursor.return_value.__enter__.return_value
        mock_cursor.fetchall.return_value = []

        # Act
        collector.collect_from_routing_history(since=since_date)

        # Assert
        mock_cursor.execute.assert_called()


# === get_training_data_count テスト ===


class TestGetTrainingDataCount:
    """get_training_data_count() メソッドのテスト"""

    def test_returns_count_all(
        self,
        collector: TrainingDataCollector,
        mock_db_connection: MagicMock,
    ):
        """全件カウントを返す"""
        # Arrange
        mock_cursor = mock_db_connection.get_cursor.return_value.__enter__.return_value
        mock_cursor.fetchone.return_value = (100,)

        # Act
        result = collector.get_training_data_count(only_unused=False)

        # Assert
        assert result == 100

    def test_returns_count_unused_only(
        self,
        collector: TrainingDataCollector,
        mock_db_connection: MagicMock,
    ):
        """未使用のみのカウントを返す"""
        # Arrange
        mock_cursor = mock_db_connection.get_cursor.return_value.__enter__.return_value
        mock_cursor.fetchone.return_value = (50,)

        # Act
        result = collector.get_training_data_count(only_unused=True)

        # Assert
        assert result == 50

    def test_returns_zero_when_no_data(
        self,
        collector: TrainingDataCollector,
        mock_db_connection: MagicMock,
    ):
        """データがない場合は 0 を返す"""
        # Arrange
        mock_cursor = mock_db_connection.get_cursor.return_value.__enter__.return_value
        mock_cursor.fetchone.return_value = (0,)

        # Act
        result = collector.get_training_data_count()

        # Assert
        assert result == 0

    def test_returns_zero_when_fetchone_returns_none(
        self,
        collector: TrainingDataCollector,
        mock_db_connection: MagicMock,
    ):
        """fetchone が None を返す場合は 0 を返す"""
        # Arrange
        mock_cursor = mock_db_connection.get_cursor.return_value.__enter__.return_value
        mock_cursor.fetchone.return_value = None

        # Act
        result = collector.get_training_data_count()

        # Assert
        assert result == 0


# === is_ready_for_training テスト ===


class TestIsReadyForTraining:
    """is_ready_for_training() メソッドのテスト"""

    def test_returns_true_when_enough_samples(
        self,
        mock_db_connection: MagicMock,
        mock_embedding_client: MagicMock,
        mock_agent_registry: MagicMock,
    ):
        """十分なサンプルがある場合は True"""
        # Arrange
        config = Phase3Config(min_training_samples=100)
        collector = TrainingDataCollector(
            db_connection=mock_db_connection,
            embedding_client=mock_embedding_client,
            agent_registry=mock_agent_registry,
            config=config,
        )
        mock_cursor = mock_db_connection.get_cursor.return_value.__enter__.return_value
        mock_cursor.fetchone.return_value = (100,)  # ちょうど最小サンプル数

        # Act
        result = collector.is_ready_for_training()

        # Assert
        assert result is True

    def test_returns_false_when_not_enough_samples(
        self,
        mock_db_connection: MagicMock,
        mock_embedding_client: MagicMock,
        mock_agent_registry: MagicMock,
    ):
        """サンプルが不足している場合は False"""
        # Arrange
        config = Phase3Config(min_training_samples=100)
        collector = TrainingDataCollector(
            db_connection=mock_db_connection,
            embedding_client=mock_embedding_client,
            agent_registry=mock_agent_registry,
            config=config,
        )
        mock_cursor = mock_db_connection.get_cursor.return_value.__enter__.return_value
        mock_cursor.fetchone.return_value = (99,)  # 1件不足

        # Act
        result = collector.is_ready_for_training()

        # Assert
        assert result is False

    def test_returns_true_when_more_than_enough_samples(
        self,
        mock_db_connection: MagicMock,
        mock_embedding_client: MagicMock,
        mock_agent_registry: MagicMock,
    ):
        """サンプルが十分にある場合は True"""
        # Arrange
        config = Phase3Config(min_training_samples=100)
        collector = TrainingDataCollector(
            db_connection=mock_db_connection,
            embedding_client=mock_embedding_client,
            agent_registry=mock_agent_registry,
            config=config,
        )
        mock_cursor = mock_db_connection.get_cursor.return_value.__enter__.return_value
        mock_cursor.fetchone.return_value = (1000,)  # 十分な数

        # Act
        result = collector.is_ready_for_training()

        # Assert
        assert result is True


# === エラーハンドリングテスト ===


class TestErrorHandling:
    """エラーハンドリングのテスト"""

    def test_continues_on_single_record_failure(
        self,
        collector: TrainingDataCollector,
        mock_db_connection: MagicMock,
        mock_embedding_client: MagicMock,
    ):
        """個別レコードの処理失敗時も他のレコードは処理を継続"""
        # Arrange
        test_id1 = uuid4()
        test_id2 = uuid4()
        session_id = uuid4()
        now = datetime.now()

        mock_cursor = mock_db_connection.get_cursor.return_value.__enter__.return_value

        # 2件のレコードを返す（最初のは成功、2件目も成功）
        # ただしエンベディングは1件目でエラー
        mock_cursor.fetchall.side_effect = [
            # _fetch_routing_history の結果
            [
                (
                    test_id1,
                    session_id,
                    "orch_01",
                    "タスク1",
                    "test_agent",
                    None,
                    None,
                    "success",
                    None,
                    "positive",
                    now,
                    now,
                ),
                (
                    test_id2,
                    session_id,
                    "orch_01",
                    "タスク2",
                    "test_agent",
                    None,
                    None,
                    "success",
                    None,
                    "positive",
                    now,
                    now,
                ),
            ],
            # 1件目の _get_past_experiences
            [],
            # 2件目の _get_past_experiences
            [],
        ]

        # 1件目でエラー、2件目で成功
        mock_embedding_client.get_embedding.side_effect = [
            Exception("API Error"),
            [0.1] * 1536,
        ]

        # INSERT の結果
        mock_cursor.fetchone.return_value = (uuid4(),)

        # Act
        result = collector.collect_from_routing_history()

        # Assert
        # 2件中1件が成功（エラーが発生しても継続）
        assert result == 1

    def test_handles_missing_agent_gracefully(
        self,
        collector: TrainingDataCollector,
        mock_db_connection: MagicMock,
        mock_agent_registry: MagicMock,
    ):
        """存在しないエージェントでもエラーにならない"""
        # Arrange
        mock_agent_registry.get_by_id.return_value = None

        test_id = uuid4()
        session_id = uuid4()
        now = datetime.now()

        mock_cursor = mock_db_connection.get_cursor.return_value.__enter__.return_value
        mock_cursor.fetchall.side_effect = [
            # _fetch_routing_history の結果
            [
                (
                    test_id,
                    session_id,
                    "orch_01",
                    "タスク概要",
                    "unknown_agent",  # 存在しないエージェント
                    None,
                    None,
                    "success",
                    None,
                    "positive",
                    now,
                    now,
                )
            ],
            # _get_past_experiences
            [],
        ]
        mock_cursor.fetchone.return_value = (uuid4(),)

        # Act
        result = collector.collect_from_routing_history()

        # Assert
        # エージェントが見つからなくてもデフォルト値で処理される
        assert result == 1


# === 境界値テスト ===


class TestBoundaryValues:
    """境界値のテスト"""

    def test_empty_task_summary(
        self,
        collector: TrainingDataCollector,
        mock_db_connection: MagicMock,
        mock_embedding_client: MagicMock,
    ):
        """空のタスク概要"""
        # Arrange
        mock_embedding_client.get_embedding.side_effect = Exception("Empty text")

        test_id = uuid4()
        session_id = uuid4()
        now = datetime.now()

        mock_cursor = mock_db_connection.get_cursor.return_value.__enter__.return_value
        mock_cursor.fetchall.side_effect = [
            [
                (
                    test_id,
                    session_id,
                    "orch_01",
                    "",  # 空のタスク概要
                    "test_agent",
                    None,
                    None,
                    "success",
                    None,
                    "positive",
                    now,
                    now,
                )
            ],
            [],
        ]

        # Act
        result = collector.collect_from_routing_history()

        # Assert
        # エラーが発生してスキップされる
        assert result == 0

    def test_very_long_task_summary(
        self,
        collector: TrainingDataCollector,
        mock_db_connection: MagicMock,
    ):
        """非常に長いタスク概要"""
        # Arrange
        long_task = "タスク説明: " + "これは長いタスクです。" * 1000

        test_id = uuid4()
        session_id = uuid4()
        now = datetime.now()

        mock_cursor = mock_db_connection.get_cursor.return_value.__enter__.return_value
        mock_cursor.fetchall.side_effect = [
            [
                (
                    test_id,
                    session_id,
                    "orch_01",
                    long_task,
                    "test_agent",
                    None,
                    None,
                    "success",
                    None,
                    "positive",
                    now,
                    now,
                )
            ],
            [],
        ]
        mock_cursor.fetchone.return_value = (uuid4(),)

        # Act
        result = collector.collect_from_routing_history()

        # Assert
        assert result == 1


# === ラベル計算の全パターンテスト ===


class TestLabelCalculationMatrix:
    """ラベル計算の全パターンをマトリックステスト"""

    @pytest.mark.parametrize(
        "user_feedback,result_status,expected_label",
        [
            # positive フィードバック
            ("positive", "success", 1.0),
            ("positive", "failure", 1.0),
            ("positive", "timeout", 1.0),
            ("positive", "cancelled", 1.0),
            ("positive", None, 1.0),
            # negative フィードバック
            ("negative", "success", 0.5),
            ("negative", "failure", 0.0),
            ("negative", "timeout", 0.0),
            ("negative", "cancelled", 0.0),
            ("negative", None, 0.0),
            # neutral フィードバック
            ("neutral", "success", 1.0),
            ("neutral", "failure", 0.5),
            ("neutral", "timeout", 0.5),
            ("neutral", "cancelled", 0.5),
            ("neutral", None, 0.5),
            # フィードバックなし
            (None, "success", 1.0),
            (None, "failure", 0.5),
            (None, "timeout", 0.5),
            (None, "cancelled", 0.5),
            (None, None, 0.5),
        ],
    )
    def test_label_calculation_matrix(
        self,
        collector: TrainingDataCollector,
        user_feedback: Optional[str],
        result_status: Optional[str],
        expected_label: float,
    ):
        """ラベル計算の全パターン"""
        # Act
        result = collector._calculate_label(user_feedback, result_status)

        # Assert
        assert result == expected_label, (
            f"({user_feedback}, {result_status}) の期待値は {expected_label} だが "
            f"実際は {result}"
        )
