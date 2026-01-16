# ExperimentManager テスト
# 仕様: docs/phase3-implementation-spec.ja.md セクション5.6
"""
ExperimentManagerの単体テスト

検証観点:
- テストカバレッジ: 全メソッド、主要パスを網羅
- 再現性: モック活用で外部依存なし
- 境界値・異常系: エラーケース、エッジケースを検証
- パフォーマンス: 軽量モックで高速実行
- 保守性: テストクラス分割、フィクスチャ活用
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest

from src.ab_testing.experiment_manager import (
    ExperimentManager,
    ExperimentNotFoundError,
    ExperimentResult,
    ExperimentStateError,
    ExperimentStatus,
    VariantStats,
)
from src.config.phase3_config import Phase3Config


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def config():
    """テスト用設定"""
    config = Phase3Config()
    config.ab_testing_enabled = True
    config.min_samples_per_variant = 10  # テスト用に小さい値
    config.significance_threshold = 0.95
    return config


@pytest.fixture
def mock_cursor():
    """モックカーソル"""
    cursor = MagicMock()
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=None)
    return cursor


@pytest.fixture
def mock_db(mock_cursor):
    """モックDB接続"""
    db = MagicMock()
    db.get_cursor = MagicMock(return_value=mock_cursor)
    return db


@pytest.fixture
def manager(mock_db, config):
    """ExperimentManagerインスタンス"""
    return ExperimentManager(mock_db, config)


@pytest.fixture
def sample_experiment_row():
    """サンプル実験データ（DBから返される行形式）"""
    return (
        "123e4567-e89b-12d3-a456-426614174000",  # id
        "similarity_test",  # name
        "similarity_threshold",  # parameter_name
        json.dumps({"control": 0.3, "variant_a": 0.25}),  # variants
        json.dumps([0.5, 0.5]),  # traffic_split
        "draft",  # status
        None,  # results
        None,  # winner_variant
        None,  # statistical_significance
        datetime.now(),  # created_at
        None,  # started_at
        None,  # ended_at
    )


@pytest.fixture
def running_experiment_row():
    """実行中の実験データ"""
    return (
        "123e4567-e89b-12d3-a456-426614174000",
        "similarity_test",
        "similarity_threshold",
        json.dumps({"control": 0.3, "variant_a": 0.25}),
        json.dumps([0.5, 0.5]),
        "running",
        None,
        None,
        None,
        datetime.now(),
        datetime.now(),
        None,
    )


def create_experiment_row(
    exp_id: str = "123e4567-e89b-12d3-a456-426614174000",
    name: str = "test_experiment",
    parameter_name: str = "test_param",
    variants: Dict[str, Any] = None,
    traffic_split: List[float] = None,
    status: str = "draft",
) -> tuple:
    """実験データ行を作成するヘルパー"""
    if variants is None:
        variants = {"control": 0.3, "variant_a": 0.25}
    if traffic_split is None:
        traffic_split = [0.5, 0.5]

    return (
        exp_id,
        name,
        parameter_name,
        json.dumps(variants),
        json.dumps(traffic_split),
        status,
        None,
        None,
        None,
        datetime.now(),
        datetime.now() if status != "draft" else None,
        None,
    )


# ============================================================================
# TestExperimentManagerInit
# ============================================================================


class TestExperimentManagerInit:
    """ExperimentManager初期化テスト"""

    def test_init_with_config(self, mock_db, config):
        """設定指定での初期化"""
        manager = ExperimentManager(mock_db, config)
        assert manager.db == mock_db
        assert manager.config == config

    def test_init_without_config(self, mock_db):
        """設定なしでの初期化（デフォルト設定使用）"""
        manager = ExperimentManager(mock_db)
        assert manager.db == mock_db
        assert isinstance(manager.config, Phase3Config)


# ============================================================================
# TestCreateExperiment
# ============================================================================


class TestCreateExperiment:
    """create_experiment メソッドのテスト"""

    @pytest.mark.asyncio
    async def test_create_experiment_success(self, manager, mock_cursor):
        """実験作成成功"""
        variants = {"control": 0.3, "variant_a": 0.25}

        exp_id = await manager.create_experiment(
            name="test_experiment",
            parameter_name="similarity_threshold",
            variants=variants,
        )

        assert isinstance(exp_id, UUID)
        mock_cursor.execute.assert_called_once()
        call_args = mock_cursor.execute.call_args[0]
        assert "INSERT INTO ab_experiments" in call_args[0]

    @pytest.mark.asyncio
    async def test_create_experiment_with_traffic_split(self, manager, mock_cursor):
        """トラフィック配分指定での実験作成"""
        variants = {"control": 0.3, "variant_a": 0.25, "variant_b": 0.35}
        traffic_split = [0.5, 0.3, 0.2]

        exp_id = await manager.create_experiment(
            name="test_experiment",
            parameter_name="similarity_threshold",
            variants=variants,
            traffic_split=traffic_split,
        )

        assert isinstance(exp_id, UUID)
        call_args = mock_cursor.execute.call_args[0]
        assert json.dumps(traffic_split) in str(call_args[1])

    @pytest.mark.asyncio
    async def test_create_experiment_empty_variants_error(self, manager):
        """空バリアントでエラー"""
        with pytest.raises(ValueError, match="variants cannot be empty"):
            await manager.create_experiment(
                name="test",
                parameter_name="param",
                variants={},
            )

    @pytest.mark.asyncio
    async def test_create_experiment_mismatched_traffic_split_error(self, manager):
        """トラフィック配分とバリアント数の不一致でエラー"""
        with pytest.raises(ValueError, match="traffic_split length"):
            await manager.create_experiment(
                name="test",
                parameter_name="param",
                variants={"control": 0.3, "variant_a": 0.25},
                traffic_split=[0.5],  # 2バリアントに対して1つ
            )

    @pytest.mark.asyncio
    async def test_create_experiment_invalid_traffic_split_sum_error(self, manager):
        """トラフィック配分の合計が1でない場合にエラー"""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            await manager.create_experiment(
                name="test",
                parameter_name="param",
                variants={"control": 0.3, "variant_a": 0.25},
                traffic_split=[0.5, 0.3],  # 合計0.8
            )


# ============================================================================
# TestStartExperiment
# ============================================================================


class TestStartExperiment:
    """start_experiment メソッドのテスト"""

    @pytest.mark.asyncio
    async def test_start_experiment_success(
        self, manager, mock_cursor, sample_experiment_row
    ):
        """実験開始成功"""
        mock_cursor.fetchone.return_value = sample_experiment_row
        exp_id = UUID("123e4567-e89b-12d3-a456-426614174000")

        await manager.start_experiment(exp_id)

        # UPDATE が呼ばれたことを確認
        calls = mock_cursor.execute.call_args_list
        assert any("UPDATE ab_experiments" in str(call) for call in calls)

    @pytest.mark.asyncio
    async def test_start_experiment_not_found_error(self, manager, mock_cursor):
        """実験が見つからない場合にエラー"""
        mock_cursor.fetchone.return_value = None
        exp_id = uuid4()

        with pytest.raises(ExperimentNotFoundError):
            await manager.start_experiment(exp_id)

    @pytest.mark.asyncio
    async def test_start_experiment_invalid_state_error(
        self, manager, mock_cursor, running_experiment_row
    ):
        """実行中の実験を開始しようとするとエラー"""
        mock_cursor.fetchone.return_value = running_experiment_row
        exp_id = UUID("123e4567-e89b-12d3-a456-426614174000")

        with pytest.raises(ExperimentStateError, match="running"):
            await manager.start_experiment(exp_id)


# ============================================================================
# TestPauseExperiment
# ============================================================================


class TestPauseExperiment:
    """pause_experiment メソッドのテスト"""

    @pytest.mark.asyncio
    async def test_pause_experiment_success(
        self, manager, mock_cursor, running_experiment_row
    ):
        """実験一時停止成功"""
        mock_cursor.fetchone.return_value = running_experiment_row
        exp_id = UUID("123e4567-e89b-12d3-a456-426614174000")

        await manager.pause_experiment(exp_id)

        calls = mock_cursor.execute.call_args_list
        assert any("UPDATE ab_experiments" in str(call) for call in calls)

    @pytest.mark.asyncio
    async def test_pause_experiment_invalid_state_error(
        self, manager, mock_cursor, sample_experiment_row
    ):
        """draft状態の実験を一時停止しようとするとエラー"""
        mock_cursor.fetchone.return_value = sample_experiment_row
        exp_id = UUID("123e4567-e89b-12d3-a456-426614174000")

        with pytest.raises(ExperimentStateError, match="draft"):
            await manager.pause_experiment(exp_id)


# ============================================================================
# TestResumeExperiment
# ============================================================================


class TestResumeExperiment:
    """resume_experiment メソッドのテスト"""

    @pytest.mark.asyncio
    async def test_resume_experiment_success(self, manager, mock_cursor):
        """一時停止中の実験を再開"""
        paused_row = create_experiment_row(status="paused")
        mock_cursor.fetchone.return_value = paused_row
        exp_id = UUID("123e4567-e89b-12d3-a456-426614174000")

        await manager.resume_experiment(exp_id)

        calls = mock_cursor.execute.call_args_list
        assert any("UPDATE ab_experiments" in str(call) for call in calls)

    @pytest.mark.asyncio
    async def test_resume_experiment_invalid_state_error(
        self, manager, mock_cursor, sample_experiment_row
    ):
        """draft状態の実験を再開しようとするとエラー"""
        mock_cursor.fetchone.return_value = sample_experiment_row
        exp_id = UUID("123e4567-e89b-12d3-a456-426614174000")

        with pytest.raises(ExperimentStateError, match="draft"):
            await manager.resume_experiment(exp_id)


# ============================================================================
# TestGetVariant
# ============================================================================


class TestGetVariant:
    """get_variant メソッドのテスト"""

    @pytest.mark.asyncio
    async def test_get_variant_running_experiment(
        self, manager, mock_cursor, running_experiment_row
    ):
        """実行中の実験でバリアント取得"""
        mock_cursor.fetchone.return_value = running_experiment_row
        exp_id = UUID("123e4567-e89b-12d3-a456-426614174000")
        session_id = uuid4()

        variant_id, value = await manager.get_variant(exp_id, session_id)

        assert variant_id in ["control", "variant_a"]
        assert value in [0.3, 0.25]

    @pytest.mark.asyncio
    async def test_get_variant_draft_experiment_returns_control(
        self, manager, mock_cursor, sample_experiment_row
    ):
        """draft状態の実験ではcontrolを返す"""
        mock_cursor.fetchone.return_value = sample_experiment_row
        exp_id = UUID("123e4567-e89b-12d3-a456-426614174000")
        session_id = uuid4()

        variant_id, value = await manager.get_variant(exp_id, session_id)

        assert variant_id == "control"
        assert value == 0.3

    @pytest.mark.asyncio
    async def test_get_variant_deterministic(
        self, manager, mock_cursor, running_experiment_row
    ):
        """同じsession_idは同じバリアントを返す（決定論的）"""
        mock_cursor.fetchone.return_value = running_experiment_row
        exp_id = UUID("123e4567-e89b-12d3-a456-426614174000")
        session_id = uuid4()

        # 同じsession_idで複数回呼び出し
        results = []
        for _ in range(5):
            variant_id, value = await manager.get_variant(exp_id, session_id)
            results.append((variant_id, value))

        # すべて同じ結果であることを確認
        assert all(r == results[0] for r in results)

    @pytest.mark.asyncio
    async def test_get_variant_different_sessions_distribute(
        self, manager, mock_cursor, running_experiment_row
    ):
        """異なるsession_idは異なるバリアントに分散する"""
        mock_cursor.fetchone.return_value = running_experiment_row
        exp_id = UUID("123e4567-e89b-12d3-a456-426614174000")

        # 多数のセッションでテスト
        variant_counts = {"control": 0, "variant_a": 0}
        for _ in range(100):
            session_id = uuid4()
            variant_id, _ = await manager.get_variant(exp_id, session_id)
            variant_counts[variant_id] += 1

        # 両方のバリアントに分散していることを確認（完全均等ではないが偏りすぎていない）
        assert variant_counts["control"] > 20
        assert variant_counts["variant_a"] > 20


# ============================================================================
# TestRecordMetric
# ============================================================================


class TestRecordMetric:
    """record_metric メソッドのテスト"""

    @pytest.mark.asyncio
    async def test_record_metric_success(
        self, manager, mock_cursor, running_experiment_row
    ):
        """メトリクス記録成功"""
        mock_cursor.fetchone.return_value = running_experiment_row
        exp_id = UUID("123e4567-e89b-12d3-a456-426614174000")
        session_id = uuid4()

        await manager.record_metric(
            experiment_id=exp_id,
            session_id=session_id,
            variant_id="control",
            metric_name="success_rate",
            metric_value=0.85,
        )

        calls = mock_cursor.execute.call_args_list
        assert any("INSERT INTO ab_experiment_logs" in str(call) for call in calls)

    @pytest.mark.asyncio
    async def test_record_metric_invalid_variant_error(
        self, manager, mock_cursor, running_experiment_row
    ):
        """無効なバリアントIDでエラー"""
        mock_cursor.fetchone.return_value = running_experiment_row
        exp_id = UUID("123e4567-e89b-12d3-a456-426614174000")

        with pytest.raises(ValueError, match="Invalid variant_id"):
            await manager.record_metric(
                experiment_id=exp_id,
                session_id=uuid4(),
                variant_id="nonexistent_variant",
                metric_name="success_rate",
                metric_value=0.85,
            )


# ============================================================================
# TestAnalyzeResults
# ============================================================================


class TestAnalyzeResults:
    """analyze_results メソッドのテスト"""

    @pytest.mark.asyncio
    async def test_analyze_results_no_data(
        self, manager, mock_cursor, running_experiment_row
    ):
        """データなしの場合"""
        mock_cursor.fetchone.side_effect = [
            running_experiment_row,  # _get_experiment
            None,  # _get_variant_metrics (no metric name)
        ]
        mock_cursor.fetchall.return_value = []
        exp_id = UUID("123e4567-e89b-12d3-a456-426614174000")

        result = await manager.analyze_results(exp_id)

        assert isinstance(result, ExperimentResult)
        assert result.is_significant is False
        assert result.winner_variant is None

    @pytest.mark.asyncio
    async def test_analyze_results_with_data(self, manager, mock_cursor):
        """データありの場合"""
        running_row = create_experiment_row(status="running")

        # メトリクスデータ: control が明らかに高い
        control_values = [(0.9,), (0.88,), (0.92,), (0.91,), (0.89,),
                          (0.90,), (0.87,), (0.93,), (0.88,), (0.91,)]
        variant_values = [(0.75,), (0.73,), (0.72,), (0.74,), (0.76,),
                          (0.71,), (0.77,), (0.73,), (0.75,), (0.74,)]

        # モックの設定
        call_count = [0]

        def mock_fetchone():
            call_count[0] += 1
            if call_count[0] == 1:
                return running_row  # _get_experiment
            elif call_count[0] == 2:
                return ("success_rate",)  # metric name detection
            return None

        def mock_fetchall():
            # バリアントごとのメトリクス
            return [
                ("control", 0.9), ("control", 0.88), ("control", 0.92),
                ("control", 0.91), ("control", 0.89), ("control", 0.90),
                ("control", 0.87), ("control", 0.93), ("control", 0.88),
                ("control", 0.91),
                ("variant_a", 0.75), ("variant_a", 0.73), ("variant_a", 0.72),
                ("variant_a", 0.74), ("variant_a", 0.76), ("variant_a", 0.71),
                ("variant_a", 0.77), ("variant_a", 0.73), ("variant_a", 0.75),
                ("variant_a", 0.74),
            ]

        mock_cursor.fetchone.side_effect = mock_fetchone
        mock_cursor.fetchall.return_value = mock_fetchall()

        exp_id = UUID("123e4567-e89b-12d3-a456-426614174000")
        result = await manager.analyze_results(exp_id)

        assert isinstance(result, ExperimentResult)
        assert result.variant_stats["control"].sample_count == 10
        assert result.variant_stats["variant_a"].sample_count == 10
        assert result.variant_stats["control"].mean > result.variant_stats["variant_a"].mean

    @pytest.mark.asyncio
    async def test_analyze_results_significant(self, manager, mock_cursor, config):
        """統計的に有意な差がある場合"""
        # min_samples_per_variant を小さくしてテスト
        config.min_samples_per_variant = 5
        manager.config = config

        running_row = create_experiment_row(status="running")

        # 明確に異なるデータ（control が高い）
        call_count = [0]

        def mock_fetchone():
            call_count[0] += 1
            if call_count[0] == 1:
                return running_row
            elif call_count[0] == 2:
                return ("success_rate",)
            return None

        mock_cursor.fetchone.side_effect = mock_fetchone
        mock_cursor.fetchall.return_value = [
            ("control", 0.95), ("control", 0.93), ("control", 0.94),
            ("control", 0.96), ("control", 0.92),
            ("variant_a", 0.60), ("variant_a", 0.58), ("variant_a", 0.62),
            ("variant_a", 0.59), ("variant_a", 0.61),
        ]

        exp_id = UUID("123e4567-e89b-12d3-a456-426614174000")
        result = await manager.analyze_results(exp_id)

        assert result.is_significant is True
        assert result.winner_variant == "control"
        assert result.p_value is not None
        assert result.p_value < 0.05


# ============================================================================
# TestCompleteExperiment
# ============================================================================


class TestCompleteExperiment:
    """complete_experiment メソッドのテスト"""

    @pytest.mark.asyncio
    async def test_complete_experiment_success(self, manager, mock_cursor, config):
        """実験完了成功"""
        config.min_samples_per_variant = 5
        manager.config = config

        running_row = create_experiment_row(status="running")
        call_count = [0]

        def mock_fetchone():
            call_count[0] += 1
            if call_count[0] in [1, 2]:  # _get_experiment calls
                return running_row
            elif call_count[0] == 3:
                return ("success_rate",)
            return None

        mock_cursor.fetchone.side_effect = mock_fetchone
        mock_cursor.fetchall.return_value = [
            ("control", 0.8), ("control", 0.82), ("control", 0.78),
            ("control", 0.81), ("control", 0.79),
            ("variant_a", 0.7), ("variant_a", 0.72), ("variant_a", 0.68),
            ("variant_a", 0.71), ("variant_a", 0.69),
        ]

        exp_id = UUID("123e4567-e89b-12d3-a456-426614174000")
        await manager.complete_experiment(exp_id, winner_variant="control")

        calls = mock_cursor.execute.call_args_list
        assert any(
            "UPDATE ab_experiments" in str(call) and "completed" in str(call).lower()
            for call in calls
        )

    @pytest.mark.asyncio
    async def test_complete_experiment_invalid_state_error(
        self, manager, mock_cursor, sample_experiment_row
    ):
        """draft状態の実験を完了しようとするとエラー"""
        mock_cursor.fetchone.return_value = sample_experiment_row
        exp_id = UUID("123e4567-e89b-12d3-a456-426614174000")

        with pytest.raises(ExperimentStateError, match="draft"):
            await manager.complete_experiment(exp_id)

    @pytest.mark.asyncio
    async def test_complete_experiment_invalid_winner_error(
        self, manager, mock_cursor
    ):
        """無効な勝者バリアントでエラー"""
        running_row = create_experiment_row(status="running")
        mock_cursor.fetchone.return_value = running_row
        exp_id = UUID("123e4567-e89b-12d3-a456-426614174000")

        with pytest.raises(ValueError, match="Invalid winner_variant"):
            await manager.complete_experiment(exp_id, winner_variant="nonexistent")


# ============================================================================
# TestListExperiments
# ============================================================================


class TestListExperiments:
    """list_experiments メソッドのテスト"""

    @pytest.mark.asyncio
    async def test_list_experiments_all(self, manager, mock_cursor):
        """全実験を取得"""
        rows = [
            create_experiment_row(exp_id="111", name="exp1", status="running"),
            create_experiment_row(exp_id="222", name="exp2", status="draft"),
            create_experiment_row(exp_id="333", name="exp3", status="completed"),
        ]
        mock_cursor.fetchall.return_value = rows

        experiments = await manager.list_experiments()

        assert len(experiments) == 3
        mock_cursor.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_experiments_by_status(self, manager, mock_cursor):
        """ステータスでフィルタ"""
        rows = [
            create_experiment_row(exp_id="111", name="exp1", status="running"),
        ]
        mock_cursor.fetchall.return_value = rows

        experiments = await manager.list_experiments(status=ExperimentStatus.RUNNING)

        assert len(experiments) == 1
        call_args = mock_cursor.execute.call_args[0]
        assert "status = %s" in call_args[0]


# ============================================================================
# TestSelectVariant
# ============================================================================


class TestSelectVariant:
    """_select_variant メソッドのテスト（内部メソッド）"""

    def test_select_variant_deterministic(self, manager):
        """決定論的なバリアント選択"""
        session_id = UUID("123e4567-e89b-12d3-a456-426614174000")
        variants = {"control": 0.3, "variant_a": 0.25}
        traffic_split = [0.5, 0.5]

        # 同じ入力は同じ結果
        results = [
            manager._select_variant(session_id, variants, traffic_split)
            for _ in range(10)
        ]
        assert all(r == results[0] for r in results)

    def test_select_variant_distribution(self, manager):
        """バリアント分布がトラフィック配分に近い"""
        variants = {"control": 0.3, "variant_a": 0.25}
        traffic_split = [0.7, 0.3]  # 70% control, 30% variant_a

        counts = {"control": 0, "variant_a": 0}
        for _ in range(1000):
            session_id = uuid4()
            variant = manager._select_variant(session_id, variants, traffic_split)
            counts[variant] += 1

        # おおよそ 70/30 に分布（誤差許容）
        control_ratio = counts["control"] / 1000
        assert 0.6 < control_ratio < 0.8

    def test_select_variant_three_variants(self, manager):
        """3バリアントの場合"""
        variants = {"control": 0.3, "variant_a": 0.25, "variant_b": 0.35}
        traffic_split = [0.34, 0.33, 0.33]

        counts = {"control": 0, "variant_a": 0, "variant_b": 0}
        for _ in range(1000):
            session_id = uuid4()
            variant = manager._select_variant(session_id, variants, traffic_split)
            counts[variant] += 1

        # すべてのバリアントが選択されている
        assert all(c > 200 for c in counts.values())


# ============================================================================
# TestCalculateStd
# ============================================================================


class TestCalculateStd:
    """_calculate_std メソッドのテスト"""

    def test_calculate_std_empty(self, manager):
        """空リストの標準偏差"""
        assert manager._calculate_std([]) == 0.0

    def test_calculate_std_single(self, manager):
        """単一要素の標準偏差"""
        assert manager._calculate_std([5.0]) == 0.0

    def test_calculate_std_identical(self, manager):
        """同じ値のリストの標準偏差"""
        assert manager._calculate_std([5.0, 5.0, 5.0]) == 0.0

    def test_calculate_std_varied(self, manager):
        """異なる値のリストの標準偏差"""
        values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        std = manager._calculate_std(values)
        # 標本標準偏差（N-1で除算）: 約2.14
        assert 2.0 < std < 2.3


# ============================================================================
# TestAnalyzeSignificance
# ============================================================================


class TestAnalyzeSignificance:
    """_analyze_significance メソッドのテスト"""

    def test_analyze_significance_insufficient_data(self, manager):
        """データ不足の場合"""
        variant_stats = {
            "control": VariantStats(
                variant_id="control",
                sample_count=1,
                mean=0.8,
                std=0.0,
                min_value=0.8,
                max_value=0.8,
                values=[0.8],
            ),
        }

        p_value, is_sig, winner, conf = manager._analyze_significance(variant_stats)
        assert p_value is None
        assert is_sig is False
        assert winner is None

    def test_analyze_significance_two_variants(self, manager, config):
        """2バリアントの有意性分析"""
        config.min_samples_per_variant = 5
        manager.config = config

        control_values = [0.9, 0.92, 0.88, 0.91, 0.89]
        variant_values = [0.6, 0.58, 0.62, 0.59, 0.61]

        variant_stats = {
            "control": VariantStats(
                variant_id="control",
                sample_count=5,
                mean=sum(control_values) / 5,
                std=manager._calculate_std(control_values),
                min_value=min(control_values),
                max_value=max(control_values),
                values=control_values,
            ),
            "variant_a": VariantStats(
                variant_id="variant_a",
                sample_count=5,
                mean=sum(variant_values) / 5,
                std=manager._calculate_std(variant_values),
                min_value=min(variant_values),
                max_value=max(variant_values),
                values=variant_values,
            ),
        }

        p_value, is_sig, winner, conf = manager._analyze_significance(variant_stats)

        assert p_value is not None
        assert p_value < 0.05  # 明確な差がある
        assert is_sig is True
        assert winner == "control"


# ============================================================================
# TestGenerateRecommendation
# ============================================================================


class TestGenerateRecommendation:
    """_generate_recommendation メソッドのテスト"""

    def test_generate_recommendation_no_data(self, manager):
        """データなしの場合"""
        variant_stats = {}
        rec = manager._generate_recommendation(variant_stats, False, None)
        assert "No data collected" in rec

    def test_generate_recommendation_insufficient_samples(self, manager, config):
        """サンプル不足の場合"""
        config.min_samples_per_variant = 100
        manager.config = config

        variant_stats = {
            "control": VariantStats(
                variant_id="control",
                sample_count=50,
                mean=0.8,
                std=0.1,
                min_value=0.6,
                max_value=1.0,
                values=[0.8] * 50,
            ),
        }

        rec = manager._generate_recommendation(variant_stats, False, None)
        assert "Insufficient samples" in rec

    def test_generate_recommendation_significant(self, manager, config):
        """有意差ありの場合"""
        config.min_samples_per_variant = 10
        manager.config = config

        variant_stats = {
            "control": VariantStats(
                variant_id="control",
                sample_count=100,
                mean=0.85,
                std=0.1,
                min_value=0.6,
                max_value=1.0,
                values=[0.85] * 100,
            ),
        }

        rec = manager._generate_recommendation(variant_stats, True, "control")
        assert "Recommended" in rec
        assert "control" in rec

    def test_generate_recommendation_not_significant(self, manager, config):
        """有意差なしの場合"""
        config.min_samples_per_variant = 10
        manager.config = config

        variant_stats = {
            "control": VariantStats(
                variant_id="control",
                sample_count=100,
                mean=0.85,
                std=0.1,
                min_value=0.6,
                max_value=1.0,
                values=[0.85] * 100,
            ),
            "variant_a": VariantStats(
                variant_id="variant_a",
                sample_count=100,
                mean=0.84,
                std=0.1,
                min_value=0.6,
                max_value=1.0,
                values=[0.84] * 100,
            ),
        }

        rec = manager._generate_recommendation(variant_stats, False, None)
        assert "No statistically significant" in rec


# ============================================================================
# TestEdgeCases
# ============================================================================


class TestEdgeCases:
    """エッジケーステスト"""

    @pytest.mark.asyncio
    async def test_create_experiment_single_variant(self, manager, mock_cursor):
        """単一バリアントの実験（有効だが意味は薄い）"""
        exp_id = await manager.create_experiment(
            name="single_variant",
            parameter_name="param",
            variants={"control": 0.5},
            traffic_split=[1.0],
        )
        assert isinstance(exp_id, UUID)

    @pytest.mark.asyncio
    async def test_get_variant_without_control(self, manager, mock_cursor):
        """controlがないバリアント"""
        row = create_experiment_row(
            variants={"variant_a": 0.3, "variant_b": 0.4},
            traffic_split=[0.5, 0.5],
            status="draft",
        )
        mock_cursor.fetchone.return_value = row
        exp_id = UUID("123e4567-e89b-12d3-a456-426614174000")

        # draft状態では最初のバリアントを返す
        variant_id, value = await manager.get_variant(exp_id, uuid4())
        assert variant_id == "variant_a"

    @pytest.mark.asyncio
    async def test_analyze_results_empty_variant(self, manager, mock_cursor, config):
        """一方のバリアントにデータがない場合"""
        config.min_samples_per_variant = 5
        manager.config = config

        running_row = create_experiment_row(status="running")
        call_count = [0]

        def mock_fetchone():
            call_count[0] += 1
            if call_count[0] == 1:
                return running_row
            elif call_count[0] == 2:
                return ("success_rate",)
            return None

        mock_cursor.fetchone.side_effect = mock_fetchone
        # controlのみデータがある
        mock_cursor.fetchall.return_value = [
            ("control", 0.8), ("control", 0.82), ("control", 0.78),
        ]

        exp_id = UUID("123e4567-e89b-12d3-a456-426614174000")
        result = await manager.analyze_results(exp_id)

        assert result.variant_stats["control"].sample_count == 3
        assert result.variant_stats["variant_a"].sample_count == 0
        assert result.is_significant is False


# ============================================================================
# TestVariantStats
# ============================================================================


class TestVariantStats:
    """VariantStats dataclass のテスト"""

    def test_variant_stats_creation(self):
        """VariantStats の作成"""
        vs = VariantStats(
            variant_id="control",
            sample_count=100,
            mean=0.85,
            std=0.05,
            min_value=0.75,
            max_value=0.95,
            values=[0.85] * 100,
        )

        assert vs.variant_id == "control"
        assert vs.sample_count == 100
        assert vs.mean == 0.85
        assert vs.std == 0.05
        assert len(vs.values) == 100

    def test_variant_stats_default_values(self):
        """デフォルト値"""
        vs = VariantStats(
            variant_id="test",
            sample_count=0,
            mean=0.0,
            std=0.0,
            min_value=0.0,
            max_value=0.0,
        )
        assert vs.values == []


# ============================================================================
# TestExperimentResult
# ============================================================================


class TestExperimentResult:
    """ExperimentResult dataclass のテスト"""

    def test_experiment_result_creation(self):
        """ExperimentResult の作成"""
        vs = VariantStats(
            variant_id="control",
            sample_count=100,
            mean=0.85,
            std=0.05,
            min_value=0.75,
            max_value=0.95,
        )

        result = ExperimentResult(
            experiment_id=uuid4(),
            experiment_name="test_exp",
            parameter_name="param",
            status=ExperimentStatus.RUNNING,
            variant_stats={"control": vs},
            p_value=0.03,
            is_significant=True,
            winner_variant="control",
            confidence_level=0.97,
            recommendation="Adopt control",
        )

        assert result.experiment_name == "test_exp"
        assert result.is_significant is True
        assert result.winner_variant == "control"


# ============================================================================
# TestExperimentStatus
# ============================================================================


class TestExperimentStatus:
    """ExperimentStatus enum のテスト"""

    def test_status_values(self):
        """ステータス値"""
        assert ExperimentStatus.DRAFT.value == "draft"
        assert ExperimentStatus.RUNNING.value == "running"
        assert ExperimentStatus.PAUSED.value == "paused"
        assert ExperimentStatus.COMPLETED.value == "completed"

    def test_status_from_string(self):
        """文字列からの変換"""
        assert ExperimentStatus("draft") == ExperimentStatus.DRAFT
        assert ExperimentStatus("running") == ExperimentStatus.RUNNING
