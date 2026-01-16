# A/Bテスト実験管理
# 仕様: docs/phase3-implementation-spec.ja.md セクション5.6
"""
ExperimentManager: A/Bテスト実験の作成・実行・分析を行うクラス

設計方針:
- 実験ライフサイクル: draft → running → paused → completed
- 決定論的バリアント割り当て: session_id のハッシュ値でバリアント選択
- 統計分析: scipy.stats.ttest_ind による t検定
"""

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

from scipy import stats

from src.config.phase3_config import Phase3Config
from src.db.connection import DatabaseConnection


class ExperimentStatus(str, Enum):
    """実験のステータス"""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"


class ExperimentNotFoundError(Exception):
    """実験が見つからない場合のエラー"""
    pass


class ExperimentStateError(Exception):
    """実験の状態が不正な場合のエラー"""
    pass


@dataclass
class VariantStats:
    """バリアントの統計情報"""
    variant_id: str
    sample_count: int
    mean: float
    std: float
    min_value: float
    max_value: float
    values: List[float] = field(default_factory=list)


@dataclass
class ExperimentResult:
    """実験結果"""
    experiment_id: UUID
    experiment_name: str
    parameter_name: str
    status: ExperimentStatus
    variant_stats: Dict[str, VariantStats]
    p_value: Optional[float] = None
    is_significant: bool = False
    winner_variant: Optional[str] = None
    confidence_level: float = 0.0
    recommendation: str = ""


class ExperimentManager:
    """A/Bテスト実験管理クラス

    A/Bテスト実験の作成・実行・メトリクス収集・統計分析を行う。

    使用例:
        db = DatabaseConnection()
        config = Phase3Config()
        manager = ExperimentManager(db, config)

        # 実験作成
        exp_id = await manager.create_experiment(
            name="similarity_threshold_test",
            parameter_name="similarity_threshold",
            variants={"control": 0.3, "variant_a": 0.25, "variant_b": 0.35},
        )

        # 実験開始
        await manager.start_experiment(exp_id)

        # バリアント取得
        variant_id, value = await manager.get_variant(exp_id, session_id)

        # メトリクス記録
        await manager.record_metric(exp_id, session_id, variant_id, "success_rate", 0.85)

        # 結果分析
        result = await manager.analyze_results(exp_id)

        # 実験完了
        await manager.complete_experiment(exp_id, winner_variant="variant_a")

    Attributes:
        db: データベース接続
        config: Phase 3 設定
    """

    def __init__(
        self,
        db_connection: DatabaseConnection,
        config: Optional[Phase3Config] = None,
    ):
        """ExperimentManagerを初期化

        Args:
            db_connection: データベース接続インスタンス
            config: Phase 3 設定。Noneの場合はデフォルト設定を使用。
        """
        self.db = db_connection
        self.config = config or Phase3Config()

    async def create_experiment(
        self,
        name: str,
        parameter_name: str,
        variants: Dict[str, Any],
        traffic_split: Optional[List[float]] = None,
    ) -> UUID:
        """実験を作成

        Args:
            name: 実験名
            parameter_name: 対象パラメータ名
            variants: バリアント定義 {"control": value1, "variant_a": value2, ...}
            traffic_split: トラフィック配分 [0.5, 0.5] など。Noneの場合は均等分割。

        Returns:
            作成された実験のID

        Raises:
            ValueError: バリアントが空の場合、またはトラフィック配分の合計が1でない場合
        """
        if not variants:
            raise ValueError("variants cannot be empty")

        # デフォルトは均等分割
        if traffic_split is None:
            n = len(variants)
            traffic_split = [1.0 / n] * n

        # トラフィック配分の検証
        if len(traffic_split) != len(variants):
            raise ValueError(
                f"traffic_split length ({len(traffic_split)}) must match "
                f"variants length ({len(variants)})"
            )

        total_split = sum(traffic_split)
        if not (0.99 <= total_split <= 1.01):  # 浮動小数点誤差を考慮
            raise ValueError(f"traffic_split must sum to 1.0, got {total_split}")

        experiment_id = uuid4()

        def _insert():
            with self.db.get_cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO ab_experiments
                    (id, name, parameter_name, variants, traffic_split, status)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        str(experiment_id),
                        name,
                        parameter_name,
                        json.dumps(variants),
                        json.dumps(traffic_split),
                        ExperimentStatus.DRAFT.value,
                    ),
                )

        await asyncio.to_thread(_insert)
        return experiment_id

    async def start_experiment(self, experiment_id: UUID) -> None:
        """実験を開始

        Args:
            experiment_id: 実験ID

        Raises:
            ExperimentNotFoundError: 実験が見つからない場合
            ExperimentStateError: 実験が draft 状態でない場合
        """
        experiment = await self._get_experiment(experiment_id)

        if experiment["status"] != ExperimentStatus.DRAFT.value:
            raise ExperimentStateError(
                f"Cannot start experiment in '{experiment['status']}' status. "
                f"Only 'draft' experiments can be started."
            )

        def _update():
            with self.db.get_cursor() as cur:
                cur.execute(
                    """
                    UPDATE ab_experiments
                    SET status = %s, started_at = NOW()
                    WHERE id = %s
                    """,
                    (ExperimentStatus.RUNNING.value, str(experiment_id)),
                )

        await asyncio.to_thread(_update)

    async def pause_experiment(self, experiment_id: UUID) -> None:
        """実験を一時停止

        Args:
            experiment_id: 実験ID

        Raises:
            ExperimentNotFoundError: 実験が見つからない場合
            ExperimentStateError: 実験が running 状態でない場合
        """
        experiment = await self._get_experiment(experiment_id)

        if experiment["status"] != ExperimentStatus.RUNNING.value:
            raise ExperimentStateError(
                f"Cannot pause experiment in '{experiment['status']}' status. "
                f"Only 'running' experiments can be paused."
            )

        def _update():
            with self.db.get_cursor() as cur:
                cur.execute(
                    """
                    UPDATE ab_experiments
                    SET status = %s
                    WHERE id = %s
                    """,
                    (ExperimentStatus.PAUSED.value, str(experiment_id)),
                )

        await asyncio.to_thread(_update)

    async def resume_experiment(self, experiment_id: UUID) -> None:
        """一時停止中の実験を再開

        Args:
            experiment_id: 実験ID

        Raises:
            ExperimentNotFoundError: 実験が見つからない場合
            ExperimentStateError: 実験が paused 状態でない場合
        """
        experiment = await self._get_experiment(experiment_id)

        if experiment["status"] != ExperimentStatus.PAUSED.value:
            raise ExperimentStateError(
                f"Cannot resume experiment in '{experiment['status']}' status. "
                f"Only 'paused' experiments can be resumed."
            )

        def _update():
            with self.db.get_cursor() as cur:
                cur.execute(
                    """
                    UPDATE ab_experiments
                    SET status = %s
                    WHERE id = %s
                    """,
                    (ExperimentStatus.RUNNING.value, str(experiment_id)),
                )

        await asyncio.to_thread(_update)

    async def get_variant(
        self,
        experiment_id: UUID,
        session_id: UUID,
    ) -> Tuple[str, Any]:
        """セッションに割り当てるバリアントを取得

        session_id のハッシュ値を使って決定論的にバリアントを選択する。
        同じ session_id は常に同じバリアントに割り当てられる。

        Args:
            experiment_id: 実験ID
            session_id: セッションID

        Returns:
            (variant_id, variant_value) のタプル

        Raises:
            ExperimentNotFoundError: 実験が見つからない場合
        """
        experiment = await self._get_experiment(experiment_id)

        variants = experiment["variants"]
        traffic_split = experiment["traffic_split"]

        # 実験が running でない場合は control を返す
        if experiment["status"] != ExperimentStatus.RUNNING.value:
            variant_ids = list(variants.keys())
            # "control" があればそれを、なければ最初のバリアントを返す
            default_variant = "control" if "control" in variants else variant_ids[0]
            return default_variant, variants[default_variant]

        # 決定論的にバリアントを選択
        variant_id = self._select_variant(session_id, variants, traffic_split)
        return variant_id, variants[variant_id]

    async def record_metric(
        self,
        experiment_id: UUID,
        session_id: UUID,
        variant_id: str,
        metric_name: str,
        metric_value: float,
    ) -> None:
        """メトリクスを記録

        Args:
            experiment_id: 実験ID
            session_id: セッションID
            variant_id: バリアントID
            metric_name: メトリクス名
            metric_value: メトリクス値

        Raises:
            ExperimentNotFoundError: 実験が見つからない場合
            ValueError: バリアントIDが無効な場合
        """
        experiment = await self._get_experiment(experiment_id)

        if variant_id not in experiment["variants"]:
            raise ValueError(
                f"Invalid variant_id '{variant_id}'. "
                f"Valid variants: {list(experiment['variants'].keys())}"
            )

        def _insert():
            with self.db.get_cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO ab_experiment_logs
                    (experiment_id, variant_id, session_id, metric_name, metric_value)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        str(experiment_id),
                        variant_id,
                        str(session_id),
                        metric_name,
                        metric_value,
                    ),
                )

        await asyncio.to_thread(_insert)

    async def analyze_results(
        self,
        experiment_id: UUID,
        metric_name: Optional[str] = None,
    ) -> ExperimentResult:
        """実験結果を分析

        各バリアントの平均・標準偏差を計算し、t検定で統計的有意性を判定する。

        Args:
            experiment_id: 実験ID
            metric_name: 分析するメトリクス名。Noneの場合は最初に記録されたメトリクスを使用。

        Returns:
            ExperimentResult: 実験結果

        Raises:
            ExperimentNotFoundError: 実験が見つからない場合
        """
        experiment = await self._get_experiment(experiment_id)
        variants = experiment["variants"]

        # メトリクスを取得
        variant_metrics = await self._get_variant_metrics(
            experiment_id, metric_name
        )

        # 各バリアントの統計を計算
        variant_stats: Dict[str, VariantStats] = {}
        for variant_id in variants.keys():
            values = variant_metrics.get(variant_id, [])
            if values:
                variant_stats[variant_id] = VariantStats(
                    variant_id=variant_id,
                    sample_count=len(values),
                    mean=sum(values) / len(values),
                    std=self._calculate_std(values),
                    min_value=min(values),
                    max_value=max(values),
                    values=values,
                )
            else:
                variant_stats[variant_id] = VariantStats(
                    variant_id=variant_id,
                    sample_count=0,
                    mean=0.0,
                    std=0.0,
                    min_value=0.0,
                    max_value=0.0,
                    values=[],
                )

        # 統計的有意性を分析
        p_value, is_significant, winner_variant, confidence_level = (
            self._analyze_significance(variant_stats)
        )

        # 推奨事項を生成
        recommendation = self._generate_recommendation(
            variant_stats, is_significant, winner_variant
        )

        # numpy型をPython標準型に変換
        return ExperimentResult(
            experiment_id=experiment_id,
            experiment_name=experiment["name"],
            parameter_name=experiment["parameter_name"],
            status=ExperimentStatus(experiment["status"]),
            variant_stats=variant_stats,
            p_value=float(p_value) if p_value is not None else None,
            is_significant=bool(is_significant),
            winner_variant=winner_variant,
            confidence_level=float(confidence_level),
            recommendation=recommendation,
        )

    async def complete_experiment(
        self,
        experiment_id: UUID,
        winner_variant: Optional[str] = None,
    ) -> None:
        """実験を完了

        Args:
            experiment_id: 実験ID
            winner_variant: 勝者バリアント。Noneの場合は自動判定結果を使用。

        Raises:
            ExperimentNotFoundError: 実験が見つからない場合
            ExperimentStateError: 実験が running または paused 状態でない場合
            ValueError: 指定された winner_variant が無効な場合
        """
        experiment = await self._get_experiment(experiment_id)

        if experiment["status"] not in [
            ExperimentStatus.RUNNING.value,
            ExperimentStatus.PAUSED.value,
        ]:
            raise ExperimentStateError(
                f"Cannot complete experiment in '{experiment['status']}' status. "
                f"Only 'running' or 'paused' experiments can be completed."
            )

        # 勝者バリアントの検証
        if winner_variant is not None:
            if winner_variant not in experiment["variants"]:
                raise ValueError(
                    f"Invalid winner_variant '{winner_variant}'. "
                    f"Valid variants: {list(experiment['variants'].keys())}"
                )

        # 結果を分析して結果を保存
        result = await self.analyze_results(experiment_id)

        # 勝者が指定されていない場合は分析結果を使用
        final_winner = winner_variant or result.winner_variant

        def _update():
            with self.db.get_cursor() as cur:
                # 結果を JSON にシリアライズ（numpy型を標準型に変換）
                results_json = {
                    "variant_stats": {
                        vid: {
                            "sample_count": int(vs.sample_count),
                            "mean": float(vs.mean),
                            "std": float(vs.std),
                            "min_value": float(vs.min_value),
                            "max_value": float(vs.max_value),
                        }
                        for vid, vs in result.variant_stats.items()
                    },
                    "p_value": float(result.p_value) if result.p_value is not None else None,
                    "is_significant": bool(result.is_significant),
                    "recommendation": result.recommendation,
                }

                cur.execute(
                    """
                    UPDATE ab_experiments
                    SET status = %s,
                        ended_at = NOW(),
                        results = %s,
                        winner_variant = %s,
                        statistical_significance = %s
                    WHERE id = %s
                    """,
                    (
                        ExperimentStatus.COMPLETED.value,
                        json.dumps(results_json),
                        final_winner,
                        result.confidence_level,
                        str(experiment_id),
                    ),
                )

        await asyncio.to_thread(_update)

    async def get_experiment(self, experiment_id: UUID) -> Dict[str, Any]:
        """実験情報を取得（公開API）

        Args:
            experiment_id: 実験ID

        Returns:
            実験情報の辞書

        Raises:
            ExperimentNotFoundError: 実験が見つからない場合
        """
        return await self._get_experiment(experiment_id)

    async def list_experiments(
        self,
        status: Optional[ExperimentStatus] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """実験一覧を取得

        Args:
            status: フィルタするステータス。Noneの場合は全て取得。
            limit: 取得件数の上限

        Returns:
            実験情報のリスト
        """
        def _query():
            with self.db.get_cursor() as cur:
                if status:
                    cur.execute(
                        """
                        SELECT id, name, parameter_name, variants, traffic_split,
                               status, results, winner_variant, statistical_significance,
                               created_at, started_at, ended_at
                        FROM ab_experiments
                        WHERE status = %s
                        ORDER BY created_at DESC
                        LIMIT %s
                        """,
                        (status.value, limit),
                    )
                else:
                    cur.execute(
                        """
                        SELECT id, name, parameter_name, variants, traffic_split,
                               status, results, winner_variant, statistical_significance,
                               created_at, started_at, ended_at
                        FROM ab_experiments
                        ORDER BY created_at DESC
                        LIMIT %s
                        """,
                        (limit,),
                    )
                rows = cur.fetchall()
                return [self._row_to_dict(row) for row in rows]

        return await asyncio.to_thread(_query)

    # ===== Private Methods =====

    async def _get_experiment(self, experiment_id: UUID) -> Dict[str, Any]:
        """実験情報を取得（内部用）"""
        def _query():
            with self.db.get_cursor() as cur:
                cur.execute(
                    """
                    SELECT id, name, parameter_name, variants, traffic_split,
                           status, results, winner_variant, statistical_significance,
                           created_at, started_at, ended_at
                    FROM ab_experiments
                    WHERE id = %s
                    """,
                    (str(experiment_id),),
                )
                row = cur.fetchone()
                return row

        row = await asyncio.to_thread(_query)

        if row is None:
            raise ExperimentNotFoundError(f"Experiment {experiment_id} not found")

        return self._row_to_dict(row)

    def _row_to_dict(self, row: tuple) -> Dict[str, Any]:
        """DBの行データを辞書に変換"""
        return {
            "id": row[0],
            "name": row[1],
            "parameter_name": row[2],
            "variants": row[3] if isinstance(row[3], dict) else json.loads(row[3]),
            "traffic_split": row[4] if isinstance(row[4], list) else json.loads(row[4]),
            "status": row[5],
            "results": row[6] if isinstance(row[6], dict) else (
                json.loads(row[6]) if row[6] else None
            ),
            "winner_variant": row[7],
            "statistical_significance": row[8],
            "created_at": row[9],
            "started_at": row[10],
            "ended_at": row[11],
        }

    def _select_variant(
        self,
        session_id: UUID,
        variants: Dict[str, Any],
        traffic_split: List[float],
    ) -> str:
        """決定論的にバリアントを選択

        session_id のハッシュ値を使って、traffic_split に基づいてバリアントを選択する。
        同じ session_id は常に同じバリアントに割り当てられる。
        """
        # session_id のハッシュ値から 0-1 の範囲の値を生成
        hash_value = int(hashlib.md5(str(session_id).encode()).hexdigest(), 16)
        normalized_value = (hash_value % 10000) / 10000.0

        # 累積確率でバリアントを選択
        variant_ids = list(variants.keys())
        cumulative = 0.0

        for i, (variant_id, split) in enumerate(zip(variant_ids, traffic_split)):
            cumulative += split
            if normalized_value < cumulative:
                return variant_id

        # フォールバック（浮動小数点の誤差対策）
        return variant_ids[-1]

    async def _get_variant_metrics(
        self,
        experiment_id: UUID,
        metric_name: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        """バリアントごとのメトリクス値を取得"""
        def _query():
            with self.db.get_cursor() as cur:
                if metric_name:
                    cur.execute(
                        """
                        SELECT variant_id, metric_value
                        FROM ab_experiment_logs
                        WHERE experiment_id = %s AND metric_name = %s
                        ORDER BY created_at
                        """,
                        (str(experiment_id), metric_name),
                    )
                else:
                    # メトリクス名を自動検出（最初に記録されたメトリクス）
                    cur.execute(
                        """
                        SELECT metric_name FROM ab_experiment_logs
                        WHERE experiment_id = %s
                        ORDER BY created_at
                        LIMIT 1
                        """,
                        (str(experiment_id),),
                    )
                    row = cur.fetchone()
                    if not row:
                        return {}

                    detected_metric = row[0]
                    cur.execute(
                        """
                        SELECT variant_id, metric_value
                        FROM ab_experiment_logs
                        WHERE experiment_id = %s AND metric_name = %s
                        ORDER BY created_at
                        """,
                        (str(experiment_id), detected_metric),
                    )

                rows = cur.fetchall()
                result: Dict[str, List[float]] = {}
                for variant_id, value in rows:
                    if variant_id not in result:
                        result[variant_id] = []
                    result[variant_id].append(float(value))
                return result

        return await asyncio.to_thread(_query)

    def _calculate_std(self, values: List[float]) -> float:
        """標準偏差を計算"""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5

    def _analyze_significance(
        self,
        variant_stats: Dict[str, VariantStats],
    ) -> Tuple[Optional[float], bool, Optional[str], float]:
        """統計的有意性を分析

        Returns:
            (p_value, is_significant, winner_variant, confidence_level)
        """
        stats_with_data = {
            vid: vs for vid, vs in variant_stats.items() if vs.sample_count > 0
        }

        # 2つ以上のバリアントが必要
        if len(stats_with_data) < 2:
            return None, False, None, 0.0

        # 最小サンプル数のチェック
        min_samples = self.config.min_samples_per_variant
        has_enough_samples = all(
            vs.sample_count >= min_samples for vs in stats_with_data.values()
        )

        # 2バリアントの場合は t検定
        if len(stats_with_data) == 2:
            variant_ids = list(stats_with_data.keys())
            values1 = stats_with_data[variant_ids[0]].values
            values2 = stats_with_data[variant_ids[1]].values

            # サンプルが少なすぎる場合
            if len(values1) < 2 or len(values2) < 2:
                return None, False, None, 0.0

            # t検定を実行
            t_stat, p_value = stats.ttest_ind(values1, values2)
            confidence_level = 1.0 - p_value

            # 有意性判定
            is_significant = bool(
                has_enough_samples
                and p_value < (1.0 - self.config.significance_threshold)
            )

            # 勝者判定（有意差があり、平均が高い方）
            winner_variant = None
            if is_significant:
                mean1 = stats_with_data[variant_ids[0]].mean
                mean2 = stats_with_data[variant_ids[1]].mean
                winner_variant = (
                    variant_ids[0] if mean1 > mean2 else variant_ids[1]
                )

            return float(p_value), is_significant, winner_variant, float(confidence_level)

        # 3バリアント以上の場合は最も平均が高いものと2番目を比較
        sorted_variants = sorted(
            stats_with_data.items(),
            key=lambda x: x[1].mean,
            reverse=True,
        )

        best_variant = sorted_variants[0]
        second_variant = sorted_variants[1]

        values1 = best_variant[1].values
        values2 = second_variant[1].values

        if len(values1) < 2 or len(values2) < 2:
            return None, False, None, 0.0

        t_stat, p_value = stats.ttest_ind(values1, values2)
        confidence_level = 1.0 - p_value

        is_significant = bool(
            has_enough_samples
            and p_value < (1.0 - self.config.significance_threshold)
        )

        winner_variant = best_variant[0] if is_significant else None

        return float(p_value), is_significant, winner_variant, float(confidence_level)

    def _generate_recommendation(
        self,
        variant_stats: Dict[str, VariantStats],
        is_significant: bool,
        winner_variant: Optional[str],
    ) -> str:
        """推奨事項を生成"""
        stats_with_data = {
            vid: vs for vid, vs in variant_stats.items() if vs.sample_count > 0
        }

        if not stats_with_data:
            return "No data collected yet. Continue running the experiment."

        min_samples = self.config.min_samples_per_variant
        insufficient_variants = [
            vid for vid, vs in stats_with_data.items()
            if vs.sample_count < min_samples
        ]

        if insufficient_variants:
            needed = {
                vid: min_samples - variant_stats[vid].sample_count
                for vid in insufficient_variants
            }
            return (
                f"Insufficient samples for variants: {insufficient_variants}. "
                f"Need additional samples: {needed}"
            )

        if is_significant and winner_variant:
            winner_stats = variant_stats[winner_variant]
            return (
                f"Statistically significant result. "
                f"Recommended: Adopt '{winner_variant}' "
                f"(mean={winner_stats.mean:.4f}, n={winner_stats.sample_count})"
            )

        return (
            "No statistically significant difference detected. "
            "Consider extending the experiment duration or accepting the control."
        )
