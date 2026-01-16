# 学習データ収集
# routing_history から学習データを収集し routing_training_data テーブルに保存
# 実装仕様: docs/phase3-implementation-spec.ja.md セクション5.1 学習パイプライン
"""
学習データ収集モジュール

routing_history テーブルからルーティング履歴を取得し、
ニューラルスコアラー用の学習データを routing_training_data テーブルに保存する。

設計方針（タスク実行フローエージェント観点）:
- API設計: シンプルなメソッド一つで学習データ収集を完結
- フロー整合性: routing_history → ラベル付け → 特徴量抽出 → routing_training_data の流れを明確に
- エラー処理: 個々のレコード処理失敗は他のレコードに影響しない
- 拡張性: ラベル計算ロジックを独立メソッドに分離し、カスタマイズ容易に
- テスト容易性: 各コンポーネントを依存性注入で受け取り、モック化が容易

データフロー:
    routing_history（Phase2作成済み）
        ↓
    ラベル付け
    ├── user_feedback == "positive" → label = 1.0
    ├── user_feedback == "negative" → label = 0.0
    └── result_status == "success" → label に 0.5 加算
        ↓
    特徴量抽出（FeatureExtractor使用）
    ├── タスクエンベディング（AzureEmbeddingClient）
    ├── タスク特徴量
    └── エージェント特徴量
        ↓
    routing_training_data テーブルへ保存
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from psycopg2.extras import Json

from src.agents.agent_registry import AgentDefinition, AgentRegistry
from src.config.phase3_config import Phase3Config
from src.db.connection import DatabaseConnection
from src.embedding.azure_client import AzureEmbeddingClient
from src.scoring.feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)


@dataclass
class RoutingHistoryRecord:
    """routing_history テーブルの1レコードを表すデータクラス

    Attributes:
        id: ルーティング履歴のID
        session_id: セッションID
        orchestrator_id: オーケストレーターID
        task_summary: タスク概要
        selected_agent_id: 選択されたエージェントID
        selection_reason: 選択理由
        candidate_agents: 候補エージェントとスコア
        result_status: 結果ステータス（success / failure / timeout / cancelled）
        result_summary: 結果サマリー
        user_feedback: ユーザーフィードバック（positive / negative / neutral）
        started_at: 開始日時
        completed_at: 完了日時
    """

    id: UUID
    session_id: UUID
    orchestrator_id: str
    task_summary: str
    selected_agent_id: str
    selection_reason: Optional[str]
    candidate_agents: Optional[List[Dict[str, Any]]]
    result_status: Optional[str]
    result_summary: Optional[str]
    user_feedback: Optional[str]
    started_at: datetime
    completed_at: Optional[datetime]

    @classmethod
    def from_row(cls, row: tuple) -> "RoutingHistoryRecord":
        """DBの行からインスタンスを生成

        Args:
            row: DBから取得した行

        Returns:
            RoutingHistoryRecord インスタンス
        """
        return cls(
            id=row[0],
            session_id=row[1],
            orchestrator_id=row[2],
            task_summary=row[3],
            selected_agent_id=row[4],
            selection_reason=row[5],
            candidate_agents=row[6],
            result_status=row[7],
            result_summary=row[8],
            user_feedback=row[9],
            started_at=row[10],
            completed_at=row[11],
        )


class TrainingDataCollector:
    """学習データ収集クラス

    routing_history からルーティング履歴を取得し、
    ニューラルスコアラー用の学習データを routing_training_data テーブルに保存する。

    使用例:
        db = DatabaseConnection()
        embedding_client = AzureEmbeddingClient()
        agent_registry = AgentRegistry(db)
        config = Phase3Config()

        collector = TrainingDataCollector(
            db_connection=db,
            embedding_client=embedding_client,
            agent_registry=agent_registry,
            config=config,
        )

        # routing_historyから学習データを収集
        saved_count = collector.collect_from_routing_history(
            since=datetime(2025, 1, 1),
            limit=1000,
        )
        print(f"保存件数: {saved_count}")

    Attributes:
        db: DatabaseConnection インスタンス
        embedding_client: AzureEmbeddingClient インスタンス
        agent_registry: AgentRegistry インスタンス
        config: Phase3Config インスタンス
        feature_extractor: FeatureExtractor インスタンス
    """

    # routing_history から取得するカラム
    _ROUTING_HISTORY_COLUMNS = """
        id, session_id, orchestrator_id, task_summary,
        selected_agent_id, selection_reason, candidate_agents,
        result_status, result_summary, user_feedback,
        started_at, completed_at
    """

    # routing_training_data への INSERT SQL
    _INSERT_TRAINING_DATA_SQL = """
        INSERT INTO routing_training_data (
            task_embedding, task_features, agent_features,
            selected_agent_id, candidate_scores,
            user_feedback, result_status, actual_score,
            created_at, used_for_training
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        RETURNING id
    """

    def __init__(
        self,
        db_connection: DatabaseConnection,
        embedding_client: AzureEmbeddingClient,
        agent_registry: AgentRegistry,
        config: Phase3Config,
    ):
        """TrainingDataCollector を初期化

        Args:
            db_connection: DatabaseConnection インスタンス
            embedding_client: AzureEmbeddingClient インスタンス
            agent_registry: AgentRegistry インスタンス
            config: Phase3Config インスタンス
        """
        self.db = db_connection
        self.embedding_client = embedding_client
        self.agent_registry = agent_registry
        self.config = config
        self.feature_extractor = FeatureExtractor(config)

    def collect_from_routing_history(
        self,
        since: Optional[datetime] = None,
        limit: int = 1000,
    ) -> int:
        """routing_history から学習データを収集して保存

        未収集のルーティング履歴からフィードバックがあるレコードを取得し、
        ラベル付け・特徴量抽出を行ってから routing_training_data に保存する。

        Args:
            since: この日時以降のレコードを取得（Noneの場合は全期間）
            limit: 取得する最大レコード数

        Returns:
            保存した学習データの件数

        Note:
            - user_feedback が NULL のレコードはスキップされる
            - 特徴量抽出やDB保存に失敗したレコードはスキップされる
            - 既に学習データとして保存済みのレコードはスキップされる
        """
        # 1. routing_history から未収集のレコードを取得
        records = self._fetch_routing_history(since, limit)
        logger.info(f"routing_history から {len(records)} 件のレコードを取得")

        if not records:
            return 0

        saved_count = 0

        for record in records:
            try:
                # フィードバックがないレコードはスキップ
                if not record.user_feedback and not record.result_status:
                    logger.debug(
                        f"レコード {record.id}: フィードバックなし、スキップ"
                    )
                    continue

                # 2. ラベル付け
                actual_score = self._calculate_label(
                    user_feedback=record.user_feedback,
                    result_status=record.result_status,
                )

                # 3. 特徴量抽出
                task_features, agent_features, task_embedding = (
                    self._extract_features(record)
                )

                # 4. routing_training_data に保存
                self._save_training_data(
                    record=record,
                    task_embedding=task_embedding,
                    task_features=task_features,
                    agent_features=agent_features,
                    actual_score=actual_score,
                )

                saved_count += 1
                logger.debug(
                    f"レコード {record.id}: 学習データとして保存完了 (score={actual_score})"
                )

            except Exception as e:
                logger.warning(
                    f"レコード {record.id}: 学習データ収集に失敗: {e}"
                )
                continue

        logger.info(
            f"学習データ収集完了: {saved_count}/{len(records)} 件を保存"
        )
        return saved_count

    def _calculate_label(
        self,
        user_feedback: Optional[str],
        result_status: Optional[str],
    ) -> float:
        """フィードバックと結果からラベルを計算

        ラベル付けロジック:
        - user_feedback == "positive" → label = 1.0
        - user_feedback == "negative" → label = 0.0
        - 上記以外の場合 → label = 0.5（中立）
        - result_status == "success" → label に 0.5 加算（最大1.0）

        Args:
            user_feedback: ユーザーフィードバック（positive / negative / neutral / None）
            result_status: 結果ステータス（success / failure / timeout / cancelled / None）

        Returns:
            0.0-1.0 の範囲のラベル値

        Examples:
            >>> self._calculate_label("positive", "success")
            1.0
            >>> self._calculate_label("negative", "success")
            0.5
            >>> self._calculate_label("positive", "failure")
            1.0
            >>> self._calculate_label("neutral", "success")
            1.0
            >>> self._calculate_label(None, "success")
            1.0
            >>> self._calculate_label(None, "failure")
            0.5
        """
        label = 0.5  # デフォルト（中立）

        # ユーザーフィードバックによるベースラベル
        if user_feedback == "positive":
            label = 1.0
        elif user_feedback == "negative":
            label = 0.0
        # neutral または None の場合は 0.5 のまま

        # result_status == "success" の場合、0.5を加算（最大1.0）
        if result_status == "success":
            label = min(label + 0.5, 1.0)

        return label

    def _fetch_routing_history(
        self,
        since: Optional[datetime],
        limit: int,
    ) -> List[RoutingHistoryRecord]:
        """routing_history から未収集のレコードを取得

        Args:
            since: この日時以降のレコードを取得
            limit: 取得する最大レコード数

        Returns:
            RoutingHistoryRecord のリスト
        """
        # 既に学習データとして保存済みの履歴IDを取得
        # （routing_training_data には routing_history の ID を直接保存していないため、
        #   task_summary と selected_agent_id と started_at の組み合わせで判定）
        sql = f"""
            SELECT {self._ROUTING_HISTORY_COLUMNS}
            FROM routing_history rh
            WHERE NOT EXISTS (
                SELECT 1 FROM routing_training_data rtd
                WHERE rtd.selected_agent_id = rh.selected_agent_id
                  AND rtd.created_at = rh.started_at
            )
            AND (rh.user_feedback IS NOT NULL OR rh.result_status IS NOT NULL)
        """
        params: List[Any] = []

        if since:
            sql += " AND rh.started_at >= %s"
            params.append(since)

        sql += " ORDER BY rh.started_at ASC LIMIT %s"
        params.append(limit)

        with self.db.get_cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
            return [RoutingHistoryRecord.from_row(row) for row in rows]

    def _extract_features(
        self,
        record: RoutingHistoryRecord,
    ) -> tuple[Dict[str, float], Dict[str, float], List[float]]:
        """特徴量を抽出

        Args:
            record: RoutingHistoryRecord インスタンス

        Returns:
            (task_features, agent_features, task_embedding) のタプル
        """
        # タスク特徴量の抽出
        task_features = self.feature_extractor.extract_task_features(
            record.task_summary
        )

        # エージェント定義の取得
        agent = self.agent_registry.get_by_id(record.selected_agent_id)
        if agent is None:
            # エージェントが見つからない場合はデフォルト値を使用
            agent = AgentDefinition(
                agent_id=record.selected_agent_id,
                name="Unknown",
                role="Unknown",
                perspectives=[],
                system_prompt="",
                capabilities=[],
            )

        # 過去の経験データを取得（routing_history から）
        past_experiences = self._get_past_experiences(record.selected_agent_id)

        # エージェント特徴量の抽出
        agent_features = self.feature_extractor.extract_agent_features(
            agent=agent,
            past_experiences=past_experiences,
        )

        # タスクエンベディングの取得
        task_embedding = self.embedding_client.get_embedding(record.task_summary)

        return task_features, agent_features, task_embedding

    def _get_past_experiences(
        self,
        agent_id: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """エージェントの過去の経験を取得

        Args:
            agent_id: エージェントID
            limit: 取得する最大件数

        Returns:
            過去の経験データのリスト
        """
        sql = """
            SELECT
                result_status,
                EXTRACT(EPOCH FROM (completed_at - started_at)) as duration_seconds,
                started_at
            FROM routing_history
            WHERE selected_agent_id = %s
              AND completed_at IS NOT NULL
            ORDER BY started_at DESC
            LIMIT %s
        """

        with self.db.get_cursor() as cur:
            cur.execute(sql, (agent_id, limit))
            rows = cur.fetchall()

            experiences = []
            for row in rows:
                experiences.append({
                    "success": row[0] == "success",
                    "duration_seconds": row[1] if row[1] else 0.0,
                    "created_at": row[2],
                })

            return experiences

    def _save_training_data(
        self,
        record: RoutingHistoryRecord,
        task_embedding: List[float],
        task_features: Dict[str, float],
        agent_features: Dict[str, float],
        actual_score: float,
    ) -> UUID:
        """学習データを routing_training_data テーブルに保存

        Args:
            record: RoutingHistoryRecord インスタンス
            task_embedding: タスクエンベディング
            task_features: タスク特徴量
            agent_features: エージェント特徴量
            actual_score: ラベル（正解スコア）

        Returns:
            保存したレコードのID
        """
        # candidate_scores の準備
        candidate_scores = record.candidate_agents or []

        with self.db.get_cursor() as cur:
            cur.execute(
                self._INSERT_TRAINING_DATA_SQL,
                (
                    task_embedding,
                    Json(task_features),
                    Json(agent_features),
                    record.selected_agent_id,
                    Json(candidate_scores),
                    record.user_feedback,
                    record.result_status,
                    actual_score,
                    record.started_at,  # created_at に started_at を使用
                    False,  # used_for_training
                ),
            )
            row = cur.fetchone()
            return row[0] if row else None

    def get_training_data_count(self, only_unused: bool = False) -> int:
        """学習データの件数を取得

        Args:
            only_unused: True の場合、未使用の学習データのみカウント

        Returns:
            学習データの件数
        """
        if only_unused:
            sql = """
                SELECT COUNT(*)
                FROM routing_training_data
                WHERE used_for_training = FALSE
            """
        else:
            sql = "SELECT COUNT(*) FROM routing_training_data"

        with self.db.get_cursor() as cur:
            cur.execute(sql)
            row = cur.fetchone()
            return row[0] if row else 0

    def is_ready_for_training(self) -> bool:
        """学習に必要な最小サンプル数が揃っているか確認

        Returns:
            True: 最小サンプル数以上のデータがある場合
            False: データが不足している場合
        """
        count = self.get_training_data_count(only_unused=True)
        return count >= self.config.min_training_samples
