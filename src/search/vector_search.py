# ベクトル検索モジュール（Stage 1: 関連性フィルタ）
# 実装仕様: docs/phase1-implementation-spec.ja.md セクション4.3
"""
ベクトル検索モジュール

pgvector の cosine 距離を使用して、クエリに類似した記憶を検索する。
Stage 1 として、類似度閾値以上の候補を取得し、Stage 2 のランキングに渡す。

設計方針（検索エンジンエージェント観点）:
- 検索精度: cosine 類似度で意味的に近い記憶を取得
- レスポンス性能: candidate_limit で候補数を制限し、Stage 2 の負荷を軽減
- スケーラビリティ: Phase 1 は線形スキャン、1万件超でインデックス追加を検討
- API連携: Azure OpenAI Embedding でクエリをベクトル化
- フォールバック: embedding 取得失敗時は空リストを返す
"""

import logging
from typing import List, Optional, Tuple

from src.config.phase1_config import Phase1Config, config as default_config
from src.db.connection import DatabaseConnection
from src.embedding.azure_client import AzureEmbeddingClient, AzureEmbeddingError
from src.models.memory import AgentMemory


logger = logging.getLogger(__name__)


class VectorSearchError(Exception):
    """ベクトル検索のエラー"""
    pass


class VectorSearch:
    """ベクトル検索エンジン（Stage 1: 関連性フィルタ）

    pgvector の cosine 距離を使用して、クエリに類似した記憶を検索する。

    使用例:
        db = DatabaseConnection()
        embedding_client = AzureEmbeddingClient()
        search = VectorSearch(db, embedding_client)

        # 候補を取得
        candidates = search.search_candidates(
            query="緊急調達のコスト",
            agent_id="procurement_agent_01"
        )

        for memory, similarity in candidates:
            print(f"{memory.content[:50]}... (similarity: {similarity:.3f})")
    """

    def __init__(
        self,
        db: DatabaseConnection,
        embedding_client: AzureEmbeddingClient,
        config: Optional[Phase1Config] = None,
    ):
        """ベクトル検索エンジンを初期化

        Args:
            db: データベース接続
            embedding_client: Azure OpenAI Embedding クライアント
            config: Phase 1 設定（省略時はデフォルト設定を使用）
        """
        self.db = db
        self.embedding_client = embedding_client
        self.config = config or default_config

        logger.info(
            f"VectorSearch 初期化完了: "
            f"similarity_threshold={self.config.similarity_threshold}, "
            f"candidate_limit={self.config.candidate_limit}"
        )

    def search_candidates(
        self,
        query: str,
        agent_id: str,
        perspective: Optional[str] = None,
    ) -> List[Tuple[AgentMemory, float]]:
        """Stage 1: ベクトル検索で候補を取得

        クエリをエンベディングし、pgvector の cosine 距離で類似記憶を検索する。
        similarity_threshold 以上の候補を candidate_limit 件まで取得する。

        Args:
            query: 検索クエリ（テキスト）
            agent_id: 検索対象のエージェントID
            perspective: 観点（将来の拡張用、現在は未使用）

        Returns:
            (AgentMemory, similarity_score) のリスト。
            類似度の高い順にソートされている。

        Raises:
            VectorSearchError: エンベディング取得またはDB検索に失敗した場合

        Note:
            - pgvector の <=> 演算子は cosine 距離（0に近いほど類似）
            - 類似度は 1 - 距離 で計算（1に近いほど類似）
            - status='active' の記憶のみを検索対象とする
        """
        if not query or not query.strip():
            logger.warning("空のクエリが渡されました")
            return []

        # 1. クエリをエンベディング
        try:
            query_embedding = self.embedding_client.get_embedding(query)
        except AzureEmbeddingError as e:
            logger.error(f"クエリのエンベディング取得に失敗: {e}")
            raise VectorSearchError(f"クエリのエンベディング取得に失敗しました: {e}") from e

        # 2. pgvector で検索
        # <=> 演算子は cosine 距離を計算（0に近いほど類似）
        # 類似度 = 1 - 距離
        sql = """
            SELECT
                id, agent_id, content, embedding, tags,
                scope_level, scope_domain, scope_project,
                strength, strength_by_perspective,
                access_count, candidate_count, last_accessed_at,
                next_review_at, review_count,
                impact_score, consolidation_level, learning,
                status, source, created_at, updated_at, last_decay_at,
                1 - (embedding <=> %s::vector) as similarity
            FROM agent_memory
            WHERE agent_id = %s
              AND status = 'active'
              AND embedding IS NOT NULL
              AND 1 - (embedding <=> %s::vector) >= %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """

        try:
            with self.db.get_cursor() as cur:
                # embedding を文字列形式に変換（pgvector が期待する形式）
                embedding_str = self._format_embedding(query_embedding)

                cur.execute(
                    sql,
                    (
                        embedding_str,
                        agent_id,
                        embedding_str,
                        self.config.similarity_threshold,
                        embedding_str,
                        self.config.candidate_limit,
                    ),
                )

                rows = cur.fetchall()

                # 結果をパース
                results: List[Tuple[AgentMemory, float]] = []
                for row in rows:
                    # row の最後のカラムが similarity
                    similarity = float(row[-1])
                    # similarity を除いた row で AgentMemory を生成
                    memory = AgentMemory.from_row(row[:-1])
                    results.append((memory, similarity))

                logger.info(
                    f"ベクトル検索完了: query={query[:30]!r}..., "
                    f"agent_id={agent_id}, "
                    f"candidates={len(results)}"
                )

                return results

        except Exception as e:
            logger.error(f"ベクトル検索に失敗: {e}")
            raise VectorSearchError(f"ベクトル検索に失敗しました: {e}") from e

    def _format_embedding(self, embedding: List[float]) -> str:
        """エンベディングを pgvector の vector 型に変換するための文字列形式に変換

        Args:
            embedding: float のリスト

        Returns:
            "[0.1, 0.2, ...]" 形式の文字列
        """
        return "[" + ",".join(str(v) for v in embedding) + "]"

    def search_by_embedding(
        self,
        query_embedding: List[float],
        agent_id: str,
        similarity_threshold: Optional[float] = None,
        candidate_limit: Optional[int] = None,
    ) -> List[Tuple[AgentMemory, float]]:
        """事前計算されたエンベディングでベクトル検索

        クエリのエンベディングが既に計算済みの場合に使用する。
        API呼び出しを節約できる。

        Args:
            query_embedding: クエリのエンベディングベクトル
            agent_id: 検索対象のエージェントID
            similarity_threshold: 類似度閾値（省略時は設定値を使用）
            candidate_limit: 最大候補数（省略時は設定値を使用）

        Returns:
            (AgentMemory, similarity_score) のリスト。
            類似度の高い順にソートされている。

        Raises:
            VectorSearchError: DB検索に失敗した場合
        """
        threshold = similarity_threshold or self.config.similarity_threshold
        limit = candidate_limit or self.config.candidate_limit

        sql = """
            SELECT
                id, agent_id, content, embedding, tags,
                scope_level, scope_domain, scope_project,
                strength, strength_by_perspective,
                access_count, candidate_count, last_accessed_at,
                next_review_at, review_count,
                impact_score, consolidation_level, learning,
                status, source, created_at, updated_at, last_decay_at,
                1 - (embedding <=> %s::vector) as similarity
            FROM agent_memory
            WHERE agent_id = %s
              AND status = 'active'
              AND embedding IS NOT NULL
              AND 1 - (embedding <=> %s::vector) >= %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """

        try:
            with self.db.get_cursor() as cur:
                embedding_str = self._format_embedding(query_embedding)

                cur.execute(
                    sql,
                    (
                        embedding_str,
                        agent_id,
                        embedding_str,
                        threshold,
                        embedding_str,
                        limit,
                    ),
                )

                rows = cur.fetchall()

                results: List[Tuple[AgentMemory, float]] = []
                for row in rows:
                    similarity = float(row[-1])
                    memory = AgentMemory.from_row(row[:-1])
                    results.append((memory, similarity))

                logger.info(
                    f"エンベディング検索完了: agent_id={agent_id}, candidates={len(results)}"
                )

                return results

        except Exception as e:
            logger.error(f"エンベディング検索に失敗: {e}")
            raise VectorSearchError(f"エンベディング検索に失敗しました: {e}") from e

    def count_candidates(
        self,
        query: str,
        agent_id: str,
    ) -> int:
        """類似度閾値以上の候補数をカウント

        検索結果の件数だけを取得したい場合に使用。
        全カラムを取得しないため、より軽量。

        Args:
            query: 検索クエリ（テキスト）
            agent_id: 検索対象のエージェントID

        Returns:
            類似度閾値以上の候補数

        Raises:
            VectorSearchError: エンベディング取得またはDB検索に失敗した場合
        """
        if not query or not query.strip():
            return 0

        try:
            query_embedding = self.embedding_client.get_embedding(query)
        except AzureEmbeddingError as e:
            logger.error(f"クエリのエンベディング取得に失敗: {e}")
            raise VectorSearchError(f"クエリのエンベディング取得に失敗しました: {e}") from e

        sql = """
            SELECT COUNT(*)
            FROM agent_memory
            WHERE agent_id = %s
              AND status = 'active'
              AND embedding IS NOT NULL
              AND 1 - (embedding <=> %s::vector) >= %s
        """

        try:
            with self.db.get_cursor() as cur:
                embedding_str = self._format_embedding(query_embedding)

                cur.execute(
                    sql,
                    (
                        agent_id,
                        embedding_str,
                        self.config.similarity_threshold,
                    ),
                )

                result = cur.fetchone()
                count = result[0] if result else 0

                logger.debug(
                    f"候補数カウント完了: query={query[:30]!r}..., "
                    f"agent_id={agent_id}, count={count}"
                )

                return count

        except Exception as e:
            logger.error(f"候補数カウントに失敗: {e}")
            raise VectorSearchError(f"候補数カウントに失敗しました: {e}") from e
