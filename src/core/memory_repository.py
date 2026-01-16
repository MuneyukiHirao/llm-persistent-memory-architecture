# メモリリポジトリ
# 外部メモリシステムのCRUD操作を担当
# 実装仕様: docs/phase1-implementation-spec.ja.md セクション3, 4
"""
メモリリポジトリモジュール

PostgreSQL + pgvector に対する CRUD 操作と 2段階強化メカニズムを提供。

設計方針（メモリ管理エージェント観点）:
- 強度の正確性: candidate_count と access_count の分離を SQL レベルで保証
- 観点別強度: strength_by_perspective の部分更新をサポート
- 原子性: 複数フィールド更新時も単一 UPDATE 文で整合性を保証
- 効率性: DatabaseConnection のコネクションプールを活用
- テスト容易性: リポジトリパターンで DB 操作を抽象化
"""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from psycopg2.extras import Json

from src.config.phase1_config import Phase1Config
from src.db.connection import DatabaseConnection
from src.models.memory import AgentMemory


class MemoryRepository:
    """メモリリポジトリクラス

    agent_memory テーブルに対する CRUD 操作と 2段階強化処理を提供。

    使用例:
        db = DatabaseConnection()
        config = Phase1Config()
        repo = MemoryRepository(db, config)

        # 新規メモリ作成
        memory = AgentMemory.create(
            agent_id="agent_01",
            content="重要な学び",
        )
        created = repo.create(memory)

        # 2段階強化
        repo.increment_candidate_count(memory.id)  # 検索候補になった
        repo.increment_access_count(memory.id, 0.1)  # 実際に使用された

    Attributes:
        db: DatabaseConnection インスタンス
        config: Phase1Config インスタンス
    """

    # SQL定義（可読性のために定数として定義）
    _COLUMNS = """
        id, agent_id, content, embedding, tags,
        scope_level, scope_domain, scope_project,
        strength, strength_by_perspective,
        access_count, candidate_count, last_accessed_at,
        next_review_at, review_count,
        impact_score, consolidation_level, learning,
        status, source, created_at, updated_at, last_decay_at
    """

    _INSERT_SQL = """
        INSERT INTO agent_memory (
            id, agent_id, content, embedding, tags,
            scope_level, scope_domain, scope_project,
            strength, strength_by_perspective,
            access_count, candidate_count, last_accessed_at,
            next_review_at, review_count,
            impact_score, consolidation_level, learning,
            status, source, created_at, updated_at, last_decay_at
        ) VALUES (
            %s, %s, %s, %s, %s,
            %s, %s, %s,
            %s, %s,
            %s, %s, %s,
            %s, %s,
            %s, %s, %s,
            %s, %s, %s, %s, %s
        )
        RETURNING {columns}
    """.format(columns=_COLUMNS)

    _SELECT_BY_ID_SQL = """
        SELECT {columns}
        FROM agent_memory
        WHERE id = %s
    """.format(columns=_COLUMNS)

    _SELECT_BY_AGENT_ID_SQL = """
        SELECT {columns}
        FROM agent_memory
        WHERE agent_id = %s AND status = %s
        ORDER BY created_at DESC
    """.format(columns=_COLUMNS)

    _UPDATE_SQL = """
        UPDATE agent_memory SET
            content = %s,
            embedding = %s,
            tags = %s,
            scope_level = %s,
            scope_domain = %s,
            scope_project = %s,
            strength = %s,
            strength_by_perspective = %s,
            access_count = %s,
            candidate_count = %s,
            last_accessed_at = %s,
            next_review_at = %s,
            review_count = %s,
            impact_score = %s,
            consolidation_level = %s,
            learning = %s,
            status = %s,
            source = %s,
            updated_at = %s,
            last_decay_at = %s
        WHERE id = %s
        RETURNING {columns}
    """.format(columns=_COLUMNS)

    _ARCHIVE_SQL = """
        UPDATE agent_memory SET
            status = 'archived',
            updated_at = %s
        WHERE id = %s
    """

    _INCREMENT_CANDIDATE_SQL = """
        UPDATE agent_memory SET
            candidate_count = candidate_count + 1,
            updated_at = %s
        WHERE id = %s
    """

    _INCREMENT_ACCESS_SQL = """
        UPDATE agent_memory SET
            access_count = access_count + 1,
            strength = strength + %s,
            last_accessed_at = %s,
            updated_at = %s
        WHERE id = %s
    """

    def __init__(self, db: DatabaseConnection, config: Phase1Config):
        """MemoryRepository を初期化

        Args:
            db: DatabaseConnection インスタンス
            config: Phase1Config インスタンス
        """
        self.db = db
        self.config = config

    def _format_embedding(self, embedding: Optional[List[float]]) -> Optional[str]:
        """embedding をPostgreSQL vector型のリテラル形式に変換

        Args:
            embedding: float のリスト、または None

        Returns:
            '[0.1, 0.2, ...]' 形式の文字列、または None
        """
        if embedding is None:
            return None
        # vector 型は文字列リテラル形式で渡す
        return "[" + ",".join(str(v) for v in embedding) + "]"

    # === Create ===
    def create(self, memory: AgentMemory) -> AgentMemory:
        """新規メモリを作成

        Args:
            memory: 作成する AgentMemory インスタンス

        Returns:
            作成された AgentMemory（DB から返却された値で更新済み）

        Note:
            - created_at, updated_at は現在時刻で設定される
            - embedding は None でも作成可能（後から非同期で設定）
        """
        now = datetime.now()

        with self.db.get_cursor() as cur:
            cur.execute(
                self._INSERT_SQL,
                (
                    str(memory.id),
                    memory.agent_id,
                    memory.content,
                    self._format_embedding(memory.embedding),
                    memory.tags,
                    memory.scope_level,
                    memory.scope_domain,
                    memory.scope_project,
                    memory.strength,
                    Json(memory.strength_by_perspective),
                    memory.access_count,
                    memory.candidate_count,
                    memory.last_accessed_at,
                    memory.next_review_at,
                    memory.review_count,
                    memory.impact_score,
                    memory.consolidation_level,
                    memory.learning,
                    memory.status,
                    memory.source,
                    now,
                    now,
                    memory.last_decay_at,
                ),
            )
            row = cur.fetchone()
            return AgentMemory.from_row(row)

    # === Read ===
    def get_by_id(self, memory_id: UUID) -> Optional[AgentMemory]:
        """ID でメモリを取得

        Args:
            memory_id: メモリの UUID

        Returns:
            AgentMemory インスタンス、見つからない場合は None
        """
        with self.db.get_cursor() as cur:
            cur.execute(self._SELECT_BY_ID_SQL, (str(memory_id),))
            row = cur.fetchone()
            if row is None:
                return None
            return AgentMemory.from_row(row)

    def get_by_agent_id(
        self, agent_id: str, status: str = "active"
    ) -> List[AgentMemory]:
        """エージェント ID でメモリを取得

        Args:
            agent_id: エージェントの ID
            status: フィルタするステータス（デフォルト: "active"）

        Returns:
            AgentMemory のリスト（created_at 降順）
        """
        with self.db.get_cursor() as cur:
            cur.execute(self._SELECT_BY_AGENT_ID_SQL, (agent_id, status))
            rows = cur.fetchall()
            return [AgentMemory.from_row(row) for row in rows]

    # === Update ===
    def update(self, memory: AgentMemory) -> AgentMemory:
        """メモリを更新

        updated_at は自動的に現在時刻で更新される。
        id, agent_id, created_at は変更されない。

        Args:
            memory: 更新する AgentMemory インスタンス

        Returns:
            更新された AgentMemory

        Raises:
            ValueError: 指定した ID のメモリが存在しない場合
        """
        now = datetime.now()

        with self.db.get_cursor() as cur:
            cur.execute(
                self._UPDATE_SQL,
                (
                    memory.content,
                    self._format_embedding(memory.embedding),
                    memory.tags,
                    memory.scope_level,
                    memory.scope_domain,
                    memory.scope_project,
                    memory.strength,
                    Json(memory.strength_by_perspective),
                    memory.access_count,
                    memory.candidate_count,
                    memory.last_accessed_at,
                    memory.next_review_at,
                    memory.review_count,
                    memory.impact_score,
                    memory.consolidation_level,
                    memory.learning,
                    memory.status,
                    memory.source,
                    now,
                    memory.last_decay_at,
                    str(memory.id),
                ),
            )
            row = cur.fetchone()
            if row is None:
                raise ValueError(f"Memory with id {memory.id} not found")
            return AgentMemory.from_row(row)

    # === Delete (論理削除) ===
    def archive(self, memory_id: UUID) -> bool:
        """メモリをアーカイブ（論理削除）

        status を 'archived' に変更する。物理削除は行わない。

        Args:
            memory_id: アーカイブするメモリの UUID

        Returns:
            True: アーカイブ成功、False: 対象のメモリが存在しない
        """
        now = datetime.now()

        with self.db.get_cursor() as cur:
            cur.execute(self._ARCHIVE_SQL, (now, str(memory_id)))
            return cur.rowcount > 0

    # === 2段階強化用 ===
    def increment_candidate_count(self, memory_id: UUID) -> None:
        """検索候補になった回数をインクリメント

        2段階強化の第1段階: 検索候補として参照されただけで
        candidate_count をインクリメントする。
        strength は変更しない。

        Args:
            memory_id: 対象メモリの UUID

        Note:
            - 冪等ではない（呼び出すたびにカウントが増える）
            - 実際に使用された場合は increment_access_count を使用
        """
        now = datetime.now()

        with self.db.get_cursor() as cur:
            cur.execute(self._INCREMENT_CANDIDATE_SQL, (now, str(memory_id)))

    def increment_access_count(
        self, memory_id: UUID, strength_increment: float
    ) -> None:
        """使用回数をインクリメントし、強度を更新

        2段階強化の第2段階: 実際に使用されたメモリの
        access_count をインクリメントし、strength を強化する。

        Args:
            memory_id: 対象メモリの UUID
            strength_increment: 強度の増分値
                               （config.strength_increment_on_use を使用）

        Note:
            - last_accessed_at も現在時刻で更新される
            - 観点別強度の更新は別途 update_perspective_strength を使用
        """
        now = datetime.now()

        with self.db.get_cursor() as cur:
            cur.execute(
                self._INCREMENT_ACCESS_SQL,
                (strength_increment, now, now, str(memory_id)),
            )

    def update_perspective_strength(
        self, memory_id: UUID, perspective: str, increment: float
    ) -> None:
        """観点別強度を更新

        特定の観点の強度のみを更新する。
        該当観点が存在しない場合は increment の値で初期化される。

        Args:
            memory_id: 対象メモリの UUID
            perspective: 観点名（例: "コスト", "納期"）
            increment: 強度の増分値
                      （config.perspective_strength_increment を使用）

        Note:
            - JSONB の部分更新を SQL で実行
            - 全観点を毎回更新しない（該当観点のみ）
        """
        now = datetime.now()

        # JSONB の部分更新: 既存値 + increment、または increment で初期化
        sql = """
            UPDATE agent_memory SET
                strength_by_perspective = jsonb_set(
                    COALESCE(strength_by_perspective, '{}'::jsonb),
                    %s,
                    to_jsonb(
                        COALESCE(
                            (strength_by_perspective->>%s)::float,
                            0.0
                        ) + %s
                    )
                ),
                updated_at = %s
            WHERE id = %s
        """

        with self.db.get_cursor() as cur:
            cur.execute(
                sql,
                (
                    [perspective],  # JSONB パス（配列形式）
                    perspective,    # 既存値取得用
                    increment,
                    now,
                    str(memory_id),
                ),
            )

    def batch_increment_candidate_count(self, memory_ids: List[UUID]) -> int:
        """複数メモリの candidate_count を一括インクリメント

        バッチ処理による効率化。検索で複数の候補が返された場合に使用。

        Args:
            memory_ids: 対象メモリの UUID リスト

        Returns:
            更新された行数
        """
        if not memory_ids:
            return 0

        now = datetime.now()

        # ANY 句を使用した一括更新（UUID型に明示的キャスト）
        sql = """
            UPDATE agent_memory SET
                candidate_count = candidate_count + 1,
                updated_at = %s
            WHERE id = ANY(%s::uuid[])
        """

        with self.db.get_cursor() as cur:
            cur.execute(sql, (now, [str(mid) for mid in memory_ids]))
            return cur.rowcount

    def get_memories_for_decay(
        self, agent_id: str, batch_size: int = 100
    ) -> List[AgentMemory]:
        """減衰処理対象のメモリを取得

        睡眠フェーズで減衰処理を行うメモリを取得する。
        status='active' のメモリを strength 昇順で取得。

        Args:
            agent_id: エージェントの ID
            batch_size: 一度に取得する件数（デフォルト: 100）

        Returns:
            AgentMemory のリスト（strength 昇順）
        """
        sql = """
            SELECT {columns}
            FROM agent_memory
            WHERE agent_id = %s AND status = 'active'
            ORDER BY strength ASC
            LIMIT %s
        """.format(columns=self._COLUMNS)

        with self.db.get_cursor() as cur:
            cur.execute(sql, (agent_id, batch_size))
            rows = cur.fetchall()
            return [AgentMemory.from_row(row) for row in rows]

    def batch_update_strength(
        self, updates: List[tuple[UUID, float]]
    ) -> int:
        """複数メモリの strength を一括更新

        睡眠フェーズの減衰処理で使用。

        Args:
            updates: (memory_id, new_strength) のタプルリスト

        Returns:
            更新された行数
        """
        if not updates:
            return 0

        now = datetime.now()

        # VALUES 句を使用した一括更新
        # PostgreSQL の UPDATE FROM 構文を使用
        values_sql = ",".join(
            f"('{str(mid)}'::uuid, {strength})"
            for mid, strength in updates
        )

        sql = f"""
            UPDATE agent_memory AS m SET
                strength = v.new_strength,
                updated_at = %s,
                last_decay_at = %s
            FROM (VALUES {values_sql}) AS v(id, new_strength)
            WHERE m.id = v.id
        """

        with self.db.get_cursor() as cur:
            cur.execute(sql, (now, now))
            return cur.rowcount

    def batch_archive(self, memory_ids: List[UUID]) -> int:
        """複数メモリを一括アーカイブ

        睡眠フェーズで archive_threshold 以下になったメモリを
        一括でアーカイブする。

        Args:
            memory_ids: アーカイブするメモリの UUID リスト

        Returns:
            アーカイブされた行数
        """
        if not memory_ids:
            return 0

        now = datetime.now()

        # UUID型に明示的キャスト
        sql = """
            UPDATE agent_memory SET
                status = 'archived',
                updated_at = %s
            WHERE id = ANY(%s::uuid[])
        """

        with self.db.get_cursor() as cur:
            cur.execute(sql, (now, [str(mid) for mid in memory_ids]))
            return cur.rowcount

    def count_active_memories(self, agent_id: str) -> int:
        """アクティブなメモリの件数を取得

        容量管理で使用。

        Args:
            agent_id: エージェントの ID

        Returns:
            アクティブなメモリの件数
        """
        sql = """
            SELECT COUNT(*)
            FROM agent_memory
            WHERE agent_id = %s AND status = 'active'
        """

        with self.db.get_cursor() as cur:
            cur.execute(sql, (agent_id,))
            row = cur.fetchone()
            return row[0] if row else 0

    def get_lowest_strength_memories(
        self, agent_id: str, limit: int
    ) -> List[AgentMemory]:
        """最も強度の低いメモリを取得

        容量超過時の整理で使用。

        Args:
            agent_id: エージェントの ID
            limit: 取得件数

        Returns:
            AgentMemory のリスト（strength 昇順）
        """
        sql = """
            SELECT {columns}
            FROM agent_memory
            WHERE agent_id = %s AND status = 'active'
            ORDER BY strength ASC
            LIMIT %s
        """.format(columns=self._COLUMNS)

        with self.db.get_cursor() as cur:
            cur.execute(sql, (agent_id, limit))
            rows = cur.fetchall()
            return [AgentMemory.from_row(row) for row in rows]
