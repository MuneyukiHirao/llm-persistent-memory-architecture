# エージェントレジストリ
# agent_definitions テーブルに対するCRUD操作を提供
# 実装仕様: docs/phase2-implementation-spec.ja.md セクション3.1, 5.3
"""
エージェントレジストリモジュール

agent_definitions テーブルに対するCRUD操作とcapabilities検索を提供。

設計方針（メモリ管理エージェント観点）:
- データ整合性: DBの制約に沿った検証（agent_id重複チェック等）
- パフォーマンス: GINインデックスを活用したcapabilities検索
- エラー耐性: 存在しないagent_idに対する操作を適切にハンドリング
- トランザクション: 単一操作の原子性をコンテキストマネージャーで保証
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any

from psycopg2.extras import Json

from src.db.connection import DatabaseConnection


@dataclass
class AgentDefinition:
    """エージェント定義

    agent_definitions テーブルの1行に対応するデータクラス。

    Attributes:
        agent_id: エージェントの一意識別子（PRIMARY KEY）
        name: エージェント名
        role: エージェントの役割説明
        perspectives: 観点のリスト（5つ程度）
        system_prompt: システムプロンプト
        tools: ツール定義（JSON Schema形式）
        capabilities: 能力タグ（ルーティング判断に使用）
        status: ステータス（active / disabled）
        created_at: 作成日時
        updated_at: 更新日時
    """

    agent_id: str
    name: str
    role: str
    perspectives: List[str]
    system_prompt: str
    tools: List[Dict[str, Any]] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    status: str = "active"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @classmethod
    def from_row(cls, row: tuple) -> "AgentDefinition":
        """DBの行からインスタンス生成

        Args:
            row: DBから取得した行（カラム順序は_COLUMNSに準拠）

        Returns:
            AgentDefinition インスタンス
        """
        return cls(
            agent_id=row[0],
            name=row[1],
            role=row[2],
            perspectives=row[3] if row[3] else [],
            system_prompt=row[4],
            tools=row[5] if row[5] else [],
            capabilities=row[6] if row[6] else [],
            status=row[7],
            created_at=row[8],
            updated_at=row[9],
        )

    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換

        Returns:
            エージェント定義を辞書形式で返却
        """
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "role": self.role,
            "perspectives": self.perspectives,
            "system_prompt": self.system_prompt,
            "tools": self.tools,
            "capabilities": self.capabilities,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class AgentRegistry:
    """エージェント登録・管理

    agent_definitions テーブルに対するCRUD操作を提供。

    使用例:
        db = DatabaseConnection()
        registry = AgentRegistry(db)

        # エージェント登録
        agent = AgentDefinition(
            agent_id="research_agent_01",
            name="調査エージェント",
            role="技術調査とリサーチを担当",
            perspectives=["正確性", "網羅性", "効率性", "信頼性", "関連性"],
            system_prompt="あなたは調査専門のエージェントです...",
            capabilities=["research", "analysis", "documentation"],
        )
        agent_id = registry.register(agent)

        # エージェント検索（capabilities使用）
        agents = registry.search_by_capabilities(["research"])

    Attributes:
        db: DatabaseConnection インスタンス
    """

    # SQL定義（可読性のために定数として定義）
    _COLUMNS = """
        agent_id, name, role, perspectives, system_prompt,
        tools, capabilities, status, created_at, updated_at
    """

    _INSERT_SQL = """
        INSERT INTO agent_definitions (
            agent_id, name, role, perspectives, system_prompt,
            tools, capabilities, status, created_at, updated_at
        ) VALUES (
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s
        )
        RETURNING {columns}
    """.format(columns=_COLUMNS)

    _SELECT_BY_ID_SQL = """
        SELECT {columns}
        FROM agent_definitions
        WHERE agent_id = %s
    """.format(columns=_COLUMNS)

    _SELECT_ACTIVE_SQL = """
        SELECT {columns}
        FROM agent_definitions
        WHERE status = 'active'
        ORDER BY name
    """.format(columns=_COLUMNS)

    _SELECT_BY_CAPABILITIES_SQL = """
        SELECT {columns}
        FROM agent_definitions
        WHERE status = 'active'
        AND capabilities && %s
        ORDER BY name
    """.format(columns=_COLUMNS)

    _UPDATE_SQL = """
        UPDATE agent_definitions SET
            name = %s,
            role = %s,
            perspectives = %s,
            system_prompt = %s,
            tools = %s,
            capabilities = %s,
            status = %s,
            updated_at = %s
        WHERE agent_id = %s
        RETURNING {columns}
    """.format(columns=_COLUMNS)

    _UPDATE_STATUS_SQL = """
        UPDATE agent_definitions SET
            status = %s,
            updated_at = %s
        WHERE agent_id = %s
    """

    _DELETE_SQL = """
        DELETE FROM agent_definitions
        WHERE agent_id = %s
    """

    def __init__(self, db: DatabaseConnection):
        """AgentRegistry を初期化

        Args:
            db: DatabaseConnection インスタンス
        """
        self.db = db

    def register(self, agent: AgentDefinition) -> str:
        """エージェントを登録

        Args:
            agent: 登録する AgentDefinition インスタンス

        Returns:
            登録されたエージェントのagent_id

        Raises:
            psycopg2.errors.UniqueViolation: agent_idが重複している場合
        """
        now = datetime.now()

        with self.db.get_cursor() as cur:
            cur.execute(
                self._INSERT_SQL,
                (
                    agent.agent_id,
                    agent.name,
                    agent.role,
                    agent.perspectives,
                    agent.system_prompt,
                    Json(agent.tools),
                    agent.capabilities,
                    agent.status,
                    now,
                    now,
                ),
            )
            row = cur.fetchone()
            return row[0] if row else agent.agent_id

    def get_by_id(self, agent_id: str) -> Optional[AgentDefinition]:
        """IDでエージェントを取得

        Args:
            agent_id: エージェントの ID

        Returns:
            AgentDefinition インスタンス、見つからない場合は None
        """
        with self.db.get_cursor() as cur:
            cur.execute(self._SELECT_BY_ID_SQL, (agent_id,))
            row = cur.fetchone()
            if row is None:
                return None
            return AgentDefinition.from_row(row)

    def get_active_agents(self) -> List[AgentDefinition]:
        """アクティブなエージェント一覧を取得

        Returns:
            status='active' の AgentDefinition リスト（name順）
        """
        with self.db.get_cursor() as cur:
            cur.execute(self._SELECT_ACTIVE_SQL)
            rows = cur.fetchall()
            return [AgentDefinition.from_row(row) for row in rows]

    def search_by_capabilities(
        self, capabilities: List[str]
    ) -> List[AgentDefinition]:
        """capabilities でエージェントを検索

        GINインデックスを使用した効率的な配列重複検索（&&演算子）。
        指定したcapabilitiesのいずれかを持つエージェントを返す。

        Args:
            capabilities: 検索する能力タグのリスト

        Returns:
            マッチした AgentDefinition リスト（name順）

        Note:
            - 空のリストを指定した場合は空のリストを返す
            - OR検索（いずれかのcapabilityを持つ）
        """
        if not capabilities:
            return []

        with self.db.get_cursor() as cur:
            cur.execute(self._SELECT_BY_CAPABILITIES_SQL, (capabilities,))
            rows = cur.fetchall()
            return [AgentDefinition.from_row(row) for row in rows]

    def update(self, agent: AgentDefinition) -> AgentDefinition:
        """エージェントを更新

        agent_id, created_at は変更されない。

        Args:
            agent: 更新する AgentDefinition インスタンス

        Returns:
            更新された AgentDefinition

        Raises:
            ValueError: 指定した agent_id のエージェントが存在しない場合
        """
        now = datetime.now()

        with self.db.get_cursor() as cur:
            cur.execute(
                self._UPDATE_SQL,
                (
                    agent.name,
                    agent.role,
                    agent.perspectives,
                    agent.system_prompt,
                    Json(agent.tools),
                    agent.capabilities,
                    agent.status,
                    now,
                    agent.agent_id,
                ),
            )
            row = cur.fetchone()
            if row is None:
                raise ValueError(f"Agent with id {agent.agent_id} not found")
            return AgentDefinition.from_row(row)

    def update_status(self, agent_id: str, status: str) -> bool:
        """エージェントのステータスを更新

        Args:
            agent_id: エージェントの ID
            status: 新しいステータス（'active' / 'disabled'）

        Returns:
            True: 更新成功、False: 対象のエージェントが存在しない
        """
        now = datetime.now()

        with self.db.get_cursor() as cur:
            cur.execute(self._UPDATE_STATUS_SQL, (status, now, agent_id))
            return cur.rowcount > 0

    def delete(self, agent_id: str) -> bool:
        """エージェントを削除

        物理削除を実行。論理削除の場合は update_status を使用。

        Args:
            agent_id: 削除するエージェントの ID

        Returns:
            True: 削除成功、False: 対象のエージェントが存在しない

        Note:
            - routing_history テーブルに外部キー制約があるため、
              関連する履歴がある場合は削除に失敗する
        """
        with self.db.get_cursor() as cur:
            cur.execute(self._DELETE_SQL, (agent_id,))
            return cur.rowcount > 0

    def get_all(self) -> List[AgentDefinition]:
        """全エージェントを取得（ステータス問わず）

        Returns:
            全ての AgentDefinition リスト（name順）
        """
        sql = """
            SELECT {columns}
            FROM agent_definitions
            ORDER BY name
        """.format(columns=self._COLUMNS)

        with self.db.get_cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
            return [AgentDefinition.from_row(row) for row in rows]

    def count_active(self) -> int:
        """アクティブなエージェント数を取得

        Returns:
            status='active' のエージェント数
        """
        sql = """
            SELECT COUNT(*)
            FROM agent_definitions
            WHERE status = 'active'
        """

        with self.db.get_cursor() as cur:
            cur.execute(sql)
            row = cur.fetchone()
            return row[0] if row else 0

    def search_by_all_capabilities(
        self, capabilities: List[str]
    ) -> List[AgentDefinition]:
        """すべてのcapabilitiesを持つエージェントを検索

        AND検索（すべてのcapabilityを持つエージェントのみ返す）。

        Args:
            capabilities: 検索する能力タグのリスト

        Returns:
            マッチした AgentDefinition リスト（name順）
        """
        if not capabilities:
            return self.get_active_agents()

        # @> 演算子: 左辺が右辺のすべての要素を含む
        sql = """
            SELECT {columns}
            FROM agent_definitions
            WHERE status = 'active'
            AND capabilities @> %s
            ORDER BY name
        """.format(columns=self._COLUMNS)

        with self.db.get_cursor() as cur:
            cur.execute(sql, (capabilities,))
            rows = cur.fetchall()
            return [AgentDefinition.from_row(row) for row in rows]
