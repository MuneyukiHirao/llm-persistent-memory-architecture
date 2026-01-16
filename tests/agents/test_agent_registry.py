# エージェントレジストリのテスト
"""
AgentRegistry の単体テスト

テスト観点:
- CRUD操作（register, get_by_id, get_active_agents, update, update_status, delete）
- capabilities検索（search_by_capabilities, search_by_all_capabilities）
- AgentDefinition のシリアライズ/デシリアライズ
- 境界値・異常系

注意: DBモック使用（実DB接続不要）
"""

from datetime import datetime
from typing import List
from unittest.mock import MagicMock

import pytest

from src.agents.agent_registry import AgentDefinition, AgentRegistry


class TestAgentDefinitionDataclass:
    """AgentDefinition データクラスのテスト"""

    def test_create_with_defaults(self):
        """デフォルト値での作成"""
        # Act
        agent = AgentDefinition(
            agent_id="test_agent",
            name="テストエージェント",
            role="テスト用のエージェント",
            perspectives=["正確性", "効率性"],
            system_prompt="あなたはテストエージェントです",
        )

        # Assert
        assert agent.agent_id == "test_agent"
        assert agent.name == "テストエージェント"
        assert agent.role == "テスト用のエージェント"
        assert agent.perspectives == ["正確性", "効率性"]
        assert agent.system_prompt == "あなたはテストエージェントです"
        assert agent.tools == []
        assert agent.capabilities == []
        assert agent.status == "active"
        assert agent.created_at is None
        assert agent.updated_at is None

    def test_create_with_all_fields(self):
        """全フィールド指定での作成"""
        # Arrange
        now = datetime.now()

        # Act
        agent = AgentDefinition(
            agent_id="full_agent",
            name="完全なエージェント",
            role="全機能テスト",
            perspectives=["観点1", "観点2", "観点3"],
            system_prompt="システムプロンプト",
            tools=[{"name": "tool1", "description": "ツール1"}],
            capabilities=["research", "analysis"],
            status="disabled",
            created_at=now,
            updated_at=now,
        )

        # Assert
        assert agent.tools == [{"name": "tool1", "description": "ツール1"}]
        assert agent.capabilities == ["research", "analysis"]
        assert agent.status == "disabled"
        assert agent.created_at == now
        assert agent.updated_at == now

    def test_from_row(self):
        """DBの行からインスタンス生成"""
        # Arrange
        now = datetime.now()
        row = (
            "agent_01",
            "エージェント1",
            "役割説明",
            ["正確性", "効率性"],
            "システムプロンプト",
            [{"name": "tool1"}],
            ["research"],
            "active",
            now,
            now,
        )

        # Act
        agent = AgentDefinition.from_row(row)

        # Assert
        assert agent.agent_id == "agent_01"
        assert agent.name == "エージェント1"
        assert agent.role == "役割説明"
        assert agent.perspectives == ["正確性", "効率性"]
        assert agent.system_prompt == "システムプロンプト"
        assert agent.tools == [{"name": "tool1"}]
        assert agent.capabilities == ["research"]
        assert agent.status == "active"
        assert agent.created_at == now
        assert agent.updated_at == now

    def test_from_row_with_none_values(self):
        """None値を含む行からインスタンス生成"""
        # Arrange
        row = (
            "agent_02",
            "エージェント2",
            "役割",
            None,  # perspectives
            "プロンプト",
            None,  # tools
            None,  # capabilities
            "active",
            None,  # created_at
            None,  # updated_at
        )

        # Act
        agent = AgentDefinition.from_row(row)

        # Assert
        assert agent.perspectives == []
        assert agent.tools == []
        assert agent.capabilities == []
        assert agent.created_at is None
        assert agent.updated_at is None

    def test_to_dict(self):
        """辞書に変換"""
        # Arrange
        now = datetime.now()
        agent = AgentDefinition(
            agent_id="dict_agent",
            name="辞書変換テスト",
            role="変換テスト用",
            perspectives=["観点1"],
            system_prompt="プロンプト",
            tools=[{"name": "tool1"}],
            capabilities=["cap1"],
            status="active",
            created_at=now,
            updated_at=now,
        )

        # Act
        result = agent.to_dict()

        # Assert
        assert result["agent_id"] == "dict_agent"
        assert result["name"] == "辞書変換テスト"
        assert result["perspectives"] == ["観点1"]
        assert result["tools"] == [{"name": "tool1"}]
        assert result["capabilities"] == ["cap1"]
        assert result["created_at"] == now.isoformat()
        assert result["updated_at"] == now.isoformat()

    def test_to_dict_with_none_timestamps(self):
        """タイムスタンプがNoneの場合の辞書変換"""
        # Arrange
        agent = AgentDefinition(
            agent_id="none_ts",
            name="Noneタイムスタンプ",
            role="テスト",
            perspectives=[],
            system_prompt="",
        )

        # Act
        result = agent.to_dict()

        # Assert
        assert result["created_at"] is None
        assert result["updated_at"] is None


class TestAgentRegistrySetup:
    """AgentRegistry 初期化のテスト"""

    @pytest.fixture
    def mock_db(self) -> MagicMock:
        """モック DatabaseConnection"""
        return MagicMock()

    def test_init(self, mock_db: MagicMock):
        """初期化時に db が設定される"""
        # Act
        registry = AgentRegistry(mock_db)

        # Assert
        assert registry.db is mock_db


class TestRegister:
    """register() メソッドのテスト"""

    @pytest.fixture
    def mock_cursor(self) -> MagicMock:
        """モックカーソル"""
        cursor = MagicMock()
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=None)
        return cursor

    @pytest.fixture
    def mock_db(self, mock_cursor: MagicMock) -> MagicMock:
        """モック DatabaseConnection"""
        db = MagicMock()
        db.get_cursor = MagicMock(return_value=mock_cursor)
        return db

    @pytest.fixture
    def registry(self, mock_db: MagicMock) -> AgentRegistry:
        """テスト用レジストリ"""
        return AgentRegistry(mock_db)

    def _create_agent(
        self,
        agent_id: str = "test_agent",
        name: str = "テストエージェント",
    ) -> AgentDefinition:
        """テスト用エージェントを作成"""
        return AgentDefinition(
            agent_id=agent_id,
            name=name,
            role="テスト用",
            perspectives=["正確性", "効率性"],
            system_prompt="テストプロンプト",
            capabilities=["research"],
        )

    def _make_db_row(self, agent: AgentDefinition) -> tuple:
        """AgentDefinition からDB行データを生成"""
        now = datetime.now()
        return (
            agent.agent_id,
            agent.name,
            agent.role,
            agent.perspectives,
            agent.system_prompt,
            agent.tools,
            agent.capabilities,
            agent.status,
            now,
            now,
        )

    def test_register_new_agent(
        self, registry: AgentRegistry, mock_cursor: MagicMock
    ):
        """新規エージェントの登録"""
        # Arrange
        agent = self._create_agent()
        mock_cursor.fetchone.return_value = self._make_db_row(agent)

        # Act
        result = registry.register(agent)

        # Assert
        assert result == agent.agent_id
        mock_cursor.execute.assert_called_once()

    def test_register_agent_with_tools(
        self, registry: AgentRegistry, mock_cursor: MagicMock
    ):
        """ツールを持つエージェントの登録"""
        # Arrange
        agent = AgentDefinition(
            agent_id="tool_agent",
            name="ツールエージェント",
            role="ツールテスト",
            perspectives=["観点1"],
            system_prompt="プロンプト",
            tools=[
                {"name": "search", "description": "検索ツール"},
                {"name": "write", "description": "書き込みツール"},
            ],
        )
        mock_cursor.fetchone.return_value = self._make_db_row(agent)

        # Act
        result = registry.register(agent)

        # Assert
        assert result == "tool_agent"
        # ツールがJSON形式で渡されていることを確認
        call_args = mock_cursor.execute.call_args[0][1]
        # tools は6番目の引数（index 5）
        assert call_args[5] is not None


class TestGetById:
    """get_by_id() メソッドのテスト"""

    @pytest.fixture
    def mock_cursor(self) -> MagicMock:
        cursor = MagicMock()
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=None)
        return cursor

    @pytest.fixture
    def mock_db(self, mock_cursor: MagicMock) -> MagicMock:
        db = MagicMock()
        db.get_cursor = MagicMock(return_value=mock_cursor)
        return db

    @pytest.fixture
    def registry(self, mock_db: MagicMock) -> AgentRegistry:
        return AgentRegistry(mock_db)

    def _make_db_row(self, agent_id: str) -> tuple:
        """テスト用DB行を作成"""
        now = datetime.now()
        return (
            agent_id,
            "テストエージェント",
            "テスト用",
            ["正確性"],
            "プロンプト",
            [],
            ["research"],
            "active",
            now,
            now,
        )

    def test_get_by_id_found(
        self, registry: AgentRegistry, mock_cursor: MagicMock
    ):
        """存在するIDでエージェントを取得"""
        # Arrange
        agent_id = "test_agent_01"
        mock_cursor.fetchone.return_value = self._make_db_row(agent_id)

        # Act
        result = registry.get_by_id(agent_id)

        # Assert
        assert result is not None
        assert result.agent_id == agent_id
        assert result.name == "テストエージェント"

    def test_get_by_id_not_found(
        self, registry: AgentRegistry, mock_cursor: MagicMock
    ):
        """存在しないIDでNoneを返す"""
        # Arrange
        mock_cursor.fetchone.return_value = None

        # Act
        result = registry.get_by_id("nonexistent")

        # Assert
        assert result is None


class TestGetActiveAgents:
    """get_active_agents() メソッドのテスト"""

    @pytest.fixture
    def mock_cursor(self) -> MagicMock:
        cursor = MagicMock()
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=None)
        return cursor

    @pytest.fixture
    def mock_db(self, mock_cursor: MagicMock) -> MagicMock:
        db = MagicMock()
        db.get_cursor = MagicMock(return_value=mock_cursor)
        return db

    @pytest.fixture
    def registry(self, mock_db: MagicMock) -> AgentRegistry:
        return AgentRegistry(mock_db)

    def _make_db_row(self, agent_id: str, name: str) -> tuple:
        now = datetime.now()
        return (
            agent_id,
            name,
            "役割",
            ["観点1"],
            "プロンプト",
            [],
            [],
            "active",
            now,
            now,
        )

    def test_get_active_agents_returns_list(
        self, registry: AgentRegistry, mock_cursor: MagicMock
    ):
        """アクティブなエージェントリストを取得"""
        # Arrange
        mock_cursor.fetchall.return_value = [
            self._make_db_row("agent_01", "エージェント1"),
            self._make_db_row("agent_02", "エージェント2"),
            self._make_db_row("agent_03", "エージェント3"),
        ]

        # Act
        result = registry.get_active_agents()

        # Assert
        assert len(result) == 3
        assert result[0].agent_id == "agent_01"
        assert result[1].agent_id == "agent_02"

    def test_get_active_agents_empty(
        self, registry: AgentRegistry, mock_cursor: MagicMock
    ):
        """アクティブなエージェントがない場合は空リスト"""
        # Arrange
        mock_cursor.fetchall.return_value = []

        # Act
        result = registry.get_active_agents()

        # Assert
        assert result == []

    def test_get_active_agents_sql_filters_status(
        self, registry: AgentRegistry, mock_cursor: MagicMock
    ):
        """SQLで status='active' がフィルタされる"""
        # Arrange
        mock_cursor.fetchall.return_value = []

        # Act
        registry.get_active_agents()

        # Assert
        sql = mock_cursor.execute.call_args[0][0]
        assert "status = 'active'" in sql


class TestSearchByCapabilities:
    """search_by_capabilities() メソッドのテスト"""

    @pytest.fixture
    def mock_cursor(self) -> MagicMock:
        cursor = MagicMock()
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=None)
        return cursor

    @pytest.fixture
    def mock_db(self, mock_cursor: MagicMock) -> MagicMock:
        db = MagicMock()
        db.get_cursor = MagicMock(return_value=mock_cursor)
        return db

    @pytest.fixture
    def registry(self, mock_db: MagicMock) -> AgentRegistry:
        return AgentRegistry(mock_db)

    def _make_db_row(
        self, agent_id: str, capabilities: List[str]
    ) -> tuple:
        now = datetime.now()
        return (
            agent_id,
            "テストエージェント",
            "役割",
            ["観点1"],
            "プロンプト",
            [],
            capabilities,
            "active",
            now,
            now,
        )

    def test_search_by_capabilities_found(
        self, registry: AgentRegistry, mock_cursor: MagicMock
    ):
        """capabilitiesでエージェントを検索"""
        # Arrange
        mock_cursor.fetchall.return_value = [
            self._make_db_row("agent_01", ["research", "analysis"]),
            self._make_db_row("agent_02", ["research"]),
        ]

        # Act
        result = registry.search_by_capabilities(["research"])

        # Assert
        assert len(result) == 2
        mock_cursor.execute.assert_called_once()
        sql = mock_cursor.execute.call_args[0][0]
        assert "capabilities &&" in sql  # OR検索（&&演算子）

    def test_search_by_capabilities_empty_list(
        self, registry: AgentRegistry, mock_cursor: MagicMock
    ):
        """空のcapabilitiesリストでは空リストを返す"""
        # Arrange & Act
        result = registry.search_by_capabilities([])

        # Assert
        assert result == []
        mock_cursor.execute.assert_not_called()

    def test_search_by_capabilities_uses_gin_index(
        self, registry: AgentRegistry, mock_cursor: MagicMock
    ):
        """GINインデックスを活用した検索"""
        # Arrange
        mock_cursor.fetchall.return_value = []

        # Act
        registry.search_by_capabilities(["analysis", "documentation"])

        # Assert
        sql = mock_cursor.execute.call_args[0][0]
        # GINインデックスを使用する && 演算子が含まれる
        assert "&&" in sql
        # パラメータが配列として渡される
        call_args = mock_cursor.execute.call_args[0][1]
        assert call_args[0] == ["analysis", "documentation"]


class TestSearchByAllCapabilities:
    """search_by_all_capabilities() メソッドのテスト"""

    @pytest.fixture
    def mock_cursor(self) -> MagicMock:
        cursor = MagicMock()
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=None)
        return cursor

    @pytest.fixture
    def mock_db(self, mock_cursor: MagicMock) -> MagicMock:
        db = MagicMock()
        db.get_cursor = MagicMock(return_value=mock_cursor)
        return db

    @pytest.fixture
    def registry(self, mock_db: MagicMock) -> AgentRegistry:
        return AgentRegistry(mock_db)

    def _make_db_row(
        self, agent_id: str, capabilities: List[str]
    ) -> tuple:
        now = datetime.now()
        return (
            agent_id,
            "テストエージェント",
            "役割",
            ["観点1"],
            "プロンプト",
            [],
            capabilities,
            "active",
            now,
            now,
        )

    def test_search_by_all_capabilities_and_search(
        self, registry: AgentRegistry, mock_cursor: MagicMock
    ):
        """すべてのcapabilitiesを持つエージェントを検索（AND検索）"""
        # Arrange
        mock_cursor.fetchall.return_value = [
            self._make_db_row("agent_01", ["research", "analysis", "documentation"]),
        ]

        # Act
        result = registry.search_by_all_capabilities(["research", "analysis"])

        # Assert
        assert len(result) == 1
        sql = mock_cursor.execute.call_args[0][0]
        assert "@>" in sql  # AND検索（@>演算子）

    def test_search_by_all_capabilities_empty_returns_all_active(
        self, registry: AgentRegistry, mock_cursor: MagicMock
    ):
        """空のリストでは全アクティブエージェントを返す"""
        # Arrange
        mock_cursor.fetchall.return_value = [
            self._make_db_row("agent_01", []),
            self._make_db_row("agent_02", ["research"]),
        ]

        # Act
        result = registry.search_by_all_capabilities([])

        # Assert
        assert len(result) == 2


class TestUpdate:
    """update() メソッドのテスト"""

    @pytest.fixture
    def mock_cursor(self) -> MagicMock:
        cursor = MagicMock()
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=None)
        return cursor

    @pytest.fixture
    def mock_db(self, mock_cursor: MagicMock) -> MagicMock:
        db = MagicMock()
        db.get_cursor = MagicMock(return_value=mock_cursor)
        return db

    @pytest.fixture
    def registry(self, mock_db: MagicMock) -> AgentRegistry:
        return AgentRegistry(mock_db)

    def _make_db_row(self, agent: AgentDefinition) -> tuple:
        now = datetime.now()
        return (
            agent.agent_id,
            agent.name,
            agent.role,
            agent.perspectives,
            agent.system_prompt,
            agent.tools,
            agent.capabilities,
            agent.status,
            now,
            now,
        )

    def test_update_existing_agent(
        self, registry: AgentRegistry, mock_cursor: MagicMock
    ):
        """既存エージェントの更新"""
        # Arrange
        agent = AgentDefinition(
            agent_id="update_test",
            name="更新後の名前",
            role="更新後の役割",
            perspectives=["新しい観点"],
            system_prompt="新しいプロンプト",
        )
        mock_cursor.fetchone.return_value = self._make_db_row(agent)

        # Act
        result = registry.update(agent)

        # Assert
        assert result.name == "更新後の名前"
        assert result.role == "更新後の役割"
        mock_cursor.execute.assert_called_once()

    def test_update_nonexistent_agent_raises_error(
        self, registry: AgentRegistry, mock_cursor: MagicMock
    ):
        """存在しないエージェントの更新でValueError"""
        # Arrange
        agent = AgentDefinition(
            agent_id="nonexistent",
            name="存在しない",
            role="役割",
            perspectives=[],
            system_prompt="",
        )
        mock_cursor.fetchone.return_value = None

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            registry.update(agent)

        assert "not found" in str(exc_info.value)


class TestUpdateStatus:
    """update_status() メソッドのテスト"""

    @pytest.fixture
    def mock_cursor(self) -> MagicMock:
        cursor = MagicMock()
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=None)
        return cursor

    @pytest.fixture
    def mock_db(self, mock_cursor: MagicMock) -> MagicMock:
        db = MagicMock()
        db.get_cursor = MagicMock(return_value=mock_cursor)
        return db

    @pytest.fixture
    def registry(self, mock_db: MagicMock) -> AgentRegistry:
        return AgentRegistry(mock_db)

    def test_update_status_to_disabled(
        self, registry: AgentRegistry, mock_cursor: MagicMock
    ):
        """ステータスをdisabledに更新"""
        # Arrange
        mock_cursor.rowcount = 1

        # Act
        result = registry.update_status("agent_01", "disabled")

        # Assert
        assert result is True
        call_args = mock_cursor.execute.call_args[0][1]
        assert call_args[0] == "disabled"

    def test_update_status_to_active(
        self, registry: AgentRegistry, mock_cursor: MagicMock
    ):
        """ステータスをactiveに更新"""
        # Arrange
        mock_cursor.rowcount = 1

        # Act
        result = registry.update_status("agent_01", "active")

        # Assert
        assert result is True

    def test_update_status_nonexistent_agent(
        self, registry: AgentRegistry, mock_cursor: MagicMock
    ):
        """存在しないエージェントのステータス更新はFalse"""
        # Arrange
        mock_cursor.rowcount = 0

        # Act
        result = registry.update_status("nonexistent", "disabled")

        # Assert
        assert result is False


class TestDelete:
    """delete() メソッドのテスト"""

    @pytest.fixture
    def mock_cursor(self) -> MagicMock:
        cursor = MagicMock()
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=None)
        return cursor

    @pytest.fixture
    def mock_db(self, mock_cursor: MagicMock) -> MagicMock:
        db = MagicMock()
        db.get_cursor = MagicMock(return_value=mock_cursor)
        return db

    @pytest.fixture
    def registry(self, mock_db: MagicMock) -> AgentRegistry:
        return AgentRegistry(mock_db)

    def test_delete_existing_agent(
        self, registry: AgentRegistry, mock_cursor: MagicMock
    ):
        """存在するエージェントの削除"""
        # Arrange
        mock_cursor.rowcount = 1

        # Act
        result = registry.delete("agent_01")

        # Assert
        assert result is True
        sql = mock_cursor.execute.call_args[0][0]
        assert "DELETE FROM" in sql

    def test_delete_nonexistent_agent(
        self, registry: AgentRegistry, mock_cursor: MagicMock
    ):
        """存在しないエージェントの削除はFalse"""
        # Arrange
        mock_cursor.rowcount = 0

        # Act
        result = registry.delete("nonexistent")

        # Assert
        assert result is False


class TestGetAll:
    """get_all() メソッドのテスト"""

    @pytest.fixture
    def mock_cursor(self) -> MagicMock:
        cursor = MagicMock()
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=None)
        return cursor

    @pytest.fixture
    def mock_db(self, mock_cursor: MagicMock) -> MagicMock:
        db = MagicMock()
        db.get_cursor = MagicMock(return_value=mock_cursor)
        return db

    @pytest.fixture
    def registry(self, mock_db: MagicMock) -> AgentRegistry:
        return AgentRegistry(mock_db)

    def _make_db_row(self, agent_id: str, status: str) -> tuple:
        now = datetime.now()
        return (
            agent_id,
            "テストエージェント",
            "役割",
            ["観点1"],
            "プロンプト",
            [],
            [],
            status,
            now,
            now,
        )

    def test_get_all_includes_disabled(
        self, registry: AgentRegistry, mock_cursor: MagicMock
    ):
        """全エージェントを取得（disabledも含む）"""
        # Arrange
        mock_cursor.fetchall.return_value = [
            self._make_db_row("agent_01", "active"),
            self._make_db_row("agent_02", "disabled"),
        ]

        # Act
        result = registry.get_all()

        # Assert
        assert len(result) == 2
        sql = mock_cursor.execute.call_args[0][0]
        assert "status = 'active'" not in sql  # ステータスフィルタなし


class TestCountActive:
    """count_active() メソッドのテスト"""

    @pytest.fixture
    def mock_cursor(self) -> MagicMock:
        cursor = MagicMock()
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=None)
        return cursor

    @pytest.fixture
    def mock_db(self, mock_cursor: MagicMock) -> MagicMock:
        db = MagicMock()
        db.get_cursor = MagicMock(return_value=mock_cursor)
        return db

    @pytest.fixture
    def registry(self, mock_db: MagicMock) -> AgentRegistry:
        return AgentRegistry(mock_db)

    def test_count_active(
        self, registry: AgentRegistry, mock_cursor: MagicMock
    ):
        """アクティブなエージェント数を取得"""
        # Arrange
        mock_cursor.fetchone.return_value = (5,)

        # Act
        result = registry.count_active()

        # Assert
        assert result == 5
        sql = mock_cursor.execute.call_args[0][0]
        assert "COUNT(*)" in sql
        assert "status = 'active'" in sql

    def test_count_active_zero(
        self, registry: AgentRegistry, mock_cursor: MagicMock
    ):
        """アクティブなエージェントがない場合は0"""
        # Arrange
        mock_cursor.fetchone.return_value = (0,)

        # Act
        result = registry.count_active()

        # Assert
        assert result == 0


class TestEdgeCases:
    """境界値・異常系のテスト"""

    @pytest.fixture
    def mock_cursor(self) -> MagicMock:
        cursor = MagicMock()
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=None)
        return cursor

    @pytest.fixture
    def mock_db(self, mock_cursor: MagicMock) -> MagicMock:
        db = MagicMock()
        db.get_cursor = MagicMock(return_value=mock_cursor)
        return db

    @pytest.fixture
    def registry(self, mock_db: MagicMock) -> AgentRegistry:
        return AgentRegistry(mock_db)

    def test_agent_with_long_system_prompt(
        self, registry: AgentRegistry, mock_cursor: MagicMock
    ):
        """長いシステムプロンプトを持つエージェント"""
        # Arrange
        long_prompt = "x" * 10000
        agent = AgentDefinition(
            agent_id="long_prompt_agent",
            name="長いプロンプト",
            role="テスト",
            perspectives=["観点1"],
            system_prompt=long_prompt,
        )
        now = datetime.now()
        mock_cursor.fetchone.return_value = (
            agent.agent_id,
            agent.name,
            agent.role,
            agent.perspectives,
            long_prompt,
            [],
            [],
            "active",
            now,
            now,
        )

        # Act
        result = registry.register(agent)

        # Assert
        assert result == "long_prompt_agent"

    def test_agent_with_many_capabilities(
        self, registry: AgentRegistry, mock_cursor: MagicMock
    ):
        """多くのcapabilitiesを持つエージェント"""
        # Arrange
        capabilities = [f"cap_{i}" for i in range(100)]
        agent = AgentDefinition(
            agent_id="many_caps_agent",
            name="多数のcapabilities",
            role="テスト",
            perspectives=["観点1"],
            system_prompt="プロンプト",
            capabilities=capabilities,
        )
        now = datetime.now()
        mock_cursor.fetchone.return_value = (
            agent.agent_id,
            agent.name,
            agent.role,
            agent.perspectives,
            agent.system_prompt,
            [],
            capabilities,
            "active",
            now,
            now,
        )

        # Act
        result = registry.register(agent)

        # Assert
        assert result == "many_caps_agent"

    def test_agent_with_complex_tools(
        self, registry: AgentRegistry, mock_cursor: MagicMock
    ):
        """複雑なtools定義を持つエージェント"""
        # Arrange
        tools = [
            {
                "name": "search",
                "description": "検索ツール",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "default": 10},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "write",
                "description": "書き込みツール",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                },
            },
        ]
        agent = AgentDefinition(
            agent_id="complex_tools_agent",
            name="複雑なtools",
            role="テスト",
            perspectives=["観点1"],
            system_prompt="プロンプト",
            tools=tools,
        )
        now = datetime.now()
        mock_cursor.fetchone.return_value = (
            agent.agent_id,
            agent.name,
            agent.role,
            agent.perspectives,
            agent.system_prompt,
            tools,
            [],
            "active",
            now,
            now,
        )

        # Act
        result = registry.register(agent)

        # Assert
        assert result == "complex_tools_agent"


class TestSqlInjectionPrevention:
    """SQLインジェクション防止のテスト"""

    @pytest.fixture
    def mock_cursor(self) -> MagicMock:
        cursor = MagicMock()
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=None)
        return cursor

    @pytest.fixture
    def mock_db(self, mock_cursor: MagicMock) -> MagicMock:
        db = MagicMock()
        db.get_cursor = MagicMock(return_value=mock_cursor)
        return db

    @pytest.fixture
    def registry(self, mock_db: MagicMock) -> AgentRegistry:
        return AgentRegistry(mock_db)

    def test_get_by_id_with_injection_attempt(
        self, registry: AgentRegistry, mock_cursor: MagicMock
    ):
        """SQLインジェクション試行がパラメータ化クエリで防止される"""
        # Arrange
        malicious_id = "'; DROP TABLE agent_definitions; --"
        mock_cursor.fetchone.return_value = None

        # Act
        registry.get_by_id(malicious_id)

        # Assert - パラメータ化クエリで安全に処理される
        call_args = mock_cursor.execute.call_args[0][1]
        assert call_args[0] == malicious_id

    def test_search_by_capabilities_with_injection_attempt(
        self, registry: AgentRegistry, mock_cursor: MagicMock
    ):
        """capabilities検索でのSQLインジェクション防止"""
        # Arrange
        malicious_caps = ["research'; DROP TABLE agent_definitions; --"]
        mock_cursor.fetchall.return_value = []

        # Act
        registry.search_by_capabilities(malicious_caps)

        # Assert
        call_args = mock_cursor.execute.call_args[0][1]
        assert call_args[0] == malicious_caps
