# ツール定義・実行フレームワークのテスト
# 対象: src/llm/tool_executor.py

import json
import pytest
from typing import Any, Dict
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

from src.llm.tool_executor import (
    Tool,
    ToolExecutor,
    ToolExecutionError,
    ToolExecutionResult,
    ToolHandler,
    SEARCH_MEMORIES_TOOL,
    RECORD_LEARNING_TOOL,
    BUILTIN_TOOLS,
    create_search_memories_handler,
    create_record_learning_handler,
    create_tool_executor_with_builtins,
)


# =============================================================================
# Tool dataclass テスト
# =============================================================================

class TestTool:
    """Tool dataclass のテスト"""

    def test_create_tool_basic(self):
        """基本的なツール作成"""
        tool = Tool(
            name="test_tool",
            description="テスト用ツール",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        )

        assert tool.name == "test_tool"
        assert tool.description == "テスト用ツール"
        assert tool.input_schema["type"] == "object"
        assert "query" in tool.input_schema["properties"]

    def test_to_dict_returns_claude_api_format(self):
        """to_dict() が Claude API 形式を返す"""
        tool = Tool(
            name="my_tool",
            description="My tool description",
            input_schema={
                "type": "object",
                "properties": {
                    "param1": {"type": "string", "description": "Parameter 1"}
                },
                "required": ["param1"]
            }
        )

        result = tool.to_dict()

        assert result["name"] == "my_tool"
        assert result["description"] == "My tool description"
        assert result["input_schema"]["type"] == "object"
        assert result["input_schema"]["properties"]["param1"]["type"] == "string"

    def test_tool_with_optional_parameters(self):
        """オプションパラメータを含むツール"""
        tool = Tool(
            name="optional_tool",
            description="Optional params tool",
            input_schema={
                "type": "object",
                "properties": {
                    "required_param": {"type": "string"},
                    "optional_param": {"type": "integer", "default": 10}
                },
                "required": ["required_param"]
            }
        )

        assert tool.input_schema["properties"]["optional_param"]["default"] == 10


# =============================================================================
# ToolExecutionResult テスト
# =============================================================================

class TestToolExecutionResult:
    """ToolExecutionResult のテスト"""

    def test_success_result(self):
        """成功結果"""
        result = ToolExecutionResult(
            tool_name="my_tool",
            success=True,
            result='{"status": "ok"}'
        )

        assert result.success is True
        assert result.error is None
        assert "ok" in result.result

    def test_failure_result(self):
        """失敗結果"""
        result = ToolExecutionResult(
            tool_name="my_tool",
            success=False,
            error="ツールが見つかりません"
        )

        assert result.success is False
        assert result.error == "ツールが見つかりません"

    def test_to_dict_success(self):
        """to_dict() 成功時"""
        result = ToolExecutionResult(
            tool_name="test",
            success=True,
            result="success"
        )

        data = result.to_dict()

        assert data["tool_name"] == "test"
        assert data["success"] is True
        assert data["result"] == "success"
        assert "error" not in data

    def test_to_dict_failure(self):
        """to_dict() 失敗時"""
        result = ToolExecutionResult(
            tool_name="test",
            success=False,
            error="Error message"
        )

        data = result.to_dict()

        assert data["success"] is False
        assert data["error"] == "Error message"


# =============================================================================
# ToolExecutor テスト
# =============================================================================

class TestToolExecutorInit:
    """ToolExecutor 初期化テスト"""

    def test_init_creates_empty_executor(self):
        """初期状態は空"""
        executor = ToolExecutor()

        assert executor.tool_count == 0
        assert executor.list_tools() == []


class TestToolExecutorRegister:
    """ToolExecutor.register_tool() テスト"""

    def test_register_tool_success(self):
        """ツール登録成功"""
        executor = ToolExecutor()
        tool = Tool(
            name="test_tool",
            description="Test",
            input_schema={"type": "object", "properties": {}}
        )

        def handler(input_data: Dict) -> str:
            return "result"

        executor.register_tool(tool, handler)

        assert executor.has_tool("test_tool")
        assert executor.tool_count == 1

    def test_register_multiple_tools(self):
        """複数ツール登録"""
        executor = ToolExecutor()

        for i in range(3):
            tool = Tool(
                name=f"tool_{i}",
                description=f"Tool {i}",
                input_schema={"type": "object", "properties": {}}
            )
            executor.register_tool(tool, lambda d: "result")

        assert executor.tool_count == 3
        assert set(executor.list_tools()) == {"tool_0", "tool_1", "tool_2"}

    def test_register_overwrites_existing(self):
        """同名ツールは上書き"""
        executor = ToolExecutor()
        tool1 = Tool(name="same", description="First", input_schema={})
        tool2 = Tool(name="same", description="Second", input_schema={})

        executor.register_tool(tool1, lambda d: "first")
        executor.register_tool(tool2, lambda d: "second")

        assert executor.tool_count == 1
        assert executor.get_tool("same").description == "Second"

    def test_register_empty_name_raises(self):
        """空のツール名はエラー"""
        executor = ToolExecutor()
        tool = Tool(name="", description="Test", input_schema={})

        with pytest.raises(ValueError, match="空にできません"):
            executor.register_tool(tool, lambda d: "result")

    def test_register_whitespace_name_raises(self):
        """空白のみのツール名はエラー"""
        executor = ToolExecutor()
        tool = Tool(name="   ", description="Test", input_schema={})

        with pytest.raises(ValueError, match="空にできません"):
            executor.register_tool(tool, lambda d: "result")


class TestToolExecutorUnregister:
    """ToolExecutor.unregister_tool() テスト"""

    def test_unregister_existing_tool(self):
        """登録済みツールの解除"""
        executor = ToolExecutor()
        tool = Tool(name="test", description="Test", input_schema={})
        executor.register_tool(tool, lambda d: "result")

        result = executor.unregister_tool("test")

        assert result is True
        assert not executor.has_tool("test")

    def test_unregister_nonexistent_tool(self):
        """存在しないツールの解除"""
        executor = ToolExecutor()

        result = executor.unregister_tool("nonexistent")

        assert result is False


class TestToolExecutorExecute:
    """ToolExecutor.execute_tool() テスト"""

    def test_execute_tool_success(self):
        """ツール実行成功"""
        executor = ToolExecutor()
        tool = Tool(
            name="echo",
            description="Echo input",
            input_schema={
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"]
            }
        )

        def echo_handler(input_data: Dict) -> str:
            return f"Echo: {input_data['message']}"

        executor.register_tool(tool, echo_handler)

        result = executor.execute_tool("echo", {"message": "hello"})

        assert result == "Echo: hello"

    def test_execute_tool_returns_json_for_dict(self):
        """dictを返すハンドラーはJSON変換される"""
        executor = ToolExecutor()
        tool = Tool(name="json_tool", description="Returns dict", input_schema={})

        def dict_handler(input_data: Dict) -> Dict:
            return {"status": "ok", "count": 42}

        executor.register_tool(tool, dict_handler)

        result = executor.execute_tool("json_tool", {})
        parsed = json.loads(result)

        assert parsed["status"] == "ok"
        assert parsed["count"] == 42

    def test_execute_nonexistent_tool_raises(self):
        """存在しないツール実行はエラー"""
        executor = ToolExecutor()

        with pytest.raises(ToolExecutionError, match="登録されていません"):
            executor.execute_tool("nonexistent", {})

    def test_execute_tool_handler_error_raises(self):
        """ハンドラー内のエラーは ToolExecutionError"""
        executor = ToolExecutor()
        tool = Tool(name="error_tool", description="Raises error", input_schema={})

        def error_handler(input_data: Dict) -> str:
            raise RuntimeError("Handler error")

        executor.register_tool(tool, error_handler)

        with pytest.raises(ToolExecutionError) as exc_info:
            executor.execute_tool("error_tool", {})

        assert exc_info.value.tool_name == "error_tool"
        assert exc_info.value.original_error is not None


class TestToolExecutorExecuteSafe:
    """ToolExecutor.execute_tool_safe() テスト"""

    def test_execute_safe_success(self):
        """安全実行：成功"""
        executor = ToolExecutor()
        tool = Tool(name="safe", description="Safe tool", input_schema={})
        executor.register_tool(tool, lambda d: "success")

        result = executor.execute_tool_safe("safe", {})

        assert result.success is True
        assert result.result == "success"
        assert result.error is None

    def test_execute_safe_nonexistent_tool(self):
        """安全実行：存在しないツール"""
        executor = ToolExecutor()

        result = executor.execute_tool_safe("nonexistent", {})

        assert result.success is False
        assert "登録されていません" in result.error

    def test_execute_safe_handler_error(self):
        """安全実行：ハンドラーエラー"""
        executor = ToolExecutor()
        tool = Tool(name="error", description="Error tool", input_schema={})
        executor.register_tool(tool, lambda d: 1/0)  # ZeroDivisionError

        result = executor.execute_tool_safe("error", {})

        assert result.success is False
        assert result.error is not None


class TestToolExecutorGetToolsForLLM:
    """ToolExecutor.get_tools_for_llm() テスト"""

    def test_get_tools_for_llm_empty(self):
        """ツールなしの場合は空リスト"""
        executor = ToolExecutor()

        result = executor.get_tools_for_llm()

        assert result == []

    def test_get_tools_for_llm_returns_claude_format(self):
        """Claude API 形式のツール一覧を返す"""
        executor = ToolExecutor()
        tool1 = Tool(
            name="tool1",
            description="Tool 1",
            input_schema={"type": "object", "properties": {"a": {"type": "string"}}}
        )
        tool2 = Tool(
            name="tool2",
            description="Tool 2",
            input_schema={"type": "object", "properties": {"b": {"type": "integer"}}}
        )

        executor.register_tool(tool1, lambda d: "")
        executor.register_tool(tool2, lambda d: "")

        result = executor.get_tools_for_llm()

        assert len(result) == 2
        tool_names = {t["name"] for t in result}
        assert tool_names == {"tool1", "tool2"}

        # 各ツールが正しい形式を持つ
        for tool_dict in result:
            assert "name" in tool_dict
            assert "description" in tool_dict
            assert "input_schema" in tool_dict


class TestToolExecutorHelpers:
    """ToolExecutor ヘルパーメソッドのテスト"""

    def test_get_tool_existing(self):
        """get_tool(): 存在するツール"""
        executor = ToolExecutor()
        tool = Tool(name="exists", description="Exists", input_schema={})
        executor.register_tool(tool, lambda d: "")

        result = executor.get_tool("exists")

        assert result is not None
        assert result.name == "exists"

    def test_get_tool_nonexistent(self):
        """get_tool(): 存在しないツール"""
        executor = ToolExecutor()

        result = executor.get_tool("nonexistent")

        assert result is None

    def test_has_tool_true(self):
        """has_tool(): 存在する場合"""
        executor = ToolExecutor()
        tool = Tool(name="exists", description="", input_schema={})
        executor.register_tool(tool, lambda d: "")

        assert executor.has_tool("exists") is True

    def test_has_tool_false(self):
        """has_tool(): 存在しない場合"""
        executor = ToolExecutor()

        assert executor.has_tool("nonexistent") is False


# =============================================================================
# 組み込みツール定義テスト
# =============================================================================

class TestBuiltinToolDefinitions:
    """組み込みツール定義のテスト"""

    def test_search_memories_tool_structure(self):
        """search_memories ツールの構造"""
        tool = SEARCH_MEMORIES_TOOL

        assert tool.name == "search_memories"
        assert "検索" in tool.description or "メモリ" in tool.description
        assert tool.input_schema["type"] == "object"
        assert "query" in tool.input_schema["properties"]
        assert "agent_id" in tool.input_schema["properties"]
        assert "perspective" in tool.input_schema["properties"]
        assert "query" in tool.input_schema["required"]
        assert "agent_id" in tool.input_schema["required"]

    def test_record_learning_tool_structure(self):
        """record_learning ツールの構造"""
        tool = RECORD_LEARNING_TOOL

        assert tool.name == "record_learning"
        assert "学び" in tool.description or "記録" in tool.description
        assert tool.input_schema["type"] == "object"
        assert "agent_id" in tool.input_schema["properties"]
        assert "content" in tool.input_schema["properties"]
        assert "learning" in tool.input_schema["properties"]
        assert "perspective" in tool.input_schema["properties"]
        assert "agent_id" in tool.input_schema["required"]
        assert "content" in tool.input_schema["required"]
        assert "learning" in tool.input_schema["required"]

    def test_builtin_tools_list(self):
        """BUILTIN_TOOLS リストの確認"""
        assert len(BUILTIN_TOOLS) == 2

        names = {t.name for t in BUILTIN_TOOLS}
        assert names == {"search_memories", "record_learning"}


# =============================================================================
# ハンドラー生成関数テスト
# =============================================================================

class TestCreateSearchMemoriesHandler:
    """create_search_memories_handler() テスト"""

    def test_handler_calls_task_executor_search(self):
        """ハンドラーが TaskExecutor.search_memories() を呼び出す"""
        # モックの TaskExecutor
        mock_executor = MagicMock()
        mock_scored_memory = MagicMock()
        mock_scored_memory.memory.id = uuid4()
        mock_scored_memory.memory.content = "Test content"
        mock_scored_memory.memory.strength = 0.8
        mock_scored_memory.final_score = 0.9
        mock_executor.search_memories.return_value = [mock_scored_memory]

        handler = create_search_memories_handler(mock_executor)

        result = handler({
            "query": "test query",
            "agent_id": "agent_01",
            "perspective": "test_perspective"
        })

        # search_memories が正しい引数で呼ばれた
        mock_executor.search_memories.assert_called_once_with(
            query="test query",
            agent_id="agent_01",
            perspective="test_perspective"
        )

        # 結果がJSON形式
        parsed = json.loads(result)
        assert parsed["count"] == 1
        assert len(parsed["memories"]) == 1
        assert parsed["memories"][0]["content"] == "Test content"

    def test_handler_empty_results(self):
        """検索結果が空の場合"""
        mock_executor = MagicMock()
        mock_executor.search_memories.return_value = []

        handler = create_search_memories_handler(mock_executor)

        result = handler({"query": "no match", "agent_id": "agent_01"})
        parsed = json.loads(result)

        assert parsed["count"] == 0
        assert parsed["memories"] == []

    def test_handler_without_perspective(self):
        """perspective なしで呼び出し"""
        mock_executor = MagicMock()
        mock_executor.search_memories.return_value = []

        handler = create_search_memories_handler(mock_executor)

        handler({"query": "test", "agent_id": "agent_01"})

        mock_executor.search_memories.assert_called_once_with(
            query="test",
            agent_id="agent_01",
            perspective=None
        )


class TestCreateRecordLearningHandler:
    """create_record_learning_handler() テスト"""

    def test_handler_calls_task_executor_record(self):
        """ハンドラーが TaskExecutor.record_learning() を呼び出す"""
        mock_executor = MagicMock()
        test_uuid = uuid4()
        mock_executor.record_learning.return_value = test_uuid

        handler = create_record_learning_handler(mock_executor)

        result = handler({
            "agent_id": "agent_01",
            "content": "学びの内容",
            "learning": "学びの詳細",
            "perspective": "test_perspective"
        })

        mock_executor.record_learning.assert_called_once_with(
            agent_id="agent_01",
            content="学びの内容",
            learning="学びの詳細",
            perspective="test_perspective"
        )

        parsed = json.loads(result)
        assert parsed["success"] is True
        assert parsed["memory_id"] == str(test_uuid)

    def test_handler_without_perspective(self):
        """perspective なしで呼び出し"""
        mock_executor = MagicMock()
        mock_executor.record_learning.return_value = uuid4()

        handler = create_record_learning_handler(mock_executor)

        handler({
            "agent_id": "agent_01",
            "content": "content",
            "learning": "learning"
        })

        mock_executor.record_learning.assert_called_once_with(
            agent_id="agent_01",
            content="content",
            learning="learning",
            perspective=None
        )


class TestCreateToolExecutorWithBuiltins:
    """create_tool_executor_with_builtins() テスト"""

    def test_creates_executor_with_builtin_tools(self):
        """組み込みツール付きの Executor を生成"""
        mock_task_executor = MagicMock()

        executor = create_tool_executor_with_builtins(mock_task_executor)

        assert executor.tool_count == 2
        assert executor.has_tool("search_memories")
        assert executor.has_tool("record_learning")

    def test_tools_are_callable(self):
        """生成されたツールが呼び出し可能"""
        mock_task_executor = MagicMock()
        mock_task_executor.search_memories.return_value = []
        mock_task_executor.record_learning.return_value = uuid4()

        executor = create_tool_executor_with_builtins(mock_task_executor)

        # search_memories 呼び出し
        result1 = executor.execute_tool("search_memories", {
            "query": "test",
            "agent_id": "agent_01"
        })
        assert json.loads(result1)["count"] == 0

        # record_learning 呼び出し
        result2 = executor.execute_tool("record_learning", {
            "agent_id": "agent_01",
            "content": "content",
            "learning": "learning"
        })
        assert json.loads(result2)["success"] is True

    def test_get_tools_for_llm_returns_both_tools(self):
        """get_tools_for_llm() が両方のツールを返す"""
        mock_task_executor = MagicMock()

        executor = create_tool_executor_with_builtins(mock_task_executor)
        tools = executor.get_tools_for_llm()

        assert len(tools) == 2
        names = {t["name"] for t in tools}
        assert names == {"search_memories", "record_learning"}


# =============================================================================
# ToolExecutionError テスト
# =============================================================================

class TestToolExecutionError:
    """ToolExecutionError のテスト"""

    def test_error_with_tool_name(self):
        """ツール名付きエラー"""
        error = ToolExecutionError(
            "エラーメッセージ",
            tool_name="my_tool"
        )

        assert error.message == "エラーメッセージ"
        assert error.tool_name == "my_tool"
        assert "my_tool" in str(error)

    def test_error_with_original_error(self):
        """元のエラー付き"""
        original = RuntimeError("Original error")
        error = ToolExecutionError(
            "Wrapped error",
            original_error=original
        )

        assert error.original_error is original
        assert "Original error" in str(error)

    def test_error_str_format(self):
        """__str__ のフォーマット"""
        error = ToolExecutionError(
            "Message",
            tool_name="test",
            original_error=ValueError("Detail")
        )

        error_str = str(error)
        assert "Message" in error_str
        assert "test" in error_str
        assert "Detail" in error_str


# =============================================================================
# 統合テスト
# =============================================================================

class TestToolExecutorIntegration:
    """ToolExecutor 統合テスト"""

    def test_full_workflow(self):
        """完全なワークフロー"""
        # 1. Executor 作成
        executor = ToolExecutor()

        # 2. カスタムツール登録
        counter = {"value": 0}

        def increment_handler(input_data: Dict) -> str:
            counter["value"] += input_data.get("amount", 1)
            return json.dumps({"new_value": counter["value"]})

        increment_tool = Tool(
            name="increment",
            description="カウンターをインクリメント",
            input_schema={
                "type": "object",
                "properties": {
                    "amount": {"type": "integer", "default": 1}
                }
            }
        )
        executor.register_tool(increment_tool, increment_handler)

        # 3. LLM用ツール一覧取得
        tools = executor.get_tools_for_llm()
        assert len(tools) == 1
        assert tools[0]["name"] == "increment"

        # 4. ツール実行
        result1 = executor.execute_tool("increment", {"amount": 5})
        assert json.loads(result1)["new_value"] == 5

        result2 = executor.execute_tool("increment", {})
        assert json.loads(result2)["new_value"] == 6

        # 5. ツール解除
        executor.unregister_tool("increment")
        assert executor.tool_count == 0

    def test_tool_with_claude_api_workflow(self):
        """Claude API ワークフローのシミュレーション"""
        # TaskExecutor モック
        mock_task_executor = MagicMock()
        mock_scored_memory = MagicMock()
        mock_scored_memory.memory.id = uuid4()
        mock_scored_memory.memory.content = "重要な学び"
        mock_scored_memory.memory.strength = 1.0
        mock_scored_memory.final_score = 0.95
        mock_task_executor.search_memories.return_value = [mock_scored_memory]
        mock_task_executor.record_learning.return_value = uuid4()

        # ToolExecutor 作成
        executor = create_tool_executor_with_builtins(mock_task_executor)

        # 1. Claude API に渡すツール定義を取得
        tools_for_claude = executor.get_tools_for_llm()
        assert len(tools_for_claude) == 2

        # 2. Claude からの tool_use をシミュレート
        tool_use = {
            "id": "toolu_123",
            "name": "search_memories",
            "input": {
                "query": "コスト最適化",
                "agent_id": "agent_01"
            }
        }

        # 3. ツール実行
        result = executor.execute_tool(
            tool_use["name"],
            tool_use["input"]
        )

        # 4. 結果を確認
        parsed = json.loads(result)
        assert parsed["count"] == 1
        assert parsed["memories"][0]["content"] == "重要な学び"
