# LLMタスク実行のテスト
# 対象: src/llm/llm_task_executor.py

import json
import pytest
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch, call
from uuid import uuid4

from src.llm.llm_task_executor import (
    LLMTaskExecutor,
    LLMTaskExecutorError,
    LLMTaskResult,
    ToolCallRecord,
)
from src.llm.claude_client import ClaudeResponse, ToolUse
from src.llm.tool_executor import ToolExecutionResult


# =============================================================================
# ToolCallRecord dataclass テスト
# =============================================================================

class TestToolCallRecord:
    """ToolCallRecord dataclass のテスト"""

    def test_create_success_record(self):
        """成功したツール呼び出し記録"""
        record = ToolCallRecord(
            tool_use_id="toolu_123",
            tool_name="search_memories",
            tool_input={"query": "test", "agent_id": "agent_01"},
            result='{"count": 1}',
            success=True,
        )

        assert record.tool_use_id == "toolu_123"
        assert record.tool_name == "search_memories"
        assert record.tool_input["query"] == "test"
        assert record.result == '{"count": 1}'
        assert record.success is True
        assert record.error is None

    def test_create_failure_record(self):
        """失敗したツール呼び出し記録"""
        record = ToolCallRecord(
            tool_use_id="toolu_456",
            tool_name="unknown_tool",
            tool_input={"data": "value"},
            result="",
            success=False,
            error="ツールが登録されていません",
        )

        assert record.success is False
        assert record.error == "ツールが登録されていません"
        assert record.result == ""

    def test_to_dict_success(self):
        """to_dict() 成功時"""
        record = ToolCallRecord(
            tool_use_id="toolu_123",
            tool_name="my_tool",
            tool_input={"key": "value"},
            result="result",
            success=True,
        )

        data = record.to_dict()

        assert data["tool_use_id"] == "toolu_123"
        assert data["tool_name"] == "my_tool"
        assert data["tool_input"] == {"key": "value"}
        assert data["result"] == "result"
        assert data["success"] is True
        assert "error" not in data

    def test_to_dict_failure(self):
        """to_dict() 失敗時"""
        record = ToolCallRecord(
            tool_use_id="toolu_123",
            tool_name="my_tool",
            tool_input={},
            success=False,
            error="Error message",
        )

        data = record.to_dict()

        assert data["success"] is False
        assert data["error"] == "Error message"


# =============================================================================
# LLMTaskResult dataclass テスト
# =============================================================================

class TestLLMTaskResult:
    """LLMTaskResult dataclass のテスト"""

    def test_create_basic_result(self):
        """基本的な結果作成"""
        result = LLMTaskResult(
            content="タスク実行結果",
            total_tokens=1000,
            iterations=1,
            stop_reason="end_turn",
        )

        assert result.content == "タスク実行結果"
        assert result.tool_calls == []
        assert result.total_tokens == 1000
        assert result.iterations == 1
        assert result.stop_reason == "end_turn"

    def test_create_result_with_tool_calls(self):
        """ツール呼び出し記録付きの結果"""
        tool_calls = [
            ToolCallRecord(
                tool_use_id="toolu_1",
                tool_name="search_memories",
                tool_input={"query": "test"},
                result='{"count": 1}',
                success=True,
            ),
            ToolCallRecord(
                tool_use_id="toolu_2",
                tool_name="record_learning",
                tool_input={"content": "learning"},
                result='{"success": true}',
                success=True,
            ),
        ]

        result = LLMTaskResult(
            content="完了",
            tool_calls=tool_calls,
            total_tokens=2000,
            iterations=3,
            searched_memories_count=5,
        )

        assert len(result.tool_calls) == 2
        assert result.searched_memories_count == 5

    def test_to_dict(self):
        """to_dict() の変換"""
        tool_call = ToolCallRecord(
            tool_use_id="toolu_1",
            tool_name="test",
            tool_input={"a": 1},
            result="result",
            success=True,
        )

        result = LLMTaskResult(
            content="result content",
            tool_calls=[tool_call],
            total_tokens=500,
            iterations=2,
            searched_memories_count=3,
            stop_reason="end_turn",
        )

        data = result.to_dict()

        assert data["content"] == "result content"
        assert len(data["tool_calls"]) == 1
        assert data["tool_calls"][0]["tool_name"] == "test"
        assert data["total_tokens"] == 500
        assert data["iterations"] == 2
        assert data["searched_memories_count"] == 3
        assert data["stop_reason"] == "end_turn"
        assert data["success"] is True

    def test_to_dict_max_iterations_not_success(self):
        """max_iterations で終了した場合は success = False"""
        result = LLMTaskResult(
            content="",
            stop_reason="max_iterations",
        )

        data = result.to_dict()

        assert data["success"] is False


# =============================================================================
# LLMTaskExecutorError テスト
# =============================================================================

class TestLLMTaskExecutorError:
    """LLMTaskExecutorError のテスト"""

    def test_basic_error(self):
        """基本的なエラー"""
        error = LLMTaskExecutorError("エラーメッセージ")

        assert error.message == "エラーメッセージ"
        assert error.iterations == 0
        assert error.original_error is None

    def test_error_with_iterations(self):
        """イテレーション回数付きエラー"""
        error = LLMTaskExecutorError(
            "最大イテレーション超過",
            iterations=10,
        )

        assert error.iterations == 10
        assert "(iterations=10)" in str(error)

    def test_error_with_original_error(self):
        """元のエラー付き"""
        original = RuntimeError("Original")
        error = LLMTaskExecutorError(
            "Wrapped",
            original_error=original,
        )

        assert error.original_error is original
        assert "Original" in str(error)


# =============================================================================
# LLMTaskExecutor 初期化テスト
# =============================================================================

class TestLLMTaskExecutorInit:
    """LLMTaskExecutor 初期化テスト"""

    def test_init_success(self):
        """初期化成功"""
        mock_claude_client = MagicMock()
        mock_tool_executor = MagicMock()
        mock_task_executor = MagicMock()

        executor = LLMTaskExecutor(
            claude_client=mock_claude_client,
            tool_executor=mock_tool_executor,
            task_executor=mock_task_executor,
        )

        assert executor.claude_client is mock_claude_client
        assert executor.tool_executor is mock_tool_executor
        assert executor.task_executor is mock_task_executor


# =============================================================================
# LLMTaskExecutor.execute_task_with_tools テスト
# =============================================================================

class TestExecuteTaskWithToolsBasic:
    """execute_task_with_tools() 基本テスト"""

    def setup_method(self):
        """テストセットアップ"""
        self.mock_claude_client = MagicMock()
        self.mock_tool_executor = MagicMock()
        self.mock_task_executor = MagicMock()

        # デフォルトでは空の検索結果
        self.mock_task_executor.search_memories.return_value = []

        # デフォルトではツールなし
        self.mock_tool_executor.get_tools_for_llm.return_value = []

        self.executor = LLMTaskExecutor(
            claude_client=self.mock_claude_client,
            tool_executor=self.mock_tool_executor,
            task_executor=self.mock_task_executor,
        )

    def test_simple_task_without_tools(self):
        """ツールなしの単純なタスク"""
        # LLMレスポンスの設定
        self.mock_claude_client.complete.return_value = ClaudeResponse(
            content="タスク完了しました。",
            stop_reason="end_turn",
            tool_uses=[],
            usage={"input_tokens": 100, "output_tokens": 50},
        )

        result = self.executor.execute_task_with_tools(
            agent_id="agent_01",
            system_prompt="You are a helpful assistant.",
            task_description="簡単なタスクを実行してください。",
        )

        assert result.content == "タスク完了しました。"
        assert result.stop_reason == "end_turn"
        assert result.iterations == 1
        assert result.total_tokens == 150
        assert len(result.tool_calls) == 0

        # search_memories が呼ばれたことを確認
        self.mock_task_executor.search_memories.assert_called_once()

    def test_task_with_memory_context(self):
        """メモリコンテキスト付きのタスク"""
        # 検索結果をモック
        mock_scored_memory = MagicMock()
        mock_scored_memory.memory.content = "過去の重要な学び"
        mock_scored_memory.memory.learning = "具体的な学びの詳細"
        mock_scored_memory.final_score = 0.85
        self.mock_task_executor.search_memories.return_value = [mock_scored_memory]

        # LLMレスポンス
        self.mock_claude_client.complete.return_value = ClaudeResponse(
            content="過去の学びを参考に回答します。",
            stop_reason="end_turn",
            tool_uses=[],
            usage={"input_tokens": 200, "output_tokens": 80},
        )

        result = self.executor.execute_task_with_tools(
            agent_id="agent_01",
            system_prompt="Assistant",
            task_description="質問に答えてください。",
            perspective="コスト",
        )

        assert result.content == "過去の学びを参考に回答します。"
        assert result.searched_memories_count == 1

        # search_memories に perspective が渡されていること
        self.mock_task_executor.search_memories.assert_called_once_with(
            query="質問に答えてください。",
            agent_id="agent_01",
            perspective="コスト",
        )

        # complete に渡されたメッセージを確認
        call_args = self.mock_claude_client.complete.call_args
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
        assert any("<context>" in str(m) for m in messages)


class TestExecuteTaskWithToolsLoop:
    """execute_task_with_tools() ツール使用ループテスト"""

    def setup_method(self):
        """テストセットアップ"""
        self.mock_claude_client = MagicMock()
        self.mock_tool_executor = MagicMock()
        self.mock_task_executor = MagicMock()

        self.mock_task_executor.search_memories.return_value = []

        # ツール一覧を返す
        self.mock_tool_executor.get_tools_for_llm.return_value = [
            {
                "name": "search_memories",
                "description": "メモリを検索",
                "input_schema": {"type": "object", "properties": {}},
            }
        ]

        self.executor = LLMTaskExecutor(
            claude_client=self.mock_claude_client,
            tool_executor=self.mock_tool_executor,
            task_executor=self.mock_task_executor,
        )

    def test_single_tool_use(self):
        """1回のツール使用"""
        # 1回目: ツール使用
        first_response = ClaudeResponse(
            content="",
            stop_reason="tool_use",
            tool_uses=[
                ToolUse(
                    id="toolu_001",
                    name="search_memories",
                    input={"query": "test", "agent_id": "agent_01"},
                )
            ],
            usage={"input_tokens": 100, "output_tokens": 50},
        )

        # 2回目: 完了
        second_response = ClaudeResponse(
            content="検索結果を基に回答します。",
            stop_reason="end_turn",
            tool_uses=[],
            usage={"input_tokens": 150, "output_tokens": 80},
        )

        self.mock_claude_client.complete.side_effect = [first_response, second_response]

        # ツール実行結果
        self.mock_tool_executor.execute_tool_safe.return_value = ToolExecutionResult(
            tool_name="search_memories",
            success=True,
            result='{"count": 1, "memories": []}',
        )

        result = self.executor.execute_task_with_tools(
            agent_id="agent_01",
            system_prompt="Assistant",
            task_description="メモリを検索して回答してください。",
        )

        assert result.content == "検索結果を基に回答します。"
        assert result.iterations == 2
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "search_memories"
        assert result.tool_calls[0].success is True
        assert result.total_tokens == 380  # 100+50+150+80

    def test_multiple_tool_uses(self):
        """複数回のツール使用"""
        # 1回目: ツール使用
        first_response = ClaudeResponse(
            content="",
            stop_reason="tool_use",
            tool_uses=[
                ToolUse(id="toolu_001", name="tool_a", input={"a": 1}),
            ],
            usage={"input_tokens": 100, "output_tokens": 50},
        )

        # 2回目: 別のツール使用
        second_response = ClaudeResponse(
            content="",
            stop_reason="tool_use",
            tool_uses=[
                ToolUse(id="toolu_002", name="tool_b", input={"b": 2}),
            ],
            usage={"input_tokens": 150, "output_tokens": 60},
        )

        # 3回目: 完了
        third_response = ClaudeResponse(
            content="完了しました。",
            stop_reason="end_turn",
            tool_uses=[],
            usage={"input_tokens": 200, "output_tokens": 30},
        )

        self.mock_claude_client.complete.side_effect = [
            first_response, second_response, third_response
        ]

        self.mock_tool_executor.execute_tool_safe.return_value = ToolExecutionResult(
            tool_name="tool",
            success=True,
            result="result",
        )

        result = self.executor.execute_task_with_tools(
            agent_id="agent_01",
            system_prompt="Assistant",
            task_description="複数のツールを使用するタスク",
        )

        assert result.iterations == 3
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].tool_name == "tool_a"
        assert result.tool_calls[1].tool_name == "tool_b"

    def test_tool_execution_failure(self):
        """ツール実行失敗時のハンドリング"""
        # 1回目: ツール使用
        first_response = ClaudeResponse(
            content="",
            stop_reason="tool_use",
            tool_uses=[
                ToolUse(id="toolu_001", name="failing_tool", input={}),
            ],
            usage={"input_tokens": 100, "output_tokens": 50},
        )

        # 2回目: 完了（エラーを受けて処理継続）
        second_response = ClaudeResponse(
            content="ツールが失敗しましたが、対応します。",
            stop_reason="end_turn",
            tool_uses=[],
            usage={"input_tokens": 150, "output_tokens": 80},
        )

        self.mock_claude_client.complete.side_effect = [first_response, second_response]

        # ツール実行失敗
        self.mock_tool_executor.execute_tool_safe.return_value = ToolExecutionResult(
            tool_name="failing_tool",
            success=False,
            error="ツールが登録されていません",
        )

        result = self.executor.execute_task_with_tools(
            agent_id="agent_01",
            system_prompt="Assistant",
            task_description="失敗するツールを呼ぶタスク",
        )

        # フローは継続
        assert result.content == "ツールが失敗しましたが、対応します。"
        assert result.stop_reason == "end_turn"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].success is False
        assert result.tool_calls[0].error == "ツールが登録されていません"


class TestExecuteTaskWithToolsMaxIterations:
    """execute_task_with_tools() 最大イテレーション回数テスト"""

    def setup_method(self):
        """テストセットアップ"""
        self.mock_claude_client = MagicMock()
        self.mock_tool_executor = MagicMock()
        self.mock_task_executor = MagicMock()

        self.mock_task_executor.search_memories.return_value = []
        self.mock_tool_executor.get_tools_for_llm.return_value = []

        self.executor = LLMTaskExecutor(
            claude_client=self.mock_claude_client,
            tool_executor=self.mock_tool_executor,
            task_executor=self.mock_task_executor,
        )

    def test_max_iterations_exceeded(self):
        """最大イテレーション回数超過"""
        # 常にツール使用を返す（無限ループ）
        tool_use_response = ClaudeResponse(
            content="処理中...",
            stop_reason="tool_use",
            tool_uses=[
                ToolUse(id="toolu_001", name="loop_tool", input={}),
            ],
            usage={"input_tokens": 100, "output_tokens": 50},
        )

        self.mock_claude_client.complete.return_value = tool_use_response
        self.mock_tool_executor.execute_tool_safe.return_value = ToolExecutionResult(
            tool_name="loop_tool",
            success=True,
            result="result",
        )

        result = self.executor.execute_task_with_tools(
            agent_id="agent_01",
            system_prompt="Assistant",
            task_description="無限ループするタスク",
            max_tool_iterations=3,
        )

        # 最大回数で停止
        assert result.stop_reason == "max_iterations"
        assert result.iterations == 3
        # complete は 3回呼ばれる
        assert self.mock_claude_client.complete.call_count == 3

    def test_custom_max_iterations(self):
        """カスタム最大イテレーション回数"""
        tool_use_response = ClaudeResponse(
            content="",
            stop_reason="tool_use",
            tool_uses=[ToolUse(id="toolu_001", name="tool", input={})],
            usage={"input_tokens": 50, "output_tokens": 25},
        )

        self.mock_claude_client.complete.return_value = tool_use_response
        self.mock_tool_executor.execute_tool_safe.return_value = ToolExecutionResult(
            tool_name="tool",
            success=True,
            result="result",
        )

        result = self.executor.execute_task_with_tools(
            agent_id="agent_01",
            system_prompt="Assistant",
            task_description="タスク",
            max_tool_iterations=5,
        )

        assert result.iterations == 5
        assert result.stop_reason == "max_iterations"


class TestExecuteTaskWithToolsStopReasons:
    """execute_task_with_tools() 停止理由テスト"""

    def setup_method(self):
        """テストセットアップ"""
        self.mock_claude_client = MagicMock()
        self.mock_tool_executor = MagicMock()
        self.mock_task_executor = MagicMock()

        self.mock_task_executor.search_memories.return_value = []
        self.mock_tool_executor.get_tools_for_llm.return_value = []

        self.executor = LLMTaskExecutor(
            claude_client=self.mock_claude_client,
            tool_executor=self.mock_tool_executor,
            task_executor=self.mock_task_executor,
        )

    def test_stop_reason_end_turn(self):
        """stop_reason = end_turn"""
        self.mock_claude_client.complete.return_value = ClaudeResponse(
            content="完了",
            stop_reason="end_turn",
            tool_uses=[],
            usage={"input_tokens": 100, "output_tokens": 50},
        )

        result = self.executor.execute_task_with_tools(
            agent_id="agent_01",
            system_prompt="Assistant",
            task_description="タスク",
        )

        assert result.stop_reason == "end_turn"

    def test_stop_reason_max_tokens(self):
        """stop_reason = max_tokens"""
        self.mock_claude_client.complete.return_value = ClaudeResponse(
            content="途中で切れた...",
            stop_reason="max_tokens",
            tool_uses=[],
            usage={"input_tokens": 100, "output_tokens": 4096},
        )

        result = self.executor.execute_task_with_tools(
            agent_id="agent_01",
            system_prompt="Assistant",
            task_description="長いタスク",
        )

        assert result.stop_reason == "max_tokens"
        assert result.content == "途中で切れた..."


# =============================================================================
# ヘルパーメソッドテスト
# =============================================================================

class TestHelperMethods:
    """ヘルパーメソッドのテスト"""

    def setup_method(self):
        """テストセットアップ"""
        self.executor = LLMTaskExecutor(
            claude_client=MagicMock(),
            tool_executor=MagicMock(),
            task_executor=MagicMock(),
        )

    def test_build_memory_context_empty(self):
        """_build_memory_context: 空の場合"""
        context = self.executor._build_memory_context([])

        assert context == ""

    def test_build_memory_context_with_memories(self):
        """_build_memory_context: メモリあり"""
        mock_memory1 = MagicMock()
        mock_memory1.memory.content = "重要な学び1"
        mock_memory1.memory.learning = "詳細1"
        mock_memory1.final_score = 0.9

        mock_memory2 = MagicMock()
        mock_memory2.memory.content = "重要な学び2"
        mock_memory2.memory.learning = None
        mock_memory2.final_score = 0.8

        context = self.executor._build_memory_context([mock_memory1, mock_memory2])

        assert "関連する過去の記憶" in context
        assert "重要な学び1" in context
        assert "詳細1" in context
        assert "重要な学び2" in context
        assert "0.90" in context
        assert "0.80" in context

    def test_build_assistant_content_text_only(self):
        """_build_assistant_content: テキストのみ"""
        response = ClaudeResponse(
            content="テキスト応答",
            stop_reason="end_turn",
            tool_uses=[],
        )

        content = self.executor._build_assistant_content(response)

        assert len(content) == 1
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "テキスト応答"

    def test_build_assistant_content_with_tool_use(self):
        """_build_assistant_content: ツール使用あり"""
        response = ClaudeResponse(
            content="考え中...",
            stop_reason="tool_use",
            tool_uses=[
                ToolUse(id="toolu_001", name="my_tool", input={"key": "value"})
            ],
        )

        content = self.executor._build_assistant_content(response)

        assert len(content) == 2
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "考え中..."
        assert content[1]["type"] == "tool_use"
        assert content[1]["id"] == "toolu_001"
        assert content[1]["name"] == "my_tool"
        assert content[1]["input"] == {"key": "value"}

    def test_extract_last_content_string(self):
        """_extract_last_content: 文字列コンテンツ"""
        messages = [
            {"role": "user", "content": "質問"},
            {"role": "assistant", "content": "回答"},
        ]

        content = self.executor._extract_last_content(messages)

        assert content == "回答"

    def test_extract_last_content_list(self):
        """_extract_last_content: リストコンテンツ"""
        messages = [
            {"role": "user", "content": "質問"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "テキスト1"},
                {"type": "tool_use", "id": "toolu_001", "name": "tool"},
                {"type": "text", "text": "テキスト2"},
            ]},
        ]

        content = self.executor._extract_last_content(messages)

        assert "テキスト1" in content
        assert "テキスト2" in content

    def test_extract_last_content_no_assistant(self):
        """_extract_last_content: アシスタントメッセージなし"""
        messages = [
            {"role": "user", "content": "質問"},
        ]

        content = self.executor._extract_last_content(messages)

        assert content == ""


# =============================================================================
# 統合テスト
# =============================================================================

class TestLLMTaskExecutorIntegration:
    """LLMTaskExecutor 統合テスト"""

    def test_full_workflow_with_tool_use(self):
        """ツール使用を含む完全なワークフロー"""
        # モックセットアップ
        mock_claude_client = MagicMock()
        mock_tool_executor = MagicMock()
        mock_task_executor = MagicMock()

        # メモリ検索結果
        mock_scored_memory = MagicMock()
        mock_scored_memory.memory.content = "過去の経験"
        mock_scored_memory.memory.learning = "学び"
        mock_scored_memory.final_score = 0.85
        mock_task_executor.search_memories.return_value = [mock_scored_memory]

        # ツール一覧
        mock_tool_executor.get_tools_for_llm.return_value = [
            {"name": "search_memories", "description": "検索", "input_schema": {}}
        ]

        # LLMレスポンス: 1回目はツール使用、2回目は完了
        first_response = ClaudeResponse(
            content="検索します。",
            stop_reason="tool_use",
            tool_uses=[
                ToolUse(
                    id="toolu_001",
                    name="search_memories",
                    input={"query": "追加検索", "agent_id": "agent_01"},
                )
            ],
            usage={"input_tokens": 200, "output_tokens": 100},
        )

        second_response = ClaudeResponse(
            content="検索結果を基にした最終回答です。",
            stop_reason="end_turn",
            tool_uses=[],
            usage={"input_tokens": 300, "output_tokens": 150},
        )

        mock_claude_client.complete.side_effect = [first_response, second_response]

        # ツール実行結果
        mock_tool_executor.execute_tool_safe.return_value = ToolExecutionResult(
            tool_name="search_memories",
            success=True,
            result='{"count": 2, "memories": [{"content": "追加メモリ"}]}',
        )

        # 実行
        executor = LLMTaskExecutor(
            claude_client=mock_claude_client,
            tool_executor=mock_tool_executor,
            task_executor=mock_task_executor,
        )

        result = executor.execute_task_with_tools(
            agent_id="agent_01",
            system_prompt="You are a helpful assistant.",
            task_description="コストについて教えてください。",
            perspective="コスト",
        )

        # 検証
        assert result.content == "検索結果を基にした最終回答です。"
        assert result.stop_reason == "end_turn"
        assert result.iterations == 2
        assert result.total_tokens == 750  # 200+100+300+150
        assert result.searched_memories_count == 1
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "search_memories"
        assert result.tool_calls[0].success is True

        # メソッド呼び出しの検証
        mock_task_executor.search_memories.assert_called_once_with(
            query="コストについて教えてください。",
            agent_id="agent_01",
            perspective="コスト",
        )

        mock_tool_executor.execute_tool_safe.assert_called_once_with(
            name="search_memories",
            input_data={"query": "追加検索", "agent_id": "agent_01"},
        )

    def test_workflow_without_tools(self):
        """ツールなしのシンプルなワークフロー"""
        mock_claude_client = MagicMock()
        mock_tool_executor = MagicMock()
        mock_task_executor = MagicMock()

        mock_task_executor.search_memories.return_value = []
        mock_tool_executor.get_tools_for_llm.return_value = []

        mock_claude_client.complete.return_value = ClaudeResponse(
            content="シンプルな回答です。",
            stop_reason="end_turn",
            tool_uses=[],
            usage={"input_tokens": 50, "output_tokens": 20},
        )

        executor = LLMTaskExecutor(
            claude_client=mock_claude_client,
            tool_executor=mock_tool_executor,
            task_executor=mock_task_executor,
        )

        result = executor.execute_task_with_tools(
            agent_id="agent_01",
            system_prompt="Assistant",
            task_description="簡単な質問",
        )

        assert result.content == "シンプルな回答です。"
        assert result.iterations == 1
        assert len(result.tool_calls) == 0
        assert result.total_tokens == 70
