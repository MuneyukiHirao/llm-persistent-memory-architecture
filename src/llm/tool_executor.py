# ツール定義・実行フレームワーク
# 実装仕様: docs/phase1-implementation-spec.ja.md
# アーキテクチャ: docs/architecture.ja.md セクション2264-2310行（ツール定義JSON構造）
"""
ツール定義・実行モジュール

LLMがツールを使用する際の定義と実行を管理します。
Claude APIおよびMCP（Model Context Protocol）互換の形式を採用。

設計方針（タスク実行フローエージェント観点）:
- API設計: Claude API互換のツール形式を提供
- フロー整合性: ツール登録→実行→結果返却の一貫したフロー
- エラー処理: ツール実行失敗時の適切なエラーハンドリング
- 拡張性: 新規ツールの動的登録をサポート
- テスト容易性: ハンドラーを依存性注入で受け取り、モック差し替え可能

使用例:
    # ToolExecutor の初期化と使用
    executor = ToolExecutor()

    # ツール登録
    def my_handler(input_data: Dict) -> str:
        return f"結果: {input_data['query']}"

    tool = Tool(
        name="my_tool",
        description="サンプルツール",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"]
        }
    )
    executor.register_tool(tool, my_handler)

    # ツール実行
    result = executor.execute_tool("my_tool", {"query": "test"})

    # Claude API 形式でツール一覧を取得
    tools_for_llm = executor.get_tools_for_llm()
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ToolExecutionError(Exception):
    """ツール実行時のエラー

    ツールが存在しない、入力が無効、ハンドラーが失敗した場合などに発生。

    Attributes:
        message: エラーメッセージ
        tool_name: 対象のツール名（存在する場合）
        original_error: 元の例外（あれば）
    """

    def __init__(
        self,
        message: str,
        tool_name: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.message = message
        self.tool_name = tool_name
        self.original_error = original_error

    def __str__(self) -> str:
        parts = [self.message]
        if self.tool_name:
            parts.append(f"tool={self.tool_name}")
        if self.original_error:
            parts.append(f"原因: {self.original_error}")
        return " ".join(parts)


@dataclass
class Tool:
    """ツール定義

    LLMが使用可能なツールを定義します。
    Claude API および MCP 互換の形式。

    Attributes:
        name: ツール名（一意識別子）
        description: ツールの説明（LLMがツール選択時に参照）
        input_schema: 入力パラメータのJSON Schema形式定義

    JSON Schema の形式例:
        {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "検索クエリ"
                },
                "limit": {
                    "type": "integer",
                    "description": "結果の最大件数",
                    "default": 10
                }
            },
            "required": ["query"]
        }
    """

    name: str
    description: str
    input_schema: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Claude API 形式の辞書に変換

        Returns:
            Claude API 互換のツール定義辞書
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


@dataclass
class ToolExecutionResult:
    """ツール実行結果

    ツール実行後の結果を構造化して保持します。

    Attributes:
        tool_name: 実行されたツール名
        success: 実行が成功したかどうか
        result: 実行結果（文字列）
        error: エラーメッセージ（失敗時）
    """

    tool_name: str
    success: bool
    result: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換

        Returns:
            辞書形式の結果
        """
        data = {
            "tool_name": self.tool_name,
            "success": self.success,
            "result": self.result,
        }
        if self.error:
            data["error"] = self.error
        return data


# ツールハンドラーの型定義
ToolHandler = Callable[[Dict[str, Any]], str]


class ToolExecutor:
    """ツール実行管理クラス

    ツールの登録・実行・一覧取得を管理します。
    Claude API 互換のツール形式をサポート。

    設計方針:
        - ツール登録: register_tool() でツール定義とハンドラーを登録
        - ツール実行: execute_tool() で名前と入力を指定して実行
        - LLM連携: get_tools_for_llm() でClaude API形式のツール一覧を取得

    使用例:
        executor = ToolExecutor()

        # カスタムツール登録
        tool = Tool(
            name="calculate",
            description="数値計算を行う",
            input_schema={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "計算式"}
                },
                "required": ["expression"]
            }
        )

        def calculate_handler(input_data: Dict) -> str:
            # 安全な計算実行...
            return str(result)

        executor.register_tool(tool, calculate_handler)

        # ツール実行
        result = executor.execute_tool("calculate", {"expression": "1 + 2"})

    Attributes:
        _tools: 登録されたツール定義（name -> Tool）
        _handlers: ツールハンドラー（name -> Callable）
    """

    def __init__(self) -> None:
        """ToolExecutor を初期化"""
        self._tools: Dict[str, Tool] = {}
        self._handlers: Dict[str, ToolHandler] = {}

        logger.info("ToolExecutor 初期化完了")

    def register_tool(self, tool: Tool, handler: ToolHandler) -> None:
        """ツールを登録

        同じ名前のツールが既に登録されている場合は上書きします。

        Args:
            tool: ツール定義
            handler: ツールハンドラー関数
                     シグネチャ: (input_data: Dict[str, Any]) -> str

        Raises:
            ValueError: ツール名が空の場合
        """
        if not tool.name or not tool.name.strip():
            raise ValueError("ツール名は空にできません")

        # 既存のツールを上書きする場合は警告
        if tool.name in self._tools:
            logger.warning(f"ツール '{tool.name}' を上書き登録します")

        self._tools[tool.name] = tool
        self._handlers[tool.name] = handler

        logger.info(f"ツール登録: name={tool.name}, description={tool.description[:50]!r}...")

    def unregister_tool(self, name: str) -> bool:
        """ツールを登録解除

        Args:
            name: 登録解除するツール名

        Returns:
            True: 登録解除成功
            False: ツールが存在しなかった場合
        """
        if name not in self._tools:
            logger.warning(f"登録解除対象のツールが見つかりません: {name}")
            return False

        del self._tools[name]
        del self._handlers[name]

        logger.info(f"ツール登録解除: name={name}")
        return True

    def execute_tool(self, name: str, input_data: Dict[str, Any]) -> str:
        """ツールを実行

        登録済みのツールを名前と入力データで実行します。

        Args:
            name: 実行するツール名
            input_data: ツールへの入力データ（JSON Schema に準拠すべき）

        Returns:
            ツール実行結果の文字列

        Raises:
            ToolExecutionError: ツールが存在しない、または実行に失敗した場合
        """
        # ツールの存在確認
        if name not in self._tools:
            raise ToolExecutionError(
                f"ツール '{name}' が登録されていません",
                tool_name=name,
            )

        handler = self._handlers[name]

        logger.info(f"ツール実行開始: name={name}, input_keys={list(input_data.keys())}")

        try:
            # ハンドラー実行
            result = handler(input_data)

            # 結果が文字列でない場合は変換
            if not isinstance(result, str):
                result = json.dumps(result, ensure_ascii=False, default=str)

            logger.info(f"ツール実行成功: name={name}, result_length={len(result)}")
            return result

        except Exception as e:
            logger.error(f"ツール実行失敗: name={name}, error={e}")
            raise ToolExecutionError(
                f"ツール '{name}' の実行に失敗しました",
                tool_name=name,
                original_error=e,
            ) from e

    def execute_tool_safe(self, name: str, input_data: Dict[str, Any]) -> ToolExecutionResult:
        """ツールを安全に実行（例外をスローしない）

        ツール実行の成功/失敗をToolExecutionResultで返します。
        例外をスローしないため、フロー継続が必要な場合に使用。

        Args:
            name: 実行するツール名
            input_data: ツールへの入力データ

        Returns:
            ToolExecutionResult: 実行結果（成功/失敗情報を含む）
        """
        try:
            result = self.execute_tool(name, input_data)
            return ToolExecutionResult(
                tool_name=name,
                success=True,
                result=result,
            )
        except ToolExecutionError as e:
            return ToolExecutionResult(
                tool_name=name,
                success=False,
                error=str(e),
            )

    def get_tools_for_llm(self) -> List[Dict[str, Any]]:
        """Claude API 形式でツール一覧を取得

        ClaudeClient.complete() の tools 引数に渡すことができる形式で
        登録済みツールの一覧を返します。

        Returns:
            Claude API 互換のツール定義リスト

        Example:
            tools = executor.get_tools_for_llm()
            # [
            #     {
            #         "name": "search_memories",
            #         "description": "メモリを検索して関連する情報を取得する",
            #         "input_schema": {
            #             "type": "object",
            #             "properties": {...},
            #             "required": [...]
            #         }
            #     },
            #     ...
            # ]
        """
        return [tool.to_dict() for tool in self._tools.values()]

    def get_tool(self, name: str) -> Optional[Tool]:
        """ツール定義を取得

        Args:
            name: ツール名

        Returns:
            Tool: ツール定義（存在しない場合は None）
        """
        return self._tools.get(name)

    def has_tool(self, name: str) -> bool:
        """ツールが登録されているか確認

        Args:
            name: ツール名

        Returns:
            True: ツールが登録されている
            False: ツールが登録されていない
        """
        return name in self._tools

    def list_tools(self) -> List[str]:
        """登録済みツール名の一覧を取得

        Returns:
            ツール名のリスト
        """
        return list(self._tools.keys())

    @property
    def tool_count(self) -> int:
        """登録されているツール数を取得

        Returns:
            ツール数
        """
        return len(self._tools)


# =============================================================================
# 組み込みツール定義
# =============================================================================

# search_memories ツール定義
SEARCH_MEMORIES_TOOL = Tool(
    name="search_memories",
    description="メモリを検索して関連する情報を取得する。過去の経験や学びを参照する際に使用。",
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "検索クエリ（テキスト）"
            },
            "agent_id": {
                "type": "string",
                "description": "検索対象のエージェントID"
            },
            "perspective": {
                "type": "string",
                "description": "観点名（オプション、指定時は観点別強度を考慮）"
            }
        },
        "required": ["query", "agent_id"]
    }
)

# record_learning ツール定義
RECORD_LEARNING_TOOL = Tool(
    name="record_learning",
    description="例外的なイベントを学びとして記録する。エラー解決、予想外の挙動発見、効率的な方法発見などの重要な学びのみを記録。",
    input_schema={
        "type": "object",
        "properties": {
            "agent_id": {
                "type": "string",
                "description": "エージェントID"
            },
            "content": {
                "type": "string",
                "description": "学びの内容（メモリのメインコンテンツ）"
            },
            "learning": {
                "type": "string",
                "description": "学びの詳細テキスト"
            },
            "perspective": {
                "type": "string",
                "description": "観点名（オプション、強度管理で使用）"
            }
        },
        "required": ["agent_id", "content", "learning"]
    }
)

# 組み込みツールのリスト
BUILTIN_TOOLS = [
    SEARCH_MEMORIES_TOOL,
    RECORD_LEARNING_TOOL,
]


def create_search_memories_handler(task_executor: Any) -> ToolHandler:
    """search_memories ツールのハンドラーを生成

    TaskExecutor.search_memories() をラップしてツールハンドラー形式で提供。

    Args:
        task_executor: TaskExecutor インスタンス

    Returns:
        ToolHandler: search_memories のハンドラー関数
    """
    def handler(input_data: Dict[str, Any]) -> str:
        query = input_data.get("query", "")
        agent_id = input_data.get("agent_id", "")
        perspective = input_data.get("perspective")

        # TaskExecutor.search_memories() を呼び出し
        results = task_executor.search_memories(
            query=query,
            agent_id=agent_id,
            perspective=perspective,
        )

        # 結果を JSON 文字列に変換
        memories_data = []
        for scored_memory in results:
            memories_data.append({
                "memory_id": str(scored_memory.memory.id),
                "content": scored_memory.memory.content,
                "final_score": scored_memory.final_score,
                "strength": scored_memory.memory.strength,
            })

        return json.dumps({
            "count": len(memories_data),
            "memories": memories_data,
        }, ensure_ascii=False)

    return handler


def create_record_learning_handler(task_executor: Any) -> ToolHandler:
    """record_learning ツールのハンドラーを生成

    TaskExecutor.record_learning() をラップしてツールハンドラー形式で提供。

    Args:
        task_executor: TaskExecutor インスタンス

    Returns:
        ToolHandler: record_learning のハンドラー関数
    """
    def handler(input_data: Dict[str, Any]) -> str:
        agent_id = input_data.get("agent_id", "")
        content = input_data.get("content", "")
        learning = input_data.get("learning", "")
        perspective = input_data.get("perspective")

        # TaskExecutor.record_learning() を呼び出し
        memory_id = task_executor.record_learning(
            agent_id=agent_id,
            content=content,
            learning=learning,
            perspective=perspective,
        )

        return json.dumps({
            "success": True,
            "memory_id": str(memory_id),
        }, ensure_ascii=False)

    return handler


def create_tool_executor_with_builtins(task_executor: Any) -> ToolExecutor:
    """組み込みツール付きの ToolExecutor を生成

    search_memories と record_learning ツールが事前登録された
    ToolExecutor インスタンスを返します。

    Args:
        task_executor: TaskExecutor インスタンス

    Returns:
        ToolExecutor: 組み込みツール登録済みのインスタンス
    """
    executor = ToolExecutor()

    # search_memories 登録
    search_handler = create_search_memories_handler(task_executor)
    executor.register_tool(SEARCH_MEMORIES_TOOL, search_handler)

    # record_learning 登録
    record_handler = create_record_learning_handler(task_executor)
    executor.register_tool(RECORD_LEARNING_TOOL, record_handler)

    logger.info(f"組み込みツール登録完了: {executor.list_tools()}")

    return executor


def register_file_tools(executor: ToolExecutor, project_root: str) -> None:
    """ファイル操作ツールを登録

    file_read, file_write, file_list ツールを ToolExecutor に登録します。
    プロジェクトルートを設定し、セキュアなファイルアクセスを提供。

    Args:
        executor: ToolExecutor インスタンス
        project_root: プロジェクトルートディレクトリのパス
    """
    from src.llm.tools import (
        file_read,
        file_write,
        file_list,
        set_project_root,
        get_file_read_tool,
        get_file_write_tool,
        get_file_list_tool,
    )

    # プロジェクトルートを設定
    set_project_root(project_root)

    # file_read ハンドラー
    def file_read_handler(input_data: Dict[str, Any]) -> str:
        path = input_data.get("path", "")
        result = file_read(path)
        return json.dumps(result, ensure_ascii=False)

    # file_write ハンドラー
    def file_write_handler(input_data: Dict[str, Any]) -> str:
        path = input_data.get("path", "")
        content = input_data.get("content", "")
        result = file_write(path, content)
        return json.dumps(result, ensure_ascii=False)

    # file_list ハンドラー
    def file_list_handler(input_data: Dict[str, Any]) -> str:
        path = input_data.get("path", "")
        pattern = input_data.get("pattern")
        result = file_list(path, pattern)
        return json.dumps(result, ensure_ascii=False)

    # ツール登録
    executor.register_tool(get_file_read_tool(), file_read_handler)
    executor.register_tool(get_file_write_tool(), file_write_handler)
    executor.register_tool(get_file_list_tool(), file_list_handler)

    logger.info("ファイル操作ツール登録完了: file_read, file_write, file_list")


def register_bash_tool(executor: ToolExecutor, working_dir: Optional[str] = None) -> None:
    """bashツールを登録

    bash_execute ツールを ToolExecutor に登録します。
    セキュリティチェック（許可リスト、危険パターンブロック）付き。

    Args:
        executor: ToolExecutor インスタンス
        working_dir: デフォルトの作業ディレクトリ（オプション）
    """
    from src.llm.tools import bash_execute, get_bash_execute_tool

    # bash_execute ハンドラー
    def bash_execute_handler(input_data: Dict[str, Any]) -> str:
        command = input_data.get("command", "")
        timeout = input_data.get("timeout", 30)
        wd = input_data.get("working_dir", working_dir)
        result = bash_execute(command, timeout=timeout, working_dir=wd)
        return json.dumps(result, ensure_ascii=False)

    executor.register_tool(get_bash_execute_tool(), bash_execute_handler)

    logger.info("bashツール登録完了: bash_execute")


def create_tool_executor_with_all_tools(
    task_executor: Optional[Any] = None,
    project_root: Optional[str] = None,
    working_dir: Optional[str] = None,
) -> ToolExecutor:
    """すべてのツール付きの ToolExecutor を生成

    メモリツール（search_memories, record_learning）とファイルツール
    （file_read, file_write, file_list）、bashツール（bash_execute）を登録します。

    Args:
        task_executor: TaskExecutor インスタンス（Noneの場合はメモリツールなし）
        project_root: プロジェクトルートディレクトリ（Noneの場合はファイルツールなし）
        working_dir: bashのデフォルト作業ディレクトリ（オプション）

    Returns:
        ToolExecutor: 全ツール登録済みのインスタンス

    使用例:
        # 全ツール登録
        executor = create_tool_executor_with_all_tools(
            task_executor=task_executor,
            project_root="/path/to/project",
            working_dir="/path/to/project",
        )

        # ファイルツールとbashツールのみ
        executor = create_tool_executor_with_all_tools(
            project_root="/path/to/project",
        )
    """
    executor = ToolExecutor()

    # メモリツール登録（task_executorがある場合）
    if task_executor is not None:
        search_handler = create_search_memories_handler(task_executor)
        executor.register_tool(SEARCH_MEMORIES_TOOL, search_handler)

        record_handler = create_record_learning_handler(task_executor)
        executor.register_tool(RECORD_LEARNING_TOOL, record_handler)

        logger.info("メモリツール登録完了: search_memories, record_learning")

    # ファイルツール登録（project_rootがある場合）
    if project_root is not None:
        register_file_tools(executor, project_root)

    # bashツール登録
    register_bash_tool(executor, working_dir)

    logger.info(f"ToolExecutor 初期化完了: {len(executor.list_tools())} ツール登録済み")

    return executor
