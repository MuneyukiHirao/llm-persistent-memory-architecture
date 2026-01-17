# LLM呼び出しとツール使用ループを統合したタスク実行
# 実装仕様: docs/phase1-implementation-spec.ja.md
# アーキテクチャ: docs/architecture.ja.md セクション 2327-2350行（execute_task_with_tools疑似コード）
"""
LLMタスク実行モジュール

LLM呼び出しとツール使用ループを統合してタスクを実行します。
メモリ検索をコンテキストに追加し、LLMがツールを使用する場合は
ツール実行→結果追加→再呼び出しのループを行います。

設計方針（タスク実行フローエージェント観点）:
- フロー整合性: メモリ検索→LLM呼び出し→ツール使用ループ→完了の順序を保証
- 無限ループ防止: max_tool_iterations で最大ループ回数を制限
- エラー処理: ツール実行失敗時もフローを継続（結果をエラーとして返す）
- ログ可視性: 各ステップでのログ出力

使用例:
    claude_client = ClaudeClient(config)
    tool_executor = create_tool_executor_with_builtins(task_executor)

    llm_executor = LLMTaskExecutor(
        claude_client=claude_client,
        tool_executor=tool_executor,
        task_executor=task_executor,
    )

    result = llm_executor.execute_task_with_tools(
        agent_id="agent_01",
        system_prompt="You are a helpful assistant.",
        task_description="調達コストについて教えてください",
        perspective="コスト",
        max_tool_iterations=10,
    )

    print(result.content)
    print(f"ツール呼び出し回数: {len(result.tool_calls)}")
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.llm.claude_client import ClaudeClient, ClaudeResponse
from src.llm.tool_executor import ToolExecutor, ToolExecutionResult
from src.core.task_executor import TaskExecutor

logger = logging.getLogger(__name__)


class LLMTaskExecutorError(Exception):
    """LLMタスク実行のエラー

    ツール使用ループの最大回数超過、コンテキスト構築失敗などで発生。

    Attributes:
        message: エラーメッセージ
        iterations: エラー発生時のイテレーション回数
        original_error: 元の例外（あれば）
    """

    def __init__(
        self,
        message: str,
        iterations: int = 0,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.message = message
        self.iterations = iterations
        self.original_error = original_error

    def __str__(self) -> str:
        parts = [self.message]
        if self.iterations > 0:
            parts.append(f"(iterations={self.iterations})")
        if self.original_error:
            parts.append(f"原因: {self.original_error}")
        return " ".join(parts)


@dataclass
class ToolCallRecord:
    """ツール呼び出しの記録

    LLMがツールを呼び出した際の情報を保持します。

    Attributes:
        tool_use_id: ツール使用の一意識別子
        tool_name: 呼び出されたツール名
        tool_input: ツールへの入力パラメータ
        result: ツール実行結果
        success: 実行が成功したかどうか
        error: エラーメッセージ（失敗時）
    """

    tool_use_id: str
    tool_name: str
    tool_input: Dict[str, Any]
    result: str = ""
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        data = {
            "tool_use_id": self.tool_use_id,
            "tool_name": self.tool_name,
            "tool_input": self.tool_input,
            "result": self.result,
            "success": self.success,
        }
        if self.error:
            data["error"] = self.error
        return data


@dataclass
class LLMTaskResult:
    """LLMタスク実行結果

    execute_task_with_tools() の戻り値として、タスク実行の全結果を格納。

    Attributes:
        content: 最終的なLLM応答テキスト
        tool_calls: 呼び出したツールの記録リスト
        total_tokens: 合計トークン使用量
        iterations: ツール使用ループのイテレーション回数
        searched_memories_count: 検索されたメモリの件数
        stop_reason: 最終的な停止理由（"end_turn", "max_tokens", "max_iterations"）
    """

    content: str
    tool_calls: List[ToolCallRecord] = field(default_factory=list)
    total_tokens: int = 0
    iterations: int = 0
    searched_memories_count: int = 0
    stop_reason: str = "end_turn"

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "content": self.content,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "total_tokens": self.total_tokens,
            "iterations": self.iterations,
            "searched_memories_count": self.searched_memories_count,
            "stop_reason": self.stop_reason,
            "success": self.stop_reason in ("end_turn", "max_tokens"),
        }


class LLMTaskExecutor:
    """LLM呼び出しとツール使用ループを統合したタスク実行クラス

    ClaudeClient、ToolExecutor、TaskExecutor を統合し、
    メモリ検索からLLM呼び出し、ツール使用ループまでの一連のフローを管理します。

    アーキテクチャ docs/architecture.ja.md 2327-2350行のフローを実装:
        1. メモリ検索（search_memories）をコンテキストに追加
        2. LLM呼び出し
        3. tool_use の場合 → ツール実行 → 結果をコンテキストに追加 → 2に戻る
        4. end_turn の場合 → 結果を返す

    Attributes:
        claude_client: ClaudeClient インスタンス（LLM呼び出し用）
        tool_executor: ToolExecutor インスタンス（ツール実行用）
        task_executor: TaskExecutor インスタンス（メモリ操作用）
    """

    # デフォルトのツール使用最大イテレーション回数
    DEFAULT_MAX_TOOL_ITERATIONS = 10

    def __init__(
        self,
        claude_client: ClaudeClient,
        tool_executor: ToolExecutor,
        task_executor: TaskExecutor,
    ):
        """LLMTaskExecutor を初期化

        Args:
            claude_client: ClaudeClient インスタンス
            tool_executor: ToolExecutor インスタンス
            task_executor: TaskExecutor インスタンス（メモリ操作用）
        """
        self.claude_client = claude_client
        self.tool_executor = tool_executor
        self.task_executor = task_executor

        logger.info("LLMTaskExecutor 初期化完了")

    def execute_task_with_tools(
        self,
        agent_id: str,
        system_prompt: str,
        task_description: str,
        perspective: Optional[str] = None,
        max_tool_iterations: int = DEFAULT_MAX_TOOL_ITERATIONS,
    ) -> LLMTaskResult:
        """ツール使用ループ付きでタスクを実行

        メモリ検索でコンテキストを構築し、LLMを呼び出します。
        LLMがツールを使用する場合は、ツール実行→結果追加→再呼び出しのループを行います。

        処理フロー:
            1. メモリ検索（search_memories）でコンテキスト取得
            2. コンテキストをメッセージに追加
            3. LLM呼び出し
            4. stop_reason == "tool_use" の場合:
               a. 各ツールを実行
               b. 結果をコンテキストに追加
               c. 3に戻る（最大 max_tool_iterations 回まで）
            5. end_turn または max_tokens の場合 → 結果を返す

        Args:
            agent_id: エージェントID（メモリ検索に使用）
            system_prompt: システムプロンプト
            task_description: タスクの説明（ユーザーメッセージとして使用）
            perspective: 観点（メモリ検索で使用、オプション）
            max_tool_iterations: ツール使用ループの最大回数（デフォルト10回）

        Returns:
            LLMTaskResult: タスク実行結果

        Raises:
            LLMTaskExecutorError: 最大イテレーション回数を超過した場合
            ClaudeClientError: LLM呼び出しに失敗した場合
        """
        logger.info(
            f"タスク実行開始: agent_id={agent_id}, "
            f"task={task_description[:50]!r}..., "
            f"perspective={perspective}, "
            f"max_iterations={max_tool_iterations}"
        )

        # 結果記録用
        tool_calls: List[ToolCallRecord] = []
        total_input_tokens = 0
        total_output_tokens = 0
        iterations = 0

        # ========================================
        # Step 1: メモリ検索でコンテキスト取得
        # ========================================
        searched_memories = self.task_executor.search_memories(
            query=task_description,
            agent_id=agent_id,
            perspective=perspective,
        )

        context_text = self._build_memory_context(searched_memories)
        logger.info(f"メモリ検索完了: {len(searched_memories)} 件")

        # ========================================
        # Step 2: 初期メッセージ構築
        # ========================================
        messages: List[Dict[str, Any]] = []

        # コンテキストがあれば追加
        if context_text:
            user_content = f"<context>\n{context_text}\n</context>\n\n{task_description}"
        else:
            user_content = task_description

        messages.append({
            "role": "user",
            "content": user_content,
        })

        # ツール一覧を取得
        tools = self.tool_executor.get_tools_for_llm()

        # ========================================
        # Step 3: LLM呼び出しループ
        # ========================================
        while True:
            iterations += 1

            # 最大イテレーション回数チェック
            if iterations > max_tool_iterations:
                logger.warning(
                    f"最大イテレーション回数を超過: {max_tool_iterations}"
                )
                # 現時点での結果を返す（最後のレスポンスがあれば）
                return LLMTaskResult(
                    content=self._extract_last_content(messages),
                    tool_calls=tool_calls,
                    total_tokens=total_input_tokens + total_output_tokens,
                    iterations=iterations - 1,
                    searched_memories_count=len(searched_memories),
                    stop_reason="max_iterations",
                )

            logger.info(f"LLM呼び出し: iteration={iterations}")

            # LLM呼び出し
            response = self.claude_client.complete(
                system_prompt=system_prompt,
                messages=messages,
                tools=tools if tools else None,
            )

            # トークン使用量を累積
            if response.usage:
                total_input_tokens += response.usage.get("input_tokens", 0)
                total_output_tokens += response.usage.get("output_tokens", 0)

            logger.debug(
                f"LLM応答: stop_reason={response.stop_reason}, "
                f"content_length={len(response.content)}, "
                f"tool_uses={len(response.tool_uses)}"
            )

            # ========================================
            # Step 4: 停止理由による分岐
            # ========================================
            if response.stop_reason == "tool_use":
                # ツール使用の場合
                # アシスタントのレスポンスをメッセージに追加
                assistant_content = self._build_assistant_content(response)
                messages.append({
                    "role": "assistant",
                    "content": assistant_content,
                })

                # 各ツールを実行
                tool_results: List[Dict[str, Any]] = []
                for tool_use in response.tool_uses:
                    logger.info(
                        f"ツール実行: name={tool_use.name}, "
                        f"id={tool_use.id}"
                    )

                    # ツール実行（安全版を使用）
                    exec_result = self.tool_executor.execute_tool_safe(
                        name=tool_use.name,
                        input_data=tool_use.input,
                    )

                    # 記録を追加
                    tool_calls.append(ToolCallRecord(
                        tool_use_id=tool_use.id,
                        tool_name=tool_use.name,
                        tool_input=tool_use.input,
                        result=exec_result.result if exec_result.success else "",
                        success=exec_result.success,
                        error=exec_result.error,
                    ))

                    # tool_result を構築
                    if exec_result.success:
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": exec_result.result,
                        })
                    else:
                        # エラーの場合もツール結果として返す
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": f"Error: {exec_result.error}",
                            "is_error": True,
                        })

                    logger.info(
                        f"ツール実行完了: name={tool_use.name}, "
                        f"success={exec_result.success}"
                    )

                # ツール結果をユーザーメッセージとして追加
                messages.append({
                    "role": "user",
                    "content": tool_results,
                })

                # 次のイテレーションへ
                continue

            else:
                # end_turn または max_tokens の場合は終了
                logger.info(
                    f"タスク実行完了: stop_reason={response.stop_reason}, "
                    f"iterations={iterations}, "
                    f"tool_calls={len(tool_calls)}"
                )

                return LLMTaskResult(
                    content=response.content,
                    tool_calls=tool_calls,
                    total_tokens=total_input_tokens + total_output_tokens,
                    iterations=iterations,
                    searched_memories_count=len(searched_memories),
                    stop_reason=response.stop_reason,
                )

    def _build_memory_context(
        self,
        searched_memories: List[Any],
    ) -> str:
        """検索されたメモリからコンテキストテキストを構築

        Args:
            searched_memories: ScoredMemory のリスト

        Returns:
            コンテキストテキスト（メモリが空の場合は空文字列）
        """
        if not searched_memories:
            return ""

        lines = ["関連する過去の記憶:"]
        for i, scored_memory in enumerate(searched_memories, 1):
            memory = scored_memory.memory
            # スコアと内容を表示
            lines.append(
                f"{i}. [スコア: {scored_memory.final_score:.2f}] {memory.content}"
            )
            # 学びがあれば追加
            if memory.learning:
                lines.append(f"   学び: {memory.learning}")

        return "\n".join(lines)

    def _build_assistant_content(
        self,
        response: ClaudeResponse,
    ) -> List[Dict[str, Any]]:
        """ClaudeResponse からアシスタントメッセージのコンテンツを構築

        Args:
            response: ClaudeResponse インスタンス

        Returns:
            アシスタントメッセージのコンテンツ配列
        """
        content: List[Dict[str, Any]] = []

        # テキストコンテンツがあれば追加
        if response.content:
            content.append({
                "type": "text",
                "text": response.content,
            })

        # ツール使用を追加
        for tool_use in response.tool_uses:
            content.append({
                "type": "tool_use",
                "id": tool_use.id,
                "name": tool_use.name,
                "input": tool_use.input,
            })

        return content

    def _extract_last_content(
        self,
        messages: List[Dict[str, Any]],
    ) -> str:
        """メッセージ履歴から最後のアシスタント応答を抽出

        max_iterations で打ち切った場合に使用。

        Args:
            messages: メッセージ履歴

        Returns:
            最後のアシスタント応答のテキスト（見つからない場合は空文字列）
        """
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                # content が配列の場合はテキストを抽出
                if isinstance(content, list):
                    texts = [
                        c.get("text", "")
                        for c in content
                        if c.get("type") == "text"
                    ]
                    return " ".join(texts)
                elif isinstance(content, str):
                    return content
        return ""
