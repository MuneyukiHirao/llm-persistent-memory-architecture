# Claude Messages API クライアント
# 実装仕様: docs/phase1-implementation-spec.ja.md
# 参考: docs/architecture.ja.md セクション 2312-2350行（ツール実行フロー）

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import anthropic
from anthropic import APIConnectionError, APIStatusError, RateLimitError

from src.config.llm_config import LLMConfig
from src.llm.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


class ClaudeClientError(Exception):
    """Claude API クライアントのエラー

    APIエラー、タイムアウト、リトライ失敗などをラップします。

    Attributes:
        message: エラーメッセージ
        original_error: 元の例外（あれば）
        is_retryable: リトライ可能かどうか
    """

    def __init__(
        self,
        message: str,
        original_error: Optional[Exception] = None,
        is_retryable: bool = False,
    ):
        super().__init__(message)
        self.message = message
        self.original_error = original_error
        self.is_retryable = is_retryable

    def __str__(self) -> str:
        if self.original_error:
            return f"{self.message}: {self.original_error}"
        return self.message


@dataclass
class ToolUse:
    """ツール呼び出し情報

    LLMがツールを使用する際の情報を保持します。

    Attributes:
        id: ツール使用の一意識別子（tool_resultで使用）
        name: ツール名
        input: ツールへの入力パラメータ
    """

    id: str
    name: str
    input: Dict[str, Any]


@dataclass
class ClaudeResponse:
    """Claude API レスポンス

    LLMからの応答を構造化して保持します。

    Attributes:
        content: テキスト応答（ツール使用時は空文字列の場合あり）
        stop_reason: 停止理由（"end_turn", "tool_use", "max_tokens"）
        tool_uses: ツール呼び出しのリスト
        usage: トークン使用量（input_tokens, output_tokens）
    """

    content: str
    stop_reason: str
    tool_uses: List[ToolUse] = field(default_factory=list)
    usage: Optional[Dict[str, int]] = None


class ClaudeClient:
    """Claude Messages API クライアント

    Anthropic Claude APIを使用してLLM呼び出しを行います。
    リトライロジック、タイムアウト処理、tool_use/tool_result対応を含みます。

    使用例:
        config = LLMConfig()
        client = ClaudeClient(config)

        # 基本的な呼び出し
        response = client.complete(
            system_prompt="You are a helpful assistant.",
            messages=[{"role": "user", "content": "Hello!"}]
        )
        print(response.content)

        # ツール付き呼び出し
        tools = [{
            "name": "get_weather",
            "description": "Get weather information",
            "input_schema": {"type": "object", "properties": {...}}
        }]
        response = client.complete(
            system_prompt="You are a helpful assistant.",
            messages=[{"role": "user", "content": "What's the weather?"}],
            tools=tools
        )
        if response.stop_reason == "tool_use":
            for tool_use in response.tool_uses:
                print(f"Tool: {tool_use.name}, Input: {tool_use.input}")
    """

    # リトライ時の指数バックオフ設定
    INITIAL_RETRY_DELAY = 1.0  # 秒
    MAX_RETRY_DELAY = 60.0  # 秒
    BACKOFF_MULTIPLIER = 2.0

    def __init__(self, config: LLMConfig):
        """クライアントを初期化

        Args:
            config: LLM設定（APIキー、モデル名、タイムアウトなど）

        Raises:
            ClaudeClientError: 設定が無効な場合
        """
        self.config = config

        # 設定の検証
        try:
            config.validate()
        except ValueError as e:
            raise ClaudeClientError(f"無効な設定: {e}") from e

        # Anthropic クライアント初期化
        self._client = anthropic.Anthropic(
            api_key=config.api_key,
            timeout=config.timeout_seconds,
        )

        # レート制限機能の初期化
        self._rate_limiter: Optional[RateLimiter] = None
        if config.enable_rate_limiting:
            self._rate_limiter = RateLimiter(config.rate_limit_config)
            logger.info("レート制限機能を有効化")

        logger.info(
            f"ClaudeClient 初期化完了: model={config.model_name}, "
            f"max_tokens={config.max_tokens}, timeout={config.timeout_seconds}s, "
            f"rate_limiting={config.enable_rate_limiting}"
        )

    def complete(
        self,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ClaudeResponse:
        """LLM呼び出しを実行

        Args:
            system_prompt: システムプロンプト
            messages: メッセージ履歴（role/content形式）
            tools: 利用可能なツール定義（オプション）

        Returns:
            ClaudeResponse: LLMからの応答

        Raises:
            ClaudeClientError: API呼び出しに失敗した場合
        """
        last_error: Optional[Exception] = None
        retry_delay = self.INITIAL_RETRY_DELAY

        for attempt in range(self.config.max_retries + 1):
            try:
                # レート制限チェック（リトライ前に毎回実行）
                if self._rate_limiter:
                    # 入力トークン数を推定（簡易的に文字数 / 4）
                    estimated_input = self._estimate_input_tokens(system_prompt, messages)
                    self._rate_limiter.wait_if_needed(estimated_input_tokens=estimated_input)

                return self._call_api(system_prompt, messages, tools)

            except RateLimitError as e:
                # レート制限は常にリトライ
                last_error = e
                logger.warning(
                    f"レート制限に達しました (試行 {attempt + 1}/{self.config.max_retries + 1}): {e}"
                )

            except APIConnectionError as e:
                # 接続エラーはリトライ
                last_error = e
                logger.warning(
                    f"API接続エラー (試行 {attempt + 1}/{self.config.max_retries + 1}): {e}"
                )

            except APIStatusError as e:
                # 5xxエラーはリトライ、4xxエラーは即座に失敗
                if e.status_code >= 500:
                    last_error = e
                    logger.warning(
                        f"APIサーバーエラー (試行 {attempt + 1}/{self.config.max_retries + 1}): {e}"
                    )
                else:
                    # 4xxエラー（認証エラー、リクエストエラーなど）
                    raise ClaudeClientError(
                        f"APIリクエストエラー (HTTP {e.status_code})",
                        original_error=e,
                        is_retryable=False,
                    ) from e

            except anthropic.APITimeoutError as e:
                # タイムアウトはリトライ
                last_error = e
                logger.warning(
                    f"APIタイムアウト (試行 {attempt + 1}/{self.config.max_retries + 1}): {e}"
                )

            except Exception as e:
                # 予期せぬエラー
                raise ClaudeClientError(
                    f"予期せぬエラー: {type(e).__name__}",
                    original_error=e,
                    is_retryable=False,
                ) from e

            # 最後の試行でなければリトライ
            if attempt < self.config.max_retries:
                logger.info(f"{retry_delay:.1f}秒後にリトライします...")
                time.sleep(retry_delay)
                retry_delay = min(
                    retry_delay * self.BACKOFF_MULTIPLIER, self.MAX_RETRY_DELAY
                )

        # 全リトライ失敗
        raise ClaudeClientError(
            f"最大リトライ回数 ({self.config.max_retries}) を超えました",
            original_error=last_error,
            is_retryable=True,
        )

    def _call_api(
        self,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ClaudeResponse:
        """API呼び出しの実行（リトライなし）

        Args:
            system_prompt: システムプロンプト
            messages: メッセージ履歴
            tools: ツール定義

        Returns:
            ClaudeResponse: パース済みのレスポンス
        """
        # API呼び出しパラメータ
        params: Dict[str, Any] = {
            "model": self.config.model_name,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "system": system_prompt,
            "messages": messages,
        }

        # ツールが指定されている場合のみ追加
        if tools:
            params["tools"] = tools

        logger.debug(f"API呼び出し: model={self.config.model_name}, messages={len(messages)}")

        # API呼び出し
        response = self._client.messages.create(**params)

        # 使用量を記録（レート制限用）
        if self._rate_limiter and response.usage:
            self._rate_limiter.record_usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )

        # レスポンスをパース
        return self._parse_response(response)

    def _estimate_input_tokens(
        self,
        system_prompt: str,
        messages: List[Dict[str, Any]],
    ) -> int:
        """入力トークン数を推定

        簡易的な推定（文字数 / 4）を使用します。
        より正確な推定が必要な場合は tokenizer を使用できます。

        Args:
            system_prompt: システムプロンプト
            messages: メッセージ履歴

        Returns:
            推定トークン数
        """
        total_chars = len(system_prompt)
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                # マルチモーダルコンテンツの場合
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        total_chars += len(block.get("text", ""))

        # 英語の場合、約4文字で1トークン（簡易推定）
        return total_chars // 4

    def _parse_response(self, response: anthropic.types.Message) -> ClaudeResponse:
        """APIレスポンスをパース

        Args:
            response: Anthropic APIのレスポンス

        Returns:
            ClaudeResponse: 構造化されたレスポンス
        """
        content_text = ""
        tool_uses: List[ToolUse] = []

        # content配列を処理
        for block in response.content:
            if block.type == "text":
                content_text += block.text
            elif block.type == "tool_use":
                tool_uses.append(
                    ToolUse(
                        id=block.id,
                        name=block.name,
                        input=block.input,
                    )
                )

        # 使用量情報
        usage = None
        if response.usage:
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }

        logger.debug(
            f"API応答: stop_reason={response.stop_reason}, "
            f"content_length={len(content_text)}, tool_uses={len(tool_uses)}"
        )

        return ClaudeResponse(
            content=content_text,
            stop_reason=response.stop_reason,
            tool_uses=tool_uses,
            usage=usage,
        )

    def is_available(self) -> bool:
        """APIが利用可能かテスト

        Returns:
            True: API呼び出しが成功した場合
            False: API呼び出しが失敗した場合
        """
        try:
            self.complete(
                system_prompt="Test",
                messages=[{"role": "user", "content": "Hi"}],
            )
            return True
        except ClaudeClientError:
            return False


# シングルトンインスタンス（遅延初期化）
_client_instance: Optional[ClaudeClient] = None


def get_claude_client(config: Optional[LLMConfig] = None) -> ClaudeClient:
    """グローバルなClaudeクライアントを取得

    初回呼び出し時にクライアントを初期化します。

    Args:
        config: LLM設定（省略時はデフォルト設定を使用）

    Returns:
        ClaudeClient インスタンス

    Raises:
        ClaudeClientError: クライアントの初期化に失敗した場合
    """
    global _client_instance
    if _client_instance is None:
        if config is None:
            config = LLMConfig()
        _client_instance = ClaudeClient(config)
    return _client_instance


def reset_client() -> None:
    """グローバルクライアントをリセット（テスト用）"""
    global _client_instance
    _client_instance = None
