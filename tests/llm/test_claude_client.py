# Claude API クライアントの単体テスト
# 実装仕様: docs/phase1-implementation-spec.ja.md

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import asdict

import anthropic
from anthropic import APIConnectionError, APIStatusError, RateLimitError

from src.config.llm_config import LLMConfig
from src.llm.claude_client import (
    ClaudeClient,
    ClaudeClientError,
    ClaudeResponse,
    ToolUse,
    get_claude_client,
    reset_client,
)


class TestToolUseDataclass:
    """ToolUse dataclass のテスト"""

    def test_tool_use_creation(self):
        """ToolUseを正しく作成できることを確認"""
        tool_use = ToolUse(
            id="tool_123",
            name="get_weather",
            input={"location": "Tokyo"},
        )
        assert tool_use.id == "tool_123"
        assert tool_use.name == "get_weather"
        assert tool_use.input == {"location": "Tokyo"}

    def test_tool_use_to_dict(self):
        """ToolUseを辞書に変換できることを確認"""
        tool_use = ToolUse(
            id="tool_456",
            name="search",
            input={"query": "test"},
        )
        data = asdict(tool_use)
        assert data == {
            "id": "tool_456",
            "name": "search",
            "input": {"query": "test"},
        }


class TestClaudeResponseDataclass:
    """ClaudeResponse dataclass のテスト"""

    def test_response_creation_minimal(self):
        """最小限のClaudeResponseを作成できることを確認"""
        response = ClaudeResponse(
            content="Hello!",
            stop_reason="end_turn",
        )
        assert response.content == "Hello!"
        assert response.stop_reason == "end_turn"
        assert response.tool_uses == []
        assert response.usage is None

    def test_response_creation_with_tools(self):
        """ツール情報付きのClaudeResponseを作成できることを確認"""
        tool_use = ToolUse(id="tool_1", name="test", input={})
        response = ClaudeResponse(
            content="",
            stop_reason="tool_use",
            tool_uses=[tool_use],
            usage={"input_tokens": 10, "output_tokens": 5},
        )
        assert response.content == ""
        assert response.stop_reason == "tool_use"
        assert len(response.tool_uses) == 1
        assert response.tool_uses[0].name == "test"
        assert response.usage == {"input_tokens": 10, "output_tokens": 5}


class TestClaudeClientErrorClass:
    """ClaudeClientError カスタムエラーのテスト"""

    def test_error_creation_simple(self):
        """シンプルなエラーを作成できることを確認"""
        error = ClaudeClientError("Test error")
        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.original_error is None
        assert error.is_retryable is False

    def test_error_creation_with_original(self):
        """元のエラー付きでエラーを作成できることを確認"""
        original = ValueError("Original error")
        error = ClaudeClientError(
            "Wrapped error",
            original_error=original,
            is_retryable=True,
        )
        assert "Wrapped error" in str(error)
        assert "Original error" in str(error)
        assert error.original_error is original
        assert error.is_retryable is True


class TestClaudeClientInitialization:
    """ClaudeClient 初期化テスト"""

    def test_init_with_valid_config(self):
        """有効な設定でクライアントを初期化できることを確認"""
        config = LLMConfig(api_key="test-api-key")
        with patch.object(anthropic, "Anthropic"):
            client = ClaudeClient(config)
            assert client.config == config

    def test_init_with_invalid_config_missing_api_key(self, monkeypatch):
        """APIキーがない場合にエラーが発生することを確認"""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        config = LLMConfig()
        with pytest.raises(ClaudeClientError, match="無効な設定"):
            ClaudeClient(config)

    def test_init_with_invalid_config_bad_temperature(self):
        """無効なtemperatureの場合にエラーが発生することを確認"""
        config = LLMConfig(api_key="test-api-key", temperature=2.0)
        with pytest.raises(ClaudeClientError, match="無効な設定"):
            ClaudeClient(config)


class TestClaudeClientComplete:
    """ClaudeClient.complete() メソッドのテスト"""

    @pytest.fixture
    def mock_client(self):
        """モック化されたClaudeClientを作成"""
        config = LLMConfig(api_key="test-api-key", max_retries=2)
        with patch.object(anthropic, "Anthropic") as mock_anthropic:
            client = ClaudeClient(config)
            yield client, mock_anthropic.return_value

    def test_complete_simple_text_response(self, mock_client):
        """シンプルなテキスト応答を処理できることを確認"""
        client, mock_api = mock_client

        # モックレスポンスを設定
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(type="text", text="Hello, world!")
        ]
        mock_response.stop_reason = "end_turn"
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)
        mock_api.messages.create.return_value = mock_response

        response = client.complete(
            system_prompt="You are a helpful assistant.",
            messages=[{"role": "user", "content": "Hi"}],
        )

        assert response.content == "Hello, world!"
        assert response.stop_reason == "end_turn"
        assert response.tool_uses == []
        assert response.usage == {"input_tokens": 10, "output_tokens": 5}

    def test_complete_with_tool_use(self, mock_client):
        """ツール使用を含む応答を処理できることを確認"""
        client, mock_api = mock_client

        # モックレスポンスを設定
        mock_tool_use = MagicMock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.id = "tool_abc123"
        mock_tool_use.name = "get_weather"
        mock_tool_use.input = {"location": "Tokyo"}

        mock_response = MagicMock()
        mock_response.content = [mock_tool_use]
        mock_response.stop_reason = "tool_use"
        mock_response.usage = MagicMock(input_tokens=20, output_tokens=15)
        mock_api.messages.create.return_value = mock_response

        tools = [{
            "name": "get_weather",
            "description": "Get weather info",
            "input_schema": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
            },
        }]

        response = client.complete(
            system_prompt="You are a helpful assistant.",
            messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
            tools=tools,
        )

        assert response.stop_reason == "tool_use"
        assert len(response.tool_uses) == 1
        assert response.tool_uses[0].id == "tool_abc123"
        assert response.tool_uses[0].name == "get_weather"
        assert response.tool_uses[0].input == {"location": "Tokyo"}

    def test_complete_with_text_and_tool_use(self, mock_client):
        """テキストとツール使用の両方を含む応答を処理できることを確認"""
        client, mock_api = mock_client

        mock_text = MagicMock()
        mock_text.type = "text"
        mock_text.text = "Let me check the weather."

        mock_tool_use = MagicMock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.id = "tool_xyz"
        mock_tool_use.name = "get_weather"
        mock_tool_use.input = {"location": "Osaka"}

        mock_response = MagicMock()
        mock_response.content = [mock_text, mock_tool_use]
        mock_response.stop_reason = "tool_use"
        mock_response.usage = MagicMock(input_tokens=30, output_tokens=25)
        mock_api.messages.create.return_value = mock_response

        response = client.complete(
            system_prompt="Test",
            messages=[{"role": "user", "content": "Weather?"}],
            tools=[{"name": "get_weather", "description": "Get weather", "input_schema": {}}],
        )

        assert response.content == "Let me check the weather."
        assert response.stop_reason == "tool_use"
        assert len(response.tool_uses) == 1


class TestClaudeClientRetryLogic:
    """ClaudeClient リトライロジックのテスト"""

    @pytest.fixture
    def mock_client_fast_retry(self):
        """リトライ間隔を短くしたモッククライアント"""
        config = LLMConfig(api_key="test-api-key", max_retries=2)
        with patch.object(anthropic, "Anthropic") as mock_anthropic:
            client = ClaudeClient(config)
            # リトライ待機時間を短縮
            client.INITIAL_RETRY_DELAY = 0.01
            client.MAX_RETRY_DELAY = 0.02
            yield client, mock_anthropic.return_value

    def test_retry_on_rate_limit(self, mock_client_fast_retry):
        """レート制限時にリトライすることを確認"""
        client, mock_api = mock_client_fast_retry

        # 最初の2回は失敗、3回目は成功
        mock_success = MagicMock()
        mock_success.content = [MagicMock(type="text", text="Success")]
        mock_success.stop_reason = "end_turn"
        mock_success.usage = MagicMock(input_tokens=1, output_tokens=1)

        mock_api.messages.create.side_effect = [
            RateLimitError(
                message="rate limited",
                response=MagicMock(status_code=429),
                body=None,
            ),
            RateLimitError(
                message="rate limited",
                response=MagicMock(status_code=429),
                body=None,
            ),
            mock_success,
        ]

        response = client.complete(
            system_prompt="Test",
            messages=[{"role": "user", "content": "Hi"}],
        )

        assert response.content == "Success"
        assert mock_api.messages.create.call_count == 3

    def test_retry_on_connection_error(self, mock_client_fast_retry):
        """接続エラー時にリトライすることを確認"""
        client, mock_api = mock_client_fast_retry

        mock_success = MagicMock()
        mock_success.content = [MagicMock(type="text", text="Connected")]
        mock_success.stop_reason = "end_turn"
        mock_success.usage = None

        mock_api.messages.create.side_effect = [
            APIConnectionError(request=MagicMock()),
            mock_success,
        ]

        response = client.complete(
            system_prompt="Test",
            messages=[{"role": "user", "content": "Hi"}],
        )

        assert response.content == "Connected"
        assert mock_api.messages.create.call_count == 2

    def test_retry_on_server_error(self, mock_client_fast_retry):
        """5xxエラー時にリトライすることを確認"""
        client, mock_api = mock_client_fast_retry

        mock_success = MagicMock()
        mock_success.content = [MagicMock(type="text", text="Recovered")]
        mock_success.stop_reason = "end_turn"
        mock_success.usage = None

        mock_api.messages.create.side_effect = [
            APIStatusError(
                message="Server error",
                response=MagicMock(status_code=500),
                body=None,
            ),
            mock_success,
        ]

        response = client.complete(
            system_prompt="Test",
            messages=[{"role": "user", "content": "Hi"}],
        )

        assert response.content == "Recovered"

    def test_no_retry_on_client_error(self, mock_client_fast_retry):
        """4xxエラー時はリトライしないことを確認"""
        client, mock_api = mock_client_fast_retry

        mock_api.messages.create.side_effect = APIStatusError(
            message="Bad request",
            response=MagicMock(status_code=400),
            body=None,
        )

        with pytest.raises(ClaudeClientError) as excinfo:
            client.complete(
                system_prompt="Test",
                messages=[{"role": "user", "content": "Hi"}],
            )

        assert "HTTP 400" in str(excinfo.value)
        assert excinfo.value.is_retryable is False
        assert mock_api.messages.create.call_count == 1

    def test_max_retries_exceeded(self, mock_client_fast_retry):
        """最大リトライ回数を超えた場合にエラーになることを確認"""
        client, mock_api = mock_client_fast_retry

        mock_api.messages.create.side_effect = RateLimitError(
            message="rate limited",
            response=MagicMock(status_code=429),
            body=None,
        )

        with pytest.raises(ClaudeClientError) as excinfo:
            client.complete(
                system_prompt="Test",
                messages=[{"role": "user", "content": "Hi"}],
            )

        assert "最大リトライ回数" in str(excinfo.value)
        assert excinfo.value.is_retryable is True
        # max_retries=2 なので、初回 + 2回リトライ = 3回
        assert mock_api.messages.create.call_count == 3


class TestClaudeClientTimeout:
    """ClaudeClient タイムアウト処理のテスト"""

    def test_timeout_error_is_retried(self):
        """タイムアウトエラーがリトライされることを確認"""
        config = LLMConfig(api_key="test-api-key", max_retries=1, timeout_seconds=10)
        with patch.object(anthropic, "Anthropic") as mock_anthropic:
            client = ClaudeClient(config)
            client.INITIAL_RETRY_DELAY = 0.01
            mock_api = mock_anthropic.return_value

            mock_success = MagicMock()
            mock_success.content = [MagicMock(type="text", text="OK")]
            mock_success.stop_reason = "end_turn"
            mock_success.usage = None

            mock_api.messages.create.side_effect = [
                anthropic.APITimeoutError(request=MagicMock()),
                mock_success,
            ]

            response = client.complete(
                system_prompt="Test",
                messages=[{"role": "user", "content": "Hi"}],
            )

            assert response.content == "OK"
            assert mock_api.messages.create.call_count == 2


class TestClaudeClientIsAvailable:
    """ClaudeClient.is_available() メソッドのテスト"""

    def test_is_available_true(self):
        """APIが利用可能な場合にTrueを返すことを確認"""
        config = LLMConfig(api_key="test-api-key")
        with patch.object(anthropic, "Anthropic") as mock_anthropic:
            client = ClaudeClient(config)
            mock_api = mock_anthropic.return_value

            mock_response = MagicMock()
            mock_response.content = [MagicMock(type="text", text="Hi")]
            mock_response.stop_reason = "end_turn"
            mock_response.usage = None
            mock_api.messages.create.return_value = mock_response

            assert client.is_available() is True

    def test_is_available_false(self):
        """APIが利用不可能な場合にFalseを返すことを確認"""
        config = LLMConfig(api_key="test-api-key", max_retries=0)
        with patch.object(anthropic, "Anthropic") as mock_anthropic:
            client = ClaudeClient(config)
            mock_api = mock_anthropic.return_value
            mock_api.messages.create.side_effect = APIConnectionError(
                request=MagicMock()
            )

            assert client.is_available() is False


class TestGlobalClientFunctions:
    """グローバルクライアント関数のテスト"""

    def setup_method(self):
        """各テストの前にクライアントをリセット"""
        reset_client()

    def teardown_method(self):
        """各テストの後にクライアントをリセット"""
        reset_client()

    def test_get_claude_client_creates_instance(self):
        """get_claude_clientがインスタンスを作成することを確認"""
        config = LLMConfig(api_key="test-api-key")
        with patch.object(anthropic, "Anthropic"):
            client = get_claude_client(config)
            assert isinstance(client, ClaudeClient)

    def test_get_claude_client_returns_same_instance(self):
        """get_claude_clientが同じインスタンスを返すことを確認"""
        config = LLMConfig(api_key="test-api-key")
        with patch.object(anthropic, "Anthropic"):
            client1 = get_claude_client(config)
            client2 = get_claude_client()
            assert client1 is client2

    def test_reset_client_clears_instance(self):
        """reset_clientがインスタンスをクリアすることを確認"""
        config = LLMConfig(api_key="test-api-key")
        with patch.object(anthropic, "Anthropic"):
            client1 = get_claude_client(config)
            reset_client()
            client2 = get_claude_client(config)
            assert client1 is not client2


class TestModuleExport:
    """モジュールエクスポートのテスト"""

    def test_import_from_llm_module(self):
        """src.llmからインポートできることを確認"""
        from src.llm import (
            ClaudeClient,
            ClaudeClientError,
            ClaudeResponse,
            ToolUse,
            get_claude_client,
            reset_client,
        )
        assert ClaudeClient is not None
        assert ClaudeClientError is not None
        assert ClaudeResponse is not None
        assert ToolUse is not None
        assert get_claude_client is not None
        assert reset_client is not None
