# LLM 設定クラスの単体テスト
# 実装仕様: docs/phase1-implementation-spec.ja.md

import os
import pytest

from src.config.llm_config import LLMConfig, llm_config


class TestLLMConfigDefaults:
    """LLMConfigのデフォルト値テスト"""

    def test_model_name_default(self):
        """model_nameのデフォルト値を確認"""
        config = LLMConfig()
        assert config.model_name == "claude-sonnet-4-5-20250929"

    def test_max_tokens_default(self):
        """max_tokensのデフォルト値（8192）を確認"""
        config = LLMConfig()
        assert config.max_tokens == 8192

    def test_temperature_default(self):
        """temperatureのデフォルト値（0.7）を確認"""
        config = LLMConfig()
        assert config.temperature == 0.7

    def test_timeout_seconds_default(self):
        """timeout_secondsのデフォルト値（120）を確認"""
        config = LLMConfig()
        assert config.timeout_seconds == 120

    def test_max_retries_default(self):
        """max_retriesのデフォルト値（3）を確認"""
        config = LLMConfig()
        assert config.max_retries == 3


class TestLLMConfigEnvironmentVariables:
    """環境変数との連携テスト"""

    def test_api_key_from_env(self, monkeypatch):
        """ANTHROPIC_API_KEYから api_key を取得できることを確認"""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key-12345")
        config = LLMConfig()
        assert config.api_key == "test-api-key-12345"

    def test_api_key_explicit_overrides_env(self, monkeypatch):
        """明示的に指定した api_key が環境変数より優先されることを確認"""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-api-key")
        config = LLMConfig(api_key="explicit-api-key")
        assert config.api_key == "explicit-api-key"

    def test_model_name_from_env(self, monkeypatch):
        """CLAUDE_MODELから model_name を取得できることを確認"""
        monkeypatch.setenv("CLAUDE_MODEL", "claude-opus-4-20250514")
        config = LLMConfig()
        assert config.model_name == "claude-opus-4-20250514"

    def test_model_name_explicit_overrides_env(self, monkeypatch):
        """環境変数がmodel_nameのデフォルト値を上書きすることを確認"""
        monkeypatch.setenv("CLAUDE_MODEL", "claude-opus-4-20250514")
        # 明示的に指定しない場合は環境変数が使用される
        config = LLMConfig()
        assert config.model_name == "claude-opus-4-20250514"

    def test_no_env_vars_set(self, monkeypatch):
        """環境変数が設定されていない場合のデフォルト動作を確認"""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("CLAUDE_MODEL", raising=False)
        config = LLMConfig()
        assert config.api_key is None
        assert config.model_name == "claude-sonnet-4-5-20250929"


class TestLLMConfigValidation:
    """設定値検証テスト"""

    def test_validate_missing_api_key(self, monkeypatch):
        """api_keyが設定されていない場合にValueErrorが発生することを確認"""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        config = LLMConfig()
        with pytest.raises(ValueError, match="Anthropic APIキーが設定されていません"):
            config.validate()

    def test_validate_with_api_key(self, monkeypatch):
        """api_keyが設定されている場合に検証が通ることを確認"""
        config = LLMConfig(api_key="valid-api-key")
        config.validate()  # エラーが発生しなければOK

    def test_validate_invalid_max_tokens(self, monkeypatch):
        """max_tokensが0以下の場合にValueErrorが発生することを確認"""
        config = LLMConfig(api_key="valid-api-key", max_tokens=0)
        with pytest.raises(ValueError, match="max_tokens は正の整数"):
            config.validate()

    def test_validate_invalid_temperature_high(self, monkeypatch):
        """temperatureが1.0より大きい場合にValueErrorが発生することを確認"""
        config = LLMConfig(api_key="valid-api-key", temperature=1.5)
        with pytest.raises(ValueError, match="temperature は 0.0-1.0 の範囲"):
            config.validate()

    def test_validate_invalid_temperature_low(self, monkeypatch):
        """temperatureが0.0未満の場合にValueErrorが発生することを確認"""
        config = LLMConfig(api_key="valid-api-key", temperature=-0.1)
        with pytest.raises(ValueError, match="temperature は 0.0-1.0 の範囲"):
            config.validate()

    def test_validate_invalid_timeout(self, monkeypatch):
        """timeout_secondsが0以下の場合にValueErrorが発生することを確認"""
        config = LLMConfig(api_key="valid-api-key", timeout_seconds=0)
        with pytest.raises(ValueError, match="timeout_seconds は正の整数"):
            config.validate()

    def test_validate_invalid_max_retries(self, monkeypatch):
        """max_retriesが負の場合にValueErrorが発生することを確認"""
        config = LLMConfig(api_key="valid-api-key", max_retries=-1)
        with pytest.raises(ValueError, match="max_retries は非負の整数"):
            config.validate()


class TestLLMConfigIsConfigured:
    """is_configuredプロパティテスト"""

    def test_is_configured_true(self):
        """api_keyが設定されている場合にTrueを返すことを確認"""
        config = LLMConfig(api_key="valid-api-key")
        assert config.is_configured is True

    def test_is_configured_false_none(self, monkeypatch):
        """api_keyがNoneの場合にFalseを返すことを確認"""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        config = LLMConfig()
        assert config.is_configured is False

    def test_is_configured_false_empty(self):
        """api_keyが空文字列の場合にFalseを返すことを確認"""
        config = LLMConfig(api_key="")
        assert config.is_configured is False


class TestLLMConfigCustomValues:
    """カスタム設定値テスト"""

    def test_custom_values(self):
        """すべてのパラメータをカスタム値で設定できることを確認"""
        config = LLMConfig(
            model_name="claude-opus-4-20250514",
            max_tokens=8192,
            temperature=0.5,
            api_key="custom-api-key",
            timeout_seconds=180,
            max_retries=5,
        )
        assert config.model_name == "claude-opus-4-20250514"
        assert config.max_tokens == 8192
        assert config.temperature == 0.5
        assert config.api_key == "custom-api-key"
        assert config.timeout_seconds == 180
        assert config.max_retries == 5


class TestLLMConfigRepr:
    """repr出力でapi_keyが非表示になることを確認"""

    def test_api_key_hidden_in_repr(self):
        """api_keyがreprに表示されないことを確認"""
        config = LLMConfig(api_key="secret-api-key")
        repr_str = repr(config)
        assert "secret-api-key" not in repr_str
        assert "api_key" not in repr_str


class TestDefaultInstance:
    """デフォルトインスタンスのテスト"""

    def test_llm_config_instance_exists(self):
        """llm_configインスタンスが存在することを確認"""
        assert llm_config is not None

    def test_llm_config_is_llm_config(self):
        """llm_configがLLMConfigインスタンスであることを確認"""
        assert isinstance(llm_config, LLMConfig)


class TestModuleExport:
    """モジュールエクスポートのテスト"""

    def test_import_from_config_module(self):
        """src.configからインポートできることを確認"""
        from src.config import LLMConfig, llm_config
        assert LLMConfig is not None
        assert llm_config is not None
