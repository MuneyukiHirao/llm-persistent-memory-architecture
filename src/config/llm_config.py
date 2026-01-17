# LLM 呼び出し設定
# 実装仕様: docs/phase1-implementation-spec.ja.md

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RateLimitConfig:
    """レート制限設定

    Anthropic API の制限値に基づくデフォルト値を提供します。
    Tier 2 制限 (Claude Sonnet 4/4.5):
    - 1,000 リクエスト/分
    - 450,000 入力トークン/分
    - 90,000 出力トークン/分
    """

    requests_per_minute: int = 1000
    """1分あたりの最大リクエスト数"""

    input_tokens_per_minute: int = 450000
    """1分あたりの最大入力トークン数"""

    output_tokens_per_minute: int = 90000
    """1分あたりの最大出力トークン数"""
    
    safety_margin: float = 0.9
    """安全マージン（制限値の90%で制限）"""
    
    window_seconds: int = 60
    """レート制限の時間窓（秒）"""


@dataclass
class LLMConfig:
    """LLM 呼び出し設定

    Claude API を使用するための設定パラメータを管理します。
    環境変数からの設定取得をサポートします。

    環境変数:
        ANTHROPIC_API_KEY: Anthropic APIキー（必須）
        CLAUDE_MODEL: モデル名（オプション、デフォルト: claude-sonnet-4-20250514）

    使用例:
        config = LLMConfig()  # 環境変数から自動取得
        config = LLMConfig(api_key="sk-...", model_name="claude-opus-4-20250514")
    """

    model_name: str = "claude-sonnet-4-5-20250929"
    """使用するClaudeモデル名"""

    max_tokens: int = 8192
    """最大出力トークン数"""

    temperature: float = 0.7
    """サンプリング温度（0.0-1.0）"""

    api_key: Optional[str] = field(default=None, repr=False)
    """Anthropic APIキー（repr=Falseでログ出力時に非表示）"""

    timeout_seconds: int = 120
    """API呼び出しのタイムアウト（秒）"""

    max_retries: int = 3
    """リトライ回数"""
    
    enable_rate_limiting: bool = True
    """レート制限機能を有効にするか"""
    
    rate_limit_config: RateLimitConfig = field(default_factory=RateLimitConfig)
    """レート制限設定"""

    def __post_init__(self) -> None:
        """初期化後の処理: 環境変数から設定を取得"""
        # APIキーが指定されていない場合は環境変数から取得
        if self.api_key is None:
            self.api_key = os.getenv("ANTHROPIC_API_KEY")

        # モデル名が環境変数で上書きされている場合は適用
        env_model = os.getenv("CLAUDE_MODEL")
        if env_model:
            self.model_name = env_model
            
        # レート制限の環境変数設定
        if os.getenv("DISABLE_RATE_LIMITING") == "true":
            self.enable_rate_limiting = False

    def validate(self) -> None:
        """設定値を検証

        Raises:
            ValueError: 必須設定が欠けている場合、または値が無効な場合
        """
        if not self.api_key:
            raise ValueError(
                "Anthropic APIキーが設定されていません。"
                "ANTHROPIC_API_KEY 環境変数を設定するか、api_key 引数を指定してください。"
            )

        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens は正の整数である必要があります: {self.max_tokens}")

        if not (0.0 <= self.temperature <= 1.0):
            raise ValueError(f"temperature は 0.0-1.0 の範囲である必要があります: {self.temperature}")

        if self.timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds は正の整数である必要があります: {self.timeout_seconds}")

        if self.max_retries < 0:
            raise ValueError(f"max_retries は非負の整数である必要があります: {self.max_retries}")

    @property
    def is_configured(self) -> bool:
        """APIキーが設定されているかを確認

        Returns:
            True: APIキーが設定されている場合
            False: APIキーが設定されていない場合
        """
        return self.api_key is not None and len(self.api_key) > 0


# デフォルト設定のインスタンス
llm_config = LLMConfig()
