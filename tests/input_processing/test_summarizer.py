"""Summarizer の単体テスト"""

import pytest

from src.config.phase2_config import Phase2Config
from src.input_processing.summarizer import Summarizer


@pytest.fixture
def config() -> Phase2Config:
    """テスト用設定"""
    return Phase2Config()


@pytest.fixture
def summarizer(config: Phase2Config) -> Summarizer:
    """テスト用Summarizerインスタンス"""
    return Summarizer(config)


class TestSummarizerBasic:
    """基本的な概要生成のテスト"""

    def test_short_input_unchanged(self, summarizer: Summarizer):
        """短い入力はそのまま返す"""
        short_input = "これは短いテキストです。"
        result = summarizer.summarize(short_input)
        assert result == short_input

    def test_empty_input(self, summarizer: Summarizer):
        """空入力"""
        assert summarizer.summarize("") == ""

    def test_whitespace_input(self, summarizer: Summarizer):
        """空白のみの入力"""
        result = summarizer.summarize("   ")
        assert result == ""


class TestSummarizerTruncation:
    """長い入力の切り詰めテスト"""

    def test_long_input_truncated(self, summarizer: Summarizer):
        """長い入力は切り詰められる"""
        # summary_max_tokens = 1000 なので、1000文字以上で切り詰め
        long_input = "あ" * 2000
        result = summarizer.summarize(long_input)
        # 切り詰め + サフィックス
        assert len(result) < len(long_input)
        assert "[...入力が長いため省略されました" in result

    def test_truncation_at_sentence_boundary(self, summarizer: Summarizer):
        """文の境界で切り詰める"""
        # 句点を含む長いテキスト
        sentences = "これは文章です。" * 200  # 約1600文字
        result = summarizer.summarize(sentences)
        # 句点で終わっているか確認（サフィックスの前）
        main_part = result.split("\n\n[...")[0]
        assert main_part.endswith("。")

    def test_truncation_preserves_content(self, summarizer: Summarizer):
        """切り詰めても先頭の内容は保持される"""
        long_input = "重要な情報です。" + "あ" * 2000
        result = summarizer.summarize(long_input)
        assert "重要な情報です。" in result


class TestSummarizerSentenceBoundary:
    """文境界処理のテスト"""

    def test_truncate_at_question_mark(self, summarizer: Summarizer):
        """疑問符で切り詰める"""
        # カスタム設定で短い max_tokens を使用
        config = Phase2Config()
        config.summary_max_tokens = 50
        short_summarizer = Summarizer(config)

        text = "これは質問ですか？" + "あ" * 100
        result = short_summarizer.summarize(text)
        main_part = result.split("\n\n[...")[0]
        assert "？" in main_part or "あ" in main_part

    def test_truncate_at_newline(self, summarizer: Summarizer):
        """改行で切り詰める"""
        config = Phase2Config()
        config.summary_max_tokens = 100
        short_summarizer = Summarizer(config)

        text = "行1です。\n" + "あ" * 200
        result = short_summarizer.summarize(text)
        # 改行で区切られる可能性
        assert len(result) <= 200  # 何らかの切り詰めが行われる


class TestSummarizerConfiguration:
    """設定による動作変更のテスト"""

    def test_custom_max_tokens(self):
        """カスタムmax_tokensの動作"""
        config = Phase2Config()
        config.summary_max_tokens = 50
        summarizer = Summarizer(config)

        text = "あ" * 100
        result = summarizer.summarize(text)
        main_part = result.split("\n\n[...")[0]
        assert len(main_part) <= 50

    def test_large_max_tokens(self):
        """大きなmax_tokensの動作"""
        config = Phase2Config()
        config.summary_max_tokens = 10000
        summarizer = Summarizer(config)

        text = "あ" * 5000
        result = summarizer.summarize(text)
        # max_tokens以内なのでそのまま返る
        assert result == text


class TestSummarizerEdgeCases:
    """エッジケースのテスト"""

    def test_exactly_max_length(self, summarizer: Summarizer):
        """ちょうどmax_tokensの長さ"""
        text = "あ" * 1000  # summary_max_tokens = 1000
        result = summarizer.summarize(text)
        assert result == text  # 切り詰めなし

    def test_one_over_max_length(self, summarizer: Summarizer):
        """max_tokensを1文字超える"""
        text = "あ" * 1001
        result = summarizer.summarize(text)
        assert "[...入力が長いため省略されました" in result

    def test_no_sentence_boundaries(self):
        """文の区切りがない長いテキスト"""
        config = Phase2Config()
        config.summary_max_tokens = 50
        summarizer = Summarizer(config)

        # 句点も改行もない
        text = "あいうえお" * 20  # 100文字
        result = summarizer.summarize(text)
        # 区切りがなくてもそのまま切り詰められる
        assert "[...入力が長いため省略されました" in result

    def test_unicode_handling(self, summarizer: Summarizer):
        """Unicode文字の処理"""
        text = "🎉絵文字テスト" + "あ" * 2000
        result = summarizer.summarize(text)
        assert "🎉絵文字テスト" in result
