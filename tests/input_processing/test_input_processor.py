"""InputProcessor の単体テスト"""

import pytest

from src.config.phase2_config import Phase2Config
from src.input_processing import InputProcessor, ProcessedInput


@pytest.fixture
def config() -> Phase2Config:
    """テスト用設定"""
    return Phase2Config()


@pytest.fixture
def processor(config: Phase2Config) -> InputProcessor:
    """テスト用InputProcessorインスタンス"""
    return InputProcessor(config)


class TestProcessedInputDataclass:
    """ProcessedInput データクラスのテスト"""

    def test_default_values(self):
        """デフォルト値の確認"""
        result = ProcessedInput(summary="test")
        assert result.summary == "test"
        assert result.detail_refs == []
        assert result.items == []
        assert result.item_count == 0
        assert result.original_size_tokens == 0
        assert result.needs_negotiation is False
        assert result.negotiation_options == []

    def test_full_initialization(self):
        """全フィールドを指定した初期化"""
        result = ProcessedInput(
            summary="概要",
            detail_refs=["ref1"],
            items=["項目1", "項目2"],
            item_count=2,
            original_size_tokens=100,
            needs_negotiation=True,
            negotiation_options=["オプション1"],
        )
        assert result.summary == "概要"
        assert len(result.detail_refs) == 1
        assert len(result.items) == 2
        assert result.item_count == 2
        assert result.original_size_tokens == 100
        assert result.needs_negotiation is True
        assert len(result.negotiation_options) == 1


class TestInputProcessorBasic:
    """InputProcessor 基本テスト"""

    def test_process_simple_input(self, processor: InputProcessor):
        """シンプルな入力の処理"""
        result = processor.process("データベースを設計してください")
        assert result.summary == "データベースを設計してください"
        assert result.detail_refs == []
        assert result.needs_negotiation is False
        assert result.original_size_tokens > 0

    def test_process_empty_input(self, processor: InputProcessor):
        """空入力の処理"""
        result = processor.process("")
        assert result.summary == ""
        assert result.item_count == 0
        assert result.needs_negotiation is False

    def test_process_whitespace_input(self, processor: InputProcessor):
        """空白のみの入力"""
        result = processor.process("   ")
        assert result.summary == ""
        assert result.item_count == 0


class TestInputProcessorItemDetection:
    """論点検出の統合テスト"""

    def test_detect_items_below_threshold(self, processor: InputProcessor):
        """閾値未満の論点数"""
        user_input = """- タスク1
- タスク2
- タスク3
"""
        result = processor.process(user_input)
        assert result.item_count == 3
        assert result.needs_negotiation is False
        assert len(result.items) == 3

    def test_detect_items_at_threshold(self, processor: InputProcessor):
        """閾値ちょうどの論点数（10個）"""
        items = "\n".join([f"- タスク{i}" for i in range(1, 11)])
        result = processor.process(items)
        assert result.item_count == 10
        assert result.needs_negotiation is True
        assert len(result.negotiation_options) == 3

    def test_detect_items_above_threshold(self, processor: InputProcessor):
        """閾値を超える論点数"""
        items = "\n".join([f"- タスク{i}" for i in range(1, 16)])
        result = processor.process(items)
        assert result.item_count == 15
        assert result.needs_negotiation is True
        assert "優先度の高い" in result.negotiation_options[0]


class TestInputProcessorSizeCheck:
    """入力サイズチェックのテスト"""

    def test_small_input_no_summary(self, processor: InputProcessor):
        """小さい入力は概要生成しない"""
        small_input = "短いテキスト"
        result = processor.process(small_input)
        assert result.summary == small_input
        assert result.detail_refs == []

    def test_large_input_generates_summary(self, processor: InputProcessor):
        """大きい入力は概要を生成"""
        # input_size_threshold = 5000
        large_input = "あ" * 6000
        result = processor.process(large_input)
        assert len(result.summary) < len(large_input)
        assert len(result.detail_refs) == 1
        assert result.original_size_tokens >= 5000

    def test_large_input_preserves_detail(self, processor: InputProcessor):
        """大きい入力の詳細を保存"""
        large_input = "重要な情報: " + "あ" * 6000
        result = processor.process(large_input)

        # 詳細を取得できることを確認
        assert len(result.detail_refs) == 1
        detail = processor.get_detail(result.detail_refs[0])
        assert detail == large_input


class TestInputProcessorNegotiation:
    """交渉オプションのテスト"""

    def test_negotiation_options_content(self, processor: InputProcessor):
        """交渉オプションの内容を確認"""
        items = "\n".join([f"- タスク{i}" for i in range(1, 12)])
        result = processor.process(items)

        assert result.needs_negotiation is True
        assert len(result.negotiation_options) == 3
        assert "優先度の高い" in result.negotiation_options[0]
        assert "全て処理します" in result.negotiation_options[1]
        assert "カテゴリ別" in result.negotiation_options[2]


class TestInputProcessorDetailStorage:
    """詳細データ保存のテスト"""

    def test_get_detail_returns_original(self, processor: InputProcessor):
        """保存した詳細データを取得"""
        large_input = "テスト" * 2000
        result = processor.process(large_input)

        if result.detail_refs:
            detail = processor.get_detail(result.detail_refs[0])
            assert detail == large_input

    def test_get_nonexistent_detail(self, processor: InputProcessor):
        """存在しない詳細参照"""
        result = processor.get_detail("nonexistent_ref")
        assert result is None


class TestInputProcessorConfiguration:
    """設定による動作変更のテスト"""

    def test_custom_item_threshold(self):
        """カスタム論点閾値"""
        config = Phase2Config()
        config.input_item_threshold = 5
        processor = InputProcessor(config)

        items = "\n".join([f"- タスク{i}" for i in range(1, 6)])
        result = processor.process(items)
        assert result.needs_negotiation is True

    def test_custom_size_threshold(self):
        """カスタムサイズ閾値"""
        config = Phase2Config()
        config.input_size_threshold = 100
        processor = InputProcessor(config)

        text = "あ" * 150
        result = processor.process(text)
        assert len(result.detail_refs) == 1


class TestInputProcessorDefaultConfig:
    """デフォルト設定のテスト"""

    def test_no_config_uses_default(self):
        """config=Noneでデフォルト設定を使用"""
        processor = InputProcessor(config=None)
        assert processor.config is not None
        assert processor.config.input_item_threshold == 10


class TestInputProcessorIntegration:
    """統合テスト"""

    def test_complex_input_with_many_items_and_large_size(self, processor: InputProcessor):
        """多数の論点と大きなサイズの入力"""
        # 大きな入力 + 多数の論点
        prefix = "概要説明文。" * 500  # 約3000文字
        items = "\n".join([f"- 詳細タスク{i}" for i in range(1, 15)])
        large_input = prefix + "\n" + items

        result = processor.process(large_input)
        assert result.item_count >= 10  # 論点数が閾値以上
        assert result.needs_negotiation is True

    def test_ambiguous_input_passed_through(self, processor: InputProcessor):
        """曖昧な入力もそのまま渡す（解釈しない）"""
        ambiguous = "いい感じにしてください"
        result = processor.process(ambiguous)
        assert result.summary == ambiguous
        # 曖昧さの解消はオーケストレーターの責務なので、ここでは何もしない
        assert result.needs_negotiation is False

    def test_token_count_estimation(self, processor: InputProcessor):
        """トークン数の推定"""
        text = "テスト文字列です。"  # 9文字
        result = processor.process(text)
        # 1文字 = 1トークン で計算
        assert result.original_size_tokens == len(text)
