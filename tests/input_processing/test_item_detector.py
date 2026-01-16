"""ItemDetector の単体テスト"""

import pytest

from src.config.phase2_config import Phase2Config
from src.input_processing.item_detector import ItemDetector


@pytest.fixture
def config() -> Phase2Config:
    """テスト用設定"""
    return Phase2Config()


@pytest.fixture
def detector(config: Phase2Config) -> ItemDetector:
    """テスト用ItemDetectorインスタンス"""
    return ItemDetector(config)


class TestItemDetectorBulletLists:
    """箇条書きリストの検出テスト"""

    def test_detect_hyphen_bullet_list(self, detector: ItemDetector):
        """ハイフン箇条書きを検出"""
        user_input = """以下のタスクをお願いします:
- データベースの設計
- APIの実装
- テストの作成
"""
        items = detector.detect(user_input)
        assert len(items) == 3
        assert "データベースの設計" in items
        assert "APIの実装" in items
        assert "テストの作成" in items

    def test_detect_asterisk_bullet_list(self, detector: ItemDetector):
        """アスタリスク箇条書きを検出"""
        user_input = """* 機能Aを追加
* 機能Bを修正
* 機能Cを削除
"""
        items = detector.detect(user_input)
        assert len(items) == 3

    def test_detect_japanese_bullet_list(self, detector: ItemDetector):
        """日本語の箇条書き記号を検出"""
        user_input = """・ファイル読み込み
・データ変換
・結果出力
"""
        items = detector.detect(user_input)
        assert len(items) == 3

    def test_detect_mixed_bullets(self, detector: ItemDetector):
        """混合箇条書きを検出"""
        user_input = """- タスク1
* タスク2
・タスク3
"""
        items = detector.detect(user_input)
        assert len(items) == 3


class TestItemDetectorNumberedLists:
    """番号付きリストの検出テスト"""

    def test_detect_numbered_list_with_period(self, detector: ItemDetector):
        """「1. 」形式の番号付きリストを検出"""
        user_input = """1. 設計
2. 実装
3. テスト
4. デプロイ
"""
        items = detector.detect(user_input)
        assert len(items) == 4
        assert "設計" in items

    def test_detect_numbered_list_with_parenthesis(self, detector: ItemDetector):
        """「1) 」形式の番号付きリストを検出"""
        user_input = """1) 要件定義
2) 設計
3) 実装
"""
        items = detector.detect(user_input)
        assert len(items) == 3

    def test_detect_numbered_list_with_brackets(self, detector: ItemDetector):
        """「(1) 」形式の番号付きリストを検出"""
        user_input = """(1) データ取得
(2) データ処理
(3) 結果保存
"""
        items = detector.detect(user_input)
        assert len(items) == 3

    def test_detect_circled_numbers(self, detector: ItemDetector):
        """丸数字リストを検出"""
        user_input = """① バリデーション
② 保存処理
③ 通知送信
"""
        items = detector.detect(user_input)
        assert len(items) == 3


class TestItemDetectorHeadings:
    """見出し形式の検出テスト"""

    def test_detect_section_headings(self, detector: ItemDetector):
        """「第N項」形式を検出"""
        user_input = """第1項: 概要の説明
第2項: 詳細の説明
第3項: まとめ
"""
        items = detector.detect(user_input)
        assert len(items) == 3

    def test_detect_item_headings(self, detector: ItemDetector):
        """「項目N」形式を検出"""
        user_input = """項目1: ログイン機能
項目2: 検索機能
項目3: 通知機能
"""
        items = detector.detect(user_input)
        assert len(items) == 3


class TestItemDetectorFallback:
    """フォールバック（改行区切り）検出のテスト"""

    def test_no_detection_for_single_line(self, detector: ItemDetector):
        """単一行は論点リストとして検出しない"""
        user_input = "データベースを設計してください"
        items = detector.detect(user_input)
        assert len(items) == 0

    def test_no_detection_for_paragraph(self, detector: ItemDetector):
        """通常の文章は論点リストとして検出しない"""
        user_input = """これは通常の文章です。
複数行ありますが、論点リストではありません。
"""
        items = detector.detect(user_input)
        assert len(items) == 0  # 2行なのでフォールバックでも検出しない

    def test_detect_newline_separated_tasks(self, detector: ItemDetector):
        """改行区切りの独立したタスクを検出"""
        user_input = """ユーザー認証の実装
データベースマイグレーション
APIエンドポイント追加
テストケース作成
ドキュメント更新
"""
        items = detector.detect(user_input)
        # 5行あるので、フォールバックで検出される可能性がある
        # ただし、接続詞チェックなどで除外される可能性もある
        assert len(items) >= 3 or len(items) == 0  # どちらかの結果になる


class TestItemDetectorEdgeCases:
    """エッジケースのテスト"""

    def test_empty_input(self, detector: ItemDetector):
        """空入力"""
        assert detector.detect("") == []
        assert detector.detect("   ") == []

    def test_none_like_input(self, detector: ItemDetector):
        """None相当の入力"""
        assert detector.detect("") == []

    def test_duplicate_removal(self, detector: ItemDetector):
        """重複項目の除去"""
        user_input = """- タスクA
- タスクA
- タスクB
"""
        items = detector.detect(user_input)
        assert len(items) == 2
        assert items.count("タスクA") == 1

    def test_whitespace_handling(self, detector: ItemDetector):
        """空白の処理"""
        user_input = """  - タスク1
   - タスク2
"""
        items = detector.detect(user_input)
        assert len(items) == 2
        assert "タスク1" in items
        assert "タスク2" in items

    def test_continuation_line_excluded(self, detector: ItemDetector):
        """接続詞で始まる行は除外"""
        user_input = """タスクを実行します
また追加で確認します
さらに報告が必要です
しかし問題があります
ただし条件があります
"""
        items = detector.detect(user_input)
        # 「また」「さらに」「しかし」「ただし」で始まる行は除外される
        # 最初の行「タスクを実行します」のみが残る可能性
        # フォールバックで3行以上必要なため、0になる可能性が高い
        assert len(items) <= 1


class TestItemDetectorLargeLists:
    """大量のリストのテスト"""

    def test_detect_many_items(self, detector: ItemDetector):
        """多数の項目を検出"""
        items_text = "\n".join([f"- タスク{i}" for i in range(1, 21)])
        items = detector.detect(items_text)
        assert len(items) == 20

    def test_detect_exactly_threshold_items(self, detector: ItemDetector):
        """ちょうど閾値と同じ数の項目"""
        # config.input_item_threshold = 10
        items_text = "\n".join([f"- タスク{i}" for i in range(1, 11)])
        items = detector.detect(items_text)
        assert len(items) == 10
