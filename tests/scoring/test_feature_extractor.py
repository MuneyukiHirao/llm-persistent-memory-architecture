# FeatureExtractor テスト
# 実装仕様: docs/phase3-implementation-spec.ja.md セクション4.2, 5.1
"""
FeatureExtractor クラスのユニットテスト

テスト観点:
- テストカバレッジ: 全メソッド（extract_task_features, extract_agent_features, 内部メソッド）
- 再現性: 同一入力に対して常に同じ結果
- 境界値・異常系: 空入力、長大入力、特殊文字
- 保守性: 明確なテストケース名、Arrange-Act-Assert構造

テスト対象メソッド:
1. extract_task_features(task_summary)
2. extract_agent_features(agent, past_experiences)
3. _count_items(text)
4. _has_keywords(text, keywords)
5. _calculate_complexity(text)
"""

import pytest
from typing import Any, Dict, List

from src.agents.agent_registry import AgentDefinition
from src.config.phase3_config import Phase3Config, TASK_FEATURES, AGENT_FEATURES
from src.scoring.feature_extractor import (
    FeatureExtractor,
    CODE_KEYWORDS,
    RESEARCH_KEYWORDS,
    TEST_KEYWORDS,
)


# === フィクスチャ ===


@pytest.fixture
def config() -> Phase3Config:
    """テスト用 Phase3Config"""
    return Phase3Config()


@pytest.fixture
def extractor(config: Phase3Config) -> FeatureExtractor:
    """テスト用 FeatureExtractor"""
    return FeatureExtractor(config)


@pytest.fixture
def sample_agent() -> AgentDefinition:
    """テスト用エージェント定義"""
    return AgentDefinition(
        agent_id="test_agent",
        name="テストエージェント",
        role="テスト作成と品質検証を担当",
        perspectives=["正確性", "網羅性", "効率性", "再現性", "保守性"],
        system_prompt="あなたはテスト専門のエージェントです",
        capabilities=["testing", "debugging", "analysis"],
        status="active",
    )


@pytest.fixture
def agent_with_minimal_data() -> AgentDefinition:
    """最小データのエージェント定義"""
    return AgentDefinition(
        agent_id="minimal_agent",
        name="最小エージェント",
        role="最小構成",
        perspectives=[],
        system_prompt="",
        capabilities=[],
        status="active",
    )


@pytest.fixture
def past_experiences_success() -> List[Dict[str, Any]]:
    """成功履歴を含む過去経験"""
    return [
        {"success": True, "duration_seconds": 120.0},
        {"success": True, "duration_seconds": 180.0},
        {"success": True, "duration_seconds": 150.0},
    ]


@pytest.fixture
def past_experiences_mixed() -> List[Dict[str, Any]]:
    """成功と失敗が混在する過去経験"""
    return [
        {"success": True, "duration_seconds": 100.0},
        {"success": False, "duration_seconds": 200.0},
        {"success": True, "duration_seconds": 150.0},
        {"success": False, "duration_seconds": 50.0},
    ]


# === FeatureExtractor 初期化テスト ===


class TestFeatureExtractorInit:
    """FeatureExtractor 初期化のテスト"""

    def test_init_with_config(self, config: Phase3Config):
        """config が正しく設定される"""
        # Act
        extractor = FeatureExtractor(config)

        # Assert
        assert extractor.config is config

    def test_init_preserves_config_values(self, config: Phase3Config):
        """config の値が保持される"""
        # Arrange
        config.neural_scorer_threshold = 0.8

        # Act
        extractor = FeatureExtractor(config)

        # Assert
        assert extractor.config.neural_scorer_threshold == 0.8


# === extract_task_features テスト ===


class TestExtractTaskFeatures:
    """extract_task_features() メソッドのテスト"""

    def test_returns_all_expected_features(self, extractor: FeatureExtractor):
        """TASK_FEATURES で定義された全ての特徴量が返される"""
        # Act
        result = extractor.extract_task_features("APIを実装してテストを書く")

        # Assert
        for feature in TASK_FEATURES:
            assert feature in result, f"特徴量 '{feature}' が結果に含まれていません"

    def test_task_length_calculation(self, extractor: FeatureExtractor):
        """task_length が正しく計算される"""
        # Arrange
        task = "これは20文字のタスク説明です"
        expected_length = float(len(task))

        # Act
        result = extractor.extract_task_features(task)

        # Assert
        assert result["task_length"] == expected_length

    def test_has_code_keywords_true(self, extractor: FeatureExtractor):
        """コードキーワードを含む場合、has_code_keywords が 1.0"""
        # Arrange
        task = "関数を実装してクラスを追加する"

        # Act
        result = extractor.extract_task_features(task)

        # Assert
        assert result["has_code_keywords"] == 1.0

    def test_has_code_keywords_false(self, extractor: FeatureExtractor):
        """コードキーワードを含まない場合、has_code_keywords が 0.0"""
        # Arrange
        task = "ミーティングの予定を確認する"

        # Act
        result = extractor.extract_task_features(task)

        # Assert
        assert result["has_code_keywords"] == 0.0

    def test_has_research_keywords_true(self, extractor: FeatureExtractor):
        """調査キーワードを含む場合、has_research_keywords が 1.0"""
        # Arrange
        task = "技術選定のための調査と分析を行う"

        # Act
        result = extractor.extract_task_features(task)

        # Assert
        assert result["has_research_keywords"] == 1.0

    def test_has_research_keywords_false(self, extractor: FeatureExtractor):
        """調査キーワードを含まない場合、has_research_keywords が 0.0"""
        # Arrange
        task = "コードを書いてプッシュする"

        # Act
        result = extractor.extract_task_features(task)

        # Assert
        assert result["has_research_keywords"] == 0.0

    def test_has_test_keywords_true(self, extractor: FeatureExtractor):
        """テストキーワードを含む場合、has_test_keywords が 1.0"""
        # Arrange
        task = "ユニットテストを作成して動作確認する"

        # Act
        result = extractor.extract_task_features(task)

        # Assert
        assert result["has_test_keywords"] == 1.0

    def test_has_test_keywords_false(self, extractor: FeatureExtractor):
        """テストキーワードを含まない場合、has_test_keywords が 0.0"""
        # Arrange
        task = "データベースを更新する"

        # Act
        result = extractor.extract_task_features(task)

        # Assert
        assert result["has_test_keywords"] == 0.0

    def test_complexity_score_range(self, extractor: FeatureExtractor):
        """complexity_score が 0.0-1.0 の範囲内"""
        # Arrange
        task = "複雑なタスク：APIを実装し、テストを書き、調査を行い、ドキュメントを更新する"

        # Act
        result = extractor.extract_task_features(task)

        # Assert
        assert 0.0 <= result["complexity_score"] <= 1.0

    def test_item_count_with_bullet_list(self, extractor: FeatureExtractor):
        """箇条書きリストの論点数が正しくカウントされる"""
        # Arrange
        task = """以下のタスクを実行:
- APIエンドポイントを作成
- データベーススキーマを更新
- テストを追加"""

        # Act
        result = extractor.extract_task_features(task)

        # Assert
        assert result["item_count"] == 3.0

    def test_item_count_with_numbered_list(self, extractor: FeatureExtractor):
        """番号付きリストの論点数が正しくカウントされる"""
        # Arrange
        task = """手順:
1. 設計を確認
2. 実装を開始
3. コードレビュー
4. マージ"""

        # Act
        result = extractor.extract_task_features(task)

        # Assert
        assert result["item_count"] == 4.0


class TestExtractTaskFeaturesEmpty:
    """extract_task_features() 空入力のテスト"""

    def test_empty_string(self, extractor: FeatureExtractor):
        """空文字列で全特徴量が 0.0"""
        # Act
        result = extractor.extract_task_features("")

        # Assert
        assert result["task_length"] == 0.0
        assert result["item_count"] == 0.0
        assert result["has_code_keywords"] == 0.0
        assert result["has_research_keywords"] == 0.0
        assert result["has_test_keywords"] == 0.0
        assert result["complexity_score"] == 0.0

    def test_whitespace_only(self, extractor: FeatureExtractor):
        """空白のみの入力"""
        # Act
        result = extractor.extract_task_features("   ")

        # Assert
        # 空白のみでも task_length はカウントされる
        assert result["task_length"] == 3.0
        # 意味のある文字がないのでキーワードは検出されない
        assert result["has_code_keywords"] == 0.0


class TestExtractTaskFeaturesKeywordCaseInsensitive:
    """キーワードの大文字小文字無視テスト"""

    def test_code_keywords_case_insensitive(self, extractor: FeatureExtractor):
        """コードキーワードは大文字小文字を無視"""
        # Arrange
        tasks = [
            "IMPORT モジュールを追加",
            "Import文を修正",
            "import を更新",
        ]

        # Act & Assert
        for task in tasks:
            result = extractor.extract_task_features(task)
            assert result["has_code_keywords"] == 1.0, f"タスク '{task}' でキーワード検出失敗"

    def test_english_keywords(self, extractor: FeatureExtractor):
        """英語キーワードの検出"""
        # Arrange
        task = "Create a new class and define the interface"

        # Act
        result = extractor.extract_task_features(task)

        # Assert
        assert result["has_code_keywords"] == 1.0


# === extract_agent_features テスト ===


class TestExtractAgentFeatures:
    """extract_agent_features() メソッドのテスト"""

    def test_returns_all_expected_features(
        self,
        extractor: FeatureExtractor,
        sample_agent: AgentDefinition,
        past_experiences_success: List[Dict[str, Any]],
    ):
        """AGENT_FEATURES で定義された全ての特徴量が返される"""
        # Act
        result = extractor.extract_agent_features(sample_agent, past_experiences_success)

        # Assert
        for feature in AGENT_FEATURES:
            assert feature in result, f"特徴量 '{feature}' が結果に含まれていません"

    def test_capability_count(
        self,
        extractor: FeatureExtractor,
        sample_agent: AgentDefinition,
    ):
        """capability_count が正しく計算される"""
        # Act
        result = extractor.extract_agent_features(sample_agent)

        # Assert
        assert result["capability_count"] == 3.0  # testing, debugging, analysis

    def test_perspective_count(
        self,
        extractor: FeatureExtractor,
        sample_agent: AgentDefinition,
    ):
        """perspective_count が正しく計算される"""
        # Act
        result = extractor.extract_agent_features(sample_agent)

        # Assert
        assert result["perspective_count"] == 5.0

    def test_without_past_experiences(
        self,
        extractor: FeatureExtractor,
        sample_agent: AgentDefinition,
    ):
        """past_experiences=None の場合のデフォルト値"""
        # Act
        result = extractor.extract_agent_features(sample_agent, None)

        # Assert
        assert result["capability_count"] == 3.0
        assert result["perspective_count"] == 5.0
        assert result["past_success_rate"] == 0.5  # デフォルト（中立）
        assert result["recent_task_count"] == 0.0
        assert result["avg_task_duration"] == 0.0

    def test_past_success_rate_all_success(
        self,
        extractor: FeatureExtractor,
        sample_agent: AgentDefinition,
        past_experiences_success: List[Dict[str, Any]],
    ):
        """全て成功の場合、past_success_rate が 1.0"""
        # Act
        result = extractor.extract_agent_features(sample_agent, past_experiences_success)

        # Assert
        assert result["past_success_rate"] == 1.0

    def test_past_success_rate_mixed(
        self,
        extractor: FeatureExtractor,
        sample_agent: AgentDefinition,
        past_experiences_mixed: List[Dict[str, Any]],
    ):
        """成功/失敗が混在する場合の past_success_rate"""
        # Act
        result = extractor.extract_agent_features(sample_agent, past_experiences_mixed)

        # Assert
        # 2成功 / 4合計 = 0.5
        assert result["past_success_rate"] == 0.5

    def test_recent_task_count(
        self,
        extractor: FeatureExtractor,
        sample_agent: AgentDefinition,
        past_experiences_success: List[Dict[str, Any]],
    ):
        """recent_task_count が過去経験の件数と一致"""
        # Act
        result = extractor.extract_agent_features(sample_agent, past_experiences_success)

        # Assert
        assert result["recent_task_count"] == 3.0

    def test_avg_task_duration(
        self,
        extractor: FeatureExtractor,
        sample_agent: AgentDefinition,
        past_experiences_success: List[Dict[str, Any]],
    ):
        """avg_task_duration が正しく計算される"""
        # Act
        result = extractor.extract_agent_features(sample_agent, past_experiences_success)

        # Assert
        # (120 + 180 + 150) / 3 = 150.0
        assert result["avg_task_duration"] == 150.0

    def test_avg_task_duration_missing_field(
        self,
        extractor: FeatureExtractor,
        sample_agent: AgentDefinition,
    ):
        """duration_seconds がない経験は平均計算から除外"""
        # Arrange
        experiences = [
            {"success": True, "duration_seconds": 100.0},
            {"success": True},  # duration_seconds なし
            {"success": True, "duration_seconds": 200.0},
        ]

        # Act
        result = extractor.extract_agent_features(sample_agent, experiences)

        # Assert
        # (100 + 200) / 2 = 150.0
        assert result["avg_task_duration"] == 150.0


class TestExtractAgentFeaturesMinimal:
    """最小データのエージェントに対するテスト"""

    def test_empty_capabilities(
        self,
        extractor: FeatureExtractor,
        agent_with_minimal_data: AgentDefinition,
    ):
        """capabilities が空の場合"""
        # Act
        result = extractor.extract_agent_features(agent_with_minimal_data)

        # Assert
        assert result["capability_count"] == 0.0

    def test_empty_perspectives(
        self,
        extractor: FeatureExtractor,
        agent_with_minimal_data: AgentDefinition,
    ):
        """perspectives が空の場合"""
        # Act
        result = extractor.extract_agent_features(agent_with_minimal_data)

        # Assert
        assert result["perspective_count"] == 0.0

    def test_empty_past_experiences_list(
        self,
        extractor: FeatureExtractor,
        sample_agent: AgentDefinition,
    ):
        """空のリストの場合はデフォルト値"""
        # Act
        result = extractor.extract_agent_features(sample_agent, [])

        # Assert - 空リストは None と同様にデフォルト値
        # 注: 実装では空リストはFalsy判定でデフォルト値になる
        assert result["past_success_rate"] == 0.5


# === 内部メソッド _count_items テスト ===


class TestCountItems:
    """_count_items() メソッドのテスト"""

    def test_bullet_list_hyphen(self, extractor: FeatureExtractor):
        """ハイフン箇条書き"""
        # Arrange
        text = """タスク一覧:
- 項目1
- 項目2
- 項目3"""

        # Act
        result = extractor._count_items(text)

        # Assert
        assert result == 3.0

    def test_bullet_list_asterisk(self, extractor: FeatureExtractor):
        """アスタリスク箇条書き"""
        # Arrange
        text = """* 項目A
* 項目B"""

        # Act
        result = extractor._count_items(text)

        # Assert
        assert result == 2.0

    def test_bullet_list_japanese(self, extractor: FeatureExtractor):
        """日本語中黒箇条書き"""
        # Arrange
        text = """・ 日本語項目1
・ 日本語項目2"""

        # Act
        result = extractor._count_items(text)

        # Assert
        assert result == 2.0

    def test_numbered_list_period(self, extractor: FeatureExtractor):
        """番号付きリスト（ピリオド）"""
        # Arrange
        text = """1. 最初の手順
2. 次の手順
3. 最後の手順"""

        # Act
        result = extractor._count_items(text)

        # Assert
        assert result == 3.0

    def test_numbered_list_parenthesis(self, extractor: FeatureExtractor):
        """番号付きリスト（括弧）"""
        # Arrange
        text = """1) ステップ1
2) ステップ2"""

        # Act
        result = extractor._count_items(text)

        # Assert
        assert result == 2.0

    def test_numbered_list_circled_numbers(self, extractor: FeatureExtractor):
        """丸数字リスト"""
        # Arrange
        text = """① 第一段階
② 第二段階
③ 第三段階"""

        # Act
        result = extractor._count_items(text)

        # Assert
        assert result == 3.0

    def test_no_list_single_line(self, extractor: FeatureExtractor):
        """リストなし（単一行）"""
        # Arrange
        text = "これは箇条書きではない単純な文章です"

        # Act
        result = extractor._count_items(text)

        # Assert
        assert result == 1.0  # 意味のある行として1とカウント

    def test_no_list_multiple_lines(self, extractor: FeatureExtractor):
        """リストなし（複数行）"""
        # Arrange
        text = """最初の段落です。十分な長さがあります。
二番目の段落です。これも十分な長さがあります。
三番目の段落です。これも十分な長さがあります。"""

        # Act
        result = extractor._count_items(text)

        # Assert
        assert result == 3.0  # 意味のある行数

    def test_mixed_list_types(self, extractor: FeatureExtractor):
        """異なるリスト形式の混在"""
        # Arrange
        text = """タスク:
- 箇条書き項目
1. 番号付き項目
* アスタリスク項目"""

        # Act
        result = extractor._count_items(text)

        # Assert
        assert result == 3.0


# === 内部メソッド _has_keywords テスト ===


class TestHasKeywords:
    """_has_keywords() メソッドのテスト"""

    def test_keyword_found(self, extractor: FeatureExtractor):
        """キーワードが見つかる場合"""
        # Act
        result = extractor._has_keywords("関数を実装する", CODE_KEYWORDS)

        # Assert
        assert result == 1.0

    def test_keyword_not_found(self, extractor: FeatureExtractor):
        """キーワードが見つからない場合"""
        # Act
        result = extractor._has_keywords("今日の天気は晴れです", CODE_KEYWORDS)

        # Assert
        assert result == 0.0

    def test_keyword_case_insensitive_lowercase(self, extractor: FeatureExtractor):
        """小文字での検索"""
        # Act
        result = extractor._has_keywords("import文を追加", CODE_KEYWORDS)

        # Assert
        assert result == 1.0

    def test_keyword_case_insensitive_uppercase(self, extractor: FeatureExtractor):
        """大文字での検索"""
        # Act
        result = extractor._has_keywords("IMPORT文を追加", CODE_KEYWORDS)

        # Assert
        assert result == 1.0

    def test_keyword_partial_match(self, extractor: FeatureExtractor):
        """部分一致"""
        # Arrange - "実装" は CODE_KEYWORDS に含まれる
        text = "機能実装完了"

        # Act
        result = extractor._has_keywords(text, CODE_KEYWORDS)

        # Assert
        assert result == 1.0

    def test_empty_text(self, extractor: FeatureExtractor):
        """空テキスト"""
        # Act
        result = extractor._has_keywords("", CODE_KEYWORDS)

        # Assert
        assert result == 0.0

    def test_empty_keywords(self, extractor: FeatureExtractor):
        """空キーワードリスト"""
        # Act
        result = extractor._has_keywords("任意のテキスト", [])

        # Assert
        assert result == 0.0


# === 内部メソッド _calculate_complexity テスト ===


class TestCalculateComplexity:
    """_calculate_complexity() メソッドのテスト"""

    def test_complexity_range(self, extractor: FeatureExtractor):
        """複雑度が 0.0-1.0 の範囲内"""
        # Arrange
        texts = [
            "短い",
            "これは少し長いテキストです",
            "非常に長く複雑なタスク：" + "あ" * 500,
        ]

        # Act & Assert
        for text in texts:
            result = extractor._calculate_complexity(text)
            assert 0.0 <= result <= 1.0, f"テキスト '{text[:20]}...' の複雑度が範囲外: {result}"

    def test_complexity_increases_with_length(self, extractor: FeatureExtractor):
        """テキストが長いほど複雑度が高い"""
        # Arrange
        short_text = "タスク"
        long_text = "これは非常に長いタスクの説明です" * 10

        # Act
        short_complexity = extractor._calculate_complexity(short_text)
        long_complexity = extractor._calculate_complexity(long_text)

        # Assert
        assert long_complexity > short_complexity

    def test_complexity_increases_with_items(self, extractor: FeatureExtractor):
        """論点数が多いほど複雑度が高い"""
        # Arrange
        single_item = "単一のタスク説明"
        multi_items = """複数のタスク:
- 項目1
- 項目2
- 項目3
- 項目4
- 項目5"""

        # Act
        single_complexity = extractor._calculate_complexity(single_item)
        multi_complexity = extractor._calculate_complexity(multi_items)

        # Assert
        assert multi_complexity > single_complexity

    def test_complexity_increases_with_keyword_diversity(
        self, extractor: FeatureExtractor
    ):
        """異なるキーワードタイプが多いほど複雑度が高い"""
        # Arrange
        single_type = "関数を実装してコードを書く"  # コードキーワードのみ
        multi_types = "関数を実装して調査を行い、テストを書く"  # 3タイプ全て

        # Act
        single_complexity = extractor._calculate_complexity(single_type)
        multi_complexity = extractor._calculate_complexity(multi_types)

        # Assert
        assert multi_complexity > single_complexity

    def test_complexity_with_conditional_keywords(self, extractor: FeatureExtractor):
        """条件分岐キーワードで複雑度が上昇"""
        # Arrange
        simple = "機能を実装する"
        conditional = "もしエラーの場合はログを出力し、そうでなければ成功を返す"

        # Act
        simple_complexity = extractor._calculate_complexity(simple)
        conditional_complexity = extractor._calculate_complexity(conditional)

        # Assert
        assert conditional_complexity > simple_complexity

    def test_complexity_max_is_one(self, extractor: FeatureExtractor):
        """最大複雑度は 1.0 を超えない"""
        # Arrange - 全ての複雑度要素を最大化
        text = """
非常に長く複雑なタスク:
- 実装を行う
- 調査を実施
- テストを作成
- ドキュメントを更新
- レビューを依頼
- マージする
- デプロイする
- 監視を設定
- アラートを確認
- 分析を実施
もし問題がある場合は修正する。
または別のアプローチを検討する。
かつパフォーマンスも考慮する。
"""
        text += "追加のテキスト" * 100

        # Act
        result = extractor._calculate_complexity(text)

        # Assert
        assert result <= 1.0


# === 境界値・異常系テスト ===


class TestEdgeCases:
    """境界値・異常系のテスト"""

    def test_very_long_task_summary(self, extractor: FeatureExtractor):
        """非常に長いタスク文字列"""
        # Arrange
        long_task = "タスク説明: " + "これは長いタスクです。" * 1000

        # Act
        result = extractor.extract_task_features(long_task)

        # Assert
        assert result["task_length"] == float(len(long_task))
        assert 0.0 <= result["complexity_score"] <= 1.0

    def test_special_characters(self, extractor: FeatureExtractor):
        """特殊文字を含むタスク"""
        # Arrange
        task = "関数を実装: <script>alert('test')</script> & SQL: DROP TABLE; -- comment"

        # Act
        result = extractor.extract_task_features(task)

        # Assert
        assert result["has_code_keywords"] == 1.0
        assert isinstance(result["task_length"], float)

    def test_unicode_characters(self, extractor: FeatureExtractor):
        """Unicode文字を含むタスク"""
        # Arrange
        task = "APIを実装 🚀 テストを追加 ✅ 調査完了 📊"

        # Act
        result = extractor.extract_task_features(task)

        # Assert
        assert result["has_code_keywords"] == 1.0
        assert result["has_test_keywords"] == 1.0

    def test_newlines_only(self, extractor: FeatureExtractor):
        """改行のみの入力"""
        # Arrange
        task = "\n\n\n"

        # Act
        result = extractor.extract_task_features(task)

        # Assert
        assert result["item_count"] == 1.0  # 最小値

    def test_agent_with_none_capabilities(self, extractor: FeatureExtractor):
        """capabilities が None のエージェント（from_row でのパース時に発生しうる）"""
        # Arrange
        agent = AgentDefinition(
            agent_id="none_caps",
            name="テスト",
            role="テスト",
            perspectives=["観点1"],
            system_prompt="",
        )
        # capabilities は空リストがデフォルト

        # Act
        result = extractor.extract_agent_features(agent)

        # Assert
        assert result["capability_count"] == 0.0

    def test_past_experiences_with_no_success_field(
        self,
        extractor: FeatureExtractor,
        sample_agent: AgentDefinition,
    ):
        """success フィールドがない過去経験"""
        # Arrange
        experiences = [
            {"duration_seconds": 100.0},  # success なし
            {"success": True, "duration_seconds": 200.0},
        ]

        # Act
        result = extractor.extract_agent_features(sample_agent, experiences)

        # Assert
        # success=False として扱われる（get で False がデフォルト）
        assert result["past_success_rate"] == 0.5  # 1/2

    def test_past_experiences_all_missing_duration(
        self,
        extractor: FeatureExtractor,
        sample_agent: AgentDefinition,
    ):
        """全ての経験で duration_seconds がない"""
        # Arrange
        experiences = [
            {"success": True},
            {"success": True},
        ]

        # Act
        result = extractor.extract_agent_features(sample_agent, experiences)

        # Assert
        assert result["avg_task_duration"] == 0.0


class TestReproducibility:
    """再現性のテスト"""

    def test_same_input_same_output_task_features(self, extractor: FeatureExtractor):
        """同一入力に対して常に同じ結果（タスク特徴量）"""
        # Arrange
        task = "APIを実装してテストを書く"

        # Act
        result1 = extractor.extract_task_features(task)
        result2 = extractor.extract_task_features(task)
        result3 = extractor.extract_task_features(task)

        # Assert
        assert result1 == result2
        assert result2 == result3

    def test_same_input_same_output_agent_features(
        self,
        extractor: FeatureExtractor,
        sample_agent: AgentDefinition,
        past_experiences_success: List[Dict[str, Any]],
    ):
        """同一入力に対して常に同じ結果（エージェント特徴量）"""
        # Act
        result1 = extractor.extract_agent_features(sample_agent, past_experiences_success)
        result2 = extractor.extract_agent_features(sample_agent, past_experiences_success)
        result3 = extractor.extract_agent_features(sample_agent, past_experiences_success)

        # Assert
        assert result1 == result2
        assert result2 == result3


class TestFeatureTypes:
    """特徴量の型チェックテスト"""

    def test_task_features_are_floats(self, extractor: FeatureExtractor):
        """タスク特徴量は全て float"""
        # Act
        result = extractor.extract_task_features("テストタスク")

        # Assert
        for key, value in result.items():
            assert isinstance(value, float), f"特徴量 '{key}' が float ではありません: {type(value)}"

    def test_agent_features_are_floats(
        self,
        extractor: FeatureExtractor,
        sample_agent: AgentDefinition,
    ):
        """エージェント特徴量は全て float"""
        # Act
        result = extractor.extract_agent_features(sample_agent)

        # Assert
        for key, value in result.items():
            assert isinstance(value, float), f"特徴量 '{key}' が float ではありません: {type(value)}"


class TestKeywordLists:
    """キーワードリストのテスト"""

    def test_code_keywords_not_empty(self):
        """CODE_KEYWORDS が空でない"""
        assert len(CODE_KEYWORDS) > 0

    def test_research_keywords_not_empty(self):
        """RESEARCH_KEYWORDS が空でない"""
        assert len(RESEARCH_KEYWORDS) > 0

    def test_test_keywords_not_empty(self):
        """TEST_KEYWORDS が空でない"""
        assert len(TEST_KEYWORDS) > 0

    def test_all_keywords_are_strings(self):
        """全キーワードが文字列"""
        for kw in CODE_KEYWORDS:
            assert isinstance(kw, str)
        for kw in RESEARCH_KEYWORDS:
            assert isinstance(kw, str)
        for kw in TEST_KEYWORDS:
            assert isinstance(kw, str)
