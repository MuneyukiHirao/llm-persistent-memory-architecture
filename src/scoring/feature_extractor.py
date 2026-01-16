# 特徴量抽出
# タスクとエージェントから特徴量を抽出してニューラルスコアラーに入力
# 実装仕様: docs/phase3-implementation-spec.ja.md セクション5.1
"""
特徴量抽出モジュール

タスク概要とエージェント定義から数値特徴量を抽出し、
ニューラルスコアラーの入力として使用する。

設計方針（タスク実行フローエージェント観点）:
- API設計: シンプルな辞書形式で特徴量を返却、呼び出し側の負担を最小化
- フロー整合性: NeuralScorerから呼ばれる前提で特徴量名を統一
- エラー処理: 欠損データに対してはデフォルト値を使用
- 拡張性: 新しい特徴量の追加が容易な構造
- テスト容易性: 外部依存なし、純粋関数として実装
"""

import re
from typing import Any, Dict, List, Optional

from src.agents.agent_registry import AgentDefinition
from src.config.phase3_config import Phase3Config


# === キーワード辞書 ===
CODE_KEYWORDS: List[str] = [
    "実装",
    "コード",
    "関数",
    "クラス",
    "バグ",
    "エラー",
    "修正",
    "追加",
    "変更",
    "リファクタリング",
    "メソッド",
    "モジュール",
    "パッケージ",
    "インポート",
    "定義",
    "宣言",
    "変数",
    "型",
    "interface",
    "class",
    "function",
    "def",
    "import",
    "module",
    "API",
    "エンドポイント",
    "スキーマ",
    "マイグレーション",
]

RESEARCH_KEYWORDS: List[str] = [
    "調査",
    "分析",
    "検討",
    "比較",
    "評価",
    "調べる",
    "確認",
    "レビュー",
    "リサーチ",
    "ヒアリング",
    "要件",
    "仕様",
    "設計",
    "アーキテクチャ",
    "選定",
    "検証",
    "PoC",
    "プロトタイプ",
    "ベンチマーク",
    "パフォーマンス",
]

TEST_KEYWORDS: List[str] = [
    "テスト",
    "検証",
    "確認",
    "動作確認",
    "単体テスト",
    "結合テスト",
    "E2E",
    "ユニットテスト",
    "pytest",
    "unittest",
    "モック",
    "スタブ",
    "カバレッジ",
    "アサーション",
    "期待値",
    "正常系",
    "異常系",
    "境界値",
    "エッジケース",
]


class FeatureExtractor:
    """特徴量抽出クラス

    タスク概要とエージェント定義から、ニューラルスコアラーの入力となる
    数値特徴量を抽出する。

    使用例:
        config = Phase3Config()
        extractor = FeatureExtractor(config)

        # タスク特徴量の抽出
        task_features = extractor.extract_task_features(
            "APIエンドポイントを実装してテストを書く"
        )
        # => {"task_length": 22, "item_count": 1, "has_code_keywords": 1.0, ...}

        # エージェント特徴量の抽出
        agent_features = extractor.extract_agent_features(
            agent_definition,
            past_experiences
        )
        # => {"capability_count": 5, "perspective_count": 5, ...}

    Attributes:
        config: Phase3Config インスタンス
    """

    def __init__(self, config: Phase3Config):
        """FeatureExtractor を初期化

        Args:
            config: Phase3Config インスタンス
        """
        self.config = config

    def extract_task_features(self, task_summary: str) -> Dict[str, float]:
        """タスク文字列から特徴量を抽出

        Args:
            task_summary: タスク概要の文字列

        Returns:
            特徴量の辞書
            - task_length: タスク文字数
            - item_count: 論点数（箇条書き等）
            - has_code_keywords: コード関連キーワードの有無（0.0 or 1.0）
            - has_research_keywords: 調査関連キーワードの有無（0.0 or 1.0）
            - has_test_keywords: テスト関連キーワードの有無（0.0 or 1.0）
            - complexity_score: 複雑度スコア（0.0-1.0）
        """
        if not task_summary:
            return self._empty_task_features()

        # タスク文字数
        task_length = float(len(task_summary))

        # 論点数（箇条書き、番号付きリスト、改行区切りをカウント）
        item_count = self._count_items(task_summary)

        # キーワード検出
        has_code_keywords = self._has_keywords(task_summary, CODE_KEYWORDS)
        has_research_keywords = self._has_keywords(task_summary, RESEARCH_KEYWORDS)
        has_test_keywords = self._has_keywords(task_summary, TEST_KEYWORDS)

        # 複雑度スコア
        complexity_score = self._calculate_complexity(task_summary)

        return {
            "task_length": task_length,
            "item_count": item_count,
            "has_code_keywords": has_code_keywords,
            "has_research_keywords": has_research_keywords,
            "has_test_keywords": has_test_keywords,
            "complexity_score": complexity_score,
        }

    def extract_agent_features(
        self,
        agent: AgentDefinition,
        past_experiences: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, float]:
        """エージェント定義と過去の経験から特徴量を抽出

        Args:
            agent: AgentDefinition インスタンス
            past_experiences: 過去のタスク実行結果のリスト
                各要素は以下のキーを持つ辞書:
                - success: bool (タスク成功フラグ)
                - duration_seconds: float (タスク処理時間)
                - created_at: datetime (実行日時)

        Returns:
            特徴量の辞書
            - capability_count: 能力タグ数
            - perspective_count: 観点数
            - past_success_rate: 過去の成功率（0.0-1.0）
            - recent_task_count: 最近のタスク数
            - avg_task_duration: 平均タスク処理時間（秒）
        """
        # エージェント定義からの特徴量
        capability_count = float(len(agent.capabilities)) if agent.capabilities else 0.0
        perspective_count = float(len(agent.perspectives)) if agent.perspectives else 0.0

        # 過去の経験からの特徴量
        if not past_experiences:
            return {
                "capability_count": capability_count,
                "perspective_count": perspective_count,
                "past_success_rate": 0.5,  # デフォルト値（中立）
                "recent_task_count": 0.0,
                "avg_task_duration": 0.0,
            }

        # 成功率
        success_count = sum(1 for exp in past_experiences if exp.get("success", False))
        past_success_rate = success_count / len(past_experiences)

        # 最近のタスク数（past_experiencesに含まれるもの全てを「最近」とみなす）
        recent_task_count = float(len(past_experiences))

        # 平均タスク処理時間
        durations = [
            exp.get("duration_seconds", 0.0)
            for exp in past_experiences
            if "duration_seconds" in exp
        ]
        avg_task_duration = sum(durations) / len(durations) if durations else 0.0

        return {
            "capability_count": capability_count,
            "perspective_count": perspective_count,
            "past_success_rate": past_success_rate,
            "recent_task_count": recent_task_count,
            "avg_task_duration": avg_task_duration,
        }

    def _empty_task_features(self) -> Dict[str, float]:
        """空のタスク特徴量を返す"""
        return {
            "task_length": 0.0,
            "item_count": 0.0,
            "has_code_keywords": 0.0,
            "has_research_keywords": 0.0,
            "has_test_keywords": 0.0,
            "complexity_score": 0.0,
        }

    def _count_items(self, text: str) -> float:
        """テキスト内の論点数をカウント

        箇条書き（-、*、・）、番号付きリスト（1.、①等）、
        改行区切りの項目をカウントする。

        Args:
            text: 分析対象のテキスト

        Returns:
            論点数（float）
        """
        # 箇条書きパターン
        bullet_pattern = r"^[\s]*[-*・•]\s+"
        # 番号付きリストパターン
        numbered_pattern = r"^[\s]*(\d+[.）\)]|[①②③④⑤⑥⑦⑧⑨⑩])\s+"

        lines = text.split("\n")
        item_count = 0

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if re.match(bullet_pattern, line) or re.match(numbered_pattern, line):
                item_count += 1

        # 箇条書きがない場合は、改行で区切られた意味のある行をカウント
        if item_count == 0:
            meaningful_lines = [
                line for line in lines if line.strip() and len(line.strip()) > 5
            ]
            item_count = max(1, len(meaningful_lines))

        return float(item_count)

    def _has_keywords(self, text: str, keywords: List[str]) -> float:
        """テキストに指定キーワードが含まれるかチェック

        Args:
            text: 分析対象のテキスト
            keywords: 検索するキーワードのリスト

        Returns:
            1.0（含む）または 0.0（含まない）
        """
        text_lower = text.lower()
        for keyword in keywords:
            if keyword.lower() in text_lower:
                return 1.0
        return 0.0

    def _calculate_complexity(self, text: str) -> float:
        """タスクの複雑度スコアを計算

        以下の要素を考慮して0.0-1.0のスコアを算出:
        - 文字数（長いほど複雑）
        - 論点数（多いほど複雑）
        - 技術キーワードの多様性（多いほど複雑）
        - 条件分岐を示す表現（「場合」「もし」等）

        Args:
            text: 分析対象のテキスト

        Returns:
            複雑度スコア（0.0-1.0）
        """
        score = 0.0
        max_score = 4.0

        # 1. 文字数による複雑度（最大1.0）
        length_score = min(len(text) / 500.0, 1.0)
        score += length_score

        # 2. 論点数による複雑度（最大1.0）
        item_count = self._count_items(text)
        item_score = min(item_count / 10.0, 1.0)
        score += item_score

        # 3. 技術キーワードの多様性（最大1.0）
        keyword_types = 0
        if self._has_keywords(text, CODE_KEYWORDS) > 0:
            keyword_types += 1
        if self._has_keywords(text, RESEARCH_KEYWORDS) > 0:
            keyword_types += 1
        if self._has_keywords(text, TEST_KEYWORDS) > 0:
            keyword_types += 1
        diversity_score = keyword_types / 3.0
        score += diversity_score

        # 4. 条件分岐表現による複雑度（最大1.0）
        conditional_keywords = [
            "場合",
            "もし",
            "ただし",
            "ただ",
            "または",
            "かつ",
            "および",
            "if",
            "else",
            "when",
            "unless",
            "条件",
            "依存",
        ]
        conditional_count = sum(
            1 for kw in conditional_keywords if kw.lower() in text.lower()
        )
        conditional_score = min(conditional_count / 5.0, 1.0)
        score += conditional_score

        return score / max_score
