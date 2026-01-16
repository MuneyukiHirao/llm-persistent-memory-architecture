# 評価フロー（フィードバック検出）
# 実装仕様: docs/phase2-implementation-spec.ja.md セクション5.4
"""
評価フローモジュール

ユーザーフィードバックを観察し、ルーティング結果を評価する。

Phase 2 MVP:
- キーワードベースの明示的フィードバック検出
- パターン分析による暗黙的フィードバック検出

フィードバックタイプ:
- positive: 成功（ありがとう、良い、完璧、OK、了解）
- neutral: 特に問題なし
- negative: 失敗（やり直し、違う、ダメ、修正して）
- redo_requested: やり直し要求（もう一度、再度、別のエージェント）
- partial_failure: 部分失敗（修正要求あり）

暗黙的フィードバック判定基準:
| シグナル | 解釈 | feedback_type |
|---------|------|---------------|
| 結果をそのまま使用 | 成功 | positive |
| 軽微な修正後に使用 | 部分成功 | neutral |
| 大幅に修正して使用 | 部分失敗 | partial_failure |
| やり直しを依頼 | 失敗 | redo_requested |
| 別のエージェントに再依頼 | ルーティング誤り | negative |

設計方針（タスク実行フローエージェント観点）:
- API設計: シンプルなインターフェース、evaluate() のみで完結
- フロー整合性: 明示的→暗黙的の順で判定、確信度を付与
- エラー処理: 検出失敗時は neutral を返しフロー継続
- 拡張性: LLMベース判定への移行を考慮した構造
- テスト容易性: 各検出メソッドを独立してテスト可能
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple

from src.config.phase2_config import Phase2Config, FEEDBACK_SIGNALS


@dataclass
class FeedbackResult:
    """フィードバック評価結果

    Attributes:
        feedback_type: フィードバックタイプ
            - positive: 成功
            - neutral: 特に問題なし
            - negative: 失敗
            - redo_requested: やり直し要求
            - partial_failure: 部分失敗
        confidence: 判定の確信度 (0.0-1.0)
        detected_signals: 検出されたシグナル（キーワードなど）
        raw_response: 元のユーザー応答
    """

    feedback_type: str
    confidence: float
    detected_signals: List[str] = field(default_factory=list)
    raw_response: str = ""

    def to_dict(self) -> dict:
        """辞書に変換"""
        return {
            "feedback_type": self.feedback_type,
            "confidence": self.confidence,
            "detected_signals": self.detected_signals,
            "raw_response": self.raw_response,
        }

    @property
    def is_positive(self) -> bool:
        """成功判定か"""
        return self.feedback_type == "positive"

    @property
    def is_negative(self) -> bool:
        """失敗判定か"""
        return self.feedback_type in ("negative", "redo_requested", "partial_failure")

    @property
    def needs_retry(self) -> bool:
        """リトライが必要か"""
        return self.feedback_type in ("redo_requested", "negative")


class Evaluator:
    """評価フロー

    ユーザー応答からフィードバックタイプを判定する。

    使用例:
        evaluator = Evaluator()

        result = evaluator.evaluate("ありがとう、良さそうです")
        print(f"タイプ: {result.feedback_type}")  # positive
        print(f"確信度: {result.confidence}")     # 0.9

        result = evaluator.evaluate("もう一度やり直して")
        print(f"タイプ: {result.feedback_type}")  # redo_requested

    Attributes:
        config: Phase2Config インスタンス
    """

    # 暗黙的フィードバック検出用のキーワード
    # 明示的シグナルとは別に、応答パターンから推測
    _IMPLICIT_PARTIAL_FAILURE_KEYWORDS: List[str] = [
        "修正", "直して", "変えて", "調整", "ここだけ", "一部",
        "少し", "ちょっと", "微調整", "手直し",
    ]

    _IMPLICIT_NEGATIVE_KEYWORDS: List[str] = [
        "全然", "まったく", "使えない", "役に立たない",
        "意味がない", "期待外れ", "がっかり",
    ]

    # 短い応答の閾値（これ以下は positive の可能性高）
    _SHORT_RESPONSE_THRESHOLD: int = 10

    # 非常に短い応答（確信度高い positive）
    _VERY_SHORT_RESPONSE_THRESHOLD: int = 5

    def __init__(self, config: Optional[Phase2Config] = None):
        """Evaluator を初期化

        Args:
            config: Phase2Config インスタンス（省略時はデフォルト設定）
        """
        self.config = config or Phase2Config()

    def evaluate(self, user_response: str) -> FeedbackResult:
        """ユーザー応答からフィードバックタイプを判定

        判定順序:
        1. 明示的フィードバック検出（キーワードマッチ）
        2. 暗黙的フィードバック検出（パターン分析）
        3. デフォルト（neutral）

        Args:
            user_response: ユーザーの応答テキスト

        Returns:
            FeedbackResult インスタンス
        """
        if not user_response or not user_response.strip():
            # 空応答は neutral（判断不能）
            return FeedbackResult(
                feedback_type="neutral",
                confidence=0.0,
                detected_signals=[],
                raw_response=user_response or "",
            )

        response_stripped = user_response.strip()

        # 1. 明示的フィードバック検出
        explicit_type, explicit_signals = self._detect_explicit_feedback(response_stripped)

        if explicit_type:
            confidence = self._calculate_confidence(
                feedback_type=explicit_type,
                signals=explicit_signals,
                is_explicit=True,
            )
            return FeedbackResult(
                feedback_type=explicit_type,
                confidence=confidence,
                detected_signals=explicit_signals,
                raw_response=user_response,
            )

        # 2. 暗黙的フィードバック検出
        if self.config.implicit_feedback_enabled:
            implicit_type, implicit_signals = self._detect_implicit_feedback(response_stripped)

            if implicit_type:
                confidence = self._calculate_confidence(
                    feedback_type=implicit_type,
                    signals=implicit_signals,
                    is_explicit=False,
                )
                return FeedbackResult(
                    feedback_type=implicit_type,
                    confidence=confidence,
                    detected_signals=implicit_signals,
                    raw_response=user_response,
                )

        # 3. デフォルト（neutral）
        return FeedbackResult(
            feedback_type="neutral",
            confidence=0.3,
            detected_signals=[],
            raw_response=user_response,
        )

    def _detect_explicit_feedback(
        self,
        response: str,
    ) -> Tuple[Optional[str], List[str]]:
        """明示的フィードバックを検出（FEEDBACK_SIGNALS使用）

        FEEDBACK_SIGNALS で定義されたキーワードをマッチング。
        複数のタイプにマッチする場合は、優先度順に判定。

        優先度: redo_requested > negative > positive

        Args:
            response: ユーザー応答（strip済み）

        Returns:
            (フィードバックタイプ, 検出されたシグナルのリスト)
            マッチしない場合は (None, [])
        """
        response_lower = response.lower()
        detected: dict[str, List[str]] = {}

        # 各タイプのキーワードをチェック
        for signal_type, keywords in FEEDBACK_SIGNALS.items():
            matched_keywords = []
            for keyword in keywords:
                if keyword.lower() in response_lower:
                    matched_keywords.append(keyword)

            if matched_keywords:
                detected[signal_type] = matched_keywords

        # 優先度順に返す
        # redo_requested が最優先（明確なやり直し要求）
        if "redo_requested" in detected:
            return "redo_requested", detected["redo_requested"]

        # negative が次（否定的フィードバック）
        if "negative" in detected:
            return "negative", detected["negative"]

        # positive は最後（肯定的フィードバック）
        if "positive" in detected:
            return "positive", detected["positive"]

        return None, []

    def _detect_implicit_feedback(
        self,
        response: str,
    ) -> Tuple[Optional[str], List[str]]:
        """暗黙的フィードバックを検出

        応答パターンから暗黙的なフィードバックを推測する。

        判定基準:
        - 短い応答（10文字以下）→ positive の可能性
        - 修正要求キーワード → partial_failure
        - 否定的キーワード → negative

        Args:
            response: ユーザー応答（strip済み）

        Returns:
            (フィードバックタイプ, 検出理由のリスト)
            判定不能な場合は (None, [])
        """
        response_lower = response.lower()
        signals: List[str] = []

        # 否定的キーワードチェック（最優先）
        for keyword in self._IMPLICIT_NEGATIVE_KEYWORDS:
            if keyword in response_lower:
                signals.append(f"否定的表現: {keyword}")

        if signals:
            return "negative", signals

        # 修正要求キーワードチェック
        for keyword in self._IMPLICIT_PARTIAL_FAILURE_KEYWORDS:
            if keyword in response_lower:
                signals.append(f"修正要求: {keyword}")

        if signals:
            return "partial_failure", signals

        # 短い応答の判定
        response_len = len(response)

        if response_len <= self._VERY_SHORT_RESPONSE_THRESHOLD:
            # 非常に短い応答（「OK」「はい」など）→ positive
            return "positive", [f"非常に短い応答({response_len}文字)"]

        if response_len <= self._SHORT_RESPONSE_THRESHOLD:
            # 短い応答 → positive の可能性
            return "positive", [f"短い応答({response_len}文字)"]

        return None, []

    def _calculate_confidence(
        self,
        feedback_type: str,
        signals: List[str],
        is_explicit: bool,
    ) -> float:
        """判定の確信度を計算

        確信度の計算基準:
        - 明示的フィードバック: 0.7-1.0
        - 暗黙的フィードバック: 0.3-0.6
        - シグナル数が多いほど確信度が高い

        Args:
            feedback_type: フィードバックタイプ
            signals: 検出されたシグナル
            is_explicit: 明示的フィードバックかどうか

        Returns:
            0.0-1.0 の確信度
        """
        if not signals:
            return 0.3

        signal_count = len(signals)

        if is_explicit:
            # 明示的フィードバック: 基本0.7、シグナル数で上昇
            base_confidence = 0.7
            # 1シグナルで+0.1、2シグナルで+0.2、最大+0.3
            signal_bonus = min(signal_count * 0.1, 0.3)
            return min(base_confidence + signal_bonus, 1.0)
        else:
            # 暗黙的フィードバック: 基本0.4、シグナル数で上昇
            base_confidence = 0.4
            # 1シグナルで+0.1、2シグナルで+0.15、最大+0.2
            signal_bonus = min(signal_count * 0.1, 0.2)
            return min(base_confidence + signal_bonus, 0.6)

    def evaluate_with_context(
        self,
        user_response: str,
        previous_result: Optional[str] = None,
        task_type: Optional[str] = None,
    ) -> FeedbackResult:
        """コンテキストを考慮したフィードバック判定（拡張用）

        Phase 2 MVP では基本的な evaluate() と同じ動作。
        将来的に LLM ベースの判定を導入する際の拡張ポイント。

        Args:
            user_response: ユーザーの応答テキスト
            previous_result: 前回のエージェント結果（オプション）
            task_type: タスクタイプ（オプション）

        Returns:
            FeedbackResult インスタンス
        """
        # Phase 2 MVP では基本評価のみ
        return self.evaluate(user_response)
