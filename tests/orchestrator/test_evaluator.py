# Evaluator テスト
# 実装仕様: docs/phase2-implementation-spec.ja.md セクション5.4
"""
Evaluator クラスのユニットテスト

テスト内容:
- 明示的フィードバック検出（キーワードマッチ）
- 暗黙的フィードバック検出（パターン分析）
- 確信度計算
- FeedbackResult のプロパティ
- エッジケース
"""

import pytest

from src.orchestrator.evaluator import Evaluator, FeedbackResult
from src.config.phase2_config import Phase2Config


@pytest.fixture
def evaluator():
    """テスト用のEvaluator"""
    return Evaluator()


@pytest.fixture
def evaluator_no_implicit():
    """暗黙的フィードバック無効のEvaluator"""
    config = Phase2Config()
    config.implicit_feedback_enabled = False
    return Evaluator(config)


class TestFeedbackResult:
    """FeedbackResult のテスト"""

    def test_create_feedback_result(self):
        """FeedbackResult の作成"""
        result = FeedbackResult(
            feedback_type="positive",
            confidence=0.9,
            detected_signals=["ありがとう"],
            raw_response="ありがとう、完璧です",
        )

        assert result.feedback_type == "positive"
        assert result.confidence == 0.9
        assert len(result.detected_signals) == 1
        assert result.raw_response == "ありがとう、完璧です"

    def test_to_dict(self):
        """to_dict メソッドのテスト"""
        result = FeedbackResult(
            feedback_type="negative",
            confidence=0.8,
            detected_signals=["やり直し"],
            raw_response="やり直してください",
        )

        data = result.to_dict()
        assert data["feedback_type"] == "negative"
        assert data["confidence"] == 0.8
        assert data["detected_signals"] == ["やり直し"]
        assert data["raw_response"] == "やり直してください"

    def test_is_positive_property(self):
        """is_positive プロパティのテスト"""
        positive_result = FeedbackResult(feedback_type="positive", confidence=0.9)
        neutral_result = FeedbackResult(feedback_type="neutral", confidence=0.5)

        assert positive_result.is_positive is True
        assert neutral_result.is_positive is False

    def test_is_negative_property(self):
        """is_negative プロパティのテスト"""
        negative_result = FeedbackResult(feedback_type="negative", confidence=0.8)
        redo_result = FeedbackResult(feedback_type="redo_requested", confidence=0.9)
        partial_result = FeedbackResult(feedback_type="partial_failure", confidence=0.7)
        positive_result = FeedbackResult(feedback_type="positive", confidence=0.9)

        assert negative_result.is_negative is True
        assert redo_result.is_negative is True
        assert partial_result.is_negative is True
        assert positive_result.is_negative is False

    def test_needs_retry_property(self):
        """needs_retry プロパティのテスト"""
        redo_result = FeedbackResult(feedback_type="redo_requested", confidence=0.9)
        negative_result = FeedbackResult(feedback_type="negative", confidence=0.8)
        partial_result = FeedbackResult(feedback_type="partial_failure", confidence=0.7)

        assert redo_result.needs_retry is True
        assert negative_result.needs_retry is True
        assert partial_result.needs_retry is False  # partial_failure はリトライ不要


class TestEvaluatorExplicitFeedback:
    """明示的フィードバック検出のテスト"""

    def test_detect_positive_arigatou(self, evaluator):
        """「ありがとう」でpositive判定"""
        result = evaluator.evaluate("ありがとうございます")

        assert result.feedback_type == "positive"
        assert "ありがとう" in result.detected_signals
        assert result.confidence >= 0.7

    def test_detect_positive_ok(self, evaluator):
        """「OK」でpositive判定"""
        result = evaluator.evaluate("OK、それで大丈夫です")

        assert result.feedback_type == "positive"
        assert "OK" in result.detected_signals

    def test_detect_positive_kanpeki(self, evaluator):
        """「完璧」でpositive判定"""
        result = evaluator.evaluate("完璧です！")

        assert result.feedback_type == "positive"
        assert "完璧" in result.detected_signals

    def test_detect_positive_yoi(self, evaluator):
        """「良い」でpositive判定"""
        result = evaluator.evaluate("良い感じです")

        assert result.feedback_type == "positive"
        assert "良い" in result.detected_signals

    def test_detect_positive_ryoukai(self, evaluator):
        """「了解」でpositive判定"""
        result = evaluator.evaluate("了解しました")

        assert result.feedback_type == "positive"
        assert "了解" in result.detected_signals

    def test_detect_negative_yarinaoshi(self, evaluator):
        """「やり直し」でnegative判定"""
        result = evaluator.evaluate("やり直してください")

        assert result.feedback_type == "negative"
        assert "やり直し" in result.detected_signals
        assert result.confidence >= 0.7

    def test_detect_negative_chigau(self, evaluator):
        """「違う」でnegative判定"""
        result = evaluator.evaluate("違う、それではない")

        assert result.feedback_type == "negative"
        assert "違う" in result.detected_signals

    def test_detect_negative_dame(self, evaluator):
        """「ダメ」でnegative判定"""
        result = evaluator.evaluate("ダメです、使えません")

        assert result.feedback_type == "negative"
        assert "ダメ" in result.detected_signals

    def test_detect_negative_shuuseishite(self, evaluator):
        """「修正して」でnegative判定"""
        result = evaluator.evaluate("修正してください")

        assert result.feedback_type == "negative"
        assert "修正して" in result.detected_signals

    def test_detect_redo_requested_mouichido(self, evaluator):
        """「もう一度」でredo_requested判定"""
        result = evaluator.evaluate("もう一度お願いします")

        assert result.feedback_type == "redo_requested"
        assert "もう一度" in result.detected_signals
        assert result.confidence >= 0.7

    def test_detect_redo_requested_saido(self, evaluator):
        """「再度」でredo_requested判定"""
        result = evaluator.evaluate("再度試してください")

        assert result.feedback_type == "redo_requested"
        assert "再度" in result.detected_signals

    def test_detect_redo_requested_betsu_agent(self, evaluator):
        """「別のエージェント」でredo_requested判定"""
        result = evaluator.evaluate("別のエージェントに頼んでください")

        assert result.feedback_type == "redo_requested"
        assert "別のエージェント" in result.detected_signals

    def test_priority_redo_over_negative(self, evaluator):
        """redo_requested が negative より優先"""
        # 「もう一度」と「やり直し」の両方を含む
        result = evaluator.evaluate("やり直しが必要なので、もう一度お願い")

        # redo_requested が優先される
        assert result.feedback_type == "redo_requested"

    def test_priority_negative_over_positive(self, evaluator):
        """negative が positive より優先"""
        # 「ありがとう」と「違う」の両方を含む
        result = evaluator.evaluate("ありがとう、でも違うので修正して")

        # negative が優先される
        assert result.feedback_type == "negative"

    def test_multiple_positive_signals(self, evaluator):
        """複数のpositiveシグナル"""
        result = evaluator.evaluate("ありがとう、完璧で良い感じです")

        assert result.feedback_type == "positive"
        assert len(result.detected_signals) >= 2
        # 複数シグナルで確信度が上がる
        assert result.confidence >= 0.8


class TestEvaluatorImplicitFeedback:
    """暗黙的フィードバック検出のテスト"""

    def test_short_response_positive(self, evaluator):
        """短い応答（10文字以下）でpositive判定"""
        result = evaluator.evaluate("はい")

        assert result.feedback_type == "positive"
        assert result.confidence <= 0.6  # 暗黙的判定は確信度低め

    def test_very_short_response_positive(self, evaluator):
        """非常に短い応答（5文字以下）でpositive判定"""
        result = evaluator.evaluate("うん")

        assert result.feedback_type == "positive"

    def test_implicit_partial_failure_shuusei(self, evaluator):
        """「修正」キーワードでpartial_failure判定"""
        result = evaluator.evaluate("ここだけ少し修正が必要ですね")

        assert result.feedback_type == "partial_failure"
        assert result.confidence <= 0.6

    def test_implicit_partial_failure_naoshite(self, evaluator):
        """「直して」キーワードでpartial_failure判定"""
        result = evaluator.evaluate("この部分を直してもらえますか")

        assert result.feedback_type == "partial_failure"

    def test_implicit_partial_failure_chousei(self, evaluator):
        """「調整」キーワードでpartial_failure判定"""
        result = evaluator.evaluate("パラメータを調整してください")

        assert result.feedback_type == "partial_failure"

    def test_implicit_negative_zenzen(self, evaluator):
        """「全然」キーワードでnegative判定"""
        result = evaluator.evaluate("全然うまくいかないですね")

        assert result.feedback_type == "negative"

    def test_implicit_negative_tsukaenai(self, evaluator):
        """「使えない」キーワードでnegative判定"""
        result = evaluator.evaluate("これは使えないです")

        assert result.feedback_type == "negative"

    def test_implicit_disabled(self, evaluator_no_implicit):
        """暗黙的フィードバック無効時"""
        # 短い応答でも neutral
        result = evaluator_no_implicit.evaluate("うん")

        assert result.feedback_type == "neutral"

    def test_implicit_neutral_long_response(self, evaluator):
        """長い応答で判定不能な場合はneutral"""
        result = evaluator.evaluate(
            "この結果について特に意見はありませんが、参考になりました"
        )

        assert result.feedback_type == "neutral"


class TestEvaluatorConfidence:
    """確信度計算のテスト"""

    def test_explicit_single_signal_confidence(self, evaluator):
        """明示的フィードバック（1シグナル）の確信度"""
        result = evaluator.evaluate("ありがとう")

        assert result.confidence >= 0.7
        assert result.confidence <= 1.0

    def test_explicit_multiple_signals_confidence(self, evaluator):
        """明示的フィードバック（複数シグナル）の確信度"""
        result = evaluator.evaluate("ありがとう、完璧です、良い")

        # 複数シグナルで確信度が上がる
        assert result.confidence >= 0.8

    def test_implicit_confidence_lower(self, evaluator):
        """暗黙的フィードバックの確信度は低め"""
        result = evaluator.evaluate("はい")  # 短い応答

        assert result.confidence >= 0.3
        assert result.confidence <= 0.6

    def test_neutral_confidence(self, evaluator):
        """neutral 判定の確信度"""
        # 注意: 「良い」などのキーワードを含まない長い応答を使用
        result = evaluator.evaluate(
            "この結果について、様々な観点から考えると検討の余地があります"
        )

        assert result.feedback_type == "neutral"
        assert result.confidence <= 0.5


class TestEvaluatorEdgeCases:
    """エッジケースのテスト"""

    def test_empty_response(self, evaluator):
        """空の応答"""
        result = evaluator.evaluate("")

        assert result.feedback_type == "neutral"
        assert result.confidence == 0.0

    def test_none_response(self, evaluator):
        """None の応答（空文字として処理）"""
        result = evaluator.evaluate(None)

        assert result.feedback_type == "neutral"
        assert result.confidence == 0.0

    def test_whitespace_only_response(self, evaluator):
        """空白のみの応答"""
        result = evaluator.evaluate("   \n\t  ")

        assert result.feedback_type == "neutral"
        assert result.confidence == 0.0

    def test_case_insensitive_keywords(self, evaluator):
        """キーワードの大文字小文字"""
        # "ok" は大文字でも小文字でもマッチ
        result_upper = evaluator.evaluate("OK")
        result_lower = evaluator.evaluate("ok")

        assert result_upper.feedback_type == "positive"
        assert result_lower.feedback_type == "positive"

    def test_unicode_response(self, evaluator):
        """Unicode文字を含む応答"""
        result = evaluator.evaluate("👍 ありがとう！")

        assert result.feedback_type == "positive"

    def test_very_long_response(self, evaluator):
        """非常に長い応答"""
        long_text = "これは非常に長い応答です。" * 100
        result = evaluator.evaluate(long_text)

        # エラーにならず処理できる
        assert result is not None
        assert result.feedback_type in ("positive", "neutral", "negative", "redo_requested", "partial_failure")

    def test_mixed_feedback_signals(self, evaluator):
        """混合したフィードバックシグナル"""
        # positive, negative, redo_requested すべて含む
        result = evaluator.evaluate("ありがとう、でも違うので、もう一度お願い")

        # redo_requested が最優先
        assert result.feedback_type == "redo_requested"


class TestEvaluatorWithContext:
    """コンテキスト付き評価のテスト"""

    def test_evaluate_with_context_basic(self, evaluator):
        """evaluate_with_context の基本動作"""
        result = evaluator.evaluate_with_context(
            user_response="ありがとう",
            previous_result="実装結果",
            task_type="implementation",
        )

        # Phase 2 MVP では基本評価と同じ
        assert result.feedback_type == "positive"

    def test_evaluate_with_context_no_context(self, evaluator):
        """コンテキストなしでの evaluate_with_context"""
        result = evaluator.evaluate_with_context(
            user_response="OK",
        )

        assert result.feedback_type == "positive"


class TestEvaluatorIntegration:
    """統合テスト"""

    def test_full_workflow_positive(self, evaluator):
        """正常系: タスク完了 → positive フィードバック"""
        # ユーザーが結果に満足
        result = evaluator.evaluate("完璧です、ありがとうございます")

        assert result.feedback_type == "positive"
        assert result.is_positive is True
        assert result.is_negative is False
        assert result.needs_retry is False

    def test_full_workflow_partial_failure(self, evaluator):
        """部分失敗: 修正要求（暗黙的フィードバック）"""
        # 注意: 「修正して」は明示的negativeシグナル、「直して」は暗黙的partial_failure
        # 暗黙的フィードバックをテストするため「直して」を使用
        result = evaluator.evaluate("概ね大丈夫ですが、この部分だけ直してもらえますか")

        assert result.feedback_type == "partial_failure"
        assert result.is_positive is False
        assert result.is_negative is True
        assert result.needs_retry is False

    def test_full_workflow_redo(self, evaluator):
        """やり直し要求"""
        result = evaluator.evaluate("期待と違うので、もう一度やり直してください")

        assert result.feedback_type == "redo_requested"
        assert result.is_positive is False
        assert result.is_negative is True
        assert result.needs_retry is True

    def test_full_workflow_neutral(self, evaluator):
        """中立: 判断保留"""
        # 注意: 短い応答は暗黙的にpositiveになるため、11文字以上の応答を使用
        result = evaluator.evaluate("しばらく検討してから返答します")

        assert result.feedback_type == "neutral"
        assert result.is_positive is False
        assert result.is_negative is False
        assert result.needs_retry is False
