"""概要生成モジュール

仕様書参照: docs/phase2-implementation-spec.ja.md セクション5.1
"""

from src.config.phase2_config import Phase2Config


class Summarizer:
    """概要生成クラス

    入力が大きすぎる場合に概要を生成する。

    Phase 2 MVP: 簡易実装（LLM呼び出しなし）
    - 先頭から summary_max_tokens 相当の文字数を切り出し
    - 将来的にはHaikuなどの軽量LLMで本格的な要約を生成
    """

    # 日本語テキストの概算: 1トークン ≒ 0.5〜1文字
    # 安全のため 1トークン = 1文字 で計算
    CHARS_PER_TOKEN = 1

    def __init__(self, config: Phase2Config):
        self.config = config
        self._max_chars = config.summary_max_tokens * self.CHARS_PER_TOKEN

    def summarize(self, user_input: str) -> str:
        """入力テキストの概要を生成

        Phase 2 MVP では簡易実装として先頭を切り出し。
        将来的にはLLMによる本格的な要約を実装予定。

        Args:
            user_input: ユーザー入力テキスト

        Returns:
            概要テキスト（summary_max_tokens 以内）
        """
        if not user_input:
            return ""

        text = user_input.strip()

        # 入力がmax_chars以内なら、そのまま返す
        if len(text) <= self._max_chars:
            return text

        # 先頭からmax_chars分を切り出し
        truncated = text[:self._max_chars]

        # 文の途中で切れないよう、最後の句点・改行で区切る
        truncated = self._truncate_at_sentence_boundary(truncated)

        # 切り詰めたことを示すサフィックスを追加
        suffix = "\n\n[...入力が長いため省略されました。詳細は detail_refs を参照してください]"

        return truncated + suffix

    def _truncate_at_sentence_boundary(self, text: str) -> str:
        """文の境界で切り詰める

        句点（。）、改行、またはその他の区切りで終わるようにする。
        """
        # 句点、疑問符、感嘆符、改行を区切りとする
        sentence_endings = ['。', '！', '？', '\n']

        # 最後の区切り位置を探す
        last_boundary = -1
        for ending in sentence_endings:
            pos = text.rfind(ending)
            if pos > last_boundary:
                last_boundary = pos

        # 区切りが見つかった場合、そこまでを返す
        if last_boundary > len(text) // 2:  # 半分以上の位置にある場合のみ
            return text[:last_boundary + 1]

        # 見つからない場合、読点（、）や空白で区切る
        fallback_endings = ['、', '，', ' ', '　']
        for ending in fallback_endings:
            pos = text.rfind(ending)
            if pos > len(text) * 0.8:  # 80%以上の位置にある場合
                return text[:pos + 1]

        # それでも見つからない場合は、そのまま返す
        return text
