# 優先度ランキングモジュール（Stage 2: スコア合成）
# 実装仕様: docs/phase1-implementation-spec.ja.md セクション4.3
"""
優先度ランキングモジュール

Stage 1（ベクトル検索）で取得した候補に対して、
類似度・強度・新鮮さを組み合わせた総合スコアで最終ランキングを行う。

設計方針（検索エンジンエージェント観点）:
- 検索精度: 線形スコア合成で多角的な評価を実現
- レスポンス性能: Stage 1 で候補を絞り込んでいるため軽量
- スケーラビリティ: Phase 1 は線形スコアで開始、Phase 2 でニューラル移行検討
- API連携: DB や外部 API への追加呼び出しなし（メモリ内計算）
- フォールバック: 観点指定時に strength_by_perspective がない場合は strength を使用
"""

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from src.config.phase1_config import Phase1Config, config as default_config
from src.models.memory import AgentMemory


logger = logging.getLogger(__name__)


@dataclass
class ScoredMemory:
    """スコア計算済みのメモリ

    Stage 2 で計算された総合スコアを持つメモリのラッパー。
    デバッグや分析のためにスコアの内訳も保持する。
    """

    memory: AgentMemory
    """元の記憶データ"""

    similarity: float
    """Stage 1 からの類似度（0-1）"""

    final_score: float
    """合成スコア（weighted sum）"""

    score_breakdown: Dict[str, float]
    """スコア内訳（デバッグ用）
    例: {"similarity": 0.45, "strength": 0.25, "recency": 0.15, "total": 0.85}
    """

    def __repr__(self) -> str:
        """デバッグ用の文字列表現"""
        return (
            f"ScoredMemory("
            f"memory_id={self.memory.id!r}, "
            f"final_score={self.final_score:.3f}, "
            f"breakdown={self.score_breakdown})"
        )


class MemoryRanker:
    """メモリランキングエンジン（Stage 2: 優先度ランキング）

    Stage 1 のベクトル検索で取得した候補に対して、
    複合スコアを計算して最終的なランキングを行う。

    スコア計算式:
        final_score =
            similarity * weight_similarity +
            normalized_strength * weight_strength +
            recency_score * weight_recency

    使用例:
        config = Phase1Config()
        ranker = MemoryRanker(config)

        # Stage 1 の結果を受け取る
        candidates = vector_search.search_candidates(query, agent_id)

        # Stage 2 でランキング
        results = ranker.rank(candidates, perspective="コスト")

        for scored in results:
            print(f"{scored.memory.content[:50]}... (score: {scored.final_score:.3f})")
    """

    # 時間減衰の基準日数（30日で半減に近い減衰）
    RECENCY_DECAY_DAYS: float = 30.0

    # 強度の正規化上限（これ以上は 1.0 にクリップ）
    MAX_STRENGTH_FOR_NORMALIZE: float = 2.0

    def __init__(self, config: Optional[Phase1Config] = None):
        """ランキングエンジンを初期化

        Args:
            config: Phase 1 設定（省略時はデフォルト設定を使用）
        """
        self.config = config or default_config

        # スコア重みを検証
        weights = self.config.score_weights
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 0.001:
            logger.warning(
                f"スコア重みの合計が 1.0 ではありません: {total_weight:.3f}"
            )

        logger.info(
            f"MemoryRanker 初期化完了: "
            f"weights={weights}, "
            f"top_k={self.config.top_k_results}"
        )

    def rank(
        self,
        candidates: List[Tuple[AgentMemory, float]],
        perspective: Optional[str] = None,
    ) -> List[ScoredMemory]:
        """Stage 2: 候補をスコア合成してランキング

        Args:
            candidates: Stage 1 からの候補リスト。
                        各要素は (AgentMemory, similarity_score) のタプル。
            perspective: 観点（指定時は strength_by_perspective[perspective] を使用）

        Returns:
            ScoredMemory のリスト。
            スコア降順でソートされ、TOP_K_RESULTS 件に制限される。

        Note:
            - similarity: Stage 1 からそのまま使用（0-1 の範囲を想定）
            - strength: 0-1 に正規化（観点指定時は該当観点の強度を使用）
            - recency: 時間減衰で 0-1 にスケール
        """
        if not candidates:
            logger.debug("候補が空のため、空リストを返します")
            return []

        # 重みを取得
        weights = self.config.score_weights
        w_similarity = weights.get("similarity", 0.50)
        w_strength = weights.get("strength", 0.30)
        w_recency = weights.get("recency", 0.20)

        scored_memories: List[ScoredMemory] = []

        for memory, similarity in candidates:
            # 強度の正規化
            strength = self._get_strength(memory, perspective)
            normalized_strength = self._normalize_strength(strength)

            # 新鮮さスコアの計算
            recency_score = self._calculate_recency(
                memory.last_accessed_at,
                memory.created_at,
            )

            # 合成スコアの計算
            final_score = (
                similarity * w_similarity +
                normalized_strength * w_strength +
                recency_score * w_recency
            )

            # スコア内訳を記録（デバッグ用）
            score_breakdown = {
                "similarity_raw": similarity,
                "similarity_weighted": similarity * w_similarity,
                "strength_raw": strength,
                "strength_normalized": normalized_strength,
                "strength_weighted": normalized_strength * w_strength,
                "recency_raw": recency_score,
                "recency_weighted": recency_score * w_recency,
                "total": final_score,
            }

            scored_memories.append(
                ScoredMemory(
                    memory=memory,
                    similarity=similarity,
                    final_score=final_score,
                    score_breakdown=score_breakdown,
                )
            )

        # スコア降順でソート
        scored_memories.sort(key=lambda x: x.final_score, reverse=True)

        # TOP_K_RESULTS 件に制限
        top_k = self.config.top_k_results
        result = scored_memories[:top_k]

        logger.info(
            f"ランキング完了: "
            f"candidates={len(candidates)}, "
            f"perspective={perspective}, "
            f"returned={len(result)}"
        )

        if result:
            logger.debug(
                f"トップスコア: {result[0].final_score:.3f}, "
                f"ボトムスコア: {result[-1].final_score:.3f}"
            )

        return result

    def _get_strength(
        self,
        memory: AgentMemory,
        perspective: Optional[str],
    ) -> float:
        """記憶の強度を取得

        Args:
            memory: 対象の記憶
            perspective: 観点（指定時は strength_by_perspective を参照）

        Returns:
            強度値。観点指定時は該当観点の強度、
            なければ全体の強度を使用。
        """
        if perspective and memory.strength_by_perspective:
            # 観点指定時は該当観点の強度を使用
            perspective_strength = memory.strength_by_perspective.get(perspective)
            if perspective_strength is not None:
                return perspective_strength
            # 該当観点がない場合は全体の強度にフォールバック
            logger.debug(
                f"観点 '{perspective}' の強度が未設定のため、"
                f"全体の強度を使用: memory_id={memory.id}"
            )

        return memory.strength

    def _normalize_strength(self, strength: float) -> float:
        """強度を 0-1 に正規化

        Args:
            strength: 元の強度値

        Returns:
            0-1 に正規化された強度

        計算式:
            normalized = min(strength, MAX_STRENGTH) / MAX_STRENGTH

        Note:
            - MAX_STRENGTH_FOR_NORMALIZE (2.0) 以上の強度は 1.0 にクリップ
            - 負の強度は 0.0 にクリップ
        """
        if strength <= 0:
            return 0.0

        normalized = min(strength, self.MAX_STRENGTH_FOR_NORMALIZE)
        return normalized / self.MAX_STRENGTH_FOR_NORMALIZE

    def _calculate_recency(
        self,
        last_accessed_at: Optional[datetime],
        created_at: datetime,
    ) -> float:
        """新鮮さスコアを計算

        Args:
            last_accessed_at: 最後にアクセスされた日時（NULL の場合あり）
            created_at: 作成日時（last_accessed_at が NULL の場合に使用）

        Returns:
            新鮮さスコア（0-1）。最近ほど高い。

        計算式:
            recency = exp(-days_since_access / RECENCY_DECAY_DAYS)

        Note:
            - last_accessed_at が NULL の場合は created_at を使用
            - 30日で約 0.37 (1/e)、60日で約 0.14、90日で約 0.05
        """
        # 基準日時を決定
        reference_time = last_accessed_at or created_at

        # 経過日数を計算
        now = datetime.now()
        delta = now - reference_time
        days_elapsed = max(0, delta.total_seconds() / 86400.0)  # 86400秒 = 1日

        # 指数減衰
        recency = math.exp(-days_elapsed / self.RECENCY_DECAY_DAYS)

        return recency
