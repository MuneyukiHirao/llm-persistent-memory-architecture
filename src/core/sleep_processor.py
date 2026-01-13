# 睡眠フェーズプロセッサ
# 強度減衰、メモリ統合、アーカイブ処理を担当
# 実装仕様: docs/phase1-implementation-spec.ja.md セクション7.1 (睡眠フェーズ実装)
# アーキテクチャ: docs/architecture.ja.md セクション3.8 (睡眠フェーズ)
"""
睡眠フェーズプロセッサモジュール

タスク完了時に実行される睡眠フェーズの処理を担当。
強度減衰、閾値ベースのアーカイブ、類似メモリの統合レベル管理を行う。

設計方針（睡眠フェーズエージェント観点）:
- データ整合性: 処理後もデータの一貫性が保たれる（バッチ更新でトランザクション保証）
- 処理効率: 大量のメモリを効率的に処理（バッチ処理、ページネーション）
- 可逆性: 誤った処理を元に戻せる（アーカイブは論理削除、物理削除なし）
- パラメータ調整: 減衰率や閾値は Phase1Config で一元管理
- 障害耐性: 処理中断時にデータが壊れない（トランザクション単位での処理）

処理順序（仕様書より）:
1. 減衰対象メモリの取得（status="active"）
2. 各メモリに減衰を適用
3. 閾値以下のメモリをアーカイブ
4. 類似メモリのグループ化
5. 統合レベルの更新
6. 処理ログの記録
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID

from src.config.phase1_config import Phase1Config
from src.db.connection import DatabaseConnection
from src.core.memory_repository import MemoryRepository
from src.core.strength_manager import StrengthManager
from src.models.memory import AgentMemory

# ロガー設定
logger = logging.getLogger(__name__)


@dataclass
class SleepPhaseResult:
    """睡眠フェーズの処理結果

    Attributes:
        agent_id: 処理対象のエージェントID
        decayed_count: 減衰処理されたメモリ数
        archived_count: アーカイブされたメモリ数
        consolidated_count: 統合レベルが更新されたメモリ数
        processed_at: 処理実行日時
        errors: 処理中に発生したエラーのリスト
    """
    agent_id: str
    decayed_count: int
    archived_count: int
    consolidated_count: int
    processed_at: datetime
    errors: List[str]

    def to_dict(self) -> Dict:
        """辞書形式に変換"""
        return {
            "agent_id": self.agent_id,
            "decayed_count": self.decayed_count,
            "archived_count": self.archived_count,
            "consolidated_count": self.consolidated_count,
            "processed_at": self.processed_at.isoformat(),
            "errors": self.errors,
            "success": len(self.errors) == 0,
        }


class SleepPhaseProcessor:
    """睡眠フェーズプロセッサ

    タスク完了時に実行される睡眠フェーズの処理を管理。

    睡眠フェーズの役割（architecture.ja.md セクション3.8より）:
    - タスク受付を停止して減衰・アーカイブ・剪定を一括実行
    - 処理の一貫性を保証（リアルタイム処理中のデータ競合を防止）
    - バッチ処理による効率的なメモリ管理

    処理内容:
    1. 強度減衰: 定着レベルに応じた減衰率で strength を減少
    2. アーカイブ: archive_threshold (0.1) 以下のメモリを archived に変更
    3. 統合処理: 類似メモリの consolidation_level を管理（Phase 1は簡易版）

    使用例:
        db = DatabaseConnection()
        config = Phase1Config()
        processor = SleepPhaseProcessor(db, config)

        # 睡眠フェーズを実行
        result = processor.process_all("agent_01")
        print(f"減衰: {result.decayed_count}, アーカイブ: {result.archived_count}")

    Attributes:
        config: Phase1Config インスタンス（減衰率、閾値等のパラメータ）
        repository: MemoryRepository インスタンス（CRUD操作）
        strength_manager: StrengthManager インスタンス（強度管理）
    """

    # バッチサイズの設定（大量メモリ処理時のメモリ効率を考慮）
    DEFAULT_BATCH_SIZE = 100

    # 類似度閾値（統合処理用）
    # Phase 1 実装仕様: similarity > 0.85 でグループ化
    SIMILARITY_THRESHOLD_FOR_CONSOLIDATION = 0.85

    def __init__(
        self,
        db: DatabaseConnection,
        config: Optional[Phase1Config] = None,
    ):
        """SleepPhaseProcessor を初期化

        Args:
            db: DatabaseConnection インスタンス
            config: Phase1Config インスタンス（省略時はデフォルト設定）

        Note:
            - MemoryRepository と StrengthManager は内部で初期化
            - config の archive_threshold, daily_decay_targets を使用
        """
        self.config = config or Phase1Config()
        self.db = db
        self.repository = MemoryRepository(db, self.config)
        self.strength_manager = StrengthManager(self.repository, self.config)

    def process_all(self, agent_id: str) -> SleepPhaseResult:
        """睡眠フェーズのメイン処理を実行

        全ての睡眠フェーズ処理を順番に実行する。

        処理順序（phase1-implementation-spec.ja.md セクション7.1より）:
        1. 減衰対象メモリの取得（status="active"）
        2. 各メモリに減衰を適用
        3. 閾値以下のメモリをアーカイブ
        4. 類似メモリのグループ化
        5. 統合レベルの更新
        6. 処理ログの記録

        Args:
            agent_id: 処理対象のエージェントID

        Returns:
            SleepPhaseResult: 処理結果（各処理の件数、エラー情報）

        Note:
            - 各処理は独立したトランザクションで実行
            - エラーが発生しても後続処理は継続
            - 全エラーは result.errors に集約
        """
        logger.info(f"睡眠フェーズ開始: agent_id={agent_id}")
        errors: List[str] = []

        # 1-2. 減衰処理
        try:
            decayed_count = self.apply_decay_all(agent_id)
            logger.info(f"減衰処理完了: {decayed_count}件")
        except Exception as e:
            error_msg = f"減衰処理でエラー: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            decayed_count = 0

        # 3. アーカイブ処理
        try:
            archived_count = self.archive_weak_memories(agent_id)
            logger.info(f"アーカイブ処理完了: {archived_count}件")
        except Exception as e:
            error_msg = f"アーカイブ処理でエラー: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            archived_count = 0

        # 4-5. 統合処理（Phase 1は簡易版）
        try:
            consolidated_count = self.consolidate_similar(agent_id)
            logger.info(f"統合処理完了: {consolidated_count}件")
        except Exception as e:
            error_msg = f"統合処理でエラー: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            consolidated_count = 0

        # 6. 処理結果の記録
        result = SleepPhaseResult(
            agent_id=agent_id,
            decayed_count=decayed_count,
            archived_count=archived_count,
            consolidated_count=consolidated_count,
            processed_at=datetime.now(),
            errors=errors,
        )

        logger.info(
            f"睡眠フェーズ完了: agent_id={agent_id}, "
            f"decayed={decayed_count}, archived={archived_count}, "
            f"consolidated={consolidated_count}, errors={len(errors)}"
        )

        return result

    def apply_decay_all(
        self,
        agent_id: str,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> int:
        """全アクティブメモリに減衰を適用

        定着レベルに応じた減衰率で各メモリの strength を減少させる。

        減衰率の計算（phase1-implementation-spec.ja.md セクション4.2より）:
        - decay_rate = daily_target ** (1 / expected_tasks_per_day)
        - Level 0: 0.95^0.1 ≒ 0.9949 (5%/日)
        - Level 5: 0.998^0.1 ≒ 0.9998 (0.2%/日)

        Args:
            agent_id: 処理対象のエージェントID
            batch_size: 一度に処理するメモリ数（デフォルト: 100）

        Returns:
            減衰処理されたメモリ数

        Note:
            - status='active' のメモリのみ対象
            - last_decay_at を更新して二重減衰を防止
            - バッチ処理で大量メモリを効率的に処理
        """
        total_processed = 0
        processed_ids: set[UUID] = set()

        while True:
            # 1. バッチ取得（strength 昇順）
            memories = self.repository.get_memories_for_decay(agent_id, batch_size)

            if not memories:
                # メモリが存在しない場合は終了
                break

            # 処理済みメモリを除外（同じメモリが再取得される可能性があるため）
            unprocessed = [m for m in memories if m.id not in processed_ids]

            if not unprocessed:
                # 取得したメモリがすべて処理済み = 全件処理完了
                break

            # 2-3. 各メモリの減衰率を取得し、新しい強度を計算
            updates: List[tuple[UUID, float]] = []
            for memory in unprocessed:
                # consolidation_level から減衰率を取得
                decay_rate = self.config.get_decay_rate(memory.consolidation_level)
                # new_strength = strength * decay_rate
                new_strength = memory.strength * decay_rate

                updates.append((memory.id, new_strength))
                processed_ids.add(memory.id)

            # 4. batch_update_strength() で一括更新
            updated_count = self.repository.batch_update_strength(updates)
            total_processed += updated_count

            logger.debug(
                f"Batch processed: {updated_count} memories, "
                f"total: {total_processed}"
            )

            # batch_size 未満の取得なら全件処理完了
            if len(memories) < batch_size:
                break

        # 5. 処理件数を返す
        return total_processed

    def archive_weak_memories(self, agent_id: str) -> int:
        """閾値以下のメモリをアーカイブ

        strength が archive_threshold (0.1) 以下のメモリを
        status='archived' に変更する。

        アーカイブ処理（architecture.ja.md セクション3.8より）:
        - 物理削除ではなく論理削除（可逆性を確保）
        - アーカイブされたメモリは検索対象外
        - 必要に応じて reactivate() で再活性化可能

        Args:
            agent_id: 処理対象のエージェントID

        Returns:
            アーカイブされたメモリ数

        Note:
            - 減衰処理の後に実行すること
            - アーカイブ前に確認ログを出力（デバッグ用）
            - batch_archive() で効率的に一括処理
        """
        # 1. アクティブメモリを取得
        active_memories = self.repository.get_by_agent_id(agent_id, status="active")

        if not active_memories:
            logger.debug(f"アーカイブ対象なし: agent_id={agent_id} (アクティブメモリなし)")
            return 0

        # 2. strength <= archive_threshold のメモリをフィルタ
        threshold = self.config.archive_threshold
        archive_candidates = [
            memory for memory in active_memories
            if memory.strength <= threshold
        ]

        if not archive_candidates:
            logger.debug(
                f"アーカイブ対象なし: agent_id={agent_id}, "
                f"threshold={threshold}, active_count={len(active_memories)}"
            )
            return 0

        # 3. アーカイブ対象のIDリストを作成
        archive_ids: List[UUID] = [memory.id for memory in archive_candidates]

        # 4. ログ出力（デバッグ用: アーカイブ対象のcontent概要）
        for memory in archive_candidates:
            # content の最初の50文字を概要として出力
            content_summary = memory.content[:50] + "..." if len(memory.content) > 50 else memory.content
            logger.debug(
                f"アーカイブ候補: id={memory.id}, "
                f"strength={memory.strength:.4f}, "
                f"content=\"{content_summary}\""
            )

        logger.info(
            f"アーカイブ実行: agent_id={agent_id}, "
            f"target_count={len(archive_ids)}, threshold={threshold}"
        )

        # 5. repository.batch_archive() で一括アーカイブ
        archived_count = self.repository.batch_archive(archive_ids)

        logger.info(f"アーカイブ完了: archived={archived_count}件")

        # 6. アーカイブ件数を返す
        return archived_count

    def consolidate_similar(self, agent_id: str) -> int:
        """類似メモリの統合レベルを更新

        類似度が高いメモリ同士をグループ化し、
        consolidation_level をインクリメントする。

        Phase 1 簡易版の仕様（phase1-implementation-spec.ja.md より）:
        - similarity > 0.85 のペアをグループ化
        - マージは行わず、consolidation_level の管理のみ
        - 将来的には類似メモリを1つに統合する機能を追加予定

        Args:
            agent_id: 処理対象のエージェントID

        Returns:
            統合レベルが更新されたメモリ数

        Note:
            - Phase 1 ではスキップ。Phase 2 で本格実装予定
            - 理由: 類似メモリの検出は計算コストが高く、
              実運用での必要性を確認してから本格実装予定
            - consolidation_level の更新は、access_count ベースで
              Phase1Config.get_consolidation_level() により自動計算される
        """
        # Phase 1 簡易版: 処理をスキップ
        logger.info(
            f"consolidate_similar スキップ: agent_id={agent_id} "
            "(Phase 1 では未実装、Phase 2 で本格実装予定)"
        )
        return 0

    # === ユーティリティメソッド ===

    def get_processing_stats(self, agent_id: str) -> Dict:
        """処理前のメモリ統計を取得

        睡眠フェーズ処理前の状態を記録するためのユーティリティ。

        Args:
            agent_id: 対象エージェントID

        Returns:
            統計情報の辞書:
            - total_active: アクティブメモリ総数
            - below_threshold: アーカイブ閾値以下のメモリ数
            - avg_strength: 平均強度
        """
        # TODO: 実装
        raise NotImplementedError("get_processing_stats is not implemented yet")

    def estimate_decay_impact(self, agent_id: str) -> Dict:
        """減衰処理のインパクトを事前推定

        減衰処理を実行した場合の影響を事前に計算する。
        本番実行前の確認用。

        Args:
            agent_id: 対象エージェントID

        Returns:
            推定結果の辞書:
            - memories_to_archive: 新たにアーカイブ対象となるメモリ数
            - avg_strength_after: 減衰後の平均強度
        """
        # TODO: 実装
        raise NotImplementedError("estimate_decay_impact is not implemented yet")
