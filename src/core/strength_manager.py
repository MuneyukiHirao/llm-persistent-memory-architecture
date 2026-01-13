# 強度管理マネージャー
# 2段階強化、インパクト反映、定着レベル管理を担当
# 実装仕様: docs/phase1-implementation-spec.ja.md セクション4.1, 4.4
"""
強度管理マネージャーモジュール

2段階強化メカニズムとインパクトベースの強度管理を提供。

設計方針（メモリ管理エージェント観点）:
- 強度の正確性: candidate_count と access_count の分離が正しく機能
- 観点別強度: strength_by_perspective の更新が適切
- 原子性: 複数フィールド更新時のトランザクション整合性
- 効率性: バッチ更新、リポジトリメソッドの活用
- テスト容易性: 依存性注入でリポジトリと設定を外部から注入可能
"""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from src.config.phase1_config import Phase1Config
from src.core.memory_repository import MemoryRepository
from src.models.memory import AgentMemory


class StrengthManager:
    """強度管理マネージャー

    2段階強化、インパクト反映、定着レベル管理、再活性化を担当。

    2段階強化の仕組み:
        1. 検索候補として参照 → candidate_count += 1 のみ（strength は変更しない）
        2. 実際に使用された → access_count += 1, strength += increment
           - 該当観点の strength_by_perspective も強化
           - last_accessed_at を更新
           - consolidation_level を更新

    使用例:
        db = DatabaseConnection()
        config = Phase1Config()
        repo = MemoryRepository(db, config)
        manager = StrengthManager(repo, config)

        # 検索候補としてマーク（strength は変更しない）
        manager.mark_as_candidate([memory_id1, memory_id2])

        # 実際に使用されたことをマーク（strength も強化）
        manager.mark_as_used(memory_id1, perspective="コスト")

        # インパクトを反映
        manager.apply_impact(memory_id1, "user_positive")

    Attributes:
        repository: MemoryRepository インスタンス
        config: Phase1Config インスタンス
    """

    # インパクトタイプとスコアのマッピング
    IMPACT_SCORES = {
        "user_positive": "impact_user_positive",      # ユーザーから肯定的フィードバック
        "task_success": "impact_task_success",        # タスク成功に貢献
        "prevented_error": "impact_prevented_error",  # エラー防止に貢献
    }

    def __init__(self, repository: MemoryRepository, config: Phase1Config):
        """StrengthManager を初期化

        Args:
            repository: MemoryRepository インスタンス
            config: Phase1Config インスタンス
        """
        self.repository = repository
        self.config = config

    # === 2段階強化 ===

    def mark_as_candidate(self, memory_ids: List[UUID]) -> int:
        """検索候補になったメモリをマーク（candidate_count++）

        2段階強化の第1段階: 検索候補として参照されただけで
        candidate_count をインクリメントする。
        strength は変更しない。

        Args:
            memory_ids: 検索候補になったメモリの UUID リスト

        Returns:
            更新された行数

        Note:
            - 空リストの場合は 0 を返す
            - バッチ更新で効率的に処理
            - 実際に使用された場合は mark_as_used を使用
        """
        if not memory_ids:
            return 0

        return self.repository.batch_increment_candidate_count(memory_ids)

    def mark_as_used(
        self, memory_id: UUID, perspective: Optional[str] = None
    ) -> Optional[AgentMemory]:
        """実際に使用されたメモリを強化

        2段階強化の第2段階: 実際に使用されたメモリの
        access_count と strength を強化する。

        Args:
            memory_id: 使用されたメモリの UUID
            perspective: 強化する観点名（省略可）
                        指定時は strength_by_perspective[perspective] も強化

        Returns:
            更新後の AgentMemory インスタンス
            メモリが存在しない場合は None

        処理内容:
            1. access_count += 1
            2. strength += strength_increment_on_use (0.1)
            3. 観点指定時: strength_by_perspective[perspective] += perspective_strength_increment (0.15)
            4. last_accessed_at = now
            5. consolidation_level を更新
        """
        # 1-2. access_count と strength を同時に更新
        self.repository.increment_access_count(
            memory_id,
            self.config.strength_increment_on_use,  # 0.1
        )

        # 3. 観点指定時は観点別強度も更新
        if perspective:
            self.repository.update_perspective_strength(
                memory_id,
                perspective,
                self.config.perspective_strength_increment,  # 0.15
            )

        # 4-5. consolidation_level を更新して結果を返す
        return self.update_consolidation_level(memory_id)

    # === インパクト反映 ===

    def apply_impact(
        self, memory_id: UUID, impact_type: str
    ) -> Optional[AgentMemory]:
        """インパクトスコアを加算し、強度に反映

        メモリがタスク成功やエラー防止に貢献した場合、
        インパクトスコアを加算し、強度にも反映する。

        Args:
            memory_id: 対象メモリの UUID
            impact_type: インパクトの種類
                - "user_positive": ユーザーから肯定的フィードバック (+2.0)
                - "task_success": タスク成功に貢献 (+1.5)
                - "prevented_error": エラー防止に貢献 (+2.0)

        Returns:
            更新後の AgentMemory インスタンス
            メモリが存在しない場合は None

        Raises:
            ValueError: 不正な impact_type が指定された場合

        処理内容:
            1. impact_score += インパクト値
            2. strength += impact * impact_to_strength_ratio (0.2)
        """
        # インパクトタイプの検証
        if impact_type not in self.IMPACT_SCORES:
            valid_types = ", ".join(self.IMPACT_SCORES.keys())
            raise ValueError(
                f"Invalid impact_type: {impact_type}. "
                f"Valid types are: {valid_types}"
            )

        # インパクト値を設定から取得
        config_attr = self.IMPACT_SCORES[impact_type]
        impact_value = getattr(self.config, config_attr)

        # 強度への反映量を計算
        strength_increment = impact_value * self.config.impact_to_strength_ratio

        # メモリを取得
        memory = self.repository.get_by_id(memory_id)
        if memory is None:
            return None

        # メモリを更新
        updated_memory = memory.copy_with(
            impact_score=memory.impact_score + impact_value,
            strength=memory.strength + strength_increment,
            updated_at=datetime.now(),
        )

        return self.repository.update(updated_memory)

    # === 定着レベル管理 ===

    def update_consolidation_level(self, memory_id: UUID) -> Optional[AgentMemory]:
        """access_countに基づいて定着レベルを更新

        メモリの使用回数に応じて定着レベルを計算し更新する。
        定着レベルが高いほど、睡眠フェーズでの減衰率が低くなる。

        Args:
            memory_id: 対象メモリの UUID

        Returns:
            更新後の AgentMemory インスタンス
            メモリが存在しない場合は None

        定着レベル閾値（access_count）:
            Level 0: 0回以上
            Level 1: 5回以上
            Level 2: 15回以上
            Level 3: 30回以上
            Level 4: 60回以上
            Level 5: 100回以上
        """
        memory = self.repository.get_by_id(memory_id)
        if memory is None:
            return None

        # access_count から定着レベルを計算
        new_level = self.config.get_consolidation_level(memory.access_count)

        # レベルが変わった場合のみ更新
        if new_level != memory.consolidation_level:
            updated_memory = memory.copy_with(
                consolidation_level=new_level,
                updated_at=datetime.now(),
            )
            return self.repository.update(updated_memory)

        return memory

    # === 再活性化 ===

    def reactivate(self, memory_id: UUID) -> Optional[AgentMemory]:
        """アーカイブされたメモリを再活性化

        アーカイブ（status='archived'）されたメモリを
        アクティブ状態に戻す。

        Args:
            memory_id: 再活性化するメモリの UUID

        Returns:
            再活性化された AgentMemory インスタンス
            メモリが存在しない場合は None

        Raises:
            ValueError: メモリが既にアクティブな場合

        処理内容:
            1. status = 'active'
            2. strength = reactivation_strength (0.5)
        """
        memory = self.repository.get_by_id(memory_id)
        if memory is None:
            return None

        # 既にアクティブな場合はエラー
        if memory.status == "active":
            raise ValueError(
                f"Memory {memory_id} is already active. "
                f"Reactivation is only for archived memories."
            )

        # 再活性化
        updated_memory = memory.copy_with(
            status="active",
            strength=self.config.reactivation_strength,  # 0.5
            updated_at=datetime.now(),
        )

        return self.repository.update(updated_memory)

    # === ユーティリティ ===

    def get_impact_value(self, impact_type: str) -> float:
        """インパクトタイプに対応する値を取得

        Args:
            impact_type: インパクトの種類

        Returns:
            インパクト値

        Raises:
            ValueError: 不正な impact_type が指定された場合
        """
        if impact_type not in self.IMPACT_SCORES:
            valid_types = ", ".join(self.IMPACT_SCORES.keys())
            raise ValueError(
                f"Invalid impact_type: {impact_type}. "
                f"Valid types are: {valid_types}"
            )

        config_attr = self.IMPACT_SCORES[impact_type]
        return getattr(self.config, config_attr)
