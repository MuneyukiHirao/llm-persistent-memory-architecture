# 2段階強化動作確認スクリプト
"""
Phase 1 2段階強化動作確認

2段階強化メカニズムが正しく機能することを検証:
1. candidate_count と access_count が分離して更新されているか
2. 検索候補になっただけで strength が上がっていないか確認
3. 実際に使用された時のみ strength が強化されることを確認

2段階強化とは:
- Stage 1（候補ブースト）: 検索候補になった → candidate_count++ のみ、strengthは変更しない
- Stage 2（使用ブースト）: 実際に使用された → access_count++, strength += 0.1

参照仕様書:
- docs/phase1-implementation-spec.ja.md セクション1.2（2段階強化）
- docs/architecture.ja.md セクション3.5（2段階強化）

テスト観点（検証エージェント視点）:
- テストカバレッジ: Stage 1, Stage 2 の両方を独立してテスト
- 再現性: 決定的な操作でDB状態を検証
- 境界値: 初回候補、初回使用、大量候補などのエッジケース
- パフォーマンス: バッチ操作の正確性
- 保守性: 各カウンターの更新を個別にアサート
"""

import os
import sys
from datetime import datetime
from typing import List, Tuple
from uuid import UUID

import pytest

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from tests.verification.conftest import (
    TEST_AGENT_PREFIX,
    create_test_memory,
)


class TestCandidateCountIncrement:
    """Stage 1: 候補カウント（candidate_count）のテスト

    検索候補になった時点で candidate_count がインクリメントされ、
    strength は変更されないことを検証
    """

    def test_mark_as_candidate_increments_count(
        self, repository, strength_manager, unique_agent_id, cleanup_test_memories
    ):
        """候補マークで candidate_count がインクリメントされるか"""
        agent_id = unique_agent_id

        # 初期メモリを作成
        memory = create_test_memory(
            agent_id=agent_id,
            content="候補カウントテストメモリ",
            strength=1.0,
            candidate_count=0,
        )
        created = repository.create(memory)

        # 初期状態を確認
        assert created.candidate_count == 0
        initial_strength = created.strength

        # Stage 1: 候補としてマーク
        strength_manager.mark_as_candidate([created.id])

        # 更新後の状態を取得
        updated = repository.get_by_id(created.id)

        # アサーション
        assert updated.candidate_count == 1, \
            f"candidate_count が 1 になっていない: {updated.candidate_count}"
        assert updated.strength == initial_strength, \
            f"strength が変更されている: {updated.strength} != {initial_strength}"
        assert updated.access_count == 0, \
            f"access_count が変更されている: {updated.access_count}"

    def test_mark_as_candidate_does_not_change_strength(
        self, repository, strength_manager, unique_agent_id, cleanup_test_memories
    ):
        """候補マークで strength が変更されないことを明示的に検証"""
        agent_id = unique_agent_id

        # 初期強度が異なるメモリを作成
        test_strengths = [0.5, 1.0, 1.5]

        for strength in test_strengths:
            memory = create_test_memory(
                agent_id=agent_id,
                content=f"強度 {strength} のテストメモリ",
                strength=strength,
            )
            created = repository.create(memory)

            # 候補としてマーク
            strength_manager.mark_as_candidate([created.id])

            # 強度が変わっていないことを確認
            updated = repository.get_by_id(created.id)
            assert updated.strength == strength, \
                f"strength が変更された: {strength} → {updated.strength}"

    def test_multiple_candidate_marks(
        self, repository, strength_manager, unique_agent_id, cleanup_test_memories
    ):
        """複数回の候補マークで candidate_count が累積するか"""
        agent_id = unique_agent_id

        memory = create_test_memory(
            agent_id=agent_id,
            content="複数候補マークテストメモリ",
        )
        created = repository.create(memory)

        # 5回候補としてマーク
        for i in range(5):
            strength_manager.mark_as_candidate([created.id])
            updated = repository.get_by_id(created.id)
            assert updated.candidate_count == i + 1, \
                f"candidate_count が {i + 1} になっていない: {updated.candidate_count}"

        # 最終確認
        final = repository.get_by_id(created.id)
        assert final.candidate_count == 5
        assert final.strength == 1.0  # 変更されていない
        assert final.access_count == 0

    def test_batch_candidate_mark(
        self, repository, strength_manager, unique_agent_id, cleanup_test_memories
    ):
        """バッチで複数メモリを候補マークできるか"""
        agent_id = unique_agent_id

        # 5件のメモリを作成
        memories = []
        for i in range(5):
            memory = create_test_memory(
                agent_id=agent_id,
                content=f"バッチ候補テストメモリ {i}",
            )
            created = repository.create(memory)
            memories.append(created)

        # 全てを一括で候補マーク
        memory_ids = [m.id for m in memories]
        updated_count = strength_manager.mark_as_candidate(memory_ids)

        # アサーション
        assert updated_count == 5

        for memory in memories:
            updated = repository.get_by_id(memory.id)
            assert updated.candidate_count == 1
            assert updated.strength == 1.0
            assert updated.access_count == 0


class TestAccessCountIncrement:
    """Stage 2: 使用カウント（access_count）と強度強化のテスト

    実際に使用された時に access_count と strength が
    正しく更新されることを検証
    """

    def test_mark_as_used_increments_counts_and_strength(
        self, repository, config, strength_manager, unique_agent_id, cleanup_test_memories
    ):
        """使用マークで access_count と strength が更新されるか"""
        agent_id = unique_agent_id

        # 初期メモリを作成
        memory = create_test_memory(
            agent_id=agent_id,
            content="使用カウントテストメモリ",
            strength=1.0,
            access_count=0,
        )
        created = repository.create(memory)

        # Stage 2: 使用としてマーク
        result = strength_manager.mark_as_used(created.id)

        # アサーション
        assert result is not None

        # DBから最新を取得して確認
        updated = repository.get_by_id(created.id)

        assert updated.access_count == 1, \
            f"access_count が 1 になっていない: {updated.access_count}"
        assert updated.strength == pytest.approx(1.0 + config.strength_increment_on_use, rel=0.01), \
            f"strength が正しく増加していない: {updated.strength}"

    def test_mark_as_used_updates_last_accessed_at(
        self, repository, strength_manager, unique_agent_id, cleanup_test_memories
    ):
        """使用マークで last_accessed_at が更新されるか"""
        agent_id = unique_agent_id

        memory = create_test_memory(
            agent_id=agent_id,
            content="last_accessed_at テストメモリ",
        )
        created = repository.create(memory)

        # 初期状態（last_accessed_at は None または古い値）
        assert created.last_accessed_at is None

        # 使用マーク前の時刻を記録
        before_mark = datetime.now()

        # 使用としてマーク
        strength_manager.mark_as_used(created.id)

        # 更新を確認
        updated = repository.get_by_id(created.id)

        assert updated.last_accessed_at is not None
        assert updated.last_accessed_at >= before_mark, \
            "last_accessed_at が更新されていない"

    def test_mark_as_used_with_perspective(
        self, repository, config, strength_manager, unique_agent_id, cleanup_test_memories
    ):
        """観点指定で使用マークした場合の strength_by_perspective 更新"""
        agent_id = unique_agent_id

        memory = create_test_memory(
            agent_id=agent_id,
            content="観点付き使用テストメモリ",
            strength=1.0,
        )
        created = repository.create(memory)

        # 観点「コスト」で使用マーク
        strength_manager.mark_as_used(created.id, perspective="コスト")

        # 更新を確認
        updated = repository.get_by_id(created.id)

        # strength_by_perspective が更新されていることを確認
        assert "コスト" in updated.strength_by_perspective
        assert updated.strength_by_perspective["コスト"] == pytest.approx(
            config.perspective_strength_increment, rel=0.01
        )

        # 全体 strength も更新されていること
        assert updated.strength == pytest.approx(
            1.0 + config.strength_increment_on_use, rel=0.01
        )


class TestTwoStagesSeparation:
    """Stage 1 と Stage 2 の分離テスト

    候補マーク（Stage 1）と使用マーク（Stage 2）が
    互いに独立して機能することを検証
    """

    def test_candidate_only_does_not_increase_access_count(
        self, repository, strength_manager, unique_agent_id, cleanup_test_memories
    ):
        """候補マークのみでは access_count が増えないことを検証"""
        agent_id = unique_agent_id

        memory = create_test_memory(
            agent_id=agent_id,
            content="候補のみテストメモリ",
        )
        created = repository.create(memory)

        # 10回候補としてマーク
        for _ in range(10):
            strength_manager.mark_as_candidate([created.id])

        # 確認
        updated = repository.get_by_id(created.id)

        assert updated.candidate_count == 10, "candidate_count が 10 でない"
        assert updated.access_count == 0, "access_count が変更されている"
        assert updated.strength == 1.0, "strength が変更されている"

    def test_used_increments_access_but_not_candidate(
        self, repository, config, strength_manager, unique_agent_id, cleanup_test_memories
    ):
        """使用マークは access_count のみ増やし candidate_count は増やさない"""
        agent_id = unique_agent_id

        memory = create_test_memory(
            agent_id=agent_id,
            content="使用マークテストメモリ",
        )
        created = repository.create(memory)

        # 5回使用マーク（候補マークなし）
        for _ in range(5):
            strength_manager.mark_as_used(created.id)

        # 確認
        updated = repository.get_by_id(created.id)

        assert updated.access_count == 5, "access_count が 5 でない"
        assert updated.candidate_count == 0, "candidate_count が変更されている"

        # strength は 5 回分増加
        expected_strength = 1.0 + (config.strength_increment_on_use * 5)
        assert updated.strength == pytest.approx(expected_strength, rel=0.01)

    def test_typical_flow_candidate_then_used(
        self, repository, config, strength_manager, unique_agent_id, cleanup_test_memories
    ):
        """典型的なフロー: 候補→使用の順序で正しく動作するか"""
        agent_id = unique_agent_id

        memory = create_test_memory(
            agent_id=agent_id,
            content="典型フローテストメモリ",
        )
        created = repository.create(memory)

        # フロー1: 検索候補になる（Stage 1）
        strength_manager.mark_as_candidate([created.id])

        state_after_candidate = repository.get_by_id(created.id)
        assert state_after_candidate.candidate_count == 1
        assert state_after_candidate.access_count == 0
        assert state_after_candidate.strength == 1.0

        # フロー2: 実際に使用される（Stage 2）
        strength_manager.mark_as_used(created.id)

        state_after_used = repository.get_by_id(created.id)
        assert state_after_used.candidate_count == 1  # 変更なし
        assert state_after_used.access_count == 1
        assert state_after_used.strength == pytest.approx(
            1.0 + config.strength_increment_on_use, rel=0.01
        )

        print(f"\n典型フローの最終状態:")
        print(f"  candidate_count: {state_after_used.candidate_count}")
        print(f"  access_count: {state_after_used.access_count}")
        print(f"  strength: {state_after_used.strength:.3f}")

    def test_multiple_candidates_single_use(
        self, repository, config, strength_manager, unique_agent_id, cleanup_test_memories
    ):
        """複数回候補になり、1回だけ使用されるケース"""
        agent_id = unique_agent_id

        memory = create_test_memory(
            agent_id=agent_id,
            content="複数候補1使用テストメモリ",
        )
        created = repository.create(memory)

        # 10回候補になる
        for _ in range(10):
            strength_manager.mark_as_candidate([created.id])

        # 1回だけ使用
        strength_manager.mark_as_used(created.id)

        # 確認
        final = repository.get_by_id(created.id)

        assert final.candidate_count == 10
        assert final.access_count == 1
        assert final.strength == pytest.approx(
            1.0 + config.strength_increment_on_use, rel=0.01
        )

        # 使用率 = 1/10 = 0.1
        usage_rate = final.access_count / final.candidate_count
        assert usage_rate == pytest.approx(0.1, rel=0.01)


class TestUsageRateCalculation:
    """使用率計算の検証

    usage_rate = access_count / candidate_count が
    正しく計算できることを検証
    """

    def test_usage_rate_typical_scenarios(
        self, repository, strength_manager, unique_agent_id, cleanup_test_memories
    ):
        """典型的な使用率シナリオ"""
        agent_id = unique_agent_id

        scenarios: List[Tuple[int, int, float]] = [
            (100, 10, 0.1),   # 10% 使用率
            (100, 20, 0.2),   # 20% 使用率
            (100, 30, 0.3),   # 30% 使用率
            (50, 25, 0.5),    # 50% 使用率
            (10, 10, 1.0),    # 100% 使用率
        ]

        for candidate_count, access_count, expected_rate in scenarios:
            memory = create_test_memory(
                agent_id=agent_id,
                content=f"使用率 {expected_rate:.0%} テストメモリ",
                candidate_count=candidate_count,
                access_count=access_count,
            )
            created = repository.create(memory)

            # 使用率を計算
            usage_rate = created.access_count / created.candidate_count

            assert usage_rate == pytest.approx(expected_rate, rel=0.01), \
                f"使用率が期待値と異なる: {usage_rate} != {expected_rate}"

    def test_zero_candidate_count_handling(
        self, repository, unique_agent_id, cleanup_test_memories
    ):
        """candidate_count = 0 の場合のハンドリング"""
        agent_id = unique_agent_id

        memory = create_test_memory(
            agent_id=agent_id,
            content="候補なしテストメモリ",
            candidate_count=0,
            access_count=0,
        )
        created = repository.create(memory)

        # ゼロ除算を回避して使用率を計算
        if created.candidate_count > 0:
            usage_rate = created.access_count / created.candidate_count
        else:
            usage_rate = 0.0

        assert usage_rate == 0.0


class TestNoiseReduction:
    """ノイズ軽減の検証

    2段階強化の目的の一つは「ノイズの強化を防ぐ」こと。
    検索候補にはなるが使用されないメモリ（ノイズ）が
    強化されないことを検証。
    """

    def test_noise_is_not_reinforced(
        self, repository, config, strength_manager, sleep_processor,
        unique_agent_id, cleanup_test_memories
    ):
        """ノイズ（候補にはなるが使用されない）が強化されないことを検証"""
        agent_id = unique_agent_id

        # 有用なメモリとノイズメモリを作成
        useful = create_test_memory(
            agent_id=agent_id,
            content="有用なメモリ: プロジェクト管理のベストプラクティス",
        )
        noise = create_test_memory(
            agent_id=agent_id,
            content="ノイズメモリ: 関連するが使われない情報",
        )

        useful_created = repository.create(useful)
        noise_created = repository.create(noise)

        # 両方が候補になる（検索でヒット）
        strength_manager.mark_as_candidate([useful_created.id, noise_created.id])

        # 有用なメモリのみが使用される
        strength_manager.mark_as_used(useful_created.id)

        # 減衰を適用
        sleep_processor.apply_decay_all(agent_id)

        # 確認
        useful_final = repository.get_by_id(useful_created.id)
        noise_final = repository.get_by_id(noise_created.id)

        print(f"\nノイズ軽減テスト結果:")
        print(f"  有用: candidate_count={useful_final.candidate_count}, "
              f"access_count={useful_final.access_count}, strength={useful_final.strength:.3f}")
        print(f"  ノイズ: candidate_count={noise_final.candidate_count}, "
              f"access_count={noise_final.access_count}, strength={noise_final.strength:.3f}")

        # アサーション
        # 1. 両方とも candidate_count は同じ
        assert useful_final.candidate_count == noise_final.candidate_count

        # 2. access_count は異なる（有用のみが使用された）
        assert useful_final.access_count == 1
        assert noise_final.access_count == 0

        # 3. 強度は有用な方が高い（使用による強化 vs 減衰のみ）
        assert useful_final.strength > noise_final.strength, \
            "ノイズの強度が有用メモリより高い"

    def test_repeated_noise_detection(
        self, repository, config, strength_manager, sleep_processor,
        unique_agent_id, cleanup_test_memories
    ):
        """繰り返し候補になるが使用されないメモリの検出"""
        agent_id = unique_agent_id

        memory = create_test_memory(
            agent_id=agent_id,
            content="繰り返しノイズテストメモリ",
        )
        created = repository.create(memory)

        # 100回候補になるが、一度も使用されない
        for _ in range(100):
            strength_manager.mark_as_candidate([created.id])

        # 確認
        final = repository.get_by_id(created.id)

        # candidate_count > 50 かつ access_count = 0 → 「候補だけで未使用」
        is_noise = final.candidate_count > 50 and final.access_count == 0

        print(f"\nノイズ検出テスト:")
        print(f"  candidate_count: {final.candidate_count}")
        print(f"  access_count: {final.access_count}")
        print(f"  ノイズ判定: {is_noise}")

        assert is_noise, "ノイズとして検出されるべき"
        assert final.strength == 1.0, "strength が変更されている（強化されてしまっている）"
