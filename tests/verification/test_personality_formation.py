# 「個性」形成検証スクリプト
"""
Phase 1「個性」形成検証

Phase 1の核心検証目標「強度管理と減衰が個性を生むか」を検証する:
1. 同じタスクを複数回実行し、検索結果の一貫性を測定
2. 使用頻度の高い情報が定着（consolidation_level上昇）しているか確認
3. 減衰により未使用情報の強度が低下しているか確認

「個性」とは:
- 頻繁に使用される有用な情報は強化され、優先的に検索される
- 使用されない情報は徐々に減衰し、最終的にアーカイブされる
- これにより、エージェントは「学習」し「忘却」する個性を持つ

参照仕様書:
- docs/phase1-implementation-spec.ja.md セクション1.2（Phase 1 で検証する核心機能）
- docs/architecture.ja.md セクション3.6（睡眠フェーズ）

テスト観点（検証エージェント視点）:
- テストカバレッジ: 強化・減衰・定着の全経路を検証
- 再現性: 決定的なテストデータとシード
- 境界値: 定着レベル閾値付近の動作
- パフォーマンス: 複数タスクサイクルのシミュレーション
- 保守性: 各検証ポイントを独立したテストに分離
"""

import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from uuid import UUID

import pytest

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from tests.verification.conftest import (
    TEST_AGENT_PREFIX,
    create_test_memory,
)


@dataclass
class PersonalityMetrics:
    """個性形成メトリクス"""
    frequently_used_strength: float
    rarely_used_strength: float
    strength_difference: float
    consolidation_progression: List[int]
    decay_observed: bool

    def __repr__(self) -> str:
        return (
            f"PersonalityMetrics(\n"
            f"  frequently_used_strength={self.frequently_used_strength:.3f},\n"
            f"  rarely_used_strength={self.rarely_used_strength:.3f},\n"
            f"  strength_difference={self.strength_difference:.3f},\n"
            f"  consolidation_progression={self.consolidation_progression},\n"
            f"  decay_observed={self.decay_observed}\n"
            f")"
        )


class TestConsolidationProgression:
    """定着レベル進行のテスト

    使用頻度の高い情報が consolidation_level を上昇させることを検証
    """

    def test_consolidation_level_increases_with_access_count(
        self, repository, config, strength_manager, unique_agent_id, cleanup_test_memories
    ):
        """使用回数増加に伴い定着レベルが上昇するか"""
        agent_id = unique_agent_id

        # 初期メモリを作成（access_count = 0, consolidation_level = 0）
        memory = create_test_memory(
            agent_id=agent_id,
            content="定着テスト用メモリ: 緊急調達のコスト管理について",
            strength=1.0,
            access_count=0,
            candidate_count=0,
            consolidation_level=0,
        )
        created = repository.create(memory)

        # 定着レベル閾値: [0, 5, 15, 30, 60, 100]
        # 各閾値を超えるまで使用をシミュレート

        expected_levels = [
            (0, 0),   # 0回使用 → Level 0
            (5, 1),   # 5回使用 → Level 1
            (15, 2),  # 15回使用 → Level 2
            (30, 3),  # 30回使用 → Level 3
            (60, 4),  # 60回使用 → Level 4
            (100, 5), # 100回使用 → Level 5
        ]

        progression = []
        for target_access_count, expected_level in expected_levels:
            # アクセス回数を直接設定（シミュレーション）
            with repository.db.get_cursor() as cur:
                cur.execute(
                    """
                    UPDATE agent_memory
                    SET access_count = %s
                    WHERE id = %s
                    """,
                    (target_access_count, str(created.id))
                )

            # consolidation_level を更新
            updated = strength_manager.update_consolidation_level(created.id)
            progression.append(updated.consolidation_level)

            # アサーション
            assert updated.consolidation_level == expected_level, \
                f"access_count={target_access_count} で consolidation_level={expected_level} を期待したが、{updated.consolidation_level} でした"

        # 進行の確認（単調増加）
        assert progression == [0, 1, 2, 3, 4, 5], f"定着レベルの進行が不正: {progression}"

    def test_consolidation_level_stays_at_boundary(
        self, repository, config, strength_manager, unique_agent_id, cleanup_test_memories
    ):
        """境界値でのconsolidation_levelの安定性"""
        agent_id = unique_agent_id

        # access_count = 4（Level 0の上限）でメモリを作成
        memory = create_test_memory(
            agent_id=agent_id,
            content="境界値テストメモリ",
            access_count=4,
            consolidation_level=0,
        )
        created = repository.create(memory)

        # Level 0 のままであることを確認
        updated = strength_manager.update_consolidation_level(created.id)
        assert updated.consolidation_level == 0

        # access_count を 5 に増加
        with repository.db.get_cursor() as cur:
            cur.execute(
                "UPDATE agent_memory SET access_count = 5 WHERE id = %s",
                (str(created.id),)
            )

        # Level 1 に上昇することを確認
        updated = strength_manager.update_consolidation_level(created.id)
        assert updated.consolidation_level == 1


class TestDecayBehavior:
    """減衰動作のテスト

    未使用情報の強度が減衰により低下することを検証
    """

    def test_decay_reduces_strength(
        self, db, repository, config, sleep_processor, unique_agent_id, cleanup_test_memories
    ):
        """睡眠フェーズで強度が減衰するか"""
        agent_id = unique_agent_id

        # 初期強度 1.0 のメモリを作成
        initial_strength = 1.0
        memory = create_test_memory(
            agent_id=agent_id,
            content="減衰テスト用メモリ",
            strength=initial_strength,
            consolidation_level=0,  # Level 0: 最大の減衰率
        )
        created = repository.create(memory)

        # 睡眠フェーズを実行
        result = sleep_processor.apply_decay_all(agent_id)

        # 減衰後の強度を取得
        updated = repository.get_by_id(created.id)

        # アサーション: 強度が減少していることを確認
        # Level 0 の減衰率: 0.9949 (タスク単位)
        expected_decay_rate = config.get_decay_rate(0)
        expected_strength = initial_strength * expected_decay_rate

        assert updated.strength < initial_strength, \
            f"強度が減衰していない: {updated.strength} >= {initial_strength}"
        assert updated.strength == pytest.approx(expected_strength, rel=0.01), \
            f"減衰量が期待値と異なる: {updated.strength} != {expected_strength}"

    def test_higher_consolidation_decays_slower(
        self, db, repository, config, sleep_processor, unique_agent_id, cleanup_test_memories
    ):
        """定着レベルが高いほど減衰が遅いか"""
        agent_id = unique_agent_id

        # 異なる定着レベルのメモリを作成
        memories = []
        for level in range(6):  # Level 0-5
            memory = create_test_memory(
                agent_id=agent_id,
                content=f"定着レベル {level} のテストメモリ",
                strength=1.0,
                consolidation_level=level,
                access_count=config.consolidation_thresholds[level] if level > 0 else 0,
            )
            created = repository.create(memory)
            memories.append(created)

        # 睡眠フェーズを実行
        sleep_processor.apply_decay_all(agent_id)

        # 減衰後の強度を取得
        updated_strengths = []
        for memory in memories:
            updated = repository.get_by_id(memory.id)
            updated_strengths.append(updated.strength)

        # アサーション: 定着レベルが高いほど強度が高い（減衰が少ない）
        for i in range(len(updated_strengths) - 1):
            assert updated_strengths[i] <= updated_strengths[i + 1], \
                f"Level {i} の強度 ({updated_strengths[i]:.4f}) > Level {i+1} の強度 ({updated_strengths[i+1]:.4f})"

        # 詳細を出力
        print("\n定着レベル別の減衰結果:")
        for i, strength in enumerate(updated_strengths):
            decay_rate = config.get_decay_rate(i)
            print(f"  Level {i}: strength={strength:.4f} (decay_rate={decay_rate:.4f})")

    def test_multiple_decay_cycles(
        self, db, repository, config, sleep_processor, unique_agent_id, cleanup_test_memories
    ):
        """複数回の減衰サイクルで累積減衰するか"""
        agent_id = unique_agent_id

        # 初期メモリを作成
        memory = create_test_memory(
            agent_id=agent_id,
            content="複数サイクル減衰テストメモリ",
            strength=1.0,
            consolidation_level=0,
        )
        created = repository.create(memory)

        # 10回の減衰サイクルを実行
        strength_history = [1.0]
        for _ in range(10):
            sleep_processor.apply_decay_all(agent_id)
            updated = repository.get_by_id(created.id)
            strength_history.append(updated.strength)

        # アサーション: 単調減少
        for i in range(len(strength_history) - 1):
            assert strength_history[i] > strength_history[i + 1], \
                f"サイクル {i} → {i+1} で強度が減少していない"

        # 詳細を出力
        print(f"\n10サイクルの強度推移: {' → '.join(f'{s:.4f}' for s in strength_history)}")
        print(f"最終強度: {strength_history[-1]:.4f} (初期比: {strength_history[-1]/strength_history[0]:.2%})")


class TestArchiveBehavior:
    """アーカイブ動作のテスト

    閾値以下に減衰したメモリがアーカイブされることを検証
    """

    def test_weak_memory_gets_archived(
        self, db, repository, config, sleep_processor, unique_agent_id, cleanup_test_memories
    ):
        """閾値以下のメモリがアーカイブされるか"""
        agent_id = unique_agent_id

        # archive_threshold (0.1) 以下のメモリを作成
        weak_memory = create_test_memory(
            agent_id=agent_id,
            content="弱いメモリ（アーカイブ対象）",
            strength=0.05,  # 閾値以下
        )
        created = repository.create(weak_memory)

        # 強いメモリも作成（比較用）
        strong_memory = create_test_memory(
            agent_id=agent_id,
            content="強いメモリ（アーカイブ対象外）",
            strength=1.0,
        )
        strong_created = repository.create(strong_memory)

        # アーカイブ処理を実行
        archived_count = sleep_processor.archive_weak_memories(agent_id)

        # アサーション
        weak_updated = repository.get_by_id(created.id)
        strong_updated = repository.get_by_id(strong_created.id)

        assert weak_updated.status == "archived", "弱いメモリがアーカイブされていない"
        assert strong_updated.status == "active", "強いメモリが誤ってアーカイブされた"
        assert archived_count >= 1

    def test_decay_eventually_leads_to_archive(
        self, db, repository, config, sleep_processor, unique_agent_id, cleanup_test_memories
    ):
        """減衰を繰り返すと最終的にアーカイブされるか"""
        agent_id = unique_agent_id

        # 初期強度 0.5 のメモリ（Level 0 の減衰率で約 100 サイクル後に閾値に到達）
        memory = create_test_memory(
            agent_id=agent_id,
            content="減衰→アーカイブテストメモリ",
            strength=0.5,
            consolidation_level=0,
        )
        created = repository.create(memory)

        # 減衰→アーカイブのサイクルを繰り返す
        max_cycles = 500
        for cycle in range(max_cycles):
            # 減衰
            sleep_processor.apply_decay_all(agent_id)

            # 現在の強度を確認
            current = repository.get_by_id(created.id)

            # 閾値以下になったらアーカイブ
            if current.strength <= config.archive_threshold:
                sleep_processor.archive_weak_memories(agent_id)

                # アーカイブされたことを確認
                updated = repository.get_by_id(created.id)
                assert updated.status == "archived", \
                    f"サイクル {cycle} で閾値以下になったがアーカイブされていない"

                print(f"\nサイクル {cycle} でアーカイブ: 最終強度={current.strength:.4f}")
                return

        pytest.fail(f"{max_cycles} サイクル以内にアーカイブされなかった")


class TestPersonalityFormation:
    """「個性」形成の統合テスト

    頻繁に使用されるメモリと使用されないメモリの差異を検証
    """

    def test_frequently_used_vs_rarely_used(
        self, db, repository, config, strength_manager, sleep_processor,
        unique_agent_id, cleanup_test_memories
    ):
        """頻繁に使用されるメモリと使用されないメモリの差異"""
        agent_id = unique_agent_id

        # 2つのメモリを作成
        frequently_used = create_test_memory(
            agent_id=agent_id,
            content="頻繁に使用されるメモリ: プロジェクト管理のベストプラクティス",
            strength=1.0,
            consolidation_level=0,
        )
        rarely_used = create_test_memory(
            agent_id=agent_id,
            content="ほとんど使用されないメモリ: レガシーシステムの情報",
            strength=1.0,
            consolidation_level=0,
        )

        freq_created = repository.create(frequently_used)
        rare_created = repository.create(rarely_used)

        # 10回のタスクサイクルをシミュレート
        for cycle in range(10):
            # frequently_used を毎回使用
            strength_manager.mark_as_used(freq_created.id)

            # rarely_used は使用しない

            # 睡眠フェーズ（減衰）
            sleep_processor.apply_decay_all(agent_id)

        # 最終状態を取得
        freq_final = repository.get_by_id(freq_created.id)
        rare_final = repository.get_by_id(rare_created.id)

        # 結果を出力
        print(f"\n「個性」形成テスト結果:")
        print(f"  頻繁に使用: strength={freq_final.strength:.3f}, "
              f"access_count={freq_final.access_count}, "
              f"consolidation_level={freq_final.consolidation_level}")
        print(f"  ほとんど未使用: strength={rare_final.strength:.3f}, "
              f"access_count={rare_final.access_count}, "
              f"consolidation_level={rare_final.consolidation_level}")

        # アサーション
        # 1. 頻繁に使用されるメモリの方が強度が高い
        assert freq_final.strength > rare_final.strength, \
            "頻繁に使用されるメモリの強度が低い"

        # 2. 頻繁に使用されるメモリの方が定着レベルが高い
        assert freq_final.consolidation_level >= rare_final.consolidation_level, \
            "頻繁に使用されるメモリの定着レベルが低い"

        # 3. 使用回数の差
        assert freq_final.access_count > rare_final.access_count, \
            "頻繁に使用されるメモリのアクセス回数が少ない"

    def test_learning_and_forgetting_cycle(
        self, db, repository, config, strength_manager, sleep_processor,
        unique_agent_id, cleanup_test_memories
    ):
        """学習と忘却のサイクル検証

        新しい知識を学習し、古い知識を忘却する過程をシミュレート
        """
        agent_id = unique_agent_id

        # フェーズ1: 古い知識を作成
        old_knowledge = create_test_memory(
            agent_id=agent_id,
            content="古い知識: 旧バージョンの API 仕様",
            strength=1.0,
        )
        old_created = repository.create(old_knowledge)

        # フェーズ2: しばらく使用しない（5サイクル減衰）
        for _ in range(5):
            sleep_processor.apply_decay_all(agent_id)

        old_after_decay = repository.get_by_id(old_created.id)
        print(f"\n古い知識の減衰後: strength={old_after_decay.strength:.3f}")

        # フェーズ3: 新しい知識を学習
        new_knowledge = create_test_memory(
            agent_id=agent_id,
            content="新しい知識: 新バージョンの API 仕様",
            strength=1.0,
        )
        new_created = repository.create(new_knowledge)

        # フェーズ4: 新しい知識を使用し、古い知識は使用しない（5サイクル）
        for _ in range(5):
            strength_manager.mark_as_used(new_created.id)
            sleep_processor.apply_decay_all(agent_id)

        # 最終状態を取得
        old_final = repository.get_by_id(old_created.id)
        new_final = repository.get_by_id(new_created.id)

        print(f"古い知識の最終状態: strength={old_final.strength:.3f}, status={old_final.status}")
        print(f"新しい知識の最終状態: strength={new_final.strength:.3f}, access_count={new_final.access_count}")

        # アサーション
        # 新しい知識の方が強い
        assert new_final.strength > old_final.strength, \
            "新しい知識の強度が古い知識より低い"

        # 古い知識は減衰している（初期値1.0からの減少を確認）
        # 注: Level 0 の減衰率は約0.9949/タスクなので、10サイクルで約5%減衰
        assert old_final.strength < 1.0, \
            "古い知識が減衰していない"


class TestSearchConsistency:
    """検索結果の一貫性テスト

    同じクエリに対して、強化されたメモリが優先されることを検証
    （注: 実際のベクトル検索は使用しない簡易版）
    """

    def test_strength_affects_ranking(
        self, db, repository, unique_agent_id, cleanup_test_memories
    ):
        """強度が検索ランキングに影響するか（シミュレーション）

        強度順でソートした場合、頻繁に使用されるメモリが上位になることを検証
        """
        agent_id = unique_agent_id

        # 異なる強度のメモリを作成
        memories = []
        for i, strength in enumerate([0.3, 0.8, 0.5, 1.2, 0.1]):
            memory = create_test_memory(
                agent_id=agent_id,
                content=f"ランキングテストメモリ {i}: 強度 {strength}",
                strength=strength,
            )
            created = repository.create(memory)
            memories.append(created)

        # 強度順で取得（検索ランキングのシミュレーション）
        with db.get_cursor() as cur:
            cur.execute(
                """
                SELECT id, strength, content
                FROM agent_memory
                WHERE agent_id = %s AND status = 'active'
                ORDER BY strength DESC
                """,
                (agent_id,)
            )
            ranked = cur.fetchall()

        # 結果を出力
        print("\n強度によるランキング:")
        for i, (mem_id, strength, content) in enumerate(ranked):
            print(f"  {i+1}. strength={strength:.2f}: {content[:50]}...")

        # アサーション: 強度順で並んでいること
        strengths = [row[1] for row in ranked]
        assert strengths == sorted(strengths, reverse=True), \
            "強度順でソートされていない"
