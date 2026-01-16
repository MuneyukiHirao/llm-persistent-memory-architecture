# 検証テスト用共通フィクスチャ
"""
Phase 1「個性」形成検証用のpytestフィクスチャ

実DBを使用した統合テストのための共通セットアップ/クリーンアップ処理を提供。
テストデータは各テストでセットアップ/クリーンアップされる。
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Generator, List
from uuid import uuid4

import pytest

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.config.phase1_config import Phase1Config
from src.db.connection import DatabaseConnection
from src.core.memory_repository import MemoryRepository
from src.core.strength_manager import StrengthManager
from src.core.sleep_processor import SleepPhaseProcessor
from src.models.memory import AgentMemory


# テスト用エージェントID（他のテストと衝突しないプレフィックス）
TEST_AGENT_PREFIX = "verify_agent_"


@pytest.fixture(scope="module")
def db() -> Generator[DatabaseConnection, None, None]:
    """データベース接続フィクスチャ（モジュールスコープ）

    テスト用の .env ファイルから DATABASE_URL を読み込み、
    テストモジュール終了時に接続をクローズする。
    """
    # .env ファイルから環境変数を読み込む
    env_file = os.path.join(project_root, ".env")
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())

    db = DatabaseConnection()

    # 接続確認
    if not db.health_check():
        pytest.skip("データベースに接続できません")

    yield db
    db.close()


@pytest.fixture(scope="module")
def config() -> Phase1Config:
    """Phase1設定フィクスチャ"""
    return Phase1Config()


@pytest.fixture(scope="module")
def repository(db: DatabaseConnection, config: Phase1Config) -> MemoryRepository:
    """MemoryRepositoryフィクスチャ"""
    return MemoryRepository(db, config)


@pytest.fixture(scope="module")
def strength_manager(repository: MemoryRepository, config: Phase1Config) -> StrengthManager:
    """StrengthManagerフィクスチャ"""
    return StrengthManager(repository, config)


@pytest.fixture(scope="module")
def sleep_processor(db: DatabaseConnection, config: Phase1Config) -> SleepPhaseProcessor:
    """SleepPhaseProcessorフィクスチャ"""
    return SleepPhaseProcessor(db, config)


@pytest.fixture
def unique_agent_id() -> str:
    """一意のエージェントIDを生成"""
    return f"{TEST_AGENT_PREFIX}{uuid4().hex[:8]}"


@pytest.fixture
def cleanup_test_memories(db: DatabaseConnection):
    """テスト後にテストメモリをクリーンアップ

    テスト用プレフィックスで始まるエージェントIDのメモリを全て削除。
    """
    yield

    # テスト後のクリーンアップ
    with db.get_cursor() as cur:
        cur.execute(
            "DELETE FROM agent_memory WHERE agent_id LIKE %s",
            (f"{TEST_AGENT_PREFIX}%",)
        )


def create_test_memory(
    agent_id: str,
    content: str,
    strength: float = 1.0,
    access_count: int = 0,
    candidate_count: int = 0,
    consolidation_level: int = 0,
    status: str = "active",
    created_days_ago: int = 0,
) -> AgentMemory:
    """テスト用メモリを作成するヘルパー関数

    Args:
        agent_id: エージェントID
        content: メモリの内容
        strength: 強度（デフォルト: 1.0）
        access_count: 使用回数（デフォルト: 0）
        candidate_count: 候補回数（デフォルト: 0）
        consolidation_level: 定着レベル（デフォルト: 0）
        status: ステータス（デフォルト: "active"）
        created_days_ago: 何日前に作成されたか（デフォルト: 0 = 今）

    Returns:
        AgentMemory インスタンス
    """
    now = datetime.now()
    created_at = now - timedelta(days=created_days_ago)

    return AgentMemory(
        id=uuid4(),
        agent_id=agent_id,
        content=content,
        embedding=None,  # 検証テストではembeddingは不要
        tags=[],
        scope_level="project",
        scope_domain=None,
        scope_project="llm-persistent-memory-phase1",
        strength=strength,
        strength_by_perspective={},
        access_count=access_count,
        candidate_count=candidate_count,
        last_accessed_at=now if access_count > 0 else None,
        impact_score=0.0,
        consolidation_level=consolidation_level,
        learning=None,
        status=status,
        source="task",
        created_at=created_at,
        updated_at=now,
        last_decay_at=None,
    )


def create_test_memories_batch(
    repository: MemoryRepository,
    agent_id: str,
    count: int,
    strength_range: tuple = (0.1, 1.5),
    access_count_range: tuple = (0, 100),
    candidate_count_range: tuple = (0, 150),
) -> List[AgentMemory]:
    """テスト用メモリをバッチで作成するヘルパー関数

    均等に分布したメモリを生成してDBに保存する。

    Args:
        repository: MemoryRepository インスタンス
        agent_id: エージェントID
        count: 作成するメモリ数
        strength_range: 強度の範囲 (min, max)
        access_count_range: 使用回数の範囲 (min, max)
        candidate_count_range: 候補回数の範囲 (min, max)

    Returns:
        作成されたメモリのリスト
    """
    import random

    memories = []
    for i in range(count):
        # 均等に分布した値を生成
        strength = strength_range[0] + (strength_range[1] - strength_range[0]) * (i / max(count - 1, 1))
        access_count = int(access_count_range[0] + (access_count_range[1] - access_count_range[0]) * random.random())
        candidate_count = int(candidate_count_range[0] + (candidate_count_range[1] - candidate_count_range[0]) * random.random())

        # candidate_count >= access_count を保証
        if candidate_count < access_count:
            candidate_count = access_count + int(random.random() * 50)

        memory = create_test_memory(
            agent_id=agent_id,
            content=f"テストメモリ {i+1}: 検証用のサンプルコンテンツです。{random.randint(1000, 9999)}",
            strength=strength,
            access_count=access_count,
            candidate_count=candidate_count,
            consolidation_level=0,  # DBに保存後に計算される
            created_days_ago=random.randint(0, 30),
        )

        created = repository.create(memory)
        memories.append(created)

    return memories
