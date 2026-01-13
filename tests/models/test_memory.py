# AgentMemory モデルのテスト
"""
AgentMemory dataclass の単体テスト

テスト観点:
- 21フィールドの初期化
- to_dict() メソッドの動作
- from_row() ファクトリメソッド（tuple/dict両対応）
- create() ファクトリメソッド
- create_from_education() ファクトリメソッド
- copy_with() メソッド
- 2段階強化関連フィールドのデフォルト値
"""

from datetime import datetime, timedelta
from uuid import UUID, uuid4

import pytest

from src.models.memory import AgentMemory


class TestAgentMemoryInitialization:
    """AgentMemory dataclass の初期化テスト"""

    def test_init_with_required_fields(self):
        """必須フィールドのみでの初期化"""
        memory_id = uuid4()
        now = datetime.now()

        memory = AgentMemory(
            id=memory_id,
            agent_id="test_agent",
            content="テスト用の記憶内容",
            created_at=now,
            updated_at=now,
        )

        assert memory.id == memory_id
        assert memory.agent_id == "test_agent"
        assert memory.content == "テスト用の記憶内容"
        assert memory.created_at == now
        assert memory.updated_at == now

    def test_init_all_21_fields(self):
        """全21フィールドの初期化"""
        memory_id = uuid4()
        now = datetime.now()
        embedding = [0.1] * 1536
        tags = ["tag1", "tag2"]
        strength_by_perspective = {"コスト": 1.2, "納期": 0.8}
        learnings = {"コスト": "緊急調達で15%コスト増", "納期": "2週間バッファが必要"}

        memory = AgentMemory(
            id=memory_id,
            agent_id="test_agent",
            content="テスト用の記憶内容",
            embedding=embedding,
            tags=tags,
            scope_level="domain",
            scope_domain="procurement",
            scope_project="project_001",
            strength=1.5,
            strength_by_perspective=strength_by_perspective,
            access_count=5,
            candidate_count=10,
            last_accessed_at=now,
            impact_score=2.0,
            consolidation_level=2,
            learnings=learnings,
            status="active",
            source="task",
            created_at=now,
            updated_at=now,
            last_decay_at=now,
        )

        # 全21フィールドを検証
        assert memory.id == memory_id
        assert memory.agent_id == "test_agent"
        assert memory.content == "テスト用の記憶内容"
        assert memory.embedding == embedding
        assert memory.tags == tags
        assert memory.scope_level == "domain"
        assert memory.scope_domain == "procurement"
        assert memory.scope_project == "project_001"
        assert memory.strength == 1.5
        assert memory.strength_by_perspective == strength_by_perspective
        assert memory.access_count == 5
        assert memory.candidate_count == 10
        assert memory.last_accessed_at == now
        assert memory.impact_score == 2.0
        assert memory.consolidation_level == 2
        assert memory.learnings == learnings
        assert memory.status == "active"
        assert memory.source == "task"
        assert memory.created_at == now
        assert memory.updated_at == now
        assert memory.last_decay_at == now

    def test_default_values(self):
        """デフォルト値の確認"""
        memory_id = uuid4()
        now = datetime.now()

        memory = AgentMemory(
            id=memory_id,
            agent_id="test_agent",
            content="テスト用の記憶内容",
            created_at=now,
            updated_at=now,
        )

        # オプショナルフィールドのデフォルト値
        assert memory.embedding is None
        assert memory.tags == []
        assert memory.scope_level == "project"
        assert memory.scope_domain is None
        assert memory.scope_project is None
        assert memory.strength == 1.0
        assert memory.strength_by_perspective == {}
        assert memory.access_count == 0
        assert memory.candidate_count == 0
        assert memory.last_accessed_at is None
        assert memory.impact_score == 0.0
        assert memory.consolidation_level == 0
        assert memory.learnings == {}
        assert memory.status == "active"
        assert memory.source is None
        assert memory.last_decay_at is None


class TestToDictMethod:
    """to_dict() メソッドのテスト"""

    @pytest.fixture
    def sample_memory(self) -> AgentMemory:
        """テスト用のメモリ"""
        return AgentMemory(
            id=UUID("12345678-1234-5678-1234-567812345678"),
            agent_id="test_agent",
            content="テスト用の記憶内容",
            embedding=[0.1, 0.2, 0.3],
            tags=["tag1", "tag2"],
            scope_level="domain",
            scope_domain="procurement",
            scope_project="project_001",
            strength=1.5,
            strength_by_perspective={"コスト": 1.2, "納期": 0.8},
            access_count=5,
            candidate_count=10,
            last_accessed_at=datetime(2024, 1, 15, 12, 0, 0),
            impact_score=2.0,
            consolidation_level=2,
            learnings={"コスト": "緊急調達で15%コスト増"},
            status="active",
            source="task",
            created_at=datetime(2024, 1, 1, 10, 0, 0),
            updated_at=datetime(2024, 1, 10, 14, 30, 0),
            last_decay_at=datetime(2024, 1, 8, 0, 0, 0),
        )

    def test_to_dict_returns_all_fields(self, sample_memory: AgentMemory):
        """to_dict() が全フィールドを含む辞書を返す"""
        result = sample_memory.to_dict()

        assert len(result) == 21
        assert "id" in result
        assert "agent_id" in result
        assert "content" in result
        assert "embedding" in result
        assert "tags" in result
        assert "scope_level" in result
        assert "scope_domain" in result
        assert "scope_project" in result
        assert "strength" in result
        assert "strength_by_perspective" in result
        assert "access_count" in result
        assert "candidate_count" in result
        assert "last_accessed_at" in result
        assert "impact_score" in result
        assert "consolidation_level" in result
        assert "learnings" in result
        assert "status" in result
        assert "source" in result
        assert "created_at" in result
        assert "updated_at" in result
        assert "last_decay_at" in result

    def test_to_dict_uuid_conversion(self, sample_memory: AgentMemory):
        """UUID が文字列に変換される"""
        result = sample_memory.to_dict()

        assert result["id"] == "12345678-1234-5678-1234-567812345678"
        assert isinstance(result["id"], str)

    def test_to_dict_datetime_conversion(self, sample_memory: AgentMemory):
        """datetime が ISO8601 形式の文字列に変換される"""
        result = sample_memory.to_dict()

        assert result["created_at"] == "2024-01-01T10:00:00"
        assert result["updated_at"] == "2024-01-10T14:30:00"
        assert result["last_accessed_at"] == "2024-01-15T12:00:00"
        assert result["last_decay_at"] == "2024-01-08T00:00:00"

    def test_to_dict_none_datetime_handling(self):
        """None の datetime フィールドが None として保持される"""
        memory = AgentMemory(
            id=uuid4(),
            agent_id="test_agent",
            content="テスト",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        result = memory.to_dict()

        assert result["last_accessed_at"] is None
        assert result["last_decay_at"] is None

    def test_to_dict_embedding_preserved(self, sample_memory: AgentMemory):
        """embedding がそのまま保持される"""
        result = sample_memory.to_dict()

        assert result["embedding"] == [0.1, 0.2, 0.3]

    def test_to_dict_none_embedding(self):
        """None の embedding が None として保持される"""
        memory = AgentMemory(
            id=uuid4(),
            agent_id="test_agent",
            content="テスト",
            embedding=None,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        result = memory.to_dict()

        assert result["embedding"] is None


class TestFromRowMethod:
    """from_row() ファクトリメソッドのテスト"""

    def test_from_row_with_dict(self):
        """dict形式の行データからインスタンス生成"""
        memory_id = uuid4()
        now = datetime.now()

        row = {
            "id": memory_id,
            "agent_id": "test_agent",
            "content": "テスト用の記憶内容",
            "embedding": [0.1, 0.2, 0.3],
            "tags": ["tag1", "tag2"],
            "scope_level": "domain",
            "scope_domain": "procurement",
            "scope_project": "project_001",
            "strength": 1.5,
            "strength_by_perspective": {"コスト": 1.2},
            "access_count": 5,
            "candidate_count": 10,
            "last_accessed_at": now,
            "impact_score": 2.0,
            "consolidation_level": 2,
            "learnings": {"コスト": "緊急調達で15%コスト増"},
            "status": "active",
            "source": "task",
            "created_at": now,
            "updated_at": now,
            "last_decay_at": now,
        }

        memory = AgentMemory.from_row(row)

        assert memory.id == memory_id
        assert memory.agent_id == "test_agent"
        assert memory.content == "テスト用の記憶内容"
        assert memory.embedding == [0.1, 0.2, 0.3]
        assert memory.tags == ["tag1", "tag2"]
        assert memory.scope_level == "domain"
        assert memory.scope_domain == "procurement"
        assert memory.scope_project == "project_001"
        assert memory.strength == 1.5
        assert memory.strength_by_perspective == {"コスト": 1.2}
        assert memory.access_count == 5
        assert memory.candidate_count == 10
        assert memory.last_accessed_at == now
        assert memory.impact_score == 2.0
        assert memory.consolidation_level == 2
        assert memory.learnings == {"コスト": "緊急調達で15%コスト増"}
        assert memory.status == "active"
        assert memory.source == "task"
        assert memory.created_at == now
        assert memory.updated_at == now
        assert memory.last_decay_at == now

    def test_from_row_with_tuple(self):
        """tuple形式の行データからインスタンス生成"""
        memory_id = uuid4()
        now = datetime.now()

        # カラム順序: id, agent_id, content, embedding, tags,
        #            scope_level, scope_domain, scope_project,
        #            strength, strength_by_perspective,
        #            access_count, candidate_count, last_accessed_at,
        #            impact_score, consolidation_level, learnings,
        #            status, source, created_at, updated_at, last_decay_at
        row = (
            memory_id,                    # 0: id
            "test_agent",                 # 1: agent_id
            "テスト用の記憶内容",          # 2: content
            [0.1, 0.2, 0.3],              # 3: embedding
            ["tag1", "tag2"],             # 4: tags
            "domain",                     # 5: scope_level
            "procurement",                # 6: scope_domain
            "project_001",                # 7: scope_project
            1.5,                          # 8: strength
            {"コスト": 1.2},              # 9: strength_by_perspective
            5,                            # 10: access_count
            10,                           # 11: candidate_count
            now,                          # 12: last_accessed_at
            2.0,                          # 13: impact_score
            2,                            # 14: consolidation_level
            {"コスト": "緊急調達で15%"},   # 15: learnings
            "active",                     # 16: status
            "task",                       # 17: source
            now,                          # 18: created_at
            now,                          # 19: updated_at
            now,                          # 20: last_decay_at
        )

        memory = AgentMemory.from_row(row)

        assert memory.id == memory_id
        assert memory.agent_id == "test_agent"
        assert memory.content == "テスト用の記憶内容"
        assert memory.embedding == [0.1, 0.2, 0.3]
        assert memory.tags == ["tag1", "tag2"]
        assert memory.scope_level == "domain"
        assert memory.scope_domain == "procurement"
        assert memory.scope_project == "project_001"
        assert memory.strength == 1.5

    def test_from_row_dict_with_string_uuid(self):
        """dict形式でUUIDが文字列の場合"""
        now = datetime.now()
        row = {
            "id": "12345678-1234-5678-1234-567812345678",
            "agent_id": "test_agent",
            "content": "テスト",
            "created_at": now,
            "updated_at": now,
        }

        memory = AgentMemory.from_row(row)

        assert isinstance(memory.id, UUID)
        assert str(memory.id) == "12345678-1234-5678-1234-567812345678"

    def test_from_row_tuple_with_string_uuid(self):
        """tuple形式でUUIDが文字列の場合"""
        now = datetime.now()
        row = (
            "12345678-1234-5678-1234-567812345678",  # id (string)
            "test_agent",
            "テスト",
            None, [], None, None, None, None, None,
            None, None, None, None, None, None,
            None, None, now, now, None,
        )

        memory = AgentMemory.from_row(row)

        assert isinstance(memory.id, UUID)
        assert str(memory.id) == "12345678-1234-5678-1234-567812345678"

    def test_from_row_dict_with_null_values(self):
        """dict形式でNULL値のハンドリング"""
        now = datetime.now()
        row = {
            "id": uuid4(),
            "agent_id": "test_agent",
            "content": "テスト",
            "embedding": None,
            "tags": None,
            "scope_level": None,
            "scope_domain": None,
            "scope_project": None,
            "strength": None,
            "strength_by_perspective": None,
            "access_count": None,
            "candidate_count": None,
            "last_accessed_at": None,
            "impact_score": None,
            "consolidation_level": None,
            "learnings": None,
            "status": None,
            "source": None,
            "created_at": now,
            "updated_at": now,
            "last_decay_at": None,
        }

        memory = AgentMemory.from_row(row)

        # デフォルト値が適用される
        assert memory.embedding is None
        assert memory.tags == []
        assert memory.scope_level == "project"
        assert memory.strength == 1.0
        assert memory.strength_by_perspective == {}
        assert memory.access_count == 0
        assert memory.candidate_count == 0
        assert memory.impact_score == 0.0
        assert memory.consolidation_level == 0
        assert memory.learnings == {}
        assert memory.status == "active"

    def test_from_row_tuple_with_null_values(self):
        """tuple形式でNULL値のハンドリング"""
        now = datetime.now()
        row = (
            uuid4(),           # 0: id
            "test_agent",      # 1: agent_id
            "テスト",          # 2: content
            None,              # 3: embedding
            None,              # 4: tags
            None,              # 5: scope_level
            None,              # 6: scope_domain
            None,              # 7: scope_project
            None,              # 8: strength
            None,              # 9: strength_by_perspective
            None,              # 10: access_count
            None,              # 11: candidate_count
            None,              # 12: last_accessed_at
            None,              # 13: impact_score
            None,              # 14: consolidation_level
            None,              # 15: learnings
            None,              # 16: status
            None,              # 17: source
            now,               # 18: created_at
            now,               # 19: updated_at
            None,              # 20: last_decay_at
        )

        memory = AgentMemory.from_row(row)

        # デフォルト値が適用される
        assert memory.embedding is None
        assert memory.tags == []
        assert memory.scope_level == "project"
        assert memory.strength == 1.0
        assert memory.strength_by_perspective == {}
        assert memory.access_count == 0
        assert memory.candidate_count == 0


class TestCreateFactoryMethod:
    """create() ファクトリメソッドのテスト"""

    def test_create_with_required_fields(self):
        """必須フィールドのみでの生成"""
        memory = AgentMemory.create(
            agent_id="test_agent",
            content="テスト用の記憶内容",
        )

        assert isinstance(memory.id, UUID)
        assert memory.agent_id == "test_agent"
        assert memory.content == "テスト用の記憶内容"

    def test_create_generates_uuid(self):
        """UUIDが自動生成される"""
        memory1 = AgentMemory.create(
            agent_id="test_agent",
            content="テスト1",
        )
        memory2 = AgentMemory.create(
            agent_id="test_agent",
            content="テスト2",
        )

        assert memory1.id != memory2.id
        assert isinstance(memory1.id, UUID)
        assert isinstance(memory2.id, UUID)

    def test_create_sets_default_values(self):
        """デフォルト値が正しく設定される"""
        memory = AgentMemory.create(
            agent_id="test_agent",
            content="テスト",
        )

        assert memory.embedding is None
        assert memory.tags == []
        assert memory.scope_level == "project"
        assert memory.scope_domain is None
        assert memory.scope_project is None
        assert memory.strength == 1.0
        assert memory.strength_by_perspective == {}
        assert memory.access_count == 0
        assert memory.candidate_count == 0
        assert memory.last_accessed_at is None
        assert memory.impact_score == 0.0
        assert memory.consolidation_level == 0
        assert memory.learnings == {}
        assert memory.status == "active"
        assert memory.source is None
        assert memory.last_decay_at is None

    def test_create_sets_timestamps(self):
        """created_at と updated_at が同じ値で設定される"""
        before = datetime.now()
        memory = AgentMemory.create(
            agent_id="test_agent",
            content="テスト",
        )
        after = datetime.now()

        assert before <= memory.created_at <= after
        assert memory.created_at == memory.updated_at

    def test_create_with_all_optional_fields(self):
        """全てのオプショナルフィールドを指定して生成"""
        embedding = [0.1] * 1536
        tags = ["tag1", "tag2"]
        strength_by_perspective = {"コスト": 1.2, "納期": 0.8}
        learnings = {"コスト": "緊急調達で15%コスト増"}

        memory = AgentMemory.create(
            agent_id="test_agent",
            content="テスト用の記憶内容",
            embedding=embedding,
            tags=tags,
            scope_level="domain",
            scope_domain="procurement",
            scope_project="project_001",
            strength=0.8,
            strength_by_perspective=strength_by_perspective,
            learnings=learnings,
            source="task",
        )

        assert memory.embedding == embedding
        assert memory.tags == tags
        assert memory.scope_level == "domain"
        assert memory.scope_domain == "procurement"
        assert memory.scope_project == "project_001"
        assert memory.strength == 0.8
        assert memory.strength_by_perspective == strength_by_perspective
        assert memory.learnings == learnings
        assert memory.source == "task"


class TestCreateFromEducationMethod:
    """create_from_education() ファクトリメソッドのテスト"""

    def test_education_initial_strength(self):
        """教育プロセスの初期強度が0.5"""
        memory = AgentMemory.create_from_education(
            agent_id="test_agent",
            content="教育コンテンツから学んだ内容",
        )

        assert memory.strength == 0.5

    def test_education_source_set(self):
        """source が "education" に設定される"""
        memory = AgentMemory.create_from_education(
            agent_id="test_agent",
            content="教育コンテンツから学んだ内容",
        )

        assert memory.source == "education"

    def test_education_with_optional_fields(self):
        """オプショナルフィールドを指定して教育メモリを生成"""
        memory = AgentMemory.create_from_education(
            agent_id="test_agent",
            content="教育コンテンツから学んだ内容",
            tags=["学習", "基礎知識"],
            scope_level="universal",
            strength_by_perspective={"理解度": 0.5},
            learnings={"基礎": "基本的な概念の理解"},
        )

        assert memory.tags == ["学習", "基礎知識"]
        assert memory.scope_level == "universal"
        assert memory.strength_by_perspective == {"理解度": 0.5}
        assert memory.learnings == {"基礎": "基本的な概念の理解"}
        assert memory.strength == 0.5  # 常に0.5
        assert memory.source == "education"  # 常にeducation


class TestCopyWithMethod:
    """copy_with() メソッドのテスト"""

    @pytest.fixture
    def original_memory(self) -> AgentMemory:
        """テスト用の元メモリ"""
        return AgentMemory.create(
            agent_id="test_agent",
            content="元の記憶内容",
            strength=1.0,
            tags=["tag1"],
        )

    def test_copy_with_single_field(self, original_memory: AgentMemory):
        """単一フィールドの変更"""
        updated = original_memory.copy_with(strength=1.5)

        assert updated.strength == 1.5
        assert updated.content == original_memory.content
        assert updated.agent_id == original_memory.agent_id
        assert updated.id == original_memory.id

    def test_copy_with_multiple_fields(self, original_memory: AgentMemory):
        """複数フィールドの変更"""
        now = datetime.now()
        updated = original_memory.copy_with(
            strength=1.5,
            access_count=1,
            last_accessed_at=now,
            updated_at=now,
        )

        assert updated.strength == 1.5
        assert updated.access_count == 1
        assert updated.last_accessed_at == now
        assert updated.updated_at == now

    def test_copy_with_preserves_unchanged(self, original_memory: AgentMemory):
        """変更されていないフィールドは保持される"""
        updated = original_memory.copy_with(strength=2.0)

        assert updated.id == original_memory.id
        assert updated.agent_id == original_memory.agent_id
        assert updated.content == original_memory.content
        assert updated.tags == original_memory.tags
        assert updated.created_at == original_memory.created_at

    def test_copy_with_returns_new_instance(self, original_memory: AgentMemory):
        """新しいインスタンスが返される（イミュータブル）"""
        updated = original_memory.copy_with(strength=2.0)

        assert updated is not original_memory
        assert original_memory.strength == 1.0  # 元は変更されない


class TestTwoStageStrengthening:
    """2段階強化関連フィールドのテスト"""

    def test_default_candidate_count(self):
        """candidate_count のデフォルト値が0"""
        memory = AgentMemory.create(
            agent_id="test_agent",
            content="テスト",
        )

        assert memory.candidate_count == 0

    def test_default_access_count(self):
        """access_count のデフォルト値が0"""
        memory = AgentMemory.create(
            agent_id="test_agent",
            content="テスト",
        )

        assert memory.access_count == 0

    def test_default_strength(self):
        """通常作成時の初期強度が1.0"""
        memory = AgentMemory.create(
            agent_id="test_agent",
            content="テスト",
        )

        assert memory.strength == 1.0

    def test_default_strength_by_perspective(self):
        """strength_by_perspective のデフォルト値が空辞書"""
        memory = AgentMemory.create(
            agent_id="test_agent",
            content="テスト",
        )

        assert memory.strength_by_perspective == {}

    def test_default_consolidation_level(self):
        """consolidation_level のデフォルト値が0"""
        memory = AgentMemory.create(
            agent_id="test_agent",
            content="テスト",
        )

        assert memory.consolidation_level == 0

    def test_default_last_accessed_at(self):
        """last_accessed_at のデフォルト値がNone"""
        memory = AgentMemory.create(
            agent_id="test_agent",
            content="テスト",
        )

        assert memory.last_accessed_at is None

    def test_two_stage_strength_simulation(self):
        """2段階強化のシミュレーション"""
        # 新規メモリ作成
        memory = AgentMemory.create(
            agent_id="test_agent",
            content="テスト用の記憶",
        )

        # Stage 1: 検索候補になった
        memory = memory.copy_with(
            candidate_count=memory.candidate_count + 1,
        )
        assert memory.candidate_count == 1
        assert memory.access_count == 0
        assert memory.strength == 1.0  # まだ強化されない

        # Stage 2: 実際に使用された
        now = datetime.now()
        memory = memory.copy_with(
            access_count=memory.access_count + 1,
            strength=memory.strength + 0.1,
            last_accessed_at=now,
            updated_at=now,
        )
        assert memory.candidate_count == 1
        assert memory.access_count == 1
        assert memory.strength == 1.1  # 強化された
        assert memory.last_accessed_at == now


class TestScopeFields:
    """スコープ関連フィールドのテスト"""

    def test_project_scope(self):
        """プロジェクトスコープ"""
        memory = AgentMemory.create(
            agent_id="test_agent",
            content="プロジェクト固有の知識",
            scope_level="project",
            scope_project="llm-persistent-memory-phase1",
        )

        assert memory.scope_level == "project"
        assert memory.scope_project == "llm-persistent-memory-phase1"

    def test_domain_scope(self):
        """ドメインスコープ"""
        memory = AgentMemory.create(
            agent_id="test_agent",
            content="調達ドメインの知識",
            scope_level="domain",
            scope_domain="procurement",
        )

        assert memory.scope_level == "domain"
        assert memory.scope_domain == "procurement"

    def test_universal_scope(self):
        """ユニバーサルスコープ"""
        memory = AgentMemory.create(
            agent_id="test_agent",
            content="普遍的な知識",
            scope_level="universal",
        )

        assert memory.scope_level == "universal"
        assert memory.scope_domain is None
        assert memory.scope_project is None


class TestReprMethod:
    """__repr__ メソッドのテスト"""

    def test_repr_format(self):
        """__repr__ の出力形式"""
        memory = AgentMemory.create(
            agent_id="test_agent",
            content="テスト用の記憶内容です。これは長いテキストになる可能性があります。",
        )

        repr_str = repr(memory)

        assert "AgentMemory(" in repr_str
        assert "agent_id='test_agent'" in repr_str
        assert "strength=1.00" in repr_str
        assert "access_count=0" in repr_str
        assert "status='active'" in repr_str
        # 長いコンテンツは省略される
        assert "..." in repr_str

    def test_repr_short_content(self):
        """短いコンテンツの場合も動作する"""
        memory = AgentMemory.create(
            agent_id="test_agent",
            content="短い",
        )

        repr_str = repr(memory)

        assert "AgentMemory(" in repr_str
        assert "content=" in repr_str


class TestEdgeCases:
    """エッジケースのテスト"""

    def test_empty_content(self):
        """空のコンテンツ"""
        memory = AgentMemory.create(
            agent_id="test_agent",
            content="",
        )

        assert memory.content == ""

    def test_unicode_content(self):
        """Unicode文字を含むコンテンツ"""
        memory = AgentMemory.create(
            agent_id="test_agent",
            content="日本語テスト 🎉 特殊文字 ™©®",
        )

        assert memory.content == "日本語テスト 🎉 特殊文字 ™©®"

    def test_very_long_content(self):
        """非常に長いコンテンツ"""
        long_content = "a" * 10000
        memory = AgentMemory.create(
            agent_id="test_agent",
            content=long_content,
        )

        assert memory.content == long_content
        assert len(memory.content) == 10000

    def test_empty_embedding(self):
        """空のembedding配列"""
        memory = AgentMemory.create(
            agent_id="test_agent",
            content="テスト",
            embedding=[],
        )

        assert memory.embedding == []

    def test_large_embedding(self):
        """大きなembedding（1536次元）"""
        embedding = [0.1] * 1536
        memory = AgentMemory.create(
            agent_id="test_agent",
            content="テスト",
            embedding=embedding,
        )

        assert len(memory.embedding) == 1536

    def test_negative_strength(self):
        """負の強度（境界値）"""
        memory = AgentMemory.create(
            agent_id="test_agent",
            content="テスト",
            strength=-0.5,
        )

        # 負の値も許容される（バリデーションは上位レイヤーで行う）
        assert memory.strength == -0.5

    def test_zero_strength(self):
        """強度0（境界値）"""
        memory = AgentMemory.create(
            agent_id="test_agent",
            content="テスト",
            strength=0.0,
        )

        assert memory.strength == 0.0

    def test_high_strength(self):
        """高い強度（上限なし）"""
        memory = AgentMemory.create(
            agent_id="test_agent",
            content="テスト",
            strength=10.0,
        )

        assert memory.strength == 10.0
