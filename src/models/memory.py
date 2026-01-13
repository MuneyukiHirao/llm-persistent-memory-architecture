# メモリモデル定義
# agent_memory テーブルに対応するデータクラス
# 実装仕様: docs/phase1-implementation-spec.ja.md セクション3.1

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4


@dataclass
class AgentMemory:
    """agent_memory テーブルに対応するデータクラス

    仕様書参照:
    - セクション3.1: メインテーブル agent_memory
    - セクション3.2: 観点別データの保存（JSONB非正規化）
    - セクション3.3: スコープによる知識の階層管理
    - セクション3.5: タイムスタンプ設計

    2段階強化の設計:
    - candidate_count: 検索候補として参照された回数
    - access_count: 実際に使用された回数
    - strength: 全体的な強度（使用時に強化される）
    - strength_by_perspective: 観点別の強度（該当観点のみ強化される）
    """

    # === 識別子 ===
    id: UUID
    """記憶の一意識別子"""

    agent_id: str
    """所属エージェントのID"""

    # === コンテンツ ===
    content: str
    """記憶の内容（テキスト）"""

    embedding: Optional[List[float]] = None
    """ベクトル表現 vector(1536)"""

    tags: List[str] = field(default_factory=list)
    """分類タグ"""

    # === スコープ（知識の階層管理） ===
    scope_level: str = "project"
    """スコープレベル: universal / domain / project"""

    scope_domain: Optional[str] = None
    """ドメイン名（domain レベルの場合に設定）"""

    scope_project: Optional[str] = None
    """プロジェクトID（project レベルの場合に設定）"""

    # === 強度管理 ===
    strength: float = 1.0
    """全体的な強度（0.0〜上限なし）
    - 新規作成時: 1.0
    - 教育プロセスで読んだだけ: 0.5
    - 使用時: +0.1
    - archive_threshold (0.1) 以下でアーカイブ
    """

    strength_by_perspective: Dict[str, float] = field(default_factory=dict)
    """観点別の強度（JSONB）
    例: {"コスト": 1.2, "納期": 0.8, ...}
    """

    # === 使用追跡（2段階強化） ===
    access_count: int = 0
    """実際に使用された回数
    - 実使用時にインクリメント
    - 定着レベルの計算に使用
    """

    candidate_count: int = 0
    """検索候補として参照された回数
    - 検索候補になっただけでインクリメント
    - 実際の使用とは区別して追跡
    """

    last_accessed_at: Optional[datetime] = None
    """最後に使用された日時（recency計算用）"""

    # === インパクト ===
    impact_score: float = 0.0
    """インパクトスコア
    - ユーザーから肯定的フィードバック: +2.0
    - タスク成功に貢献: +1.5
    - エラー防止に貢献: +2.0
    """

    # === 定着管理 ===
    consolidation_level: int = 0
    """定着レベル (0-5)
    - access_count に基づいて計算
    - 高いほど減衰率が低い
    閾値: [0, 5, 15, 30, 60, 100]
    """

    # === 学び（観点別） ===
    learnings: Dict[str, str] = field(default_factory=dict)
    """観点別の学び内容（JSONB）
    例: {"コスト": "緊急調達で15%コスト増", "納期": "2週間バッファが必要"}
    """

    # === 状態 ===
    status: str = "active"
    """記憶の状態: active / archived"""

    source: Optional[str] = None
    """記憶のソース: education / task / manual"""

    # === タイムスタンプ ===
    created_at: datetime = field(default_factory=datetime.now)
    """記憶の作成日時"""

    updated_at: datetime = field(default_factory=datetime.now)
    """最終更新日時（強度変更等）"""

    last_decay_at: Optional[datetime] = None
    """最後に減衰処理が適用された日時（睡眠フェーズ追跡）"""

    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換

        Returns:
            全フィールドを含む辞書

        Note:
            - UUID は文字列に変換
            - datetime は ISO8601 形式の文字列に変換
            - embedding は None の場合そのまま None
        """
        return {
            "id": str(self.id),
            "agent_id": self.agent_id,
            "content": self.content,
            "embedding": self.embedding,
            "tags": self.tags,
            "scope_level": self.scope_level,
            "scope_domain": self.scope_domain,
            "scope_project": self.scope_project,
            "strength": self.strength,
            "strength_by_perspective": self.strength_by_perspective,
            "access_count": self.access_count,
            "candidate_count": self.candidate_count,
            "last_accessed_at": (
                self.last_accessed_at.isoformat() if self.last_accessed_at else None
            ),
            "impact_score": self.impact_score,
            "consolidation_level": self.consolidation_level,
            "learnings": self.learnings,
            "status": self.status,
            "source": self.source,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_decay_at": (
                self.last_decay_at.isoformat() if self.last_decay_at else None
            ),
        }

    @classmethod
    def from_row(cls, row: tuple | dict) -> AgentMemory:
        """DBレコードからインスタンスを生成

        Args:
            row: psycopg2 の fetchone() で取得した行データ
                 tuple（cursor.description でカラム名取得可能）または
                 RealDictCursor 使用時の dict

        Returns:
            AgentMemory インスタンス

        Note:
            - JSONB フィールドは自動的にPython辞書に変換される（psycopg2の機能）
            - vector フィールドは float のリストとして取得される
            - TEXT[] フィールドは Python のリストに変換される
        """
        # dict形式の場合
        if isinstance(row, dict):
            return cls._from_dict_row(row)

        # tuple形式の場合（カラム順序に依存）
        # カラム順序: id, agent_id, content, embedding, tags,
        #            scope_level, scope_domain, scope_project,
        #            strength, strength_by_perspective,
        #            access_count, candidate_count, last_accessed_at,
        #            impact_score, consolidation_level, learnings,
        #            status, source, created_at, updated_at, last_decay_at
        return cls(
            id=row[0] if isinstance(row[0], UUID) else UUID(str(row[0])),
            agent_id=row[1],
            content=row[2],
            embedding=list(row[3]) if row[3] else None,
            tags=list(row[4]) if row[4] else [],
            scope_level=row[5] or "project",
            scope_domain=row[6],
            scope_project=row[7],
            strength=float(row[8]) if row[8] is not None else 1.0,
            strength_by_perspective=row[9] or {},
            access_count=int(row[10]) if row[10] is not None else 0,
            candidate_count=int(row[11]) if row[11] is not None else 0,
            last_accessed_at=row[12],
            impact_score=float(row[13]) if row[13] is not None else 0.0,
            consolidation_level=int(row[14]) if row[14] is not None else 0,
            learnings=row[15] or {},
            status=row[16] or "active",
            source=row[17],
            created_at=row[18] or datetime.now(),
            updated_at=row[19] or datetime.now(),
            last_decay_at=row[20],
        )

    @classmethod
    def _from_dict_row(cls, row: dict) -> AgentMemory:
        """dict形式の行データからインスタンスを生成（内部メソッド）"""
        return cls(
            id=(
                row["id"]
                if isinstance(row["id"], UUID)
                else UUID(str(row["id"]))
            ),
            agent_id=row["agent_id"],
            content=row["content"],
            embedding=list(row["embedding"]) if row.get("embedding") else None,
            tags=list(row.get("tags") or []),
            scope_level=row.get("scope_level") or "project",
            scope_domain=row.get("scope_domain"),
            scope_project=row.get("scope_project"),
            strength=float(row["strength"]) if row.get("strength") is not None else 1.0,
            strength_by_perspective=row.get("strength_by_perspective") or {},
            access_count=(
                int(row["access_count"]) if row.get("access_count") is not None else 0
            ),
            candidate_count=(
                int(row["candidate_count"])
                if row.get("candidate_count") is not None
                else 0
            ),
            last_accessed_at=row.get("last_accessed_at"),
            impact_score=(
                float(row["impact_score"])
                if row.get("impact_score") is not None
                else 0.0
            ),
            consolidation_level=(
                int(row["consolidation_level"])
                if row.get("consolidation_level") is not None
                else 0
            ),
            learnings=row.get("learnings") or {},
            status=row.get("status") or "active",
            source=row.get("source"),
            created_at=row.get("created_at") or datetime.now(),
            updated_at=row.get("updated_at") or datetime.now(),
            last_decay_at=row.get("last_decay_at"),
        )

    @classmethod
    def create(
        cls,
        agent_id: str,
        content: str,
        *,
        embedding: Optional[List[float]] = None,
        tags: Optional[List[str]] = None,
        scope_level: str = "project",
        scope_domain: Optional[str] = None,
        scope_project: Optional[str] = None,
        strength: float = 1.0,
        strength_by_perspective: Optional[Dict[str, float]] = None,
        learnings: Optional[Dict[str, str]] = None,
        source: Optional[str] = None,
    ) -> AgentMemory:
        """デフォルト値を設定してインスタンスを生成するファクトリメソッド

        Args:
            agent_id: エージェントID（必須）
            content: 記憶の内容（必須）
            embedding: ベクトル表現（後から非同期で設定可能）
            tags: 分類タグ
            scope_level: スコープレベル (universal/domain/project)
            scope_domain: ドメイン名
            scope_project: プロジェクトID
            strength: 初期強度（デフォルト: 1.0）
            strength_by_perspective: 観点別強度の初期値
            learnings: 観点別の学び内容
            source: 記憶のソース (education/task/manual)

        Returns:
            新規 AgentMemory インスタンス

        Example:
            >>> memory = AgentMemory.create(
            ...     agent_id="procurement_agent_01",
            ...     content="緊急調達では15%のコスト増を見込む",
            ...     tags=["コスト", "緊急調達"],
            ...     scope_level="domain",
            ...     scope_domain="procurement",
            ...     source="task",
            ... )
        """
        now = datetime.now()
        return cls(
            id=uuid4(),
            agent_id=agent_id,
            content=content,
            embedding=embedding,
            tags=tags or [],
            scope_level=scope_level,
            scope_domain=scope_domain,
            scope_project=scope_project,
            strength=strength,
            strength_by_perspective=strength_by_perspective or {},
            access_count=0,
            candidate_count=0,
            last_accessed_at=None,
            impact_score=0.0,
            consolidation_level=0,
            learnings=learnings or {},
            status="active",
            source=source,
            created_at=now,
            updated_at=now,
            last_decay_at=None,
        )

    @classmethod
    def create_from_education(
        cls,
        agent_id: str,
        content: str,
        *,
        embedding: Optional[List[float]] = None,
        tags: Optional[List[str]] = None,
        scope_level: str = "project",
        scope_domain: Optional[str] = None,
        scope_project: Optional[str] = None,
        strength_by_perspective: Optional[Dict[str, float]] = None,
        learnings: Optional[Dict[str, str]] = None,
    ) -> AgentMemory:
        """教育プロセスからの記憶を生成するファクトリメソッド

        教育プロセスで「読んだだけ」の記憶は初期強度が低い (0.5)。
        実際にタスクで使用されて初めて強化される。

        Args:
            agent_id: エージェントID（必須）
            content: 記憶の内容（必須）
            その他: create() メソッドと同様

        Returns:
            初期強度 0.5 の AgentMemory インスタンス

        Note:
            仕様書 セクション4.1:
            INITIAL_STRENGTH_EDUCATION = 0.5
        """
        return cls.create(
            agent_id=agent_id,
            content=content,
            embedding=embedding,
            tags=tags,
            scope_level=scope_level,
            scope_domain=scope_domain,
            scope_project=scope_project,
            strength=0.5,  # 教育プロセスの初期強度
            strength_by_perspective=strength_by_perspective,
            learnings=learnings,
            source="education",
        )

    def copy_with(self, **kwargs: Any) -> AgentMemory:
        """指定したフィールドを変更した新しいインスタンスを生成

        イミュータブルな更新パターンをサポート。

        Args:
            **kwargs: 変更するフィールドと値

        Returns:
            変更後の新しい AgentMemory インスタンス

        Example:
            >>> updated = memory.copy_with(
            ...     strength=memory.strength + 0.1,
            ...     access_count=memory.access_count + 1,
            ...     updated_at=datetime.now(),
            ... )
        """
        data = self.to_dict()
        # datetime や UUID を復元
        data["id"] = self.id
        data["created_at"] = self.created_at
        data["updated_at"] = self.updated_at
        data["last_accessed_at"] = self.last_accessed_at
        data["last_decay_at"] = self.last_decay_at

        # 引数で上書き
        data.update(kwargs)

        return AgentMemory(
            id=data["id"],
            agent_id=data["agent_id"],
            content=data["content"],
            embedding=data["embedding"],
            tags=data["tags"],
            scope_level=data["scope_level"],
            scope_domain=data["scope_domain"],
            scope_project=data["scope_project"],
            strength=data["strength"],
            strength_by_perspective=data["strength_by_perspective"],
            access_count=data["access_count"],
            candidate_count=data["candidate_count"],
            last_accessed_at=data["last_accessed_at"],
            impact_score=data["impact_score"],
            consolidation_level=data["consolidation_level"],
            learnings=data["learnings"],
            status=data["status"],
            source=data["source"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            last_decay_at=data["last_decay_at"],
        )

    def __repr__(self) -> str:
        """デバッグ用の文字列表現"""
        return (
            f"AgentMemory("
            f"id={self.id!r}, "
            f"agent_id={self.agent_id!r}, "
            f"content={self.content[:50]!r}..., "
            f"strength={self.strength:.2f}, "
            f"access_count={self.access_count}, "
            f"status={self.status!r})"
        )
