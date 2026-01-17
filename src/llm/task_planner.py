# タスク計画モジュール
# トークン効率改善: 大きな仕様書を読んで計画を作成し、以降は計画ファイルだけ参照
"""
タスク計画モジュール

複雑なタスクを「計画作成」と「計画実行」の2フェーズに分離することで、
トークン効率を改善します。

問題:
- 仕様書 25,000文字（約8,000トークン）を毎回のAPI呼び出しで含めると非効率
- 11回のAPI呼び出しで累積80,000トークン消費

解決策:
- Phase 1: 仕様書を読んで計画ファイルを作成（1回だけ）
- Phase 2: 計画ファイルを参照して実装（計画は小さい）

使用例:
    planner = TaskPlanner(project_root="/path/to/project")

    # 計画作成
    plan = planner.create_plan(
        spec_file="docs/phase5-cli-spec.ja.md",
        task_description="Phase 5 CLIを実装",
    )

    # 計画をファイルに保存
    plan_path = planner.save_plan(plan, "phase5-cli")

    # 計画を読み込み
    loaded_plan = planner.load_plan(plan_path)

    # サブタスクを順次実行
    for subtask in loaded_plan.subtasks:
        runner.run_task(agent_id, subtask.task_description)
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


class TaskPlannerError(Exception):
    """TaskPlanner のエラー

    計画作成・読み込み・保存時のエラー。

    Attributes:
        message: エラーメッセージ
        original_error: 元の例外（あれば）
    """

    def __init__(
        self,
        message: str,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.message = message
        self.original_error = original_error

    def __str__(self) -> str:
        if self.original_error:
            return f"{self.message}: {self.original_error}"
        return self.message


@dataclass
class Subtask:
    """サブタスク定義

    計画内の個別タスク。依存関係と対象ファイルを明示。

    Attributes:
        id: サブタスクID（例: cli_001）
        description: タスク説明
        task_description: エージェントに渡すタスク記述
        files: 対象ファイルリスト
        dependencies: 依存するサブタスクIDリスト
        status: 状態（pending/in_progress/completed/failed）
        result: 実行結果（完了時）
    """

    id: str
    description: str
    task_description: str
    files: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"
    result: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        data = {
            "id": self.id,
            "description": self.description,
            "task_description": self.task_description,
            "files": self.files,
            "dependencies": self.dependencies,
            "status": self.status,
        }
        if self.result:
            data["result"] = self.result
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Subtask":
        """辞書から生成"""
        return cls(
            id=data["id"],
            description=data["description"],
            task_description=data.get("task_description", data["description"]),
            files=data.get("files", []),
            dependencies=data.get("dependencies", []),
            status=data.get("status", "pending"),
            result=data.get("result"),
        )


@dataclass
class TaskPlan:
    """タスク計画

    仕様書から生成された実装計画。サブタスクのリストと依存関係を含む。

    Attributes:
        plan_id: 計画ID
        task_description: 元のタスク説明
        spec_file: 参照した仕様書パス
        created_at: 作成日時
        subtasks: サブタスクリスト
        context_summary: 仕様書の要約（サブタスクで参照可能）
        total_estimated_tokens: 推定総トークン数
    """

    plan_id: str
    task_description: str
    spec_file: str
    created_at: str
    subtasks: List[Subtask] = field(default_factory=list)
    context_summary: str = ""
    total_estimated_tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "plan_id": self.plan_id,
            "task_description": self.task_description,
            "spec_file": self.spec_file,
            "created_at": self.created_at,
            "context_summary": self.context_summary,
            "total_estimated_tokens": self.total_estimated_tokens,
            "subtasks": [st.to_dict() for st in self.subtasks],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskPlan":
        """辞書から生成"""
        subtasks = [Subtask.from_dict(st) for st in data.get("subtasks", [])]
        return cls(
            plan_id=data["plan_id"],
            task_description=data["task_description"],
            spec_file=data["spec_file"],
            created_at=data["created_at"],
            subtasks=subtasks,
            context_summary=data.get("context_summary", ""),
            total_estimated_tokens=data.get("total_estimated_tokens", 0),
        )

    def get_next_subtask(self) -> Optional[Subtask]:
        """次に実行可能なサブタスクを取得

        依存関係を考慮し、実行可能な pending 状態のサブタスクを返す。

        Returns:
            実行可能なサブタスク、またはなければ None
        """
        completed_ids = {st.id for st in self.subtasks if st.status == "completed"}

        for subtask in self.subtasks:
            if subtask.status != "pending":
                continue

            # 依存関係がすべて完了しているか確認
            deps_satisfied = all(dep in completed_ids for dep in subtask.dependencies)
            if deps_satisfied:
                return subtask

        return None

    def mark_subtask_completed(self, subtask_id: str, result: str = "") -> bool:
        """サブタスクを完了としてマーク

        Args:
            subtask_id: サブタスクID
            result: 実行結果

        Returns:
            成功したかどうか
        """
        for subtask in self.subtasks:
            if subtask.id == subtask_id:
                subtask.status = "completed"
                subtask.result = result
                return True
        return False

    def mark_subtask_failed(self, subtask_id: str, error: str = "") -> bool:
        """サブタスクを失敗としてマーク

        Args:
            subtask_id: サブタスクID
            error: エラーメッセージ

        Returns:
            成功したかどうか
        """
        for subtask in self.subtasks:
            if subtask.id == subtask_id:
                subtask.status = "failed"
                subtask.result = f"ERROR: {error}"
                return True
        return False

    def is_complete(self) -> bool:
        """計画が完了したかどうか"""
        return all(st.status in ("completed", "failed") for st in self.subtasks)

    def get_progress(self) -> Dict[str, int]:
        """進捗状況を取得"""
        counts = {"pending": 0, "in_progress": 0, "completed": 0, "failed": 0}
        for subtask in self.subtasks:
            counts[subtask.status] = counts.get(subtask.status, 0) + 1
        counts["total"] = len(self.subtasks)
        return counts


class TaskPlanner:
    """タスク計画管理クラス

    計画の作成・保存・読み込みを管理します。

    Attributes:
        project_root: プロジェクトルートディレクトリ
        plans_dir: 計画ファイル保存ディレクトリ
    """

    def __init__(self, project_root: str, plans_dir: Optional[str] = None):
        """TaskPlanner を初期化

        Args:
            project_root: プロジェクトルートディレクトリのパス
            plans_dir: 計画ファイル保存ディレクトリ（省略時は plans/）
        """
        self.project_root = Path(project_root).resolve()

        if plans_dir:
            self.plans_dir = Path(plans_dir).resolve()
        else:
            self.plans_dir = self.project_root / "plans"

        # ディレクトリがなければ作成
        self.plans_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"TaskPlanner 初期化完了: plans_dir={self.plans_dir}")

    def generate_plan_id(self) -> str:
        """一意の計画IDを生成"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = str(uuid.uuid4())[:8]
        return f"plan_{timestamp}_{short_uuid}"

    def create_plan_from_spec(
        self,
        spec_file: str,
        task_description: str,
        subtasks: List[Dict[str, Any]],
        context_summary: str = "",
    ) -> TaskPlan:
        """仕様書とサブタスク定義から計画を作成

        Args:
            spec_file: 仕様書ファイルパス
            task_description: タスク説明
            subtasks: サブタスク定義リスト
            context_summary: 仕様書の要約

        Returns:
            TaskPlan インスタンス
        """
        plan_id = self.generate_plan_id()
        created_at = datetime.now().isoformat()

        # サブタスクを作成
        subtask_objects = []
        for st_data in subtasks:
            subtask = Subtask(
                id=st_data.get("id", f"task_{len(subtask_objects) + 1:03d}"),
                description=st_data["description"],
                task_description=st_data.get("task_description", st_data["description"]),
                files=st_data.get("files", []),
                dependencies=st_data.get("dependencies", []),
            )
            subtask_objects.append(subtask)

        # 推定トークン数（context_summary + 各サブタスク説明）
        estimated_tokens = len(context_summary) // 4  # 大まかな見積もり
        for st in subtask_objects:
            estimated_tokens += len(st.task_description) // 4

        plan = TaskPlan(
            plan_id=plan_id,
            task_description=task_description,
            spec_file=spec_file,
            created_at=created_at,
            subtasks=subtask_objects,
            context_summary=context_summary,
            total_estimated_tokens=estimated_tokens,
        )

        logger.info(
            f"計画作成完了: plan_id={plan_id}, "
            f"subtasks={len(subtask_objects)}, "
            f"estimated_tokens={estimated_tokens}"
        )

        return plan

    def save_plan(self, plan: TaskPlan, name: Optional[str] = None) -> Path:
        """計画をYAMLファイルに保存

        Args:
            plan: 保存する計画
            name: ファイル名（省略時は plan_id を使用）

        Returns:
            保存したファイルパス
        """
        filename = name or plan.plan_id
        if not filename.endswith(".yaml"):
            filename += ".yaml"

        file_path = self.plans_dir / filename

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                yaml.dump(
                    plan.to_dict(),
                    f,
                    allow_unicode=True,
                    default_flow_style=False,
                    sort_keys=False,
                )

            logger.info(f"計画を保存しました: {file_path}")
            return file_path

        except Exception as e:
            raise TaskPlannerError(
                f"計画の保存に失敗しました: {file_path}",
                original_error=e,
            ) from e

    def load_plan(self, plan_path: str) -> TaskPlan:
        """YAMLファイルから計画を読み込み

        Args:
            plan_path: 計画ファイルパス（相対パスの場合は plans_dir からの相対）

        Returns:
            TaskPlan インスタンス
        """
        path = Path(plan_path)
        if not path.is_absolute():
            path = self.plans_dir / path

        if not path.exists():
            raise TaskPlannerError(f"計画ファイルが見つかりません: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            plan = TaskPlan.from_dict(data)
            logger.info(f"計画を読み込みました: {path}")
            return plan

        except yaml.YAMLError as e:
            raise TaskPlannerError(
                f"計画ファイルの解析に失敗しました: {path}",
                original_error=e,
            ) from e
        except Exception as e:
            raise TaskPlannerError(
                f"計画の読み込みに失敗しました: {path}",
                original_error=e,
            ) from e

    def update_plan(self, plan: TaskPlan, plan_path: str) -> Path:
        """計画を更新して保存

        Args:
            plan: 更新された計画
            plan_path: 計画ファイルパス

        Returns:
            保存したファイルパス
        """
        path = Path(plan_path)
        if not path.is_absolute():
            path = self.plans_dir / path

        return self.save_plan(plan, path.name)

    def list_plans(self) -> List[Dict[str, Any]]:
        """保存された計画一覧を取得

        Returns:
            計画情報のリスト
        """
        plans = []

        for yaml_file in self.plans_dir.glob("*.yaml"):
            try:
                with open(yaml_file, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)

                plans.append({
                    "file": str(yaml_file),
                    "plan_id": data.get("plan_id", yaml_file.stem),
                    "task_description": data.get("task_description", ""),
                    "created_at": data.get("created_at", ""),
                    "subtasks_count": len(data.get("subtasks", [])),
                })
            except Exception:
                # 読み込めないファイルはスキップ
                continue

        return sorted(plans, key=lambda x: x.get("created_at", ""), reverse=True)


# 計画作成プロンプトテンプレート
PLANNING_PROMPT_TEMPLATE = """あなたは実装計画を作成するエキスパートです。

以下の仕様書を読んで、タスクを実行するための詳細な計画を作成してください。

## タスク
{task_description}

## 仕様書
{spec_content}

## 出力形式
以下のYAML形式で計画を出力してください：

```yaml
context_summary: |
  仕様書の要点を簡潔にまとめてください（500文字以内）

subtasks:
  - id: task_001
    description: 最初に行うタスクの説明
    task_description: |
      エージェントに渡す詳細なタスク指示
      必要なコンテキストを含める
    files:
      - 対象ファイル1
      - 対象ファイル2
    dependencies: []

  - id: task_002
    description: 次に行うタスクの説明
    task_description: |
      エージェントに渡す詳細なタスク指示
    files:
      - 対象ファイル
    dependencies:
      - task_001
```

## 注意事項
- 各サブタスクは独立して実行可能なように設計してください
- 依存関係を明示してください
- task_description には、仕様書を見なくても実行できる程度の情報を含めてください
- ファイルパスは具体的に指定してください
"""


def get_planning_prompt(task_description: str, spec_content: str) -> str:
    """計画作成用のプロンプトを生成

    Args:
        task_description: タスク説明
        spec_content: 仕様書の内容

    Returns:
        計画作成プロンプト
    """
    return PLANNING_PROMPT_TEMPLATE.format(
        task_description=task_description,
        spec_content=spec_content,
    )
