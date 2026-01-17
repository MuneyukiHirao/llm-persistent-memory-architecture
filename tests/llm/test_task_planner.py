# テスト: タスク計画モジュール
"""
TaskPlanner のユニットテスト

トークン効率改善のための計画ファイル方式をテスト。
"""

import os
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from src.llm.task_planner import (
    Subtask,
    TaskPlan,
    TaskPlanner,
    TaskPlannerError,
    get_planning_prompt,
)


# =============================================================================
# Subtask のテスト
# =============================================================================

class TestSubtask:
    """Subtask dataclass のテスト"""

    def test_create_subtask(self):
        """サブタスク作成"""
        subtask = Subtask(
            id="cli_001",
            description="CLI基盤を実装",
            task_description="src/cli/main.py を作成してください",
        )

        assert subtask.id == "cli_001"
        assert subtask.description == "CLI基盤を実装"
        assert subtask.task_description == "src/cli/main.py を作成してください"
        assert subtask.files == []
        assert subtask.dependencies == []
        assert subtask.status == "pending"
        assert subtask.result is None

    def test_create_subtask_with_all_fields(self):
        """全フィールド指定でサブタスク作成"""
        subtask = Subtask(
            id="cli_002",
            description="コマンド実装",
            task_description="register コマンドを実装",
            files=["src/cli/commands/register.py"],
            dependencies=["cli_001"],
            status="completed",
            result="完了しました",
        )

        assert subtask.files == ["src/cli/commands/register.py"]
        assert subtask.dependencies == ["cli_001"]
        assert subtask.status == "completed"
        assert subtask.result == "完了しました"

    def test_to_dict(self):
        """辞書変換"""
        subtask = Subtask(
            id="cli_001",
            description="テスト",
            task_description="テストタスク",
            files=["a.py"],
            dependencies=["dep_001"],
        )

        data = subtask.to_dict()

        assert data["id"] == "cli_001"
        assert data["description"] == "テスト"
        assert data["task_description"] == "テストタスク"
        assert data["files"] == ["a.py"]
        assert data["dependencies"] == ["dep_001"]
        assert data["status"] == "pending"
        assert "result" not in data  # result が None の場合は含まれない

    def test_to_dict_with_result(self):
        """結果あり辞書変換"""
        subtask = Subtask(
            id="cli_001",
            description="テスト",
            task_description="テストタスク",
            result="完了",
        )

        data = subtask.to_dict()
        assert data["result"] == "完了"

    def test_from_dict(self):
        """辞書から生成"""
        data = {
            "id": "cli_001",
            "description": "テスト",
            "task_description": "タスク指示",
            "files": ["a.py", "b.py"],
            "dependencies": ["dep_001"],
            "status": "in_progress",
            "result": "進行中",
        }

        subtask = Subtask.from_dict(data)

        assert subtask.id == "cli_001"
        assert subtask.description == "テスト"
        assert subtask.task_description == "タスク指示"
        assert subtask.files == ["a.py", "b.py"]
        assert subtask.dependencies == ["dep_001"]
        assert subtask.status == "in_progress"
        assert subtask.result == "進行中"

    def test_from_dict_minimal(self):
        """最小限の辞書から生成"""
        data = {
            "id": "task_001",
            "description": "タスク",
        }

        subtask = Subtask.from_dict(data)

        assert subtask.id == "task_001"
        assert subtask.task_description == "タスク"  # description がフォールバック
        assert subtask.files == []
        assert subtask.dependencies == []
        assert subtask.status == "pending"


# =============================================================================
# TaskPlan のテスト
# =============================================================================

class TestTaskPlan:
    """TaskPlan dataclass のテスト"""

    def test_create_plan(self):
        """計画作成"""
        plan = TaskPlan(
            plan_id="plan_001",
            task_description="CLI実装",
            spec_file="docs/spec.md",
            created_at="2026-01-16T12:00:00",
        )

        assert plan.plan_id == "plan_001"
        assert plan.task_description == "CLI実装"
        assert plan.spec_file == "docs/spec.md"
        assert plan.subtasks == []
        assert plan.context_summary == ""

    def test_create_plan_with_subtasks(self):
        """サブタスク付き計画作成"""
        subtasks = [
            Subtask(id="t1", description="タスク1", task_description="タスク1"),
            Subtask(id="t2", description="タスク2", task_description="タスク2", dependencies=["t1"]),
        ]

        plan = TaskPlan(
            plan_id="plan_001",
            task_description="CLI実装",
            spec_file="docs/spec.md",
            created_at="2026-01-16T12:00:00",
            subtasks=subtasks,
            context_summary="要約",
        )

        assert len(plan.subtasks) == 2
        assert plan.context_summary == "要約"

    def test_to_dict(self):
        """辞書変換"""
        subtasks = [
            Subtask(id="t1", description="タスク1", task_description="タスク1"),
        ]

        plan = TaskPlan(
            plan_id="plan_001",
            task_description="CLI実装",
            spec_file="docs/spec.md",
            created_at="2026-01-16T12:00:00",
            subtasks=subtasks,
            context_summary="要約",
            total_estimated_tokens=1000,
        )

        data = plan.to_dict()

        assert data["plan_id"] == "plan_001"
        assert data["task_description"] == "CLI実装"
        assert data["spec_file"] == "docs/spec.md"
        assert data["created_at"] == "2026-01-16T12:00:00"
        assert len(data["subtasks"]) == 1
        assert data["context_summary"] == "要約"
        assert data["total_estimated_tokens"] == 1000

    def test_from_dict(self):
        """辞書から生成"""
        data = {
            "plan_id": "plan_001",
            "task_description": "CLI実装",
            "spec_file": "docs/spec.md",
            "created_at": "2026-01-16T12:00:00",
            "subtasks": [
                {"id": "t1", "description": "タスク1", "task_description": "タスク1"},
                {"id": "t2", "description": "タスク2", "task_description": "タスク2"},
            ],
            "context_summary": "要約",
            "total_estimated_tokens": 500,
        }

        plan = TaskPlan.from_dict(data)

        assert plan.plan_id == "plan_001"
        assert len(plan.subtasks) == 2
        assert plan.subtasks[0].id == "t1"
        assert plan.context_summary == "要約"
        assert plan.total_estimated_tokens == 500

    def test_get_next_subtask_no_dependencies(self):
        """依存関係なしで次のサブタスクを取得"""
        subtasks = [
            Subtask(id="t1", description="タスク1", task_description="タスク1"),
            Subtask(id="t2", description="タスク2", task_description="タスク2"),
        ]

        plan = TaskPlan(
            plan_id="plan_001",
            task_description="テスト",
            spec_file="spec.md",
            created_at="2026-01-16T12:00:00",
            subtasks=subtasks,
        )

        next_task = plan.get_next_subtask()

        assert next_task is not None
        assert next_task.id == "t1"

    def test_get_next_subtask_with_dependencies(self):
        """依存関係ありで次のサブタスクを取得"""
        subtasks = [
            Subtask(id="t1", description="タスク1", task_description="タスク1"),
            Subtask(id="t2", description="タスク2", task_description="タスク2", dependencies=["t1"]),
        ]

        plan = TaskPlan(
            plan_id="plan_001",
            task_description="テスト",
            spec_file="spec.md",
            created_at="2026-01-16T12:00:00",
            subtasks=subtasks,
        )

        # t1 を完了
        plan.mark_subtask_completed("t1", "完了")

        next_task = plan.get_next_subtask()

        assert next_task is not None
        assert next_task.id == "t2"

    def test_get_next_subtask_dependencies_not_met(self):
        """依存関係が満たされていない場合"""
        subtasks = [
            Subtask(id="t1", description="タスク1", task_description="タスク1", status="in_progress"),
            Subtask(id="t2", description="タスク2", task_description="タスク2", dependencies=["t1"]),
        ]

        plan = TaskPlan(
            plan_id="plan_001",
            task_description="テスト",
            spec_file="spec.md",
            created_at="2026-01-16T12:00:00",
            subtasks=subtasks,
        )

        next_task = plan.get_next_subtask()

        # t1 が in_progress なので、t2 は実行できない
        # pending のタスクがないので None
        assert next_task is None

    def test_mark_subtask_completed(self):
        """サブタスクを完了としてマーク"""
        subtasks = [
            Subtask(id="t1", description="タスク1", task_description="タスク1"),
        ]

        plan = TaskPlan(
            plan_id="plan_001",
            task_description="テスト",
            spec_file="spec.md",
            created_at="2026-01-16T12:00:00",
            subtasks=subtasks,
        )

        result = plan.mark_subtask_completed("t1", "成功")

        assert result is True
        assert plan.subtasks[0].status == "completed"
        assert plan.subtasks[0].result == "成功"

    def test_mark_subtask_completed_not_found(self):
        """存在しないサブタスクを完了としてマーク"""
        plan = TaskPlan(
            plan_id="plan_001",
            task_description="テスト",
            spec_file="spec.md",
            created_at="2026-01-16T12:00:00",
            subtasks=[],
        )

        result = plan.mark_subtask_completed("nonexistent", "成功")

        assert result is False

    def test_mark_subtask_failed(self):
        """サブタスクを失敗としてマーク"""
        subtasks = [
            Subtask(id="t1", description="タスク1", task_description="タスク1"),
        ]

        plan = TaskPlan(
            plan_id="plan_001",
            task_description="テスト",
            spec_file="spec.md",
            created_at="2026-01-16T12:00:00",
            subtasks=subtasks,
        )

        result = plan.mark_subtask_failed("t1", "エラー発生")

        assert result is True
        assert plan.subtasks[0].status == "failed"
        assert plan.subtasks[0].result == "ERROR: エラー発生"

    def test_is_complete_all_completed(self):
        """全タスク完了で is_complete = True"""
        subtasks = [
            Subtask(id="t1", description="タスク1", task_description="タスク1", status="completed"),
            Subtask(id="t2", description="タスク2", task_description="タスク2", status="completed"),
        ]

        plan = TaskPlan(
            plan_id="plan_001",
            task_description="テスト",
            spec_file="spec.md",
            created_at="2026-01-16T12:00:00",
            subtasks=subtasks,
        )

        assert plan.is_complete() is True

    def test_is_complete_mixed_status(self):
        """完了・失敗混在で is_complete = True"""
        subtasks = [
            Subtask(id="t1", description="タスク1", task_description="タスク1", status="completed"),
            Subtask(id="t2", description="タスク2", task_description="タスク2", status="failed"),
        ]

        plan = TaskPlan(
            plan_id="plan_001",
            task_description="テスト",
            spec_file="spec.md",
            created_at="2026-01-16T12:00:00",
            subtasks=subtasks,
        )

        assert plan.is_complete() is True

    def test_is_complete_pending(self):
        """pending があると is_complete = False"""
        subtasks = [
            Subtask(id="t1", description="タスク1", task_description="タスク1", status="completed"),
            Subtask(id="t2", description="タスク2", task_description="タスク2", status="pending"),
        ]

        plan = TaskPlan(
            plan_id="plan_001",
            task_description="テスト",
            spec_file="spec.md",
            created_at="2026-01-16T12:00:00",
            subtasks=subtasks,
        )

        assert plan.is_complete() is False

    def test_get_progress(self):
        """進捗状況を取得"""
        subtasks = [
            Subtask(id="t1", description="タスク1", task_description="タスク1", status="completed"),
            Subtask(id="t2", description="タスク2", task_description="タスク2", status="pending"),
            Subtask(id="t3", description="タスク3", task_description="タスク3", status="in_progress"),
            Subtask(id="t4", description="タスク4", task_description="タスク4", status="failed"),
        ]

        plan = TaskPlan(
            plan_id="plan_001",
            task_description="テスト",
            spec_file="spec.md",
            created_at="2026-01-16T12:00:00",
            subtasks=subtasks,
        )

        progress = plan.get_progress()

        assert progress["completed"] == 1
        assert progress["pending"] == 1
        assert progress["in_progress"] == 1
        assert progress["failed"] == 1
        assert progress["total"] == 4


# =============================================================================
# TaskPlanner のテスト
# =============================================================================

class TestTaskPlanner:
    """TaskPlanner クラスのテスト"""

    def test_init(self):
        """初期化"""
        with tempfile.TemporaryDirectory() as tmpdir:
            planner = TaskPlanner(project_root=tmpdir)

            assert planner.project_root == Path(tmpdir).resolve()
            assert planner.plans_dir == Path(tmpdir).resolve() / "plans"
            assert planner.plans_dir.exists()

    def test_init_custom_plans_dir(self):
        """カスタム plans_dir で初期化"""
        with tempfile.TemporaryDirectory() as tmpdir:
            plans_dir = Path(tmpdir) / "custom_plans"

            planner = TaskPlanner(project_root=tmpdir, plans_dir=str(plans_dir))

            assert planner.plans_dir == plans_dir.resolve()
            assert plans_dir.exists()

    def test_generate_plan_id(self):
        """計画ID生成"""
        with tempfile.TemporaryDirectory() as tmpdir:
            planner = TaskPlanner(project_root=tmpdir)

            plan_id = planner.generate_plan_id()

            assert plan_id.startswith("plan_")
            assert len(plan_id) > 20  # timestamp + uuid

    def test_create_plan_from_spec(self):
        """仕様から計画作成"""
        with tempfile.TemporaryDirectory() as tmpdir:
            planner = TaskPlanner(project_root=tmpdir)

            subtasks = [
                {"id": "t1", "description": "タスク1", "task_description": "タスク1"},
                {"id": "t2", "description": "タスク2", "task_description": "タスク2", "dependencies": ["t1"]},
            ]

            plan = planner.create_plan_from_spec(
                spec_file="docs/spec.md",
                task_description="CLI実装",
                subtasks=subtasks,
                context_summary="要約テキスト",
            )

            assert plan.plan_id.startswith("plan_")
            assert plan.task_description == "CLI実装"
            assert plan.spec_file == "docs/spec.md"
            assert len(plan.subtasks) == 2
            assert plan.context_summary == "要約テキスト"
            assert plan.subtasks[1].dependencies == ["t1"]

    def test_save_and_load_plan(self):
        """計画の保存と読み込み"""
        with tempfile.TemporaryDirectory() as tmpdir:
            planner = TaskPlanner(project_root=tmpdir)

            # 計画作成
            subtasks = [
                {"id": "t1", "description": "タスク1", "task_description": "タスク1"},
            ]
            plan = planner.create_plan_from_spec(
                spec_file="docs/spec.md",
                task_description="テスト",
                subtasks=subtasks,
                context_summary="要約",
            )

            # 保存
            plan_path = planner.save_plan(plan, "test-plan")

            assert plan_path.exists()
            assert plan_path.name == "test-plan.yaml"

            # 読み込み
            loaded_plan = planner.load_plan("test-plan.yaml")

            assert loaded_plan.plan_id == plan.plan_id
            assert loaded_plan.task_description == plan.task_description
            assert len(loaded_plan.subtasks) == 1
            assert loaded_plan.context_summary == "要約"

    def test_save_plan_auto_extension(self):
        """保存時に拡張子を自動追加"""
        with tempfile.TemporaryDirectory() as tmpdir:
            planner = TaskPlanner(project_root=tmpdir)

            plan = planner.create_plan_from_spec(
                spec_file="spec.md",
                task_description="テスト",
                subtasks=[],
            )

            plan_path = planner.save_plan(plan, "my-plan")

            assert plan_path.name == "my-plan.yaml"

    def test_load_plan_not_found(self):
        """存在しない計画の読み込み"""
        with tempfile.TemporaryDirectory() as tmpdir:
            planner = TaskPlanner(project_root=tmpdir)

            with pytest.raises(TaskPlannerError) as exc_info:
                planner.load_plan("nonexistent.yaml")

            assert "見つかりません" in str(exc_info.value)

    def test_update_plan(self):
        """計画の更新"""
        with tempfile.TemporaryDirectory() as tmpdir:
            planner = TaskPlanner(project_root=tmpdir)

            # 計画作成・保存
            plan = planner.create_plan_from_spec(
                spec_file="spec.md",
                task_description="テスト",
                subtasks=[{"id": "t1", "description": "タスク1", "task_description": "タスク1"}],
            )
            plan_path = planner.save_plan(plan, "update-test")

            # 状態を変更
            plan.mark_subtask_completed("t1", "完了")

            # 更新
            updated_path = planner.update_plan(plan, "update-test.yaml")

            # 再読み込み
            reloaded = planner.load_plan("update-test.yaml")

            assert reloaded.subtasks[0].status == "completed"
            assert reloaded.subtasks[0].result == "完了"

    def test_list_plans(self):
        """計画一覧取得"""
        with tempfile.TemporaryDirectory() as tmpdir:
            planner = TaskPlanner(project_root=tmpdir)

            # 計画を複数作成
            for i in range(3):
                plan = planner.create_plan_from_spec(
                    spec_file=f"spec{i}.md",
                    task_description=f"テスト{i}",
                    subtasks=[{"id": f"t{i}", "description": f"タスク{i}"}],
                )
                planner.save_plan(plan, f"plan-{i}")

            # 一覧取得
            plans = planner.list_plans()

            assert len(plans) == 3
            assert all("plan_id" in p for p in plans)
            assert all("task_description" in p for p in plans)
            assert all("subtasks_count" in p for p in plans)


# =============================================================================
# ユーティリティ関数のテスト
# =============================================================================

class TestGetPlanningPrompt:
    """get_planning_prompt 関数のテスト"""

    def test_get_planning_prompt(self):
        """計画作成プロンプト生成"""
        prompt = get_planning_prompt(
            task_description="CLI を実装してください",
            spec_content="# 仕様書\n\n機能一覧...",
        )

        assert "CLI を実装してください" in prompt
        assert "# 仕様書" in prompt
        assert "機能一覧" in prompt
        assert "context_summary" in prompt
        assert "subtasks" in prompt

    def test_get_planning_prompt_large_spec(self):
        """大きな仕様書でもプロンプト生成"""
        large_spec = "# 仕様書\n" + ("機能説明。" * 1000)

        prompt = get_planning_prompt(
            task_description="テスト",
            spec_content=large_spec,
        )

        assert len(prompt) > len(large_spec)
        assert "テスト" in prompt


# =============================================================================
# エラーハンドリングのテスト
# =============================================================================

class TestTaskPlannerError:
    """TaskPlannerError のテスト"""

    def test_error_message(self):
        """エラーメッセージ"""
        error = TaskPlannerError("計画の読み込みに失敗")

        assert str(error) == "計画の読み込みに失敗"

    def test_error_with_original(self):
        """元の例外付きエラー"""
        original = ValueError("YAML parse error")
        error = TaskPlannerError("計画の読み込みに失敗", original_error=original)

        assert "計画の読み込みに失敗" in str(error)
        assert "YAML parse error" in str(error)
        assert error.original_error == original


# =============================================================================
# YAML 保存形式のテスト
# =============================================================================

class TestYAMLFormat:
    """YAML ファイル形式のテスト"""

    def test_yaml_format_readable(self):
        """保存された YAML が人間に読める形式か"""
        with tempfile.TemporaryDirectory() as tmpdir:
            planner = TaskPlanner(project_root=tmpdir)

            plan = planner.create_plan_from_spec(
                spec_file="docs/phase5-cli-spec.ja.md",
                task_description="Phase 5 CLI を実装",
                subtasks=[
                    {
                        "id": "cli_001",
                        "description": "CLI基盤を実装",
                        "task_description": "src/cli/main.py を作成してください。\n\n機能:\n- init コマンド\n- list コマンド",
                        "files": ["src/cli/main.py"],
                    },
                    {
                        "id": "cli_002",
                        "description": "register コマンド",
                        "task_description": "register コマンドを実装",
                        "files": ["src/cli/commands/register.py"],
                        "dependencies": ["cli_001"],
                    },
                ],
                context_summary="Phase 5 CLI は Python コードなしでエージェントを操作する CLI。",
            )

            plan_path = planner.save_plan(plan, "readable-test")

            # ファイル内容を読み込み
            with open(plan_path, "r", encoding="utf-8") as f:
                content = f.read()

            # 人間に読める形式か確認
            assert "plan_id:" in content
            assert "task_description:" in content
            assert "context_summary:" in content
            assert "subtasks:" in content
            assert "cli_001" in content
            assert "dependencies:" in content

            # YAML として再パース可能か
            parsed = yaml.safe_load(content)
            assert parsed["plan_id"] == plan.plan_id
