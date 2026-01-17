# file_tools テスト
"""
file_tools モジュールのテスト

テスト対象:
- file_read: ファイル読み込み
- file_write: ファイル書き込み
- file_list: ディレクトリ一覧
- パストラバーサル防止機能
"""

import os
import tempfile
from pathlib import Path

import pytest

from src.llm.tools import (
    file_read,
    file_write,
    file_list,
    set_project_root,
    get_project_root,
    ToolResult,
)
from src.llm.tools.file_tools import _validate_path, _PROJECT_ROOT


@pytest.fixture
def temp_project(tmp_path):
    """テスト用の一時プロジェクトディレクトリを作成"""
    # プロジェクトルートを設定
    set_project_root(str(tmp_path))

    # テスト用ファイルを作成
    test_file = tmp_path / "test.txt"
    test_file.write_text("Hello, World!", encoding="utf-8")

    # テスト用ディレクトリを作成
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (subdir / "file1.py").write_text("# Python file 1", encoding="utf-8")
    (subdir / "file2.py").write_text("# Python file 2", encoding="utf-8")
    (subdir / "readme.md").write_text("# README", encoding="utf-8")

    yield tmp_path

    # クリーンアップ後にプロジェクトルートをリセット
    import src.llm.tools.file_tools as ft
    ft._PROJECT_ROOT = None


class TestFileRead:
    """file_read のテスト"""

    def test_read_existing_file(self, temp_project):
        """存在するファイルを読み込めることを確認"""
        result = file_read("test.txt")

        assert result["success"] is True
        assert result["data"]["content"] == "Hello, World!"
        assert result["data"]["size"] == 13

    def test_read_file_in_subdirectory(self, temp_project):
        """サブディレクトリ内のファイルを読み込めることを確認"""
        result = file_read("subdir/file1.py")

        assert result["success"] is True
        assert result["data"]["content"] == "# Python file 1"

    def test_read_nonexistent_file(self, temp_project):
        """存在しないファイルの読み込みでエラーを返すことを確認"""
        result = file_read("nonexistent.txt")

        assert result["success"] is False
        assert "存在しません" in result["error"]

    def test_read_directory_fails(self, temp_project):
        """ディレクトリを読み込もうとするとエラーを返すことを確認"""
        result = file_read("subdir")

        assert result["success"] is False
        assert "ファイルではありません" in result["error"]

    def test_path_traversal_prevention(self, temp_project):
        """パストラバーサルを防止することを確認"""
        result = file_read("../../../etc/passwd")

        assert result["success"] is False
        assert "パストラバーサル" in result["error"]

    def test_absolute_path_outside_project(self, temp_project):
        """プロジェクト外の絶対パスを拒否することを確認"""
        result = file_read("/etc/passwd")

        assert result["success"] is False
        assert "パストラバーサル" in result["error"]


class TestFileWrite:
    """file_write のテスト"""

    def test_write_new_file(self, temp_project):
        """新しいファイルを書き込めることを確認"""
        result = file_write("new_file.txt", "New content")

        assert result["success"] is True
        assert result["data"]["size"] == 11

        # 実際にファイルが作成されたことを確認
        content = (temp_project / "new_file.txt").read_text(encoding="utf-8")
        assert content == "New content"

    def test_write_file_in_new_directory(self, temp_project):
        """存在しないディレクトリにファイルを書き込めることを確認"""
        result = file_write("new_dir/nested/file.txt", "Nested content")

        assert result["success"] is True

        # 実際にファイルが作成されたことを確認
        content = (temp_project / "new_dir/nested/file.txt").read_text(encoding="utf-8")
        assert content == "Nested content"

    def test_overwrite_existing_file(self, temp_project):
        """既存ファイルを上書きできることを確認"""
        result = file_write("test.txt", "Updated content")

        assert result["success"] is True

        # 内容が更新されたことを確認
        content = (temp_project / "test.txt").read_text(encoding="utf-8")
        assert content == "Updated content"

    def test_path_traversal_prevention(self, temp_project):
        """パストラバーサルを防止することを確認"""
        result = file_write("../outside.txt", "Malicious content")

        assert result["success"] is False
        assert "パストラバーサル" in result["error"]


class TestFileList:
    """file_list のテスト"""

    def test_list_directory(self, temp_project):
        """ディレクトリ一覧を取得できることを確認"""
        result = file_list(".")

        assert result["success"] is True
        assert "test.txt" in result["data"]["files"]
        assert "subdir" in result["data"]["directories"]

    def test_list_subdirectory(self, temp_project):
        """サブディレクトリの一覧を取得できることを確認"""
        result = file_list("subdir")

        assert result["success"] is True
        assert "file1.py" in result["data"]["files"]
        assert "file2.py" in result["data"]["files"]
        assert "readme.md" in result["data"]["files"]
        assert result["data"]["total_files"] == 3
        assert result["data"]["total_directories"] == 0

    def test_list_with_pattern(self, temp_project):
        """パターンでフィルタリングできることを確認"""
        result = file_list("subdir", pattern="*.py")

        assert result["success"] is True
        assert "file1.py" in result["data"]["files"]
        assert "file2.py" in result["data"]["files"]
        assert "readme.md" not in result["data"]["files"]
        assert result["data"]["total_files"] == 2

    def test_list_nonexistent_directory(self, temp_project):
        """存在しないディレクトリでエラーを返すことを確認"""
        result = file_list("nonexistent")

        assert result["success"] is False
        assert "存在しません" in result["error"]

    def test_list_file_instead_of_directory(self, temp_project):
        """ファイルに対してエラーを返すことを確認"""
        result = file_list("test.txt")

        assert result["success"] is False
        assert "ディレクトリではありません" in result["error"]

    def test_path_traversal_prevention(self, temp_project):
        """パストラバーサルを防止することを確認"""
        result = file_list("../")

        assert result["success"] is False
        assert "パストラバーサル" in result["error"]


class TestProjectRoot:
    """プロジェクトルート設定のテスト"""

    def test_project_root_not_set(self):
        """プロジェクトルート未設定時にエラーを返すことを確認"""
        import src.llm.tools.file_tools as ft
        original = ft._PROJECT_ROOT
        ft._PROJECT_ROOT = None

        try:
            result = file_read("test.txt")
            assert result["success"] is False
            assert "プロジェクトルートが設定されていません" in result["error"]
        finally:
            ft._PROJECT_ROOT = original


class TestToolResult:
    """ToolResult のテスト"""

    def test_to_dict_success(self):
        """成功時の辞書変換を確認"""
        result = ToolResult(
            success=True,
            data={"content": "test"},
        )
        d = result.to_dict()

        assert d["success"] is True
        assert d["data"] == {"content": "test"}
        assert "error" not in d
        assert "escalation_required" not in d

    def test_to_dict_error(self):
        """エラー時の辞書変換を確認"""
        result = ToolResult(
            success=False,
            error="Something went wrong",
        )
        d = result.to_dict()

        assert d["success"] is False
        assert d["error"] == "Something went wrong"
        assert "data" not in d

    def test_to_dict_escalation(self):
        """エスカレーション時の辞書変換を確認"""
        result = ToolResult(
            success=False,
            escalation_required=True,
            escalation_reason="Not allowed",
            requested_action="rm -rf /",
            risk_level="high",
        )
        d = result.to_dict()

        assert d["success"] is False
        assert d["escalation_required"] is True
        assert d["escalation_reason"] == "Not allowed"
        assert d["requested_action"] == "rm -rf /"
        assert d["risk_level"] == "high"

    def test_from_dict(self):
        """辞書からの変換を確認"""
        d = {
            "success": True,
            "data": {"key": "value"},
        }
        result = ToolResult.from_dict(d)

        assert result.success is True
        assert result.data == {"key": "value"}
        assert result.error is None
        assert result.escalation_required is False
