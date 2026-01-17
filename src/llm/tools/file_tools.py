# ファイル操作ツール
# 実装仕様: docs/phase1-implementation-spec.ja.md
"""
ファイル操作ツール: file_read, file_write, file_list

セキュリティ機能:
- パストラバーサル防止: プロジェクトディレクトリ外へのアクセスを禁止
- 絶対パス変換: すべてのパスを正規化して検証
"""

import fnmatch
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# ToolResult は __init__.py で定義されているため、循環参照を避けるため遅延インポート
# 実際の使用時は from src.llm.tools import ToolResult でインポート

logger = logging.getLogger(__name__)

# プロジェクトルートディレクトリ（デフォルト）
# 実際の使用時は set_project_root() で設定
_PROJECT_ROOT: Optional[Path] = None


def set_project_root(path: str) -> None:
    """プロジェクトルートディレクトリを設定

    Args:
        path: プロジェクトルートディレクトリのパス
    """
    global _PROJECT_ROOT
    _PROJECT_ROOT = Path(path).resolve()
    logger.info(f"プロジェクトルート設定: {_PROJECT_ROOT}")


def get_project_root() -> Path:
    """プロジェクトルートディレクトリを取得

    Returns:
        プロジェクトルートディレクトリのパス

    Raises:
        RuntimeError: プロジェクトルートが設定されていない場合
    """
    if _PROJECT_ROOT is None:
        raise RuntimeError(
            "プロジェクトルートが設定されていません。set_project_root() を呼び出してください"
        )
    return _PROJECT_ROOT


def _validate_path(path: str) -> tuple[bool, Path, Optional[str]]:
    """パスを検証し、プロジェクト内かどうかを確認

    Args:
        path: 検証するパス

    Returns:
        (is_valid, resolved_path, error_message)
    """
    try:
        project_root = get_project_root()
    except RuntimeError as e:
        return False, Path(path), str(e)

    try:
        # パスを解決（シンボリックリンクも解決）
        resolved = (project_root / path).resolve()

        # プロジェクトルート内かどうかを確認
        try:
            resolved.relative_to(project_root)
            return True, resolved, None
        except ValueError:
            return False, resolved, f"パストラバーサル検出: {path} はプロジェクト外を参照しています"

    except Exception as e:
        return False, Path(path), f"パス解決エラー: {e}"


def _make_tool_result(
    success: bool,
    data: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    escalation_required: bool = False,
    escalation_reason: Optional[str] = None,
    requested_action: Optional[str] = None,
    risk_level: Optional[str] = None,
) -> Dict[str, Any]:
    """ToolResult 互換の辞書を生成

    循環参照を避けるため、直接辞書を返す
    """
    result: Dict[str, Any] = {"success": success}
    if data is not None:
        result["data"] = data
    if error is not None:
        result["error"] = error
    if escalation_required:
        result["escalation_required"] = True
        if escalation_reason:
            result["escalation_reason"] = escalation_reason
        if requested_action:
            result["requested_action"] = requested_action
        if risk_level:
            result["risk_level"] = risk_level
    return result


def file_read(path: str) -> Dict[str, Any]:
    """ファイルを読み込む

    Args:
        path: 読み込むファイルのパス（プロジェクトルートからの相対パス）

    Returns:
        ToolResult 互換の辞書
        - 成功時: {"success": True, "data": {"content": "...", "path": "..."}}
        - 失敗時: {"success": False, "error": "..."}
    """
    is_valid, resolved_path, error = _validate_path(path)
    if not is_valid:
        logger.warning(f"file_read パス検証失敗: {error}")
        return _make_tool_result(success=False, error=error)

    try:
        if not resolved_path.exists():
            return _make_tool_result(
                success=False,
                error=f"ファイルが存在しません: {path}"
            )

        if not resolved_path.is_file():
            return _make_tool_result(
                success=False,
                error=f"ファイルではありません: {path}"
            )

        content = resolved_path.read_text(encoding="utf-8")
        logger.info(f"file_read 成功: {path} ({len(content)} 文字)")

        return _make_tool_result(
            success=True,
            data={
                "content": content,
                "path": str(resolved_path),
                "size": len(content),
            }
        )

    except PermissionError:
        return _make_tool_result(
            success=False,
            error=f"ファイルの読み取り権限がありません: {path}"
        )
    except UnicodeDecodeError:
        return _make_tool_result(
            success=False,
            error=f"ファイルの文字エンコーディングを解読できません（UTF-8以外）: {path}"
        )
    except Exception as e:
        logger.error(f"file_read エラー: {path}, {e}")
        return _make_tool_result(
            success=False,
            error=f"ファイル読み込みエラー: {e}"
        )


def file_write(path: str, content: str) -> Dict[str, Any]:
    """ファイルを書き込む

    Args:
        path: 書き込むファイルのパス（プロジェクトルートからの相対パス）
        content: 書き込む内容

    Returns:
        ToolResult 互換の辞書
        - 成功時: {"success": True, "data": {"path": "...", "size": ...}}
        - 失敗時: {"success": False, "error": "..."}
    """
    is_valid, resolved_path, error = _validate_path(path)
    if not is_valid:
        logger.warning(f"file_write パス検証失敗: {error}")
        return _make_tool_result(success=False, error=error)

    try:
        # 親ディレクトリが存在しない場合は作成
        resolved_path.parent.mkdir(parents=True, exist_ok=True)

        resolved_path.write_text(content, encoding="utf-8")
        logger.info(f"file_write 成功: {path} ({len(content)} 文字)")

        return _make_tool_result(
            success=True,
            data={
                "path": str(resolved_path),
                "size": len(content),
            }
        )

    except PermissionError:
        return _make_tool_result(
            success=False,
            error=f"ファイルの書き込み権限がありません: {path}"
        )
    except Exception as e:
        logger.error(f"file_write エラー: {path}, {e}")
        return _make_tool_result(
            success=False,
            error=f"ファイル書き込みエラー: {e}"
        )


def file_list(path: str, pattern: Optional[str] = None) -> Dict[str, Any]:
    """ディレクトリ内のファイル一覧を取得

    Args:
        path: ディレクトリのパス（プロジェクトルートからの相対パス）
        pattern: ファイル名のパターン（オプション、fnmatch 形式）

    Returns:
        ToolResult 互換の辞書
        - 成功時: {"success": True, "data": {"files": [...], "directories": [...], "path": "..."}}
        - 失敗時: {"success": False, "error": "..."}
    """
    is_valid, resolved_path, error = _validate_path(path)
    if not is_valid:
        logger.warning(f"file_list パス検証失敗: {error}")
        return _make_tool_result(success=False, error=error)

    try:
        if not resolved_path.exists():
            return _make_tool_result(
                success=False,
                error=f"ディレクトリが存在しません: {path}"
            )

        if not resolved_path.is_dir():
            return _make_tool_result(
                success=False,
                error=f"ディレクトリではありません: {path}"
            )

        files: List[str] = []
        directories: List[str] = []

        for item in resolved_path.iterdir():
            name = item.name

            # パターンが指定されている場合はフィルタリング
            if pattern and not fnmatch.fnmatch(name, pattern):
                continue

            if item.is_file():
                files.append(name)
            elif item.is_dir():
                directories.append(name)

        # ソートして返す
        files.sort()
        directories.sort()

        logger.info(f"file_list 成功: {path} ({len(files)} ファイル, {len(directories)} ディレクトリ)")

        return _make_tool_result(
            success=True,
            data={
                "files": files,
                "directories": directories,
                "path": str(resolved_path),
                "total_files": len(files),
                "total_directories": len(directories),
            }
        )

    except PermissionError:
        return _make_tool_result(
            success=False,
            error=f"ディレクトリの読み取り権限がありません: {path}"
        )
    except Exception as e:
        logger.error(f"file_list エラー: {path}, {e}")
        return _make_tool_result(
            success=False,
            error=f"ディレクトリ一覧取得エラー: {e}"
        )


# =============================================================================
# ツール定義（ToolExecutor 用）
# =============================================================================

# 遅延インポートで Tool を取得
def _get_tool_class():
    from ..tool_executor import Tool
    return Tool


# file_read ツール定義
FILE_READ_TOOL = None  # 遅延初期化


def get_file_read_tool():
    """file_read ツール定義を取得"""
    global FILE_READ_TOOL
    if FILE_READ_TOOL is None:
        Tool = _get_tool_class()
        FILE_READ_TOOL = Tool(
            name="file_read",
            description="ファイルを読み込む。プロジェクトディレクトリ内のファイルのみ読み込み可能。",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "読み込むファイルのパス（プロジェクトルートからの相対パス）"
                    }
                },
                "required": ["path"]
            }
        )
    return FILE_READ_TOOL


# file_write ツール定義
FILE_WRITE_TOOL = None  # 遅延初期化


def get_file_write_tool():
    """file_write ツール定義を取得"""
    global FILE_WRITE_TOOL
    if FILE_WRITE_TOOL is None:
        Tool = _get_tool_class()
        FILE_WRITE_TOOL = Tool(
            name="file_write",
            description="ファイルを書き込む。プロジェクトディレクトリ内のファイルのみ書き込み可能。",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "書き込むファイルのパス（プロジェクトルートからの相対パス）"
                    },
                    "content": {
                        "type": "string",
                        "description": "書き込む内容"
                    }
                },
                "required": ["path", "content"]
            }
        )
    return FILE_WRITE_TOOL


# file_list ツール定義
FILE_LIST_TOOL = None  # 遅延初期化


def get_file_list_tool():
    """file_list ツール定義を取得"""
    global FILE_LIST_TOOL
    if FILE_LIST_TOOL is None:
        Tool = _get_tool_class()
        FILE_LIST_TOOL = Tool(
            name="file_list",
            description="ディレクトリ内のファイル一覧を取得。プロジェクトディレクトリ内のみ。",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "ディレクトリのパス（プロジェクトルートからの相対パス）"
                    },
                    "pattern": {
                        "type": "string",
                        "description": "ファイル名のパターン（オプション、fnmatch 形式、例: '*.py'）"
                    }
                },
                "required": ["path"]
            }
        )
    return FILE_LIST_TOOL
