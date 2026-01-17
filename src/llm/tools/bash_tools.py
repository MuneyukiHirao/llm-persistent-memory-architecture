# bashツール
# 実装仕様: docs/phase1-implementation-spec.ja.md
"""
bashツール: bash_execute

セキュリティ機能:
- 許可リスト: 安全なコマンドのみ実行可能
- 危険パターンブロック: 破壊的なコマンドをブロック
- エスカレーション機構: 許可リストにないコマンドは承認を要求
"""

import logging
import re
import shlex
import subprocess
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# =============================================================================
# 許可リストと危険パターン
# =============================================================================

# 許可されたコマンド
ALLOWED_COMMANDS: Set[str] = {
    "pytest",
    "python",
    "pip",
    "docker",
    "git",
    "ls",
    "cat",
    "echo",
    "mkdir",
    "cp",
    "mv",
}

# 危険なパターン（正規表現）
DANGEROUS_PATTERNS: List[tuple[str, str, str]] = [
    # (パターン, 説明, リスクレベル)
    (r"rm\s+-rf\s+/", "ルートディレクトリの再帰的削除", "high"),
    (r"rm\s+-rf\s+~", "ホームディレクトリの再帰的削除", "high"),
    (r"rm\s+-rf\s+\*", "カレントディレクトリの再帰的削除", "high"),
    (r"sudo\s+", "sudo コマンドの使用", "high"),
    (r"chmod\s+777", "全権限付与", "medium"),
    (r">\s*/etc/", "/etc への書き込み", "high"),
    (r">>\s*/etc/", "/etc への追記", "high"),
    (r":\(\)\s*\{\s*:\|\:\s*&\s*\}\s*;", "フォークボム", "high"),
    (r"mkfs\.", "ファイルシステムのフォーマット", "high"),
    (r"dd\s+if=.+of=/dev/", "デバイスへの直接書き込み", "high"),
    (r">\s*/dev/sd[a-z]", "ディスクデバイスへの書き込み", "high"),
    (r"curl\s+.+\|\s*(bash|sh)", "リモートスクリプトの実行", "high"),
    (r"wget\s+.+\|\s*(bash|sh)", "リモートスクリプトの実行", "high"),
    (r"eval\s+", "eval コマンドの使用", "medium"),
]


def _make_tool_result(
    success: bool,
    data: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    escalation_required: bool = False,
    escalation_reason: Optional[str] = None,
    requested_action: Optional[str] = None,
    risk_level: Optional[str] = None,
) -> Dict[str, Any]:
    """ToolResult 互換の辞書を生成"""
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


def _extract_command(command_str: str) -> Optional[str]:
    """コマンド文字列から最初のコマンド名を抽出

    Args:
        command_str: コマンド文字列

    Returns:
        コマンド名（抽出できない場合は None）
    """
    try:
        # パイプやリダイレクトを考慮して最初のコマンドを取得
        # 簡易的な処理として、最初の単語を取得
        parts = shlex.split(command_str)
        if parts:
            # パスからコマンド名のみを抽出
            cmd = parts[0].split("/")[-1]
            return cmd
        return None
    except ValueError:
        # シェル構文エラー
        # 空白で分割してみる
        parts = command_str.strip().split()
        if parts:
            return parts[0].split("/")[-1]
        return None


def _check_dangerous_patterns(command: str) -> Optional[tuple[str, str]]:
    """危険なパターンをチェック

    Args:
        command: チェックするコマンド

    Returns:
        (説明, リスクレベル) のタプル、危険でない場合は None
    """
    for pattern, description, risk_level in DANGEROUS_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            return description, risk_level
    return None


def _is_command_allowed(command_name: str) -> bool:
    """コマンドが許可リストに含まれているか確認

    Args:
        command_name: コマンド名

    Returns:
        許可されている場合は True
    """
    return command_name in ALLOWED_COMMANDS


def bash_execute(
    command: str,
    timeout: int = 30,
    working_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """bashコマンドを実行

    セキュリティチェック:
    1. 危険なパターンをブロック
    2. 許可リストにないコマンドはエスカレーション

    Args:
        command: 実行するコマンド
        timeout: タイムアウト（秒）
        working_dir: 作業ディレクトリ（オプション）

    Returns:
        ToolResult 互換の辞書
        - 成功時: {"success": True, "data": {"stdout": "...", "stderr": "...", "return_code": 0}}
        - 失敗時: {"success": False, "error": "..."}
        - エスカレーション: {"success": False, "escalation_required": True, ...}
    """
    # 空コマンドチェック
    if not command or not command.strip():
        return _make_tool_result(
            success=False,
            error="コマンドが空です"
        )

    # 危険パターンチェック
    dangerous = _check_dangerous_patterns(command)
    if dangerous:
        description, risk_level = dangerous
        logger.warning(f"危険なコマンドをブロック: {command} ({description})")
        return _make_tool_result(
            success=False,
            error=f"危険なコマンドがブロックされました: {description}",
            escalation_required=True,
            escalation_reason=f"セキュリティポリシー違反: {description}",
            requested_action=command,
            risk_level=risk_level,
        )

    # コマンド名を抽出
    command_name = _extract_command(command)
    if not command_name:
        return _make_tool_result(
            success=False,
            error="コマンド名を抽出できません"
        )

    # 許可リストチェック
    if not _is_command_allowed(command_name):
        logger.info(f"許可リストにないコマンド: {command_name}")
        return _make_tool_result(
            success=False,
            escalation_required=True,
            escalation_reason=f"コマンド '{command_name}' は許可リストに含まれていません",
            requested_action=command,
            risk_level="medium",
        )

    # コマンド実行
    try:
        logger.info(f"bash_execute 開始: {command}")

        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=working_dir,
        )

        logger.info(f"bash_execute 完了: return_code={result.returncode}")

        return _make_tool_result(
            success=result.returncode == 0,
            data={
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
                "command": command,
            },
            error=result.stderr if result.returncode != 0 else None,
        )

    except subprocess.TimeoutExpired:
        logger.warning(f"bash_execute タイムアウト: {command}")
        return _make_tool_result(
            success=False,
            error=f"コマンドがタイムアウトしました（{timeout}秒）"
        )
    except Exception as e:
        logger.error(f"bash_execute エラー: {command}, {e}")
        return _make_tool_result(
            success=False,
            error=f"コマンド実行エラー: {e}"
        )


def add_allowed_command(command_name: str) -> None:
    """許可リストにコマンドを追加

    Args:
        command_name: 追加するコマンド名
    """
    ALLOWED_COMMANDS.add(command_name)
    logger.info(f"許可リストにコマンド追加: {command_name}")


def remove_allowed_command(command_name: str) -> bool:
    """許可リストからコマンドを削除

    Args:
        command_name: 削除するコマンド名

    Returns:
        削除成功なら True
    """
    if command_name in ALLOWED_COMMANDS:
        ALLOWED_COMMANDS.discard(command_name)
        logger.info(f"許可リストからコマンド削除: {command_name}")
        return True
    return False


def get_allowed_commands() -> Set[str]:
    """許可リストを取得

    Returns:
        許可されたコマンドのセット（コピー）
    """
    return ALLOWED_COMMANDS.copy()


# =============================================================================
# ツール定義（ToolExecutor 用）
# =============================================================================

BASH_EXECUTE_TOOL = None  # 遅延初期化


def get_bash_execute_tool():
    """bash_execute ツール定義を取得"""
    global BASH_EXECUTE_TOOL
    if BASH_EXECUTE_TOOL is None:
        from ..tool_executor import Tool
        BASH_EXECUTE_TOOL = Tool(
            name="bash_execute",
            description="bashコマンドを実行する。許可リストにあるコマンドのみ実行可能。許可されていないコマンドはエスカレーションが必要。",
            input_schema={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "実行するbashコマンド"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "タイムアウト秒数（デフォルト: 30）",
                        "default": 30
                    },
                    "working_dir": {
                        "type": "string",
                        "description": "作業ディレクトリ（オプション）"
                    }
                },
                "required": ["command"]
            }
        )
    return BASH_EXECUTE_TOOL
