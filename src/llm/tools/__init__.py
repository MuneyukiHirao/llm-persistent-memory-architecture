# ツール実装モジュール
# 実装仕様: docs/phase1-implementation-spec.ja.md
# アーキテクチャ: docs/architecture.ja.md
"""
ファイル操作ツールとbashツールを提供するモジュール

設計方針:
- セキュリティ: パストラバーサル防止、許可リスト方式
- エスカレーション: 危険な操作は escalation_required を返す
- 互換性: 既存の ToolExecutor と連携可能

使用例:
    from src.llm.tools import (
        ToolResult,
        file_read,
        file_write,
        file_list,
        bash_execute,
        set_project_root,
    )

    # プロジェクトルートを設定
    set_project_root("/path/to/project")

    # ファイル読み込み
    result = file_read("src/main.py")
    if result["success"]:
        print(result["data"]["content"])

    # bashコマンド実行
    result = bash_execute("pytest tests/")
    if result["escalation_required"]:
        print(f"エスカレーション必要: {result['escalation_reason']}")
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .file_tools import (
    file_read,
    file_write,
    file_list,
    set_project_root,
    get_project_root,
    get_file_read_tool,
    get_file_write_tool,
    get_file_list_tool,
)
from .bash_tools import (
    bash_execute,
    get_bash_execute_tool,
    add_allowed_command,
    remove_allowed_command,
    get_allowed_commands,
    ALLOWED_COMMANDS,
)


@dataclass
class ToolResult:
    """ツール実行結果（エスカレーション機構対応）

    Attributes:
        success: 実行が成功したかどうか
        data: 実行結果データ（成功時）
        error: エラーメッセージ（失敗時）
        escalation_required: エスカレーションが必要かどうか
        escalation_reason: エスカレーション理由
        requested_action: 要求されたアクション
        risk_level: リスクレベル（low/medium/high）
    """

    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    escalation_required: bool = False
    escalation_reason: Optional[str] = None
    requested_action: Optional[str] = None
    risk_level: Optional[str] = None  # low/medium/high

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        result: Dict[str, Any] = {
            "success": self.success,
        }
        if self.data is not None:
            result["data"] = self.data
        if self.error is not None:
            result["error"] = self.error
        if self.escalation_required:
            result["escalation_required"] = True
            if self.escalation_reason:
                result["escalation_reason"] = self.escalation_reason
            if self.requested_action:
                result["requested_action"] = self.requested_action
            if self.risk_level:
                result["risk_level"] = self.risk_level
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolResult":
        """辞書からToolResultを生成"""
        return cls(
            success=data.get("success", False),
            data=data.get("data"),
            error=data.get("error"),
            escalation_required=data.get("escalation_required", False),
            escalation_reason=data.get("escalation_reason"),
            requested_action=data.get("requested_action"),
            risk_level=data.get("risk_level"),
        )


__all__ = [
    # データクラス
    "ToolResult",
    # ファイルツール関数
    "file_read",
    "file_write",
    "file_list",
    "set_project_root",
    "get_project_root",
    # ファイルツール定義ゲッター
    "get_file_read_tool",
    "get_file_write_tool",
    "get_file_list_tool",
    # bashツール関数
    "bash_execute",
    "add_allowed_command",
    "remove_allowed_command",
    "get_allowed_commands",
    "ALLOWED_COMMANDS",
    # bashツール定義ゲッター
    "get_bash_execute_tool",
]
