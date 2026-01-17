# CLI commands module
"""
CLIコマンド実装パッケージ

各コマンドは独立したモジュールとして実装され、
main.py から登録されます。
"""

from .register import register_agent_command

__all__ = [
    "register_agent_command",
]