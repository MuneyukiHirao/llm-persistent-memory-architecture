"""CLI-specific configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CLIConfig:
    """CLI settings."""

    default_output_format: str = "table"
    max_list_limit: int = 1000

    def validate(self) -> None:
        if self.default_output_format not in {"table", "json"}:
            raise ValueError("default_output_format は table/json のいずれかです")
        if self.max_list_limit <= 0:
            raise ValueError("max_list_limit は正の整数である必要があります")

