"""Output formatting helpers for CLI."""

from __future__ import annotations

import json
from typing import Iterable, List, Sequence

import click


def format_table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    """Format a simple table with padded columns."""
    rows_list: List[List[str]] = [list(map(str, row)) for row in rows]
    widths = [len(str(h)) for h in headers]
    for row in rows_list:
        for idx, cell in enumerate(row):
            if idx >= len(widths):
                widths.append(len(cell))
            else:
                widths[idx] = max(widths[idx], len(cell))

    header_line = " ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers))
    separator = "-" * len(header_line)
    body_lines = [
        " ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))
        for row in rows_list
    ]
    return "\n".join([header_line, separator] + body_lines)


def echo_table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> None:
    """Echo a simple table."""
    click.echo(format_table(headers, rows))


def echo_json(data) -> None:
    """Echo JSON with UTF-8 characters preserved."""
    click.echo(json.dumps(data, ensure_ascii=False, indent=2))

