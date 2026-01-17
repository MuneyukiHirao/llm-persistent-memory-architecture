"""Minimal progress helpers for CLI."""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass

import click


@dataclass
class StepProgress:
    """Simple step-based progress reporter."""

    total_steps: int
    current_step: int = 0

    def step(self, message: str) -> None:
        """Advance one step with a message."""
        self.current_step += 1
        click.echo(f"[{self.current_step}/{self.total_steps}] {message}")


class Timer:
    """Context manager for timing blocks."""

    def __init__(self, label: str) -> None:
        self.label = label
        self._start = 0.0

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = time.time() - self._start
        click.echo(f"{self.label} 完了 ({elapsed:.2f}秒)")


class Spinner:
    """Simple animated spinner for long-running operations.

    Usage:
        with Spinner("処理中"):
            # long running operation
            time.sleep(5)
    """

    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, message: str = "処理中", show_elapsed: bool = True):
        """Initialize spinner.

        Args:
            message: Message to display next to spinner
            show_elapsed: Whether to show elapsed time
        """
        self.message = message
        self.show_elapsed = show_elapsed
        self._stop_event = threading.Event()
        self._thread = None
        self._start_time = 0.0

    def __enter__(self):
        """Start the spinner."""
        self._start_time = time.time()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the spinner."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)

        # Clear the line
        sys.stdout.write("\r" + " " * 80 + "\r")
        sys.stdout.flush()

        # Show completion message
        elapsed = time.time() - self._start_time
        if exc_type is None:
            click.echo(f"✓ {self.message} 完了 ({elapsed:.1f}秒)")
        else:
            click.echo(f"✗ {self.message} 失敗 ({elapsed:.1f}秒)")

    def _spin(self):
        """Internal method to animate the spinner."""
        idx = 0
        while not self._stop_event.is_set():
            elapsed = int(time.time() - self._start_time)
            frame = self.FRAMES[idx % len(self.FRAMES)]

            if self.show_elapsed:
                msg = f"\r{frame} {self.message}... ({elapsed}秒経過)"
            else:
                msg = f"\r{frame} {self.message}..."

            sys.stdout.write(msg)
            sys.stdout.flush()

            idx += 1
            time.sleep(0.1)

    def update_message(self, message: str):
        """Update the spinner message dynamically.

        Args:
            message: New message to display
        """
        self.message = message

