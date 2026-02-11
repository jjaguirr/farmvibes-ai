# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Progress tracking for long-running CLI operations."""

import time
from contextlib import contextmanager
from typing import Optional

from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text

console = Console(stderr=True)


class ProgressTracker:
    """Track progress through multi-step operations with live display."""

    def __init__(self, title: str):
        self.title = title
        self.steps: list = []  # (name, status, duration)
        self.current_step: Optional[str] = None
        self.current_start: float = 0

    def start(self):
        """Start the progress display."""
        console.print(f"\n[bold]{self.title}[/bold]\n")

    def _format_duration(self, seconds: float) -> str:
        if seconds < 60:
            return f"{int(seconds)}s"
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"

    def _render(self, elapsed: float = 0.0) -> Text:
        text = Text()
        for step_name, status, duration in self.steps:
            if status == "done":
                text.append("✓ ", style="green")
                text.append(f"{step_name} ", style="dim")
                text.append(f"({self._format_duration(duration)})\n", style="dim")
            elif status == "failed":
                text.append("✗ ", style="red")
                text.append(f"{step_name}\n", style="red")
        if self.current_step:
            spinner = Spinner("dots", style="cyan")
            text.append(spinner.render(elapsed))
            text.append(f" {self.current_step}... ", style="cyan")
            text.append(f"({self._format_duration(elapsed)})", style="dim cyan")
        return text

    @contextmanager
    def step(self, name: str):
        """Context manager for a single step."""
        self.current_step = name
        self.current_start = time.time()
        try:
            yield
            duration = time.time() - self.current_start
            self.steps.append((name, "done", duration))
            console.print(
                f"[green]✓[/green] [dim]{name} ({self._format_duration(duration)})[/dim]"
            )
        except Exception:
            self.steps.append((name, "failed", 0))
            console.print(f"[red]✗[/red] [red]{name}[/red]")
            raise
        finally:
            self.current_step = None

    @contextmanager
    def live_step(self, name: str):
        """Context manager for long-running steps with a live spinner."""
        self.current_step = name
        self.current_start = time.time()
        with Live(console=console, refresh_per_second=4) as live:
            try:
                while True:
                    elapsed = time.time() - self.current_start
                    live.update(self._render(elapsed))
                    yield live
                    break
                duration = time.time() - self.current_start
                self.steps.append((name, "done", duration))
                self.current_step = None
                live.update(self._render())
            except Exception:
                self.steps.append((name, "failed", 0))
                self.current_step = None
                live.update(self._render())
                raise
