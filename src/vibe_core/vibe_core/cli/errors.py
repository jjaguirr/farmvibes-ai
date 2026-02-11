# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Smart error extraction and actionable error messages."""

import re
from typing import Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console(stderr=True)

ERROR_PATTERNS = {
    "auth": {
        "pattern": r"(authentication failed|unauthorized|az login|token.*expired|credential.*expired)",
        "message": "Authentication failed",
        "action": "Run `az login` and try again",
    },
    "port": {
        "pattern": r"(port.*already in use|bind.*address already in use|address already in use.*port)",
        "message": "Port already in use",
        "action": "Check what's using the port with `lsof -i :<port>` or stop the existing cluster",
    },
    "disk": {
        "pattern": r"(no space left|disk.*full|insufficient.*space)",
        "message": "Insufficient disk space",
        "action": "Free up space. Remove unused Docker images with `docker system prune`",
    },
    "network": {
        "pattern": r"(connection refused|network.*unreachable|timeout|could not resolve|dns.*fail)",
        "message": "Network connection failed",
        "action": "Check your internet connection and try again",
    },
    "docker": {
        "pattern": r"(docker.*not.*running|cannot connect to.*docker|docker daemon.*not.*running)",
        "message": "Docker not available",
        "action": "Start Docker and try again",
    },
    "kubectl": {
        "pattern": r"(kubectl.*not found|unable to connect to.*cluster|cluster.*not.*reachable)",
        "message": "Cannot connect to cluster",
        "action": "Ensure the cluster is running with `farmvibes-ai local status`",
    },
    "permission": {
        "pattern": r"(permission denied|access denied|forbidden)",
        "message": "Permission denied",
        "action": "Check file permissions or run with appropriate privileges",
    },
}


def detect_error_pattern(output: str) -> Optional[Tuple[str, str]]:
    output_lower = output.lower()
    for pattern_info in ERROR_PATTERNS.values():
        if re.search(pattern_info["pattern"], output_lower, re.IGNORECASE):
            return (pattern_info["message"], pattern_info["action"])
    return None


def show_error(title: str, output: str, log_file: Optional[str] = None):
    """Display a formatted error with smart extraction."""
    console.print()
    detected = detect_error_pattern(output)
    if detected:
        message, action = detected
        error_text = Text()
        error_text.append(f"✗ {title}\n\n", style="bold red")
        error_text.append(f"{message}\n\n", style="red")
        error_text.append("→ Fix: ", style="bold yellow")
        error_text.append(f"{action}\n", style="yellow")
        console.print(Panel(error_text, border_style="red", expand=False))
    else:
        console.print(Panel(Text(f"✗ {title}", style="bold red"), border_style="red", expand=False))

    lines = output.strip().split("\n")
    context = lines[-10:] if len(lines) > 10 else lines
    if context:
        console.print("\n[dim]Last output:[/dim]")
        for line in context:
            console.print(f"  [dim]│[/dim] {line}")

    if log_file:
        console.print(f"\n[dim]Full logs: {log_file}[/dim]")
    console.print()


def show_success(message: str):
    console.print(f"[green]✓[/green] {message}", style="green")
