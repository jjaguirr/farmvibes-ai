# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CLI handler for 'farmvibes-ai workflow' subcommands."""

import argparse
import os
import sys
from typing import List, Optional

from rich.console import Console

from .workflow_validator import (
    ValidationIssue,
    get_default_ops_dir,
    get_default_workflows_dir,
    validate_all,
    validate_file,
)

console = Console()
err_console = Console(stderr=True)


class WorkflowCliParser:
    """Argument parser for 'farmvibes-ai workflow <command>'."""

    SUPPORTED_COMMANDS = [
        ("validate", "Validate one or all workflow YAML files (no cluster needed)"),
    ]

    def __init__(self) -> None:
        self.parser = self._build()

    def _build(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            prog="farmvibes-ai workflow",
            description="FarmVibes.AI workflow tooling (runs locally, no cluster required).",
        )
        sub = parser.add_subparsers(dest="action", help="Action to perform", required=True)

        validate_p = sub.add_parser(
            "validate",
            help="Validate one or all workflow YAML files",
        )
        validate_p.add_argument(
            "path",
            nargs="?",
            default=None,
            help=(
                "Path to a workflow YAML file to validate. "
                "Omit to validate every workflow under the workflows directory."
            ),
        )
        validate_p.add_argument(
            "--workflows-dir",
            default=None,
            metavar="DIR",
            help="Workflows directory (auto-detected from repo layout if not given)",
        )
        validate_p.add_argument(
            "--ops-dir",
            default=None,
            metavar="DIR",
            help="Ops directory (auto-detected from repo layout if not given)",
        )

        return parser

    def parse(self, args: List[str]) -> argparse.Namespace:
        return self.parser.parse_args(args)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _display_path(path: str, workflows_dir: str) -> str:
    """Return a short display path: relative to workflows_dir if inside it, else absolute."""
    try:
        rel = os.path.relpath(path, workflows_dir)
        # If relpath climbs up with '..', the file is outside the tree — use absolute
        if rel.startswith(".."):
            return path
        return rel
    except ValueError:
        return path


def _print_file_results(path: str, issues: List[ValidationIssue], workflows_dir: str) -> None:
    rel = _display_path(path, workflows_dir)

    errors = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity == "warning"]

    if not issues:
        console.print(f"[green]✓[/green] [dim]{rel}[/dim]")
        return

    if errors:
        console.print(f"[red]✗[/red] [bold]{rel}[/bold]")
    else:
        console.print(f"[yellow]⚠[/yellow] [bold]{rel}[/bold]")

    for issue in issues:
        sev_style = "red" if issue.severity == "error" else "yellow"
        sev_label = issue.severity.upper().ljust(7)
        line_part = f"line {issue.line}  " if issue.line else ""
        console.print(
            f"  [{sev_style}]{sev_label}[/{sev_style}] "
            f"[dim]{line_part}[/dim]"
            f"[bold]{issue.location}[/bold]: {issue.message}"
        )
        if issue.suggestion:
            console.print(f"          [cyan]Did you mean:[/cyan] {issue.suggestion}")


def _print_summary(
    total_files: int, error_files: int, warning_files: int, total_errors: int, total_warnings: int
) -> None:
    console.print()
    if error_files == 0 and warning_files == 0:
        console.print(
            f"[green bold]All {total_files} workflow(s) valid.[/green bold]"
        )
    else:
        parts = []
        if error_files:
            parts.append(f"[red]{error_files} file(s) with errors[/red]")
        if warning_files:
            parts.append(f"[yellow]{warning_files} file(s) with warnings[/yellow]")
        console.print(
            f"Checked {total_files} file(s): " + ", ".join(parts)
            + f"  ({total_errors} error(s), {total_warnings} warning(s))"
        )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def dispatch(args: argparse.Namespace) -> int:
    """Run the requested workflow command.  Returns an exit code (0 = success)."""
    if args.action != "validate":
        err_console.print(f"[red]Unknown workflow command: {args.action}[/red]")
        return 1

    workflows_dir: str = args.workflows_dir or get_default_workflows_dir()
    ops_dir: str = args.ops_dir or get_default_ops_dir()

    if args.path:
        # Single-file mode
        path = os.path.abspath(args.path)
        if not os.path.exists(path):
            err_console.print(f"[red]File not found:[/red] {path}")
            return 1
        issues = validate_file(path, workflows_dir, ops_dir)
        _print_file_results(path, issues, workflows_dir)
        has_errors = any(i.severity == "error" for i in issues)
        return 1 if has_errors else 0

    # All-workflows mode
    if not os.path.isdir(workflows_dir):
        err_console.print(
            f"[red]Workflows directory not found:[/red] {workflows_dir}\n"
            "Pass --workflows-dir to specify the location."
        )
        return 1

    results = validate_all(workflows_dir, ops_dir)

    if not results:
        err_console.print(
            f"[yellow]No workflow YAML files found under {workflows_dir}[/yellow]"
        )
        return 0

    total_errors = 0
    total_warnings = 0
    error_files = 0
    warning_files = 0

    for path, issues in results:
        _print_file_results(path, issues, workflows_dir)
        file_errors = sum(1 for i in issues if i.severity == "error")
        file_warnings = sum(1 for i in issues if i.severity == "warning")
        total_errors += file_errors
        total_warnings += file_warnings
        if file_errors:
            error_files += 1
        elif file_warnings:
            warning_files += 1

    _print_summary(len(results), error_files, warning_files, total_errors, total_warnings)
    return 1 if total_errors > 0 else 0
