# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CLI commands for workflow operations."""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.table import Table

from vibe_core.workflow.validator import (
    IssueSeverity,
    ValidationResult,
    WorkflowValidator,
)

console = Console()

# Default paths relative to this file's location (for dev environment)
HERE = Path(__file__).parent.absolute()
CORE_DIR = HERE.parent
SRC_DIR = CORE_DIR.parent.parent
REPO_ROOT = SRC_DIR.parent

DEV_WORKFLOWS_DIR = REPO_ROOT / "workflows"
DEV_OPS_DIR = REPO_ROOT / "ops"

# Runtime paths (when running inside cluster)
RUN_WORKFLOWS_DIR = Path("/app/workflows")
RUN_OPS_DIR = Path("/app/ops")


def get_default_dirs() -> tuple:
    """Get the default workflows and ops directories."""
    # Prefer dev directories if they exist
    if DEV_WORKFLOWS_DIR.exists():
        return str(DEV_WORKFLOWS_DIR), str(DEV_OPS_DIR)
    elif RUN_WORKFLOWS_DIR.exists():
        return str(RUN_WORKFLOWS_DIR), str(RUN_OPS_DIR)
    else:
        # Fallback to current directory structure
        cwd = Path.cwd()
        return str(cwd / "workflows"), str(cwd / "ops")


def build_workflow_parser() -> argparse.ArgumentParser:
    """Build the argument parser for workflow commands."""
    parser = argparse.ArgumentParser(
        description="FarmVibes.AI workflow tools",
        prog="farmvibes-ai workflow",
    )

    subparsers = parser.add_subparsers(
        dest="workflow_action",
        help="Workflow action to perform",
        required=True,
    )

    # validate subcommand
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate workflow YAML files",
        description=(
            "Validate FarmVibes.AI workflow files for structural errors, "
            "missing references, cycles, and other issues. "
            "Runs locally without requiring a cluster."
        ),
    )
    validate_parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help=(
            "Path to a workflow file or directory to validate. "
            "If not specified, validates all workflows in the workflows directory."
        ),
    )
    validate_parser.add_argument(
        "--workflows-dir",
        type=str,
        default=None,
        help="Root directory containing workflow files (default: auto-detected)",
    )
    validate_parser.add_argument(
        "--ops-dir",
        type=str,
        default=None,
        help="Root directory containing operation files (default: auto-detected)",
    )
    validate_parser.add_argument(
        "--format",
        choices=["text", "table"],
        default="text",
        help="Output format (default: text)",
    )
    validate_parser.add_argument(
        "--warnings-as-errors",
        "-W",
        action="store_true",
        help="Treat warnings as errors (affects exit code)",
    )
    validate_parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Don't recursively validate sub-workflows (default: recursive)",
    )
    validate_parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Only output errors and summary",
    )

    return parser


def print_result_text(result: ValidationResult, quiet: bool = False):
    """Print validation result in text format."""
    for issue in result.issues:
        if quiet and issue.severity == IssueSeverity.WARNING:
            continue
        style = "red" if issue.severity == IssueSeverity.ERROR else "yellow"
        console.print(f"[{style}]{issue}[/{style}]")


def print_result_table(result: ValidationResult, quiet: bool = False):
    """Print validation result in table format."""
    if not result.issues:
        return

    table = Table(title="Validation Results")
    table.add_column("File", style="cyan")
    table.add_column("Line", style="magenta")
    table.add_column("Severity", style="bold")
    table.add_column("Message")
    table.add_column("Suggestion", style="green")

    for issue in result.issues:
        if quiet and issue.severity == IssueSeverity.WARNING:
            continue
        severity_style = "red" if issue.severity == IssueSeverity.ERROR else "yellow"
        table.add_row(
            issue.file,
            str(issue.line) if issue.line else "-",
            f"[{severity_style}]{issue.severity.value}[/{severity_style}]",
            issue.message + (f" (in {issue.context})" if issue.context else ""),
            issue.suggestion or "-",
        )

    console.print(table)


def cmd_validate(args: argparse.Namespace) -> int:
    """Execute the validate command."""
    # Determine directories
    default_wf_dir, default_ops_dir = get_default_dirs()
    workflows_dir = args.workflows_dir or default_wf_dir
    ops_dir = args.ops_dir or default_ops_dir

    # Check directories exist
    if not os.path.isdir(workflows_dir):
        console.print(f"[red]Error: Workflows directory not found: {workflows_dir}[/red]")
        return 1

    if not os.path.isdir(ops_dir):
        console.print(f"[red]Error: Operations directory not found: {ops_dir}[/red]")
        return 1

    # Determine what to validate
    target_path = args.path
    if target_path:
        # If path is relative and doesn't exist, try relative to workflows_dir
        if not os.path.isabs(target_path) and not os.path.exists(target_path):
            candidate = os.path.join(workflows_dir, target_path)
            if os.path.exists(candidate):
                target_path = candidate
            elif os.path.exists(candidate + ".yaml"):
                target_path = candidate + ".yaml"

    # Create validator and run
    recursive = not args.no_recursive
    validator = WorkflowValidator(workflows_dir, ops_dir, recursive=recursive)

    if not args.quiet:
        if target_path:
            console.print(f"Validating: {target_path}")
        else:
            console.print(f"Validating all workflows in: {workflows_dir}")

    result = validator.validate_all(target_path)

    # Print results
    if args.format == "table":
        print_result_table(result, args.quiet)
    else:
        print_result_text(result, args.quiet)

    # Print summary
    error_count = len(result.errors)
    warning_count = len(result.warnings)

    if error_count == 0 and warning_count == 0:
        if not args.quiet:
            console.print("[green]All workflows valid.[/green]")
        return 0

    summary_parts = []
    if error_count > 0:
        summary_parts.append(f"[red]{error_count} error{'s' if error_count != 1 else ''}[/red]")
    if warning_count > 0:
        summary_parts.append(
            f"[yellow]{warning_count} warning{'s' if warning_count != 1 else ''}[/yellow]"
        )

    console.print(f"\nValidation complete: {', '.join(summary_parts)}")

    # Determine exit code
    if result.has_errors:
        return 1
    if args.warnings_as_errors and result.has_warnings:
        return 1
    return 0


def dispatch(args: argparse.Namespace) -> int:
    """Dispatch to the appropriate workflow command handler."""
    if args.workflow_action == "validate":
        return cmd_validate(args)
    else:
        console.print(f"[red]Unknown workflow action: {args.workflow_action}[/red]")
        return 1


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for workflow commands."""
    parser = build_workflow_parser()
    args = parser.parse_args(argv)
    return dispatch(args)


if __name__ == "__main__":
    sys.exit(main())
