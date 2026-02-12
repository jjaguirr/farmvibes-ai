# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""farmvibes-ai workflow <subcommand> dispatcher."""

import argparse
import sys
from typing import List, Optional


VALIDATE_HELP = """\
Validate one or all FarmVibes.AI workflow YAML files without a running cluster.

Checks structure (required fields, task references, edge validity), detects
cycles in the DAG, validates parameter references, and suggests corrections
for misspelled workflow/op names.

Exit code 0 if all workflows are clean, 1 if any errors are found.
Warnings (e.g. unused parameters) print but do not affect the exit code."""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="farmvibes-ai workflow",
        description="FarmVibes.AI workflow development tools.",
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    validate = sub.add_parser(
        "validate",
        help="Validate workflow YAML files locally",
        description=VALIDATE_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    validate.add_argument(
        "path",
        nargs="?",
        default=None,
        metavar="PATH",
        help="Path to a workflow YAML file. Omit to validate all workflows.",
    )
    validate.add_argument(
        "--workflows-dir",
        default=None,
        metavar="DIR",
        help="Root workflows directory (auto-detected from PATH when omitted).",
    )
    validate.add_argument(
        "--ops-dir",
        default=None,
        metavar="DIR",
        help="Root ops directory (auto-detected when omitted).",
    )

    return parser


def dispatch(args: List[str]) -> int:
    """Parse workflow sub-args and dispatch.  Returns an exit code."""
    parser = _build_parser()
    parsed = parser.parse_args(args)

    if parsed.subcommand == "validate":
        from .wf_validator import validate_command
        return validate_command(
            path=parsed.path,
            workflows_dir=parsed.workflows_dir,
            ops_dir=parsed.ops_dir,
        )

    parser.print_help()
    return 1
