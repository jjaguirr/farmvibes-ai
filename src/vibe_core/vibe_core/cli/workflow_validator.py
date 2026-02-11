# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CLI workflow validator.

Delegates parsing and structural validation to ``vibe_common.validation``
(the single source of truth shared with the server).  The CLI layer adds
three things that are not present in the server-side validator:

- Line-number reporting (maps errors back to YAML source positions)
- DAG cycle detection with full path reporting
- Edit-distance suggestions for misspelled op / workflow references
"""

import os
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

try:
    from vibe_common.validation import (
        WorkflowParser,
        WorkflowSpec,
        WorkflowSpecValidator,
        detect_cycle,
        extract_line_map,
        find_ops,
        find_workflows,
        line_for,
        suggest,
    )
    from vibe_common.validation.spec_parser import DEV_WORKFLOW_DIR, RUN_WORKFLOW_DIR

    _HAS_VIBE_COMMON = True
except ImportError:
    _HAS_VIBE_COMMON = False  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Default path resolution (mirrors spec_parser logic, without importing it)
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
_DEV_WORKFLOWS = os.path.join(_REPO_ROOT, "workflows")
_DEV_OPS = os.path.join(_REPO_ROOT, "ops")


def get_default_workflows_dir() -> str:
    return _DEV_WORKFLOWS if os.path.exists(_DEV_WORKFLOWS) else "/app/workflows"


def get_default_ops_dir() -> str:
    return _DEV_OPS if os.path.exists(_DEV_OPS) else "/app/ops"


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class ValidationIssue:
    severity: str           # "error" | "warning"
    file: str
    line: Optional[int]
    location: str
    message: str
    suggestion: Optional[str] = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_line_map(path: str) -> dict:
    try:
        with open(path) as fh:
            return extract_line_map(fh.read())
    except OSError:
        return {}


def _err(path: str, location: str, msg: str,
         line: Optional[int] = None,
         suggestion: Optional[str] = None) -> ValidationIssue:
    return ValidationIssue("error", path, line, location, msg, suggestion)


def _warn(path: str, location: str, msg: str,
          line: Optional[int] = None,
          suggestion: Optional[str] = None) -> ValidationIssue:
    return ValidationIssue("warning", path, line, location, msg, suggestion)


def _run_section(validator_method, spec, path, line_map, location_hint="validation"):
    """Call a single WorkflowSpecValidator class-method and convert any
    ValueError / TypeError it raises into a ValidationIssue list."""
    issues = []
    try:
        validator_method(spec)
    except (ValueError, TypeError) as exc:
        issues.append(_err(path, location_hint, str(exc)))
    return issues


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_file(
    path: str,
    workflows_dir: Optional[str] = None,
    ops_dir: Optional[str] = None,
    _visited: Optional[Set[str]] = None,
) -> List[ValidationIssue]:
    """Validate a single workflow YAML file.  Returns a list of issues."""
    if not _HAS_VIBE_COMMON:
        raise RuntimeError(
            "'farmvibes-ai workflow validate' requires vibe-common.\n"
            "Install with:  pip install -e src/vibe_common/"
        )

    if workflows_dir is None:
        workflows_dir = get_default_workflows_dir()
    if ops_dir is None:
        ops_dir = get_default_ops_dir()
    if _visited is None:
        _visited = set()

    abs_path = os.path.abspath(path)
    if abs_path in _visited:
        return []
    _visited.add(abs_path)

    issues: List[ValidationIssue] = []

    # Pre-pass: extract line numbers before the real parser runs.
    line_map = _load_line_map(abs_path)

    # ── Step 1: parse with the real WorkflowParser ───────────────────────
    avail_workflows = find_workflows(workflows_dir)
    avail_ops = find_ops(ops_dir)

    try:
        spec: WorkflowSpec = WorkflowParser.parse(abs_path, ops_dir, workflows_dir)
    except FileNotFoundError as exc:
        # A referenced op or sub-workflow file is missing.
        msg = str(exc)
        # Try to extract the missing name for a suggestion.
        missing = os.path.splitext(os.path.basename(msg.split("'")[-2] if "'" in msg else msg))[0]
        hint = suggest(missing, avail_ops + avail_workflows)
        # Find the task that references it.
        task_line = _find_task_line_for_error(msg, line_map)
        issues.append(_err(abs_path, "tasks", msg, task_line, hint))
        return issues
    except (ValueError, TypeError) as exc:
        issues.append(_err(abs_path, "workflow", str(exc)))
        return issues

    # ── Step 2: run each validation section independently ────────────────
    issues += _run_section(
        WorkflowSpecValidator._validate_sources, spec, abs_path, line_map, "sources"
    )
    issues += _run_section(
        WorkflowSpecValidator._validate_sinks, spec, abs_path, line_map, "sinks"
    )
    issues += _run_section(
        WorkflowSpecValidator._validate_edges, spec, abs_path, line_map, "edges"
    )
    issues += _run_section(
        WorkflowSpecValidator._validate_parameters, spec, abs_path, line_map, "parameters"
    )

    # ── Step 3: cycle detection (not in WorkflowSpecValidator) ───────────
    origins = [e.origin for e in spec.edges]
    dests = [e.destination for e in spec.edges]
    cycle = detect_cycle(list(spec.tasks.keys()), origins, dests)
    if cycle:
        issues.append(
            _err(abs_path, "edges", f"DAG contains a cycle: {' → '.join(cycle)}")
        )

    return issues


def validate_all(
    workflows_dir: Optional[str] = None,
    ops_dir: Optional[str] = None,
) -> List[Tuple[str, List[ValidationIssue]]]:
    """Validate every workflow YAML under *workflows_dir*.

    Returns a list of ``(abs_path, issues)`` for every file found.
    """
    if workflows_dir is None:
        workflows_dir = get_default_workflows_dir()
    if ops_dir is None:
        ops_dir = get_default_ops_dir()

    files: List[str] = []
    if os.path.isdir(workflows_dir):
        for root, _dirs, filenames in os.walk(workflows_dir):
            for fname in sorted(filenames):
                if fname.endswith(".yaml"):
                    files.append(os.path.join(root, fname))

    results: List[Tuple[str, List[ValidationIssue]]] = []
    for path in sorted(files):
        abs_path = os.path.abspath(path)
        # Fresh visited set per top-level file; sub-workflow recursion is
        # handled inside WorkflowParser.parse() itself.
        issues = validate_file(abs_path, workflows_dir, ops_dir, _visited=set())
        results.append((abs_path, issues))

    return results


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _find_task_line_for_error(error_msg: str, line_map: dict) -> Optional[int]:
    """Best-effort: scan line_map for a task key that appears in the error."""
    for key, lineno in line_map.items():
        if key.startswith("tasks.") and key.count(".") == 1:
            task_name = key.split(".")[1]
            if task_name in error_msg:
                return lineno
    return None
