# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Standalone workflow YAML validator for FarmVibes.AI.

Runs without a cluster or any server-side dependencies (no vibe_common, no vibe_server).
"""

import difflib
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import yaml

# ---------------------------------------------------------------------------
# Pattern for @from(param_name) parameter references
# ---------------------------------------------------------------------------
_PARAM_REF_RE = re.compile(r"@from\(([^)]*)\)")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class WorkflowIssue:
    severity: str       # "error" | "warning"
    file: str
    line: Optional[int]  # 1-indexed; None when location unknown
    context: str        # e.g. "task 'foo'", "edge 0"
    message: str
    suggestion: Optional[str] = None

    def __str__(self) -> str:
        loc = f":{self.line}" if self.line is not None else ""
        tag = "ERROR  " if self.severity == "error" else "WARNING"
        sug = f"\n         Suggestion: {self.suggestion}" if self.suggestion else ""
        return f"  {tag}  {self.file}{loc}  [{self.context}]  {self.message}{sug}"


# ---------------------------------------------------------------------------
# YAML line-number extraction
# ---------------------------------------------------------------------------

def _extract_line_map(path: str) -> Dict[str, int]:
    """Return 1-indexed line numbers for notable positions in a workflow YAML.

    Keys produced (when present):
      "tasks.<name>"   – line of each task name key
      "edges.<i>"      – line of edge i's origin value
      "sources"        – line of the sources key
      "sinks"          – line of the sinks key
      "parameters"     – line of the parameters key
    """
    line_map: Dict[str, int] = {}
    try:
        with open(path) as fh:
            root = yaml.compose(fh)
    except Exception:
        return line_map

    if root is None or not hasattr(root, "value"):
        return line_map

    # root is a MappingNode; root.value = [(key_node, val_node), ...]
    top: Dict[str, Any] = {}
    for k_node, v_node in root.value:
        top[k_node.value] = (k_node, v_node)

    for section in ("sources", "sinks", "parameters"):
        if section in top:
            k_node, _ = top[section]
            line_map[section] = k_node.start_mark.line + 1

    # tasks
    if "tasks" in top:
        _, tasks_node = top["tasks"]
        if hasattr(tasks_node, "value"):
            for tk_node, _ in tasks_node.value:
                line_map[f"tasks.{tk_node.value}"] = tk_node.start_mark.line + 1

    # edges
    if "edges" in top:
        _, edges_node = top["edges"]
        if hasattr(edges_node, "value"):
            for i, edge_item in enumerate(edges_node.value):
                # edge_item is a MappingNode with origin/destination keys
                if hasattr(edge_item, "value"):
                    for ek_node, ev_node in edge_item.value:
                        if ek_node.value == "origin":
                            line_map[f"edges.{i}"] = ev_node.start_mark.line + 1

    return line_map


# ---------------------------------------------------------------------------
# Candidate discovery for suggestions
# ---------------------------------------------------------------------------

def _collect_workflow_names(workflows_dir: str) -> List[str]:
    """Return all workflow names (relative path, no .yaml) found under workflows_dir."""
    names: List[str] = []
    for dirpath, _, filenames in os.walk(workflows_dir):
        for fname in filenames:
            if fname.endswith(".yaml"):
                rel = os.path.relpath(os.path.join(dirpath, fname), workflows_dir)
                names.append(rel[:-5].replace(os.sep, "/"))
    return names


def _collect_op_names(ops_dir: str) -> List[str]:
    """Return all op names (stem of .yaml files, one level deep) found under ops_dir."""
    names: List[str] = []
    if not os.path.isdir(ops_dir):
        return names
    for dirpath, _, filenames in os.walk(ops_dir):
        for fname in filenames:
            if fname.endswith(".yaml"):
                names.append(fname[:-5])
    return names


def _suggest(name: str, candidates: List[str], cutoff: float = 0.55) -> Optional[str]:
    matches = difflib.get_close_matches(name, candidates, n=1, cutoff=cutoff)
    return matches[0] if matches else None


# ---------------------------------------------------------------------------
# Graph / cycle detection (no external deps)
# ---------------------------------------------------------------------------

def _detect_cycle(task_names: Set[str], edges: List[Dict[str, Any]]) -> Optional[str]:
    """Return a string describing a cycle if one exists, else None.

    edges is the raw list of edge dicts from YAML (origin/destination).
    Only considers tasks that are declared in task_names.
    """
    # Build adjacency: task -> set of tasks it feeds
    adj: Dict[str, Set[str]] = {t: set() for t in task_names}
    for edge in edges:
        origin_str = edge.get("origin", "")
        if not isinstance(origin_str, str):
            continue
        origin_task = origin_str.split(".")[0]
        for dest_str in edge.get("destination", []):
            if not isinstance(dest_str, str):
                continue
            dest_task = dest_str.split(".")[0]
            if origin_task in adj and dest_task in adj:
                adj[origin_task].add(dest_task)

    WHITE, GRAY, BLACK = 0, 1, 2
    color: Dict[str, int] = {t: WHITE for t in task_names}
    cycle_path: List[str] = []

    def dfs(u: str) -> bool:
        color[u] = GRAY
        cycle_path.append(u)
        for v in sorted(adj.get(u, set())):
            if color[v] == GRAY:
                # Found cycle – trim cycle_path to just the cycle
                idx = cycle_path.index(v)
                cycle_path.append(v)
                return True
            if color[v] == WHITE:
                if dfs(v):
                    return True
        cycle_path.pop()
        color[u] = BLACK
        return False

    for t in sorted(task_names):
        if color[t] == WHITE:
            if dfs(t):
                return " -> ".join(cycle_path)

    return None


# ---------------------------------------------------------------------------
# Core validation logic
# ---------------------------------------------------------------------------

def _parse_task_name(edge_str: str) -> str:
    """Return the top-level task name from a dotted edge string."""
    return edge_str.split(".")[0]


def _flat_params(params: Any) -> Iterator[str]:
    """Yield all leaf string values from a nested parameter dict."""
    if isinstance(params, dict):
        for v in params.values():
            yield from _flat_params(v)
    elif isinstance(params, str):
        yield params


def validate_workflow_file(
    path: str,
    workflows_dir: Optional[str] = None,
    ops_dir: Optional[str] = None,
) -> List[WorkflowIssue]:
    """Validate a single workflow YAML file.  Returns a list of WorkflowIssue."""
    issues: List[WorkflowIssue] = []

    # ------------------------------------------------------------------
    # Resolve directories
    # ------------------------------------------------------------------
    if workflows_dir is None:
        # Infer from the path itself: walk up until we find a "workflows" dir
        workflows_dir = _find_workflows_dir(path)
    if ops_dir is None:
        ops_dir = _find_ops_dir(workflows_dir) if workflows_dir else ""

    all_workflow_names: List[str] = (
        _collect_workflow_names(workflows_dir) if workflows_dir else []
    )
    all_op_names: List[str] = _collect_op_names(ops_dir) if ops_dir else []

    # ------------------------------------------------------------------
    # Load YAML
    # ------------------------------------------------------------------
    try:
        with open(path) as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        issues.append(WorkflowIssue("error", path, None, "yaml", f"YAML parse error: {exc}"))
        return issues
    except OSError as exc:
        issues.append(WorkflowIssue("error", path, None, "file", f"Cannot read file: {exc}"))
        return issues

    if not isinstance(data, dict):
        issues.append(WorkflowIssue("error", path, None, "yaml", "Top-level structure must be a mapping"))
        return issues

    line_map = _extract_line_map(path)

    def issue(severity: str, context: str, message: str, suggestion: Optional[str] = None, line_key: Optional[str] = None) -> None:
        line = line_map.get(line_key or context)
        issues.append(WorkflowIssue(severity, path, line, context, message, suggestion))

    # ------------------------------------------------------------------
    # Required fields
    # ------------------------------------------------------------------
    for req in ("name", "sources", "sinks", "tasks"):
        if req not in data:
            issue("error", "schema", f"Missing required field '{req}'")

    if any(req not in data for req in ("name", "sources", "sinks", "tasks")):
        return issues  # can't continue without the basics

    wf_name = data.get("name", "<unnamed>")
    sources = data.get("sources", {}) or {}
    sinks = data.get("sinks", {}) or {}
    tasks = data.get("tasks", {}) or {}
    edges_raw: List[Any] = data.get("edges") or []
    parameters = data.get("parameters") or {}

    # ------------------------------------------------------------------
    # Type checks
    # ------------------------------------------------------------------
    if not isinstance(sources, dict):
        issue("error", "sources", "Field 'sources' must be a mapping")
        sources = {}
    if not isinstance(sinks, dict):
        issue("error", "sinks", "Field 'sinks' must be a mapping")
        sinks = {}
    if not isinstance(tasks, dict):
        issue("error", "tasks", "Field 'tasks' must be a mapping")
        tasks = {}
    if not isinstance(edges_raw, list):
        issue("error", "edges", "Field 'edges' must be a list")
        edges_raw = []

    task_names: Set[str] = set(tasks.keys())

    # ------------------------------------------------------------------
    # Validate task specs and check op/workflow existence
    # ------------------------------------------------------------------
    for tname, tspec in tasks.items():
        ctx = f"task '{tname}'"
        ln_key = f"tasks.{tname}"
        if not isinstance(tspec, dict):
            issue("error", ctx, "Task spec must be a mapping", line_key=ln_key)
            continue
        if "op" not in tspec and "workflow" not in tspec:
            issue("error", ctx, "Task must have either 'op' or 'workflow' field", line_key=ln_key)
            continue

        if "workflow" in tspec:
            wf_ref = tspec["workflow"]
            if not isinstance(wf_ref, str):
                issue("error", ctx, "'workflow' field must be a string", line_key=ln_key)
            elif workflows_dir:
                wf_path = os.path.join(workflows_dir, f"{wf_ref}.yaml")
                if not os.path.isfile(wf_path):
                    sug = _suggest(wf_ref, all_workflow_names)
                    issue(
                        "error", ctx,
                        f"Workflow '{wf_ref}' not found",
                        suggestion=f"did you mean '{sug}'?" if sug else None,
                        line_key=ln_key,
                    )
        elif "op" in tspec:
            op_ref = tspec["op"]
            op_dir = tspec.get("op_dir", op_ref)
            if not isinstance(op_ref, str):
                issue("error", ctx, "'op' field must be a string", line_key=ln_key)
            elif ops_dir and os.path.isdir(ops_dir):
                op_path = os.path.join(ops_dir, str(op_dir), f"{op_ref}.yaml")
                if not os.path.isfile(op_path):
                    sug = _suggest(op_ref, all_op_names)
                    issue(
                        "error", ctx,
                        f"Op '{op_ref}' not found (expected at {op_path})",
                        suggestion=f"did you mean '{sug}'?" if sug else None,
                        line_key=ln_key,
                    )

    # ------------------------------------------------------------------
    # Validate sources point to existing tasks
    # ------------------------------------------------------------------
    if len(sources) == 0:
        issue("error", "sources", f"Workflow must have at least one source")

    for src_name, src_ports in sources.items():
        if not isinstance(src_ports, list):
            issue("error", f"source '{src_name}'", "Source value must be a list of task ports")
            continue
        if len(src_ports) == 0:
            issue("error", f"source '{src_name}'", "Source must reference at least one task port")
        for port in src_ports:
            if not isinstance(port, str):
                issue("error", f"source '{src_name}'", f"Port reference must be a string, got {type(port).__name__}")
                continue
            task = _parse_task_name(port)
            if task not in task_names:
                sug = _suggest(task, list(task_names))
                issue(
                    "error", f"source '{src_name}'",
                    f"References task '{task}' which is not defined",
                    suggestion=f"did you mean task '{sug}'?" if sug else None,
                )

    # ------------------------------------------------------------------
    # Validate sinks point to existing tasks
    # ------------------------------------------------------------------
    for sink_name, sink_port in sinks.items():
        if not isinstance(sink_port, str):
            issue("error", f"sink '{sink_name}'", f"Sink value must be a string, got {type(sink_port).__name__}")
            continue
        task = _parse_task_name(sink_port)
        if task not in task_names:
            sug = _suggest(task, list(task_names))
            issue(
                "error", f"sink '{sink_name}'",
                f"References task '{task}' which is not defined",
                suggestion=f"did you mean task '{sug}'?" if sug else None,
            )

    # ------------------------------------------------------------------
    # Validate edges
    # ------------------------------------------------------------------
    source_ports: Set[str] = {
        port for ports in sources.values() if isinstance(ports, list) for port in ports
    }

    edges: List[Dict[str, Any]] = []
    for i, edge in enumerate(edges_raw):
        ctx = f"edge {i}"
        ln_key = f"edges.{i}"
        if not isinstance(edge, dict):
            issue("error", ctx, "Edge must be a mapping with 'origin' and 'destination'", line_key=ln_key)
            continue
        if "origin" not in edge:
            issue("error", ctx, "Edge missing 'origin' field", line_key=ln_key)
            continue
        if "destination" not in edge:
            issue("error", ctx, "Edge missing 'destination' field", line_key=ln_key)
            continue

        origin = edge["origin"]
        dests = edge["destination"]

        if not isinstance(origin, str):
            issue("error", ctx, f"Edge 'origin' must be a string", line_key=ln_key)
            continue
        if not isinstance(dests, list):
            issue("error", ctx, f"Edge 'destination' must be a list", line_key=ln_key)
            continue

        origin_task = _parse_task_name(origin)
        if origin_task not in task_names:
            sug = _suggest(origin_task, list(task_names))
            issue(
                "error", ctx,
                f"Origin '{origin}' references unknown task '{origin_task}'",
                suggestion=f"did you mean '{sug}'?" if sug else None,
                line_key=ln_key,
            )

        for dest in dests:
            if not isinstance(dest, str):
                issue("error", ctx, f"Destination entry must be a string", line_key=ln_key)
                continue
            # A source port must not also be a destination
            if dest in source_ports:
                issue(
                    "error", ctx,
                    f"'{dest}' is a workflow source port and cannot also be an edge destination",
                    line_key=ln_key,
                )
            dest_task = _parse_task_name(dest)
            if dest_task not in task_names:
                sug = _suggest(dest_task, list(task_names))
                issue(
                    "error", ctx,
                    f"Destination '{dest}' references unknown task '{dest_task}'",
                    suggestion=f"did you mean '{sug}'?" if sug else None,
                    line_key=ln_key,
                )

        edges.append(edge)

    # ------------------------------------------------------------------
    # DAG cycle detection
    # ------------------------------------------------------------------
    if edges:
        cycle = _detect_cycle(task_names, edges)
        if cycle:
            issue("error", "edges", f"Cycle detected in task dependency graph: {cycle}")

    # ------------------------------------------------------------------
    # Parameter reference validation
    # ------------------------------------------------------------------
    if isinstance(parameters, dict) and parameters:
        # Collect all @from() references used in task parameters
        used_refs: Set[str] = set()
        for tspec in tasks.values():
            if not isinstance(tspec, dict):
                continue
            for leaf in _flat_params(tspec.get("parameters", {})):
                m = _PARAM_REF_RE.match(leaf)
                if m:
                    ref = m.group(1).strip()
                    if ref:
                        used_refs.add(ref)

        # Workflow params not referenced by any task
        unused = [p for p in parameters if p not in used_refs]
        for p in unused:
            issue(
                "warning", f"parameter '{p}'",
                f"Workflow parameter '{p}' is defined but not referenced by any task",
            )

        # @from() references to undefined workflow params
        undefined = [r for r in used_refs if r not in parameters]
        for r in undefined:
            sug = _suggest(r, list(parameters.keys()))
            issue(
                "error", f"parameter ref '@from({r})'",
                f"Task references undefined workflow parameter '{r}'",
                suggestion=f"did you mean '{sug}'?" if sug else None,
            )

    return issues


# ---------------------------------------------------------------------------
# Directory resolution helpers
# ---------------------------------------------------------------------------

def _find_workflows_dir(yaml_path: str) -> Optional[str]:
    """Walk up from the given YAML path to find a 'workflows' directory."""
    # The file itself may be inside workflows/ already
    candidate = os.path.abspath(yaml_path)
    for _ in range(10):
        candidate = os.path.dirname(candidate)
        if os.path.basename(candidate) == "workflows":
            return candidate
        workflows_sub = os.path.join(candidate, "workflows")
        if os.path.isdir(workflows_sub):
            return workflows_sub
    return None


def _find_ops_dir(workflows_dir: str) -> str:
    """Resolve the ops directory relative to the workflows directory."""
    # Standard repo layout: workflows/ and ops/ are siblings at repo root
    repo_root = os.path.dirname(workflows_dir)
    ops_candidate = os.path.join(repo_root, "ops")
    if os.path.isdir(ops_candidate):
        return ops_candidate
    # Installed layout
    installed = os.path.join("/", "app", "ops")
    if os.path.isdir(installed):
        return installed
    return ops_candidate  # return even if missing; existence checked later


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def validate_command(
    path: Optional[str],
    workflows_dir: Optional[str],
    ops_dir: Optional[str],
) -> int:
    """Run validation and print results.  Returns exit code (0 = clean)."""
    import sys

    # Resolve workflows_dir
    if workflows_dir is None:
        if path:
            workflows_dir = _find_workflows_dir(path)
        else:
            # Try CWD-relative
            cwd_wf = os.path.join(os.getcwd(), "workflows")
            if os.path.isdir(cwd_wf):
                workflows_dir = cwd_wf

    if path:
        targets = [os.path.abspath(path)]
    else:
        if not workflows_dir or not os.path.isdir(workflows_dir):
            print("Error: cannot find workflows directory. Specify --workflows-dir or pass a file path.", file=sys.stderr)
            return 1
        targets = []
        for dirpath, _, filenames in os.walk(workflows_dir):
            for fname in sorted(filenames):
                if fname.endswith(".yaml"):
                    targets.append(os.path.join(dirpath, fname))
        if not targets:
            print(f"No workflow YAML files found in {workflows_dir}", file=sys.stderr)
            return 0

    all_issues: List[WorkflowIssue] = []
    for target in sorted(targets):
        file_issues = validate_workflow_file(target, workflows_dir=workflows_dir, ops_dir=ops_dir)
        all_issues.extend(file_issues)

    errors = [i for i in all_issues if i.severity == "error"]
    warnings = [i for i in all_issues if i.severity == "warning"]

    if not all_issues:
        n = len(targets)
        print(f"OK  {n} workflow{'s' if n != 1 else ''} validated, no issues found.")
        return 0

    # Group by file for readable output
    files_seen: List[str] = []
    by_file: Dict[str, List[WorkflowIssue]] = {}
    for issue in all_issues:
        if issue.file not in by_file:
            by_file[issue.file] = []
            files_seen.append(issue.file)
        by_file[issue.file].append(issue)

    for fpath in files_seen:
        rel = _rel_or_abs(fpath, workflows_dir)
        file_issues = by_file[fpath]
        n_err = sum(1 for i in file_issues if i.severity == "error")
        n_warn = sum(1 for i in file_issues if i.severity == "warning")
        parts = []
        if n_err:
            parts.append(f"{n_err} error{'s' if n_err != 1 else ''}")
        if n_warn:
            parts.append(f"{n_warn} warning{'s' if n_warn != 1 else ''}")
        print(f"\n{rel}  ({', '.join(parts)})")
        for issue in file_issues:
            # Print relative path in issue too
            issue_copy = WorkflowIssue(
                severity=issue.severity,
                file=_rel_or_abs(issue.file, workflows_dir),
                line=issue.line,
                context=issue.context,
                message=issue.message,
                suggestion=issue.suggestion,
            )
            print(str(issue_copy))

    print(f"\nSummary: {len(errors)} error{'s' if len(errors) != 1 else ''}, "
          f"{len(warnings)} warning{'s' if len(warnings) != 1 else ''} "
          f"across {len(targets)} file{'s' if len(targets) != 1 else ''}.")

    return 1 if errors else 0


def _rel_or_abs(path: str, base: Optional[str]) -> str:
    if base:
        abs_path = os.path.abspath(path)
        abs_base = os.path.abspath(base)
        if abs_path.startswith(abs_base + os.sep):
            return os.path.relpath(abs_path, abs_base)
    return os.path.abspath(path)
