# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Standalone workflow validation for FarmVibes.AI.

This module provides workflow validation that can run locally without a cluster,
catching errors before API submission.
"""

import difflib
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml


class IssueSeverity(Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass
class ValidationIssue:
    """A validation problem found in a workflow."""

    file: str
    line: Optional[int]
    severity: IssueSeverity
    message: str
    context: Optional[str] = None  # task name, edge, etc.
    suggestion: Optional[str] = None

    def __str__(self) -> str:
        loc = f"{self.file}"
        if self.line is not None:
            loc += f":{self.line}"
        prefix = "error" if self.severity == IssueSeverity.ERROR else "warning"
        msg = f"{loc}: {prefix}: {self.message}"
        if self.context:
            msg += f" (in {self.context})"
        if self.suggestion:
            msg += f". Did you mean '{self.suggestion}'?"
        return msg


@dataclass
class ValidationResult:
    """Result of validating one or more workflows."""

    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(i.severity == IssueSeverity.ERROR for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(i.severity == IssueSeverity.WARNING for i in self.issues)

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == IssueSeverity.ERROR]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == IssueSeverity.WARNING]

    def add_error(
        self,
        file: str,
        message: str,
        line: Optional[int] = None,
        context: Optional[str] = None,
        suggestion: Optional[str] = None,
    ):
        self.issues.append(
            ValidationIssue(
                file=file,
                line=line,
                severity=IssueSeverity.ERROR,
                message=message,
                context=context,
                suggestion=suggestion,
            )
        )

    def add_warning(
        self,
        file: str,
        message: str,
        line: Optional[int] = None,
        context: Optional[str] = None,
        suggestion: Optional[str] = None,
    ):
        self.issues.append(
            ValidationIssue(
                file=file,
                line=line,
                severity=IssueSeverity.WARNING,
                message=message,
                context=context,
                suggestion=suggestion,
            )
        )


class LineTrackingLoader(yaml.SafeLoader):
    """YAML loader that tracks line numbers for keys."""

    pass


def _construct_mapping_with_line_numbers(loader: LineTrackingLoader, node: yaml.Node):
    """Construct a mapping while tracking line numbers."""
    loader.flatten_mapping(node)
    pairs = loader.construct_pairs(node)
    mapping = {}
    line_info = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        value = loader.construct_object(value_node)
        mapping[key] = value
        line_info[key] = key_node.start_mark.line + 1  # 1-indexed
    # Store line info as a special attribute
    mapping["__line_info__"] = line_info
    return mapping


LineTrackingLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping_with_line_numbers,
)


def _get_line(data: Dict[str, Any], key: str) -> Optional[int]:
    """Get the line number for a key in the parsed YAML data."""
    line_info = data.get("__line_info__", {})
    return line_info.get(key)


def _strip_line_info(data: Any) -> Any:
    """Remove line info metadata from parsed data."""
    if isinstance(data, dict):
        return {k: _strip_line_info(v) for k, v in data.items() if k != "__line_info__"}
    elif isinstance(data, list):
        return [_strip_line_info(item) for item in data]
    return data


class WorkflowValidator:
    """Validates FarmVibes.AI workflow YAML files."""

    REQUIRED_FIELDS = ["name", "sources", "sinks", "tasks"]
    OPTIONAL_FIELDS = ["parameters", "edges", "description", "default_parameters"]
    TASK_OP_FIELDS = ["op", "parameters", "op_dir"]
    TASK_WF_FIELDS = ["workflow", "parameters"]
    PARAM_PATTERN = re.compile(r"@from\((.*)\)")

    def __init__(self, workflows_dir: str, ops_dir: str, recursive: bool = True):
        """
        Initialize the validator.

        Args:
            workflows_dir: Root directory containing workflow YAML files
            ops_dir: Root directory containing operation YAML files
            recursive: Whether to recursively validate sub-workflows (default True)
        """
        self.workflows_dir = workflows_dir
        self.ops_dir = ops_dir
        self.recursive = recursive
        self._workflow_cache: Dict[str, Dict[str, Any]] = {}
        self._op_cache: Dict[str, Dict[str, Any]] = {}
        self._available_workflows: Optional[Set[str]] = None
        self._available_ops: Optional[Dict[str, Set[str]]] = None
        # Track validated workflows to avoid infinite recursion
        self._validated_workflows: Set[str] = set()

    def _index_available_workflows(self) -> Set[str]:
        """Index all available workflow names."""
        if self._available_workflows is not None:
            return self._available_workflows

        workflows = set()
        workflows_path = Path(self.workflows_dir)
        if workflows_path.exists():
            for yaml_file in workflows_path.rglob("*.yaml"):
                # Convert to workflow reference path (relative, without .yaml)
                rel_path = yaml_file.relative_to(workflows_path)
                wf_name = str(rel_path.with_suffix(""))
                workflows.add(wf_name)
        self._available_workflows = workflows
        return workflows

    def _index_available_ops(self) -> Dict[str, Set[str]]:
        """Index all available operation names by directory."""
        if self._available_ops is not None:
            return self._available_ops

        ops: Dict[str, Set[str]] = defaultdict(set)
        ops_path = Path(self.ops_dir)
        if ops_path.exists():
            for yaml_file in ops_path.rglob("*.yaml"):
                rel_path = yaml_file.relative_to(ops_path)
                # op_dir is the parent directory, op_name is the file stem
                op_dir = str(rel_path.parent)
                op_name = yaml_file.stem
                ops[op_dir].add(op_name)
                # Also index by full path for direct matching
                ops["__all__"].add(f"{op_dir}/{op_name}" if op_dir != "." else op_name)
        self._available_ops = ops
        return ops

    def _suggest_workflow(self, name: str) -> Optional[str]:
        """Suggest a similar workflow name using edit distance."""
        available = self._index_available_workflows()
        matches = difflib.get_close_matches(name, available, n=1, cutoff=0.6)
        return matches[0] if matches else None

    def _suggest_op(self, name: str, op_dir: Optional[str] = None) -> Optional[str]:
        """Suggest a similar operation name using edit distance."""
        available = self._index_available_ops()
        if op_dir and op_dir in available:
            candidates = available[op_dir]
        else:
            # Search all ops
            candidates = available.get("__all__", set())
        matches = difflib.get_close_matches(name, candidates, n=1, cutoff=0.6)
        return matches[0] if matches else None

    def _load_yaml_with_lines(self, filepath: str) -> Tuple[Dict[str, Any], Optional[str]]:
        """Load a YAML file with line number tracking."""
        try:
            with open(filepath) as f:
                content = f.read()
            data = yaml.load(content, Loader=LineTrackingLoader)
            if data is None:
                return {}, None
            return data, None
        except yaml.YAMLError as e:
            return {}, f"YAML parse error: {e}"
        except FileNotFoundError:
            return {}, f"File not found: {filepath}"
        except Exception as e:
            return {}, f"Error reading file: {e}"

    def _load_workflow(self, workflow_ref: str) -> Tuple[Dict[str, Any], Optional[str]]:
        """Load a workflow by reference name."""
        if workflow_ref in self._workflow_cache:
            return self._workflow_cache[workflow_ref], None

        filepath = os.path.join(self.workflows_dir, f"{workflow_ref}.yaml")
        data, error = self._load_yaml_with_lines(filepath)
        if error:
            return {}, error
        self._workflow_cache[workflow_ref] = data
        return data, None

    def _load_op(self, op_name: str, op_dir: str) -> Tuple[Dict[str, Any], Optional[str]]:
        """Load an operation by name and directory."""
        cache_key = f"{op_dir}/{op_name}"
        if cache_key in self._op_cache:
            return self._op_cache[cache_key], None

        filepath = os.path.join(self.ops_dir, op_dir, f"{op_name}.yaml")
        data, error = self._load_yaml_with_lines(filepath)
        if error:
            return {}, error
        self._op_cache[cache_key] = data
        return data, None

    def _get_op_ports(self, op_data: Dict[str, Any]) -> Tuple[Set[str], Set[str]]:
        """Get input and output port names from an operation spec.

        Returns:
            Tuple of (input_ports, output_ports)
        """
        inputs = set()
        outputs = set()

        # Ops have 'inputs' dict and 'output' dict
        inputs_spec = op_data.get("inputs", {})
        if isinstance(inputs_spec, dict):
            inputs = set(k for k in inputs_spec.keys() if k != "__line_info__")

        output_spec = op_data.get("output", {})
        if isinstance(output_spec, dict):
            outputs = set(k for k in output_spec.keys() if k != "__line_info__")

        return inputs, outputs

    def _get_workflow_ports(self, wf_data: Dict[str, Any]) -> Tuple[Set[str], Set[str]]:
        """Get input and output port names from a workflow spec.

        For workflows, sources are inputs and sinks are outputs.

        Returns:
            Tuple of (input_ports, output_ports)
        """
        inputs = set()
        outputs = set()

        sources = wf_data.get("sources", {})
        if isinstance(sources, dict):
            inputs = set(k for k in sources.keys() if k != "__line_info__")

        sinks = wf_data.get("sinks", {})
        if isinstance(sinks, dict):
            outputs = set(k for k in sinks.keys() if k != "__line_info__")

        return inputs, outputs

    def _get_task_ports(
        self, task_spec: Dict[str, Any]
    ) -> Tuple[Set[str], Set[str], Optional[str]]:
        """Get input and output ports for a task.

        Returns:
            Tuple of (input_ports, output_ports, error_message)
        """
        if "op" in task_spec:
            op_name = task_spec["op"]
            op_dir = task_spec.get("op_dir", op_name)
            op_data, error = self._load_op(op_name, op_dir)
            if error:
                return set(), set(), error
            return (*self._get_op_ports(op_data), None)
        elif "workflow" in task_spec:
            wf_ref = task_spec["workflow"]
            wf_data, error = self._load_workflow(wf_ref)
            if error:
                return set(), set(), error
            return (*self._get_workflow_ports(wf_data), None)
        return set(), set(), "Task has no op or workflow"

    def _parse_edge_string(self, edge_str: str) -> Tuple[str, str]:
        """Parse 'task.port' or 'task.subtask.port' into (top-level-task, rest).

        For nested references like 'ndvi_summary.compute_ndvi.compute_index.index',
        returns ('ndvi_summary', 'compute_ndvi.compute_index.index').
        """
        parts = edge_str.split(".", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        return edge_str, ""

    def _get_parameter_references(self, params: Dict[str, Any]) -> Set[str]:
        """Extract all @from() parameter references from a params dict."""
        refs = set()

        def extract(value: Any):
            if isinstance(value, str):
                match = self.PARAM_PATTERN.match(value)
                if match:
                    refs.add(match.group(1))
            elif isinstance(value, dict):
                for v in value.values():
                    extract(v)
            elif isinstance(value, list):
                for item in value:
                    extract(item)

        extract(params)
        return refs

    def _validate_port_reference(
        self,
        port_ref: str,
        task_spec: Dict[str, Any],
        is_input: bool,
        filepath: str,
        result: ValidationResult,
        line: Optional[int] = None,
        context: Optional[str] = None,
    ) -> bool:
        """Validate that a port exists on a task.

        Args:
            port_ref: The port part of a reference (e.g., 'user_input' from 'task.user_input')
            task_spec: The task specification dict
            is_input: True if checking input port, False for output port
            filepath: File being validated (for error reporting)
            result: ValidationResult to add errors to
            line: Line number for error reporting
            context: Context string for error reporting

        Returns:
            True if port exists or couldn't be validated (no error added for missing spec)
        """
        # For nested references like 'subtask.port', we can't validate without
        # loading the full sub-workflow graph. Skip validation for these.
        if "." in port_ref:
            return True

        input_ports, output_ports, error = self._get_task_ports(task_spec)
        if error:
            # Task spec couldn't be loaded - error already reported elsewhere
            return True

        ports = input_ports if is_input else output_ports
        port_type = "input" if is_input else "output"

        if port_ref not in ports:
            suggestion = None
            if ports:
                matches = difflib.get_close_matches(port_ref, ports, n=1, cutoff=0.6)
                suggestion = matches[0] if matches else None
            result.add_error(
                filepath,
                f"Port '{port_ref}' does not exist as {port_type} on task",
                line=line,
                context=context,
                suggestion=suggestion,
            )
            return False
        return True

    def validate_file(self, filepath: str) -> ValidationResult:
        """Validate a single workflow file."""
        result = ValidationResult()
        rel_path = filepath

        # Track this workflow as validated (use relative path from workflows_dir if possible)
        try:
            wf_ref = os.path.relpath(filepath, self.workflows_dir)
            if wf_ref.endswith(".yaml"):
                wf_ref = wf_ref[:-5]
            self._validated_workflows.add(wf_ref)
        except ValueError:
            pass  # filepath not relative to workflows_dir

        # Load and parse
        data, error = self._load_yaml_with_lines(filepath)
        if error:
            result.add_error(rel_path, error)
            return result

        if not data:
            result.add_error(rel_path, "Empty workflow file")
            return result

        # Get workflow name for context
        wf_name = data.get("name", os.path.basename(filepath))

        # Validate required fields
        for field in self.REQUIRED_FIELDS:
            if field not in data or data[field] is None:
                result.add_error(
                    rel_path,
                    f"Missing required field '{field}'",
                    line=_get_line(data, field),
                )

        # Check for unknown fields
        all_fields = self.REQUIRED_FIELDS + self.OPTIONAL_FIELDS + ["__line_info__"]
        for key in data.keys():
            if key not in all_fields:
                result.add_error(
                    rel_path,
                    f"Unknown field '{key}'",
                    line=_get_line(data, key),
                )

        # Validate sources
        sources = data.get("sources", {})
        if sources is not None:
            self._validate_sources(rel_path, data, sources, result)

        # Validate sinks
        sinks = data.get("sinks", {})
        if sinks is not None:
            self._validate_sinks(rel_path, data, sinks, result)

        # Validate tasks
        tasks = data.get("tasks", {})
        if tasks is not None:
            self._validate_tasks(rel_path, data, tasks, result)

        # Validate edges
        edges = data.get("edges", [])
        if edges is not None:
            self._validate_edges(rel_path, data, tasks or {}, sources or {}, edges, result)

        # Validate parameters
        params = data.get("parameters", {})
        if params is not None and tasks is not None:
            self._validate_parameters(rel_path, data, params, tasks, result)

        # Validate DAG (cycle detection)
        if tasks and edges:
            self._validate_dag(rel_path, tasks, sources or {}, sinks or {}, edges, result)

        return result

    def _validate_sources(
        self,
        filepath: str,
        data: Dict[str, Any],
        sources: Dict[str, Any],
        result: ValidationResult,
    ):
        """Validate sources field."""
        if not isinstance(sources, dict):
            result.add_error(
                filepath,
                "Sources field must be a mapping",
                line=_get_line(data, "sources"),
            )
            return

        if len(sources) == 0:
            result.add_error(
                filepath,
                "Workflow must have at least one source",
                line=_get_line(data, "sources"),
            )
            return

        tasks = data.get("tasks", {}) or {}
        task_names = set(tasks.keys()) - {"__line_info__"}

        for source_name, ports in sources.items():
            if source_name == "__line_info__":
                continue
            if not isinstance(ports, list):
                result.add_error(
                    filepath,
                    f"Source '{source_name}' must map to a list of task ports",
                    line=_get_line(sources, source_name),
                )
                continue

            if len(ports) == 0:
                result.add_error(
                    filepath,
                    f"Source '{source_name}' must have at least one task port",
                    line=_get_line(sources, source_name),
                )
                continue

            for port in ports:
                task_name, port_name = self._parse_edge_string(port)
                if task_name not in task_names:
                    suggestion = difflib.get_close_matches(task_name, task_names, n=1, cutoff=0.6)
                    result.add_error(
                        filepath,
                        f"Source '{source_name}' references unknown task '{task_name}'",
                        line=_get_line(sources, source_name),
                        suggestion=suggestion[0] if suggestion else None,
                    )
                elif port_name and task_name in tasks:
                    # Validate port exists on task (sources connect to task inputs)
                    task_spec = tasks[task_name]
                    if isinstance(task_spec, dict):
                        self._validate_port_reference(
                            port_name,
                            task_spec,
                            is_input=True,
                            filepath=filepath,
                            result=result,
                            line=_get_line(sources, source_name),
                            context=f"source '{source_name}' -> {task_name}.{port_name}",
                        )

    def _validate_sinks(
        self,
        filepath: str,
        data: Dict[str, Any],
        sinks: Dict[str, Any],
        result: ValidationResult,
    ):
        """Validate sinks field."""
        if not isinstance(sinks, dict):
            result.add_error(
                filepath,
                "Sinks field must be a mapping",
                line=_get_line(data, "sinks"),
            )
            return

        if len(sinks) == 0:
            result.add_warning(
                filepath,
                "Workflow has no sinks (side-effects only?)",
                line=_get_line(data, "sinks"),
            )
            return

        tasks = data.get("tasks", {}) or {}
        task_names = set(tasks.keys()) - {"__line_info__"}

        for sink_name, port in sinks.items():
            if sink_name == "__line_info__":
                continue
            if not isinstance(port, str):
                result.add_error(
                    filepath,
                    f"Sink '{sink_name}' must map to a string (task.port)",
                    line=_get_line(sinks, sink_name),
                )
                continue

            task_name, port_name = self._parse_edge_string(port)
            if task_name not in task_names:
                suggestion = difflib.get_close_matches(task_name, task_names, n=1, cutoff=0.6)
                result.add_error(
                    filepath,
                    f"Sink '{sink_name}' references unknown task '{task_name}'",
                    line=_get_line(sinks, sink_name),
                    suggestion=suggestion[0] if suggestion else None,
                )
            elif port_name and task_name in tasks:
                # Validate port exists on task (sinks connect from task outputs)
                task_spec = tasks[task_name]
                if isinstance(task_spec, dict):
                    self._validate_port_reference(
                        port_name,
                        task_spec,
                        is_input=False,
                        filepath=filepath,
                        result=result,
                        line=_get_line(sinks, sink_name),
                        context=f"sink '{sink_name}' <- {task_name}.{port_name}",
                    )

    def _validate_tasks(
        self,
        filepath: str,
        data: Dict[str, Any],
        tasks: Dict[str, Any],
        result: ValidationResult,
    ):
        """Validate tasks field."""
        if not isinstance(tasks, dict):
            result.add_error(
                filepath,
                "Tasks field must be a mapping",
                line=_get_line(data, "tasks"),
            )
            return

        if len(tasks) == 0:
            result.add_error(
                filepath,
                "Workflow must have at least one task",
                line=_get_line(data, "tasks"),
            )
            return

        wf_name = data.get("name", "")

        for task_name, task_spec in tasks.items():
            if task_name == "__line_info__":
                continue

            if not isinstance(task_spec, dict):
                result.add_error(
                    filepath,
                    f"Task '{task_name}' must be a mapping",
                    line=_get_line(tasks, task_name),
                )
                continue

            # Check task type (op or workflow)
            has_op = "op" in task_spec
            has_workflow = "workflow" in task_spec

            if not has_op and not has_workflow:
                result.add_error(
                    filepath,
                    f"Task '{task_name}' must have either 'op' or 'workflow' field",
                    line=_get_line(tasks, task_name),
                )
                continue

            if has_op and has_workflow:
                result.add_error(
                    filepath,
                    f"Task '{task_name}' cannot have both 'op' and 'workflow' fields",
                    line=_get_line(tasks, task_name),
                )
                continue

            # Validate task fields
            allowed_fields = (
                self.TASK_OP_FIELDS if has_op else self.TASK_WF_FIELDS
            ) + ["__line_info__"]
            for key in task_spec.keys():
                if key not in allowed_fields:
                    result.add_error(
                        filepath,
                        f"Unknown field '{key}' in task '{task_name}'",
                        line=_get_line(task_spec, key),
                        context=f"task '{task_name}'",
                    )

            # Validate task reference exists
            if has_workflow:
                wf_ref = task_spec["workflow"]
                if not isinstance(wf_ref, str):
                    result.add_error(
                        filepath,
                        f"Task '{task_name}' workflow field must be a string",
                        line=_get_line(task_spec, "workflow"),
                    )
                else:
                    # Check for recursive definition
                    if wf_ref == wf_name or wf_ref.endswith(f"/{wf_name}"):
                        result.add_error(
                            filepath,
                            f"Recursive workflow definition: task '{task_name}' references "
                            f"its own workflow '{wf_ref}'",
                            line=_get_line(task_spec, "workflow"),
                        )
                    else:
                        # Check workflow exists
                        wf_data, error = self._load_workflow(wf_ref)
                        if error:
                            suggestion = self._suggest_workflow(wf_ref)
                            result.add_error(
                                filepath,
                                f"Task '{task_name}' references unknown workflow '{wf_ref}'",
                                line=_get_line(task_spec, "workflow"),
                                suggestion=suggestion,
                            )
                        elif self.recursive and wf_ref not in self._validated_workflows:
                            # Recursively validate sub-workflow
                            self._validated_workflows.add(wf_ref)
                            wf_path = os.path.join(self.workflows_dir, f"{wf_ref}.yaml")
                            sub_result = self.validate_file(wf_path)
                            result.issues.extend(sub_result.issues)

            if has_op:
                op_name = task_spec["op"]
                op_dir = task_spec.get("op_dir", op_name)
                if not isinstance(op_name, str):
                    result.add_error(
                        filepath,
                        f"Task '{task_name}' op field must be a string",
                        line=_get_line(task_spec, "op"),
                    )
                else:
                    # Check op exists
                    op_data, error = self._load_op(op_name, op_dir)
                    if error:
                        suggestion = self._suggest_op(op_name, op_dir)
                        result.add_error(
                            filepath,
                            f"Task '{task_name}' references unknown operation '{op_name}' "
                            f"in '{op_dir}'",
                            line=_get_line(task_spec, "op"),
                            suggestion=suggestion,
                        )

    def _validate_edges(
        self,
        filepath: str,
        data: Dict[str, Any],
        tasks: Dict[str, Any],
        sources: Dict[str, Any],
        edges: List[Any],
        result: ValidationResult,
    ):
        """Validate edges field."""
        if edges is None:
            edges = []

        if not isinstance(edges, list):
            result.add_error(
                filepath,
                "Edges field must be a list",
                line=_get_line(data, "edges"),
            )
            return

        task_names = set(tasks.keys()) - {"__line_info__"}

        # Collect all source ports (these cannot be edge destinations)
        source_ports = set()
        for ports in sources.values():
            if isinstance(ports, list):
                source_ports.update(ports)

        for i, edge in enumerate(edges):
            edge_ctx = f"edge[{i}]"

            if not isinstance(edge, dict):
                result.add_error(
                    filepath,
                    f"Edge must be a mapping with 'origin' and 'destination'",
                    context=edge_ctx,
                )
                continue

            # Check required edge fields
            if "origin" not in edge:
                result.add_error(
                    filepath,
                    "Edge missing 'origin' field",
                    line=_get_line(edge, "origin") if isinstance(edge, dict) else None,
                    context=edge_ctx,
                )
                continue

            if "destination" not in edge:
                result.add_error(
                    filepath,
                    "Edge missing 'destination' field",
                    line=_get_line(edge, "destination") if isinstance(edge, dict) else None,
                    context=edge_ctx,
                )
                continue

            origin = edge["origin"]
            destination = edge["destination"]

            # Validate origin
            if not isinstance(origin, str):
                result.add_error(
                    filepath,
                    "Edge origin must be a string",
                    line=_get_line(edge, "origin"),
                    context=edge_ctx,
                )
            else:
                origin_task, origin_port = self._parse_edge_string(origin)
                if origin_task not in task_names:
                    suggestion = difflib.get_close_matches(origin_task, task_names, n=1, cutoff=0.6)
                    result.add_error(
                        filepath,
                        f"Edge origin references unknown task '{origin_task}'",
                        line=_get_line(edge, "origin"),
                        context=edge_ctx,
                        suggestion=suggestion[0] if suggestion else None,
                    )
                elif origin_port and origin_task in tasks:
                    # Validate origin port exists (edges originate from outputs)
                    task_spec = tasks[origin_task]
                    if isinstance(task_spec, dict):
                        self._validate_port_reference(
                            origin_port,
                            task_spec,
                            is_input=False,
                            filepath=filepath,
                            result=result,
                            line=_get_line(edge, "origin"),
                            context=f"{edge_ctx}: {origin_task}.{origin_port}",
                        )

            # Validate destination
            if not isinstance(destination, list):
                result.add_error(
                    filepath,
                    "Edge destination must be a list",
                    line=_get_line(edge, "destination"),
                    context=edge_ctx,
                )
            else:
                for dest in destination:
                    if not isinstance(dest, str):
                        result.add_error(
                            filepath,
                            "Edge destination items must be strings",
                            context=edge_ctx,
                        )
                        continue

                    dest_task, dest_port = self._parse_edge_string(dest)
                    if dest_task not in task_names:
                        suggestion = difflib.get_close_matches(
                            dest_task, task_names, n=1, cutoff=0.6
                        )
                        result.add_error(
                            filepath,
                            f"Edge destination references unknown task '{dest_task}'",
                            line=_get_line(edge, "destination"),
                            context=edge_ctx,
                            suggestion=suggestion[0] if suggestion else None,
                        )
                    elif dest_port and dest_task in tasks:
                        # Validate destination port exists (edges go to inputs)
                        task_spec = tasks[dest_task]
                        if isinstance(task_spec, dict):
                            self._validate_port_reference(
                                dest_port,
                                task_spec,
                                is_input=True,
                                filepath=filepath,
                                result=result,
                                line=_get_line(edge, "destination"),
                                context=f"{edge_ctx}: {dest_task}.{dest_port}",
                            )

                    # Check source ports are not destinations
                    if dest in source_ports:
                        result.add_error(
                            filepath,
                            f"Source port '{dest}' cannot also be an edge destination",
                            line=_get_line(edge, "destination"),
                            context=edge_ctx,
                        )

    def _validate_parameters(
        self,
        filepath: str,
        data: Dict[str, Any],
        params: Dict[str, Any],
        tasks: Dict[str, Any],
        result: ValidationResult,
    ):
        """Validate parameter references."""
        if params is None:
            params = {}

        defined_params = set(params.keys()) - {"__line_info__"}

        # Collect all parameter references from tasks
        referenced_params: Set[str] = set()
        for task_name, task_spec in tasks.items():
            if task_name == "__line_info__" or not isinstance(task_spec, dict):
                continue
            task_params = task_spec.get("parameters", {})
            if task_params:
                refs = self._get_parameter_references(task_params)
                referenced_params.update(refs)

        # Check for unused parameters
        unused = defined_params - referenced_params
        for param in unused:
            result.add_warning(
                filepath,
                f"Parameter '{param}' is defined but never used",
                line=_get_line(params, param),
            )

        # Check for undefined parameter references
        undefined = referenced_params - defined_params
        for param in undefined:
            suggestion = difflib.get_close_matches(param, defined_params, n=1, cutoff=0.6)
            result.add_error(
                filepath,
                f"Task references undefined parameter '{param}'",
                suggestion=suggestion[0] if suggestion else None,
            )

    def _validate_dag(
        self,
        filepath: str,
        tasks: Dict[str, Any],
        sources: Dict[str, Any],
        sinks: Dict[str, Any],
        edges: List[Any],
        result: ValidationResult,
    ):
        """Validate the workflow forms a valid DAG (no cycles)."""
        task_names = set(tasks.keys()) - {"__line_info__"}

        # Build adjacency list from edges
        graph: Dict[str, Set[str]] = defaultdict(set)
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            origin = edge.get("origin", "")
            destinations = edge.get("destination", [])
            if not isinstance(origin, str) or not isinstance(destinations, list):
                continue

            origin_task, _ = self._parse_edge_string(origin)
            for dest in destinations:
                if isinstance(dest, str):
                    dest_task, _ = self._parse_edge_string(dest)
                    if origin_task in task_names and dest_task in task_names:
                        graph[origin_task].add(dest_task)

        # Also add edges from sources to tasks
        for ports in sources.values():
            if isinstance(ports, list):
                for port in ports:
                    if isinstance(port, str):
                        task, _ = self._parse_edge_string(port)
                        if task in task_names:
                            graph["__source__"].add(task)

        # DFS-based cycle detection
        WHITE, GRAY, BLACK = 0, 1, 2
        color: Dict[str, int] = {task: WHITE for task in task_names}
        cycle_path: List[str] = []

        def dfs(node: str) -> bool:
            color[node] = GRAY
            cycle_path.append(node)
            for neighbor in graph.get(node, set()):
                if neighbor not in color:
                    continue
                if color[neighbor] == GRAY:
                    # Found cycle
                    cycle_start = cycle_path.index(neighbor)
                    cycle = cycle_path[cycle_start:] + [neighbor]
                    result.add_error(
                        filepath,
                        f"Workflow contains a cycle: {' -> '.join(cycle)}",
                    )
                    return True
                if color[neighbor] == WHITE:
                    if dfs(neighbor):
                        return True
            cycle_path.pop()
            color[node] = BLACK
            return False

        for task in task_names:
            if color[task] == WHITE:
                if dfs(task):
                    break  # Stop after first cycle found

    def validate_all(self, path: Optional[str] = None) -> ValidationResult:
        """
        Validate workflows.

        Args:
            path: Optional path to a specific workflow file or directory.
                  If None, validates all workflows in workflows_dir.

        Returns:
            ValidationResult with all issues found.
        """
        result = ValidationResult()

        if path is None:
            # Validate all workflows
            workflows_path = Path(self.workflows_dir)
            if not workflows_path.exists():
                result.add_error(
                    self.workflows_dir,
                    f"Workflows directory not found: {self.workflows_dir}",
                )
                return result

            yaml_files = list(workflows_path.rglob("*.yaml"))
            if not yaml_files:
                result.add_warning(self.workflows_dir, "No workflow files found")
                return result

            for yaml_file in sorted(yaml_files):
                file_result = self.validate_file(str(yaml_file))
                result.issues.extend(file_result.issues)

        elif os.path.isfile(path):
            # Validate single file
            file_result = self.validate_file(path)
            result.issues.extend(file_result.issues)

        elif os.path.isdir(path):
            # Validate all in directory
            dir_path = Path(path)
            yaml_files = list(dir_path.rglob("*.yaml"))
            if not yaml_files:
                result.add_warning(path, "No workflow files found in directory")
                return result

            for yaml_file in sorted(yaml_files):
                file_result = self.validate_file(str(yaml_file))
                result.issues.extend(file_result.issues)

        else:
            result.add_error(path, f"Path not found: {path}")

        return result
