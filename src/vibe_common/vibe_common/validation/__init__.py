# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Workflow validation package — shared between the CLI validator and the server.

Public API
----------
WorkflowParser          Parse a workflow YAML file into a WorkflowSpec.
WorkflowSpec            Dataclass representing a parsed workflow.
WorkflowSpecValidator   Validate a WorkflowSpec (sources, sinks, edges, parameters).
WorkflowDescriptionValidator  Validate workflow documentation completeness.
ParameterResolver       Resolve workflow parameter defaults across task hierarchy.
detect_cycle            Detect DAG cycles and return the offending path.
extract_line_map        Map YAML key-paths to 1-based source line numbers.
find_workflows          Enumerate available workflow names for suggestions.
find_ops                Enumerate available op names for suggestions.
suggest                 Return edit-distance suggestions for a misspelled reference.
"""

from .cycle_detector import detect_cycle
from .description_validator import WorkflowDescriptionValidator, unpack_description
from .line_tracker import extract_line_map, line_for
from .parameter import Parameter, ParameterResolver
from .spec_parser import (
    WorkflowParser,
    WorkflowSpec,
    WorkflowSpecEdge,
    WorkflowSpecNode,
    check_config_fields,
    flat_params,
    get_parameter_reference,
    get_workflow_dir,
    parse_edge_string,
)
from .spec_validator import WorkflowSpecValidator
from .suggestions import find_ops, find_workflows, suggest

__all__ = [
    "WorkflowParser",
    "WorkflowSpec",
    "WorkflowSpecEdge",
    "WorkflowSpecNode",
    "WorkflowSpecValidator",
    "WorkflowDescriptionValidator",
    "ParameterResolver",
    "Parameter",
    "detect_cycle",
    "extract_line_map",
    "line_for",
    "find_workflows",
    "find_ops",
    "suggest",
    "unpack_description",
    "check_config_fields",
    "flat_params",
    "get_parameter_reference",
    "get_workflow_dir",
    "parse_edge_string",
]
