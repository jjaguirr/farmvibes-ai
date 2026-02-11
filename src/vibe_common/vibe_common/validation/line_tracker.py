# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Map YAML key-paths to 1-based line numbers using PyYAML's AST."""

from typing import Any, Dict, Optional

import yaml


def extract_line_map(content: str) -> Dict[str, int]:
    """Parse *content* as YAML and return a dict mapping key-paths to line numbers.

    Key-paths follow the pattern ``"parent.child"`` for mapping keys and
    ``"parent[0]"`` for sequence items.  Line numbers are 1-based.

    Returns an empty dict if the YAML cannot be parsed.
    """
    try:
        doc = yaml.compose(content)
    except yaml.YAMLError:
        return {}

    result: Dict[str, int] = {}

    def walk(node: Any, prefix: str) -> None:
        if node is None:
            return
        if isinstance(node, yaml.MappingNode):
            for kn, vn in node.value:
                key = str(kn.value)
                p = f"{prefix}.{key}" if prefix else key
                result[p] = kn.start_mark.line + 1
                walk(vn, p)
        elif isinstance(node, yaml.SequenceNode):
            for i, item in enumerate(node.value):
                p = f"{prefix}[{i}]"
                result[p] = item.start_mark.line + 1
                walk(item, p)

    walk(doc, "")
    return result


def line_for(line_map: Dict[str, int], *keys: str) -> Optional[int]:
    """Return the first matching line number from *line_map*, or None."""
    for key in keys:
        if key in line_map:
            return line_map[key]
    return None
