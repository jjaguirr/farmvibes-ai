# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Edit-distance suggestions for misspelled op and workflow references."""

import difflib
import os
from typing import List, Optional


def find_workflows(workflows_dir: str) -> List[str]:
    """Return all workflow names (relative path without .yaml) under *workflows_dir*."""
    results: List[str] = []
    if not os.path.isdir(workflows_dir):
        return results
    for root, _dirs, files in os.walk(workflows_dir):
        for fname in sorted(files):
            if fname.endswith(".yaml"):
                rel = os.path.relpath(os.path.join(root, fname), workflows_dir)
                results.append(rel[:-5])  # strip .yaml
    return results


def find_ops(ops_dir: str) -> List[str]:
    """Return all op names (yaml basename without .yaml) under *ops_dir*."""
    results: List[str] = []
    if not os.path.isdir(ops_dir):
        return results
    for root, _dirs, files in os.walk(ops_dir):
        for fname in sorted(files):
            if fname.endswith(".yaml"):
                results.append(fname[:-5])
    return results


def suggest(ref: str, candidates: List[str], n: int = 3, cutoff: float = 0.5) -> Optional[str]:
    """Return a human-readable suggestion string for *ref*, or None if no close match."""
    matches = difflib.get_close_matches(ref, list(dict.fromkeys(candidates)), n=n, cutoff=cutoff)
    return ", ".join(f"'{m}'" for m in matches) if matches else None
