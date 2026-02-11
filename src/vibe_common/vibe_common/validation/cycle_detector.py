# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""DAG cycle detection for workflow task graphs.

The server-side ``Graph`` class in ``vibe_server.workflow.graph`` raises on
cycles during topological sort but does not return the cycle path.  This
module adds cycle-path reporting needed by the CLI validator.
"""

from typing import Dict, List, Optional


def _task_name(node_port: str) -> str:
    """Return the task portion of a ``'task.port'`` or ``'task.sub.port'`` string."""
    return node_port.split(".")[0]


def detect_cycle(
    task_names: List[str],
    edge_origins: List[str],
    edge_destinations: List[List[str]],
) -> Optional[List[str]]:
    """Return an ordered list of task names forming a cycle, or ``None``.

    Parameters
    ----------
    task_names:
        All task names in the workflow.
    edge_origins:
        List of edge origin strings (``'task.port'`` format), one per edge.
    edge_destinations:
        List of destination-port lists, aligned with *edge_origins*.
    """
    graph: Dict[str, List[str]] = {name: [] for name in task_names}

    for origin, dests in zip(edge_origins, edge_destinations):
        src = _task_name(origin)
        for dest_port in dests:
            dst = _task_name(dest_port)
            if src in graph and dst in graph:
                graph[src].append(dst)

    WHITE, GRAY, BLACK = 0, 1, 2
    color: Dict[str, int] = {n: WHITE for n in task_names}
    parent: Dict[str, Optional[str]] = {n: None for n in task_names}

    def dfs(node: str) -> Optional[List[str]]:
        color[node] = GRAY
        for nbr in graph[node]:
            if color[nbr] == GRAY:
                # Reconstruct cycle path back to nbr
                cycle = [nbr]
                cur: Optional[str] = node
                while cur is not None and cur != nbr:
                    cycle.append(cur)
                    cur = parent[cur]
                cycle.reverse()
                return cycle
            if color[nbr] == WHITE:
                parent[nbr] = node
                result = dfs(nbr)
                if result is not None:
                    return result
        color[node] = BLACK
        return None

    for node in task_names:
        if color[node] == WHITE:
            result = dfs(node)
            if result is not None:
                return result
    return None
