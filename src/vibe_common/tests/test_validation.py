# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests for vibe_common.validation leaf modules:
   cycle_detector, line_tracker, suggestions.
"""

import os
from typing import List, Optional

import pytest

from vibe_common.validation.cycle_detector import detect_cycle
from vibe_common.validation.line_tracker import extract_line_map, line_for
from vibe_common.validation.suggestions import find_ops, find_workflows, suggest


# ---------------------------------------------------------------------------
# detect_cycle
# ---------------------------------------------------------------------------


class TestCycleDetector:
    def _edges(self, pairs):
        """Convert [(src_task, dst_task), ...] to (origins, destinations) lists."""
        origins = [f"{s}.out" for s, _ in pairs]
        dests = [[f"{d}.inp"] for _, d in pairs]
        return origins, dests

    def test_no_cycle_linear(self):
        tasks = ["A", "B", "C"]
        origins, dests = self._edges([("A", "B"), ("B", "C")])
        assert detect_cycle(tasks, origins, dests) is None

    def test_simple_three_node_cycle(self):
        tasks = ["A", "B", "C"]
        origins, dests = self._edges([("A", "B"), ("B", "C"), ("C", "A")])
        cycle = detect_cycle(tasks, origins, dests)
        assert cycle is not None
        # Every node in the returned path must be a task name
        assert all(n in tasks for n in cycle)
        # The path must form an actual cycle (first and last are the same task,
        # or consecutive pairs are edges in the graph)
        assert len(cycle) >= 2

    def test_self_loop(self):
        tasks = ["A", "B"]
        origins, dests = self._edges([("A", "A"), ("A", "B")])
        assert detect_cycle(tasks, origins, dests) is not None

    def test_two_node_cycle(self):
        tasks = ["A", "B"]
        origins, dests = self._edges([("A", "B"), ("B", "A")])
        assert detect_cycle(tasks, origins, dests) is not None

    def test_disconnected_no_cycle(self):
        # C is isolated; A→B is acyclic
        tasks = ["A", "B", "C"]
        origins, dests = self._edges([("A", "B")])
        assert detect_cycle(tasks, origins, dests) is None

    def test_empty_graph(self):
        assert detect_cycle([], [], []) is None

    def test_single_node_no_edges(self):
        assert detect_cycle(["A"], [], []) is None

    def test_cycle_in_subgraph(self):
        # D→E→D cycle; A→B→C is clean
        tasks = ["A", "B", "C", "D", "E"]
        origins, dests = self._edges(
            [("A", "B"), ("B", "C"), ("D", "E"), ("E", "D")]
        )
        cycle = detect_cycle(tasks, origins, dests)
        assert cycle is not None
        assert "D" in cycle or "E" in cycle

    def test_returns_none_for_dag(self):
        # Diamond: A→B, A→C, B→D, C→D — no cycle
        tasks = ["A", "B", "C", "D"]
        origins, dests = self._edges(
            [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")]
        )
        assert detect_cycle(tasks, origins, dests) is None

    def test_multi_destination_edge(self):
        # Edge from A fans out to both B and C, B→A creates a cycle
        tasks = ["A", "B", "C"]
        origins = ["A.out"]
        dests = [["B.inp", "C.inp"]]
        origins += ["B.out"]
        dests += [["A.inp"]]
        cycle = detect_cycle(tasks, origins, dests)
        assert cycle is not None

    def test_port_names_are_ignored_for_task_extraction(self):
        # Port names with dots (sub.port format) — task name is still first segment
        tasks = ["alpha", "beta"]
        origins = ["alpha.sub.out"]
        dests = [["beta.sub.inp"]]
        assert detect_cycle(tasks, origins, dests) is None

    def test_unknown_task_in_edge_is_ignored(self):
        # Edge references a task not in task_names — should not crash
        tasks = ["A", "B"]
        origins = ["A.out", "X.out"]   # X not in tasks
        dests = [["B.inp"], ["A.inp"]]
        assert detect_cycle(tasks, origins, dests) is None


# ---------------------------------------------------------------------------
# extract_line_map / line_for
# ---------------------------------------------------------------------------


_SIMPLE_YAML = """\
name: helloworld
tasks:
  hello:
    op: helloworld
edges:
sources:
  input:
    - hello.user_data
sinks:
  output: hello.processed_data
"""

_EDGE_YAML = """\
name: wf
tasks:
  a:
    op: foo
edges:
  - origin: a.out
    destination:
      - b.inp
sources:
  x:
    - a.inp
sinks:
  y: a.out
"""


class TestLineTracker:
    def test_top_level_keys_are_mapped(self):
        lm = extract_line_map(_SIMPLE_YAML)
        assert "name" in lm
        assert "tasks" in lm
        assert "sources" in lm
        assert "sinks" in lm

    def test_nested_task_key_is_mapped(self):
        lm = extract_line_map(_SIMPLE_YAML)
        assert "tasks.hello" in lm

    def test_line_numbers_are_one_based(self):
        lm = extract_line_map(_SIMPLE_YAML)
        assert lm["name"] == 1
        assert lm["tasks"] == 2
        assert lm["tasks.hello"] == 3

    def test_sequence_items_are_indexed(self):
        lm = extract_line_map(_EDGE_YAML)
        # The 'edges' key exists and has an index-0 item
        assert "edges" in lm
        assert "edges[0]" in lm

    def test_empty_mapping_value_does_not_crash(self):
        # 'edges:' with no value — should not raise
        lm = extract_line_map(_SIMPLE_YAML)
        assert isinstance(lm, dict)

    def test_invalid_yaml_returns_empty_dict(self):
        assert extract_line_map("}{invalid yaml") == {}

    def test_empty_string_returns_empty_dict(self):
        assert extract_line_map("") == {}

    def test_line_for_returns_first_match(self):
        lm = {"tasks.hello": 3, "tasks.world": 5}
        assert line_for(lm, "tasks.hello") == 3

    def test_line_for_returns_first_of_multiple_candidates(self):
        lm = {"tasks.hello": 3}
        assert line_for(lm, "missing_key", "tasks.hello") == 3

    def test_line_for_returns_none_when_no_match(self):
        assert line_for({}, "any.key") is None

    def test_line_for_returns_none_for_empty_map(self):
        assert line_for({}, "tasks.hello") is None


# ---------------------------------------------------------------------------
# suggest / find_workflows / find_ops
# ---------------------------------------------------------------------------


class TestSuggestions:
    def test_suggest_finds_close_match(self):
        result = suggest("hellowrld", ["helloworld", "canopy_cover"])
        assert result == "'helloworld'"

    def test_suggest_returns_none_for_no_match(self):
        assert suggest("zzz_nomatch", ["helloworld", "ndvi"]) is None

    def test_suggest_returns_none_for_empty_candidates(self):
        assert suggest("helloworld", []) is None

    def test_suggest_deduplicates_candidates(self):
        # Same name in both ops and workflows lists should appear only once
        result = suggest("hellowrld", ["helloworld", "helloworld"])
        assert result == "'helloworld'"

    def test_suggest_multiple_matches(self):
        result = suggest("ndv", ["ndvi", "ndwi", "totally_different"])
        # Both ndvi and ndwi are close; should get at least one
        assert result is not None
        assert "ndvi" in result or "ndwi" in result

    def test_suggest_exact_match(self):
        result = suggest("helloworld", ["helloworld", "other"])
        assert result is not None
        assert "helloworld" in result

    def test_suggest_empty_ref_returns_none(self):
        # Empty string has no close matches
        assert suggest("", ["helloworld", "ndvi"]) is None

    def test_find_workflows_returns_relative_names_without_extension(
        self, tmp_path
    ):
        # Create a small fake workflow directory
        (tmp_path / "wf_a.yaml").write_text("name: wf_a\n")
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "wf_b.yaml").write_text("name: wf_b\n")
        # Non-yaml file should be excluded
        (tmp_path / "readme.md").write_text("")

        results = find_workflows(str(tmp_path))
        assert "wf_a" in results
        assert os.path.join("subdir", "wf_b") in results
        assert not any(r.endswith(".yaml") for r in results)
        assert not any("readme" in r for r in results)

    def test_find_workflows_empty_dir_returns_empty(self, tmp_path):
        assert find_workflows(str(tmp_path)) == []

    def test_find_workflows_nonexistent_dir_returns_empty(self):
        assert find_workflows("/nonexistent/path/12345") == []

    def test_find_ops_returns_basenames_without_extension(self, tmp_path):
        (tmp_path / "op_a").mkdir()
        (tmp_path / "op_a" / "op_a.yaml").write_text("name: op_a\n")
        (tmp_path / "op_b").mkdir()
        (tmp_path / "op_b" / "op_b.yaml").write_text("name: op_b\n")
        (tmp_path / "op_b" / "not_an_op.txt").write_text("")

        results = find_ops(str(tmp_path))
        assert "op_a" in results
        assert "op_b" in results
        assert not any(r.endswith(".yaml") for r in results)
        assert "not_an_op" not in results

    def test_find_ops_nonexistent_dir_returns_empty(self):
        assert find_ops("/nonexistent/path/12345") == []
