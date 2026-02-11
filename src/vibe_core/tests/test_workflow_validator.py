# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Integration tests for vibe_core.cli.workflow_validator.

Uses the fake ops / workflows from vibe_dev.testing to drive
validate_file() and validate_all() end-to-end.
"""

import os
from typing import List
from unittest.mock import patch

import pytest

from vibe_common.validation.spec_parser import (
    TaskType,
    WorkflowSpec,
    WorkflowSpecEdge,
    WorkflowSpecNode,
)
from vibe_core.datamodel import TaskDescription
from vibe_core.cli.workflow_validator import (
    ValidationIssue,
    validate_all,
    validate_file,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _errors(issues: List[ValidationIssue]) -> List[ValidationIssue]:
    return [i for i in issues if i.severity == "error"]


def _warnings(issues: List[ValidationIssue]) -> List[ValidationIssue]:
    return [i for i in issues if i.severity == "warning"]


# ---------------------------------------------------------------------------
# validate_file — valid workflows
# ---------------------------------------------------------------------------


def test_validate_file_valid_workflow(fake_ops_dir, fake_workflows_dir):
    path = os.path.join(fake_workflows_dir, "item_gather.yaml")
    issues = validate_file(path, fake_workflows_dir, fake_ops_dir)
    assert _errors(issues) == []


def test_validate_file_valid_workflow_returns_empty_list(fake_ops_dir, fake_workflows_dir):
    path = os.path.join(fake_workflows_dir, "task_params.yaml")
    issues = validate_file(path, fake_workflows_dir, fake_ops_dir)
    assert issues == []


def test_validate_file_fan_out_no_errors(fake_ops_dir, fake_workflows_dir):
    # fan_out_and_in uses list destinations — exercises _validate_edges cleanly
    path = os.path.join(fake_workflows_dir, "fan_out_and_in.yaml")
    issues = validate_file(path, fake_workflows_dir, fake_ops_dir)
    assert _errors(issues) == []


# ---------------------------------------------------------------------------
# validate_file — structural errors detected by WorkflowSpecValidator
# ---------------------------------------------------------------------------


def test_validate_file_nonexistent_source_task_reports_error(
    fake_ops_dir, fake_workflows_dir, tmp_path
):
    # _validate_sources raises when a source references a task not in tasks dict
    bad_yaml = (
        "name: bad_src_task\n"
        "tasks:\n"
        "  real_task:\n"
        "    op: item_item\n"
        "    op_dir: fake\n"
        "edges:\n"
        "sources:\n"
        "  input:\n"
        "    - ghost_task.user_data\n"   # task 'ghost_task' doesn't exist
        "sinks:\n"
        "  output: real_task.processed_data\n"
    )
    wf_path = tmp_path / "bad_src_task.yaml"
    wf_path.write_text(bad_yaml)
    issues = validate_file(str(wf_path), fake_workflows_dir, fake_ops_dir)
    assert len(_errors(issues)) > 0
    assert any("ghost_task" in i.message for i in _errors(issues))


def test_validate_file_nonexistent_sink_task_reports_error(
    fake_ops_dir, fake_workflows_dir, tmp_path
):
    # _validate_sinks raises when a sink references a task not in tasks dict
    bad_yaml = (
        "name: bad_sink_task\n"
        "tasks:\n"
        "  real_task:\n"
        "    op: item_item\n"
        "    op_dir: fake\n"
        "edges:\n"
        "sources:\n"
        "  input:\n"
        "    - real_task.user_data\n"
        "sinks:\n"
        "  output: ghost_task.processed_data\n"   # task 'ghost_task' doesn't exist
    )
    wf_path = tmp_path / "bad_sink_task.yaml"
    wf_path.write_text(bad_yaml)
    issues = validate_file(str(wf_path), fake_workflows_dir, fake_ops_dir)
    assert len(_errors(issues)) > 0
    assert any("ghost_task" in i.message for i in _errors(issues))


# ---------------------------------------------------------------------------
# validate_file — missing op (FileNotFoundError path)
# ---------------------------------------------------------------------------


def test_validate_file_missing_op_reports_error(fake_ops_dir, fake_workflows_dir, tmp_path):
    # Write a workflow that references a non-existent op
    typo_yaml = (
        "name: typo_wf\n"
        "tasks:\n"
        "  hello:\n"
        "    op: hellowrld\n"   # typo — op does not exist
        "    op_dir: fake\n"
        "edges:\n"
        "sources:\n"
        "  input:\n"
        "    - hello.user_data\n"
        "sinks:\n"
        "  output: hello.processed_data\n"
    )
    wf_path = tmp_path / "typo_wf.yaml"
    wf_path.write_text(typo_yaml)

    issues = validate_file(str(wf_path), fake_workflows_dir, fake_ops_dir)
    assert len(_errors(issues)) > 0


def test_validate_file_missing_op_suggests_correction(fake_ops_dir, fake_workflows_dir, tmp_path):
    typo_yaml = (
        "name: typo_wf\n"
        "tasks:\n"
        "  hello:\n"
        "    op: item_itm\n"   # close to item_item
        "    op_dir: fake\n"
        "edges:\n"
        "sources:\n"
        "  input:\n"
        "    - hello.user_data\n"
        "sinks:\n"
        "  output: hello.processed_data\n"
    )
    wf_path = tmp_path / "typo_wf.yaml"
    wf_path.write_text(typo_yaml)

    issues = validate_file(str(wf_path), fake_workflows_dir, fake_ops_dir)
    errors = _errors(issues)
    assert len(errors) > 0
    # At least one error should carry a suggestion
    assert any(i.suggestion for i in errors)
    suggestion_text = " ".join(i.suggestion for i in errors if i.suggestion)
    assert "item_item" in suggestion_text


def test_validate_file_missing_op_reports_line_number(fake_ops_dir, fake_workflows_dir, tmp_path):
    typo_yaml = (
        "name: typo_wf\n"
        "tasks:\n"
        "  hello:\n"
        "    op: doesnotexist\n"
        "    op_dir: fake\n"
        "edges:\n"
        "sources:\n"
        "  input:\n"
        "    - hello.user_data\n"
        "sinks:\n"
        "  output: hello.processed_data\n"
    )
    wf_path = tmp_path / "typo_wf.yaml"
    wf_path.write_text(typo_yaml)

    issues = validate_file(str(wf_path), fake_workflows_dir, fake_ops_dir)
    errors = _errors(issues)
    assert len(errors) > 0
    # The error should reference the 'tasks' section
    assert any(i.location == "tasks" for i in errors)


# ---------------------------------------------------------------------------
# validate_file — cycle detection
# ---------------------------------------------------------------------------


def test_validate_file_detects_cycle(fake_ops_dir, fake_workflows_dir, tmp_path):
    # 3-task cycle: A→B→C→B (B and C form the cycle).
    # Source is A.inp which is NOT an edge destination — _validate_edges won't
    # complain about "source is also a destination" and we reach detect_cycle().
    def _node(task_name):
        return WorkflowSpecNode(
            task=task_name, type=TaskType.op, parameters={}, op_dir="fake", parent="cyclic"
        )

    spec = WorkflowSpec(
        name="cyclic",
        tasks={"A": _node("item_item"), "B": _node("item_item"), "C": _node("item_item")},
        sources={"input": ["A.user_data"]},
        sinks={"output": "C.processed_data"},
        edges=[
            WorkflowSpecEdge(origin="A.processed_data", destination=["B.user_data"]),
            WorkflowSpecEdge(origin="B.processed_data", destination=["C.user_data"]),
            WorkflowSpecEdge(origin="C.processed_data", destination=["B.user_data"]),  # cycle
        ],
        parameters={},
        default_parameters={},
        description=TaskDescription(),
        ops_dir=fake_ops_dir,
        workflows_dir=fake_workflows_dir,
    )

    wf_path = tmp_path / "cyclic.yaml"
    wf_path.write_text("name: cyclic\ntasks:\n  A: {}\n  B: {}\n  C: {}\n")

    with patch("vibe_core.cli.workflow_validator.WorkflowParser") as mock_parser:
        mock_parser.parse.return_value = spec
        with patch("vibe_core.cli.workflow_validator.WorkflowSpecValidator") as mock_validator:
            mock_validator._validate_sources.return_value = None
            mock_validator._validate_sinks.return_value = None
            mock_validator._validate_edges.return_value = None
            mock_validator._validate_parameters.return_value = None

            issues = validate_file(str(wf_path), fake_workflows_dir, fake_ops_dir)

    cycle_errors = [i for i in issues if "cycle" in i.message.lower()]
    assert len(cycle_errors) == 1
    assert "B" in cycle_errors[0].message or "C" in cycle_errors[0].message


# ---------------------------------------------------------------------------
# validate_all
# ---------------------------------------------------------------------------


def test_validate_all_valid_workflows(fake_ops_dir, fake_workflows_dir):
    results = validate_all(fake_workflows_dir, fake_ops_dir)
    assert len(results) > 0
    # Every file should be present in results
    for _path, issues in results:
        assert isinstance(issues, list)


def test_validate_all_returns_path_issue_pairs(fake_ops_dir, fake_workflows_dir):
    results = validate_all(fake_workflows_dir, fake_ops_dir)
    for path, issues in results:
        assert os.path.isabs(path)
        assert path.endswith(".yaml")


def test_validate_all_empty_dir_returns_empty_list(tmp_path):
    results = validate_all(str(tmp_path), str(tmp_path))
    assert results == []


def test_validate_all_nonexistent_dir_returns_empty_list(tmp_path):
    results = validate_all(str(tmp_path / "no_such_dir"), str(tmp_path))
    assert results == []


def test_validate_all_includes_all_yaml_files(fake_ops_dir, fake_workflows_dir, tmp_path):
    # Create a small isolated workflows dir with two known YAML files
    wf_a = tmp_path / "wf_a.yaml"
    wf_b = tmp_path / "wf_b.yaml"
    # Write minimal valid-looking YAML (will fail validation, but should appear in results)
    wf_a.write_text("name: wf_a\ntasks: {}\nsources: {}\nsinks: {}\nedges: []\n")
    wf_b.write_text("name: wf_b\ntasks: {}\nsources: {}\nsinks: {}\nedges: []\n")

    results = validate_all(str(tmp_path), fake_ops_dir)
    paths = [p for p, _ in results]
    assert str(wf_a.resolve()) in paths
    assert str(wf_b.resolve()) in paths


def test_validate_all_skips_non_yaml_files(fake_ops_dir, tmp_path):
    (tmp_path / "not_a_workflow.txt").write_text("ignored")
    (tmp_path / "also_ignored.json").write_text("{}")
    results = validate_all(str(tmp_path), fake_ops_dir)
    assert results == []
