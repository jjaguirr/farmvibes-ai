import pytest
import yaml
from vibe_dev.testing.integration import ClusterClient

pytestmark = pytest.mark.fast


class TestWorkflowDescribe:
    def test_describe_returns_200(self, cluster_client: ClusterClient):
        """Sanity check: at least helloworld is describable."""
        r = cluster_client.get("/v0/workflows/helloworld", params={"return_format": "description"})
        assert r.status_code == 200

    def test_describe_has_required_fields(self, cluster_client: ClusterClient, all_workflows):
        """Every workflow must have name, inputs, outputs in its description."""
        errors = []
        for wf in all_workflows:
            try:
                desc = cluster_client.describe_workflow(wf)
                for field in ("name", "inputs", "outputs"):
                    if field not in desc:
                        errors.append(f"{wf}: missing field '{field}'")
            except Exception as e:
                errors.append(f"{wf}: {e}")
        assert not errors, "Schema failures:\n" + "\n".join(errors)


class TestWorkflowYaml:
    def test_yaml_is_valid(self, cluster_client: ClusterClient, all_workflows):
        """Every workflow's YAML representation must parse with required keys."""
        errors = []
        for wf in all_workflows:
            try:
                yaml_str = cluster_client.get_workflow_yaml(wf)
                parsed = yaml.safe_load(yaml_str)
                if not isinstance(parsed, dict):
                    errors.append(f"{wf}: YAML did not parse as dict")
                    continue
                for key in ("name", "sources", "sinks", "tasks"):
                    if key not in parsed:
                        errors.append(f"{wf}: missing YAML key '{key}'")
            except Exception as e:
                errors.append(f"{wf}: {e}")
        assert not errors, "YAML validation failures:\n" + "\n".join(errors)
