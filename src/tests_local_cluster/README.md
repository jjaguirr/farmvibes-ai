# Live-cluster integration tests

Tests that run against a real FarmVibes.AI deployment. No mocks.

## Running

```bash
# point at your cluster (or let it read ~/.config/farmvibes-ai/service_url)
export FARMVIBES_AI_SERVICE_URL=http://127.0.0.1:31109/

pytest src/tests_local_cluster -m fast              # seconds — health + input validation
pytest src/tests_local_cluster -m slow              # minutes — actual workflow execution
pytest src/tests_local_cluster -m "fast or slow"    # the normal CI suite
pytest src/tests_local_cluster -m cli               # CLI shell-out tests (needs kubeconfig)
pytest src/tests_local_cluster -m external          # hits USDA/OSM — opt-in, can flake
pytest src/tests_local_cluster                      # everything except external
```

If the cluster is unreachable, the whole suite skips with one message.
It does not report forty green passes against a dead cluster.

## Markers

| marker     | runtime | what it covers                                    |
|------------|---------|---------------------------------------------------|
| `fast`     | seconds | health endpoints, error handling, schema checks   |
| `slow`     | minutes | end-to-end workflow execution                     |
| `cli`      | seconds | `farmvibes-ai` subcommands (status, health, logs) |
| `newapi`   | seconds | `/v0/health`, `/healthz/*` — skip on old images   |
| `external` | minutes | real imagery/data downloads — skip unless `-m external` |

## Files

| file                            | tier      | contents                                    |
|---------------------------------|-----------|---------------------------------------------|
| `helpers.py`                    | library   | reusable fixtures — import these anywhere   |
| `conftest.py`                   | pytest    | marker registration, session fixtures       |
| `test_health.py`                | fast      | every endpoint responds with expected shape |
| `test_failures.py`              | fast      | bad input → 4xx + useful message, no zombies |
| `test_cli.py`                   | fast, cli | `farmvibes-ai local` subcommands            |
| `test_smoke.py`                 | slow      | THE smoke test — helloworld end-to-end      |
| `test_workflow_regression.py`   | mixed     | schema stability + cache/cancel paths       |
| `test_cluster_integration.py`   | legacy    | pre-existing — pixel-level raster checks    |

## Using the helpers elsewhere

```python
from tests_local_cluster.helpers import (
    resolve_service_url, WorkflowSubmitter, tracked_runs, poll_until,
)
from vibe_core.client import FarmvibesAiClient

client = FarmvibesAiClient(resolve_service_url())
with tracked_runs(client) as sub:
    rec, status = sub.submit_and_wait("helloworld", timeout_s=300)
    assert status == RunStatus.done
# cleanup happens automatically on context exit
```

## Known quirks

- `farmvibes-ai local logs <bad-service>` exits 0. There's an xfail for
  this in `test_cli.py`. It's a real bug in the CLI's exit-code path.
- `/v0/health` is newer than some cluster images. Tests that need it are
  marked `newapi` and skip cleanly on 404.
- k3d reassigns the loadbalancer host port on cluster restart. Re-run
  `farmvibes-ai local status` to refresh `~/.config/farmvibes-ai/service_url`,
  or set `FARMVIBES_AI_SERVICE_URL` explicitly.
