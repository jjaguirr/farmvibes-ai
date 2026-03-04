# Deployment Profiles Design

## Overview

Named deployment profiles for FarmVibes.AI local k3d clusters. A profile is a YAML file containing overrides for worker count, resource limits, log level, and other settings that should vary by environment (laptop vs beefy VM).

## Profile Schema

Standalone schema dict in `profiles.py` defining valid keys, types, and constraints:

```python
PROFILE_SCHEMA = {
    "worker_replicas": {"type": int, "min": 1, "max": 64},
    "worker_memory_request": {"type": str, "pattern": r"^[1-9]\d*([EPTGMK]i?)?$"},
    "log_level": {"type": str, "choices": ["DEBUG", "INFO", "WARNING", "ERROR"]},
    "max_log_file_bytes": {"type": int, "min": 0, "optional": True},
    "log_backup_count": {"type": int, "min": 0, "optional": True},
    "enable_telemetry": {"type": bool},
    "servers": {"type": int, "min": 1, "max": 8},
    "agents": {"type": int, "min": 0, "max": 16},
}
```

Adding new profile keys is a schema change only — no code changes needed.

## Built-in Profiles

Location: `vibe_core/cli/profiles/`

- **`minimal.yaml`** — 1 worker, 64Mi memory, INFO log level, telemetry off
- **`default.yaml`** — current behavior (worker_replicas: "auto" → cpu_count//2-1, 100Mi, DEBUG)
- **`production.yaml`** — more workers (cpu_count-2), 512Mi memory, INFO, telemetry off

Each file has comments documenting every available key and its constraints. The `"auto"` sentinel for `worker_replicas` triggers the existing CPU-based calculation.

## Profile Loading & Validation

New module: `vibe_core/cli/profiles.py`

### `validate_profile(profile_dict, schema)`

- Unknown keys → `"Unknown profile key 'workerz_count'. Valid keys: worker_replicas, ..."`
- Wrong types → `"Key 'worker_replicas' expects int, got str"`
- Out-of-range → `"Key 'worker_replicas' must be >= 1, got 0"`
- Pattern mismatch → `"Key 'worker_memory_request' invalid format: 'abc'"`

Standalone function — not coupled to FarmVibes config keys. Takes any dict and any schema.

### `load_profile(name, search_paths)`

Search order:
1. User config dir: `~/.config/farmvibes-ai/profiles/{name}.yaml`
2. Built-in: `vibe_core/cli/profiles/{name}.yaml`

User dir takes precedence, enabling custom profiles and built-in overrides.

### `ProfileValidationError`

`ValueError` subclass. Raised before any Terraform/k3d work begins.

## Config Layering

Priority (lowest → highest):

```
FarmVibesConfig defaults → profile YAML → FARMVIBES_* env vars → CLI args
```

### How it works

1. `--profile` flag added to `setup` and `update` in `LocalCliParser`
2. `FARMVIBES_PROFILE` env var as fallback. Flag wins if both set.
3. In `local.py:dispatch()`, before calling `setup()`:
   - Load and validate profile if selected
   - Profile values become defaults; explicit CLI args and env vars still override
4. `"auto"` sentinel for `worker_replicas` triggers `cpu_count() // 2 - 1`

## Files Changed

| File | Change |
|------|--------|
| `vibe_core/cli/profiles.py` | New — schema, validation, loading, layering logic |
| `vibe_core/cli/profiles/minimal.yaml` | New — built-in minimal profile |
| `vibe_core/cli/profiles/default.yaml` | New — built-in default profile |
| `vibe_core/cli/profiles/production.yaml` | New — built-in production profile |
| `vibe_core/cli/parsers.py` | Add `--profile` flag to setup/update |
| `vibe_core/cli/local.py` | Load profile in `dispatch()`, merge into args |
| `vibe_core/cli/wrappers.py` | `ensure_local_cluster()` accepts/passes `worker_memory_request` |
| `terraform/local/main.tf` | Use `var.worker_memory_request` instead of hardcoded `"100Mi"` |
| `terraform/local/variables.tf` | Add `worker_memory_request` variable |
| `tests/test_profiles.py` | New — unit tests for validation, loading, layering |

## Testing

### Unit tests

- `validate_profile`: valid passes, unknown key errors, wrong type, out-of-range, pattern mismatch, empty profile
- `load_profile`: loads built-in, user dir precedence, missing profile error, invalid YAML error
- Layering: profile overrides defaults, env vars override profile, CLI args override all, `"auto"` sentinel resolution

### VM integration tests

- `--profile minimal` → 1 worker, 64Mi
- `--profile production` → more workers, 512Mi
- `FARMVIBES_PROFILE=minimal` → same as flag
- Flag overrides env var
- Custom profile with typo key → fails before Terraform
- `kubectl describe` pods show matching resource limits
- `--worker-replicas 2 --profile production` → CLI arg wins

### VM isolation

Worktree at `~/farmvibes-ai-work/13_model_a` with dedicated venv. Every VM command prefixed with `source ~/farmvibes-env_13_model_a.sh &&`.
