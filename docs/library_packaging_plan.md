# Plan: Make JacobianODE consumable as an installable library

## Context

We want JacobianODE to be reusable as a `pip install`-able dependency from
downstream repos (e.g. MindControl) without requiring a fork. The downstream
consumer should be able to:
- Bring their own dataset via Hydra `_target_:`
- Bring their own experiment YAMLs that extend JacobianODE-side templates via `defaults:`
- Call training as a Python function: `from JacobianODE import train; train(cfg)`
- Write their own ~10-line `@hydra.main` entry script

We deliberately do **not** add `[project.scripts]` CLI entry points — the library
API is the contract. Most ML libraries (Lightning, HF Transformers, sklearn)
follow this pattern. Sweep automation (`jsweep`, engaging-controller, jacobian-reports)
stays personal infra and is **not** part of the library surface.

The intended consumer pattern:
```python
# consumer repo: train.py
import hydra
from JacobianODE import train

@hydra.main(config_path="conf", config_name="train", version_base=None)
def main(cfg):
    train(cfg)

if __name__ == "__main__":
    main()
```
with the consumer's `conf/experiment/<name>.yaml` extending JacobianODE-side
templates via `defaults:` (e.g. `defaults: [override /model: latent_additive_coupling]`).

## Status quo (audited 2026-05-02)

The repo is ~85% there; the audit confirmed several blockers the original
draft missed:

- `_run_training(cfg)` already exists at `run_jacobians.py:71-383`, takes only
  `cfg`, returns `float` (best val loss for Optuna). Imports clean.
- `train_jacobians` (the `@hydra.main` wrapper at line 31) handles
  faulthandler + `error_traceback.txt` writing OUTSIDE `_run_training`.
- `JacobianODE/__init__.py` is empty (3 blank lines). No side effects.
- No module-level state mutation, signal handlers, or CUDA init at import.
- Existing custom Hydra plugin at `hydra_plugins/list_sweeper_plugin/`
  (a Sweeper, not a SearchPathPlugin — no conflict).

**Real blockers identified:**
1. `sys.exit(0)` at `run_jacobians.py:92` (done-marker short-circuit) — kills
   the caller's Python process; library users can't recover.
2. YAML config files **not included** in the wheel — `pyproject.toml` has no
   `[tool.hatch.build.targets.wheel]` `include` patterns; only `*.py` files
   are bundled by default. Consumers would `pip install JacobianODE` and
   then `from JacobianODE import train` works but `defaults: [override
   /model: latent_additive_coupling]` would fail to resolve.
3. `wmtask @ git+ssh://...` is a hard dependency in `[project.dependencies]`
   (pyproject.toml:43). Anyone without SSH access to that repo cannot
   `pip install JacobianODE`. Code-wise wmtask is loaded only via Hydra
   `_target_:` when `cfg.data.data_type == "wmtask"` — never imported
   directly — so making it optional is safe.
4. `wandb.Api()` at `logging.py:225` (run-name dedup) has no error handling
   — for a consumer without a wandb login, the call throws at training
   start. Wrap + add explicit disable flag.

**Non-blockers (informational):**
- Hardcoded `/orcd/...` paths live in some experiment YAMLs (chain_smoke_*,
  some sweep YAMLs). Consumers must override these in their own YAMLs.
  Don't strip from the JacobianODE-side YAMLs — they're load-bearing for
  the maintainer's engaging runs.
- `wandb_entity: JacobianODE` default is consumer-overridable; document.
- Diagnostics dir uses `os.getcwd()` (Hydra's chdir-managed dir at
  runtime) — works fine for library callers as long as they Hydra-decorate
  their entry; document the alternative for non-Hydra callers.

## Plan — single PR

### 1. Library entry point (`JacobianODE/__init__.py`, `run_jacobians.py`)

- Rename `_run_training` → `train` in `JacobianODE/jacobians/run_jacobians.py:71`.
  Update the `train_jacobians` wrapper at line 62 to call `train(cfg)`.
  (The leading underscore wrongly signaled "private" for what's now the
  public library entry.)
- **Move faulthandler + error_traceback wrapping INSIDE `train`** so library
  callers get the same crash-dump diagnostics. Currently lines 54–68 of
  `train_jacobians`. After the move, `train_jacobians` becomes a 2-line
  Hydra-decorated shim that just calls `train(cfg)`.
- Replace `sys.exit(0)` at line 92 (done-marker short-circuit) with
  `return None`. The Hydra wrapper exits 0 cleanly when the function
  returns; library callers can catch `None` and decide how to handle.
- Add to `JacobianODE/__init__.py`:
  ```python
  from .jacobians.run_jacobians import train
  __all__ = ["train"]
  ```

### 2. wandb hardening (`logging.py`, `conf/config.yaml`, `trainer.py`)

- `JacobianODE/jacobians/training/logging.py:225` — wrap the `wandb.Api()`
  dedup call in `try/except Exception`. On failure: log a warning, return
  the candidate name as-is (skip dedup). Network / auth failures should
  not crash training.
- Add `wandb.disabled: false` (default) to `JacobianODE/jacobians/conf/config.yaml`.
- In `JacobianODE/jacobians/training/trainer.py` (around line 189 where
  `WandbLogger` is instantiated): if `cfg.wandb.disabled`, skip the
  `WandbLogger` construction entirely and pass `logger=None` (or a
  CSVLogger fallback) to the Lightning Trainer. Also gate the upstream
  `setup_wandb` call in `_run_training`/`train` on the same flag.

### 3. Packaging (`pyproject.toml`)

- Move `wmtask @ git+ssh://...` from `[project.dependencies]` to
  `[project.optional-dependencies] wmtask = [...]`. Consumers who use the
  WMTask data path do `uv sync --extra wmtask` (or `pip install
  JacobianODE[wmtask]`). Default install becomes consumer-friendly.
  **Important**: also keep `wmtask` listed under `[dependency-groups] dev`
  (or equivalent) so the maintainer's local `uv sync` continues to pull
  it in. Only the consumer-facing `[project.dependencies]` loses it.
- Add explicit hatch wheel-build config so YAML configs ship with the wheel:
  ```toml
  [tool.hatch.build.targets.wheel]
  packages = ["JacobianODE"]
  include = [
      "JacobianODE/**/*.yaml",
      "JacobianODE/**/*.yml",
  ]
  ```
- Verify with `uv build && unzip -l dist/*.whl | grep yaml` — should list
  every YAML under `JacobianODE/jacobians/conf/`.

### 4. Hydra SearchPathPlugin

Create `JacobianODE/_hydra_plugins/__init__.py` (empty) and
`JacobianODE/_hydra_plugins/jacobianode_searchpath.py`:

```python
from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


class JacobianODESearchPathPlugin(SearchPathPlugin):
    """Make JacobianODE-side conf YAMLs discoverable in consumer projects."""
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(
            provider="jacobianode",
            path="pkg://JacobianODE.jacobians.conf",
        )
```

Hydra auto-discovers plugin packages under any importable `hydra_plugins.*`
namespace; for first-party in-package plugins, register via the entry point
in `pyproject.toml`:

```toml
[project.entry-points."hydra_plugins"]
jacobianode = "JacobianODE._hydra_plugins.jacobianode_searchpath"
```

This makes `defaults: [override /model: latent_additive_coupling]` resolve
in any consumer config without manual `hydra.searchpath` setup.

### 5. README — "Use as a library" section

Add a short section to `README.md` (~25 lines) covering:
- The 10-line consumer `train.py` template.
- How to set up `conf/config.yaml` with a `defaults:` block referencing
  JacobianODE-side templates (e.g. `- override /model: latent_additive_coupling`).
- How to disable wandb when running locally (`wandb.disabled: true`).
- Note on the optional `wmtask` extra.
- Pointer that sweep automation (`jsweep`, controller, jacobian-reports)
  is personal infra — consumers should use Submitit / sbatch directly.

## Critical files

| File | Change |
|---|---|
| `JacobianODE/__init__.py` | Add `from .jacobians.run_jacobians import train` |
| `JacobianODE/jacobians/run_jacobians.py` | Rename `_run_training` → `train`; move faulthandler inside; `return None` instead of `sys.exit(0)` at L92 |
| `JacobianODE/jacobians/training/logging.py` | Try/except around `wandb.Api()` dedup at L225 |
| `JacobianODE/jacobians/training/trainer.py` | Gate `WandbLogger` construction on `cfg.wandb.disabled` |
| `JacobianODE/jacobians/conf/config.yaml` | Add `wandb.disabled: false` default |
| `pyproject.toml` | Move `wmtask` to extras (keep in dev group); add `[tool.hatch.build.targets.wheel]` include patterns; add `[project.entry-points."hydra_plugins"]` |
| `JacobianODE/_hydra_plugins/__init__.py` | New (empty) |
| `JacobianODE/_hydra_plugins/jacobianode_searchpath.py` | New SearchPathPlugin |
| `README.md` | New "Use as a library" section |

## Verification

After the changes:

1. **Tests**: `uv run --no-sync pytest JacobianODE/jacobians/tests/` — all
   197 must pass (baseline).
2. **Import smoke**: `uv run --no-sync python -c "from JacobianODE import
   train; print(train)"` — confirms top-level export.
3. **Wheel build**: `uv build`. Then `unzip -l dist/JacobianODE-0.1.0-*.whl
   | grep yaml | wc -l` — should print > 10 (all the conf YAMLs).
4. **Wheel-from-outside smoke** (the load-bearing test):
   ```bash
   mkdir /tmp/jaco_consumer && cd /tmp/jaco_consumer
   uv venv && source .venv/bin/activate
   uv pip install <repo>/dist/JacobianODE-0.1.0-*.whl
   python -c "from JacobianODE import train; print('OK')"
   # Hydra search-path test:
   python -c "
   from hydra import compose, initialize_config_dir
   import tempfile, pathlib
   tmp = pathlib.Path(tempfile.mkdtemp())
   (tmp / 'config.yaml').write_text('defaults:\n  - override /model: latent_additive_coupling\n  - _self_\n')
   with initialize_config_dir(version_base=None, config_dir=str(tmp)):
       cfg = compose('config')
       print('model resolved:', cfg.model.encoder._target_)
   "
   ```
   Both should succeed cleanly without referencing the source repo.
5. **wandb-disabled smoke**: in the same external venv, run a minimal
   `train(cfg)` with `cfg.wandb.disabled=true` and a tiny synthetic
   dataset — should complete without ever calling wandb.
6. **Back-compat**: existing in-repo training (`uv run python -m
   JacobianODE.jacobians.run_jacobians experiment=lorenz_full3_...`) must
   still work end-to-end with no behavior change. Confirm by spot-running
   a minimal experiment for 1 epoch.

## Out of scope (deferred)

- **Sweep entry-point library-ification.** `JacobianODE/jacobians/tuning/sweep.py`
  (844 lines) likely has script-only assumptions about filesystem layout
  and the engaging-controller contract. Not investigated. Library-ifying
  training is small and low-risk; library-ifying sweeps is a larger
  audit. Defer until there's a concrete consumer demand.
- **`jsweep` / `engaging-controller` / `jacobian-reports`** automation.
  Personal infra. Other consumers use Submitit / sbatch directly.
- **`[project.scripts]` CLI entry points.** Explicitly chosen against —
  library API is the contract.
- **Stripping hardcoded `/orcd/...` paths from JacobianODE-side YAMLs.**
  They're load-bearing for the maintainer's engaging runs; consumers
  override in their own YAMLs. Document, don't remove.
- **PyPI publishing.** Out of scope for this PR — wheel-from-local-path
  install + git+https for downstream pyproject is enough for MindControl.

## Risks / what could go wrong

1. **Hydra entry-point plugin discovery silently fails.** The
   `hydra_plugins.*` auto-discovery only works for top-level
   `hydra_plugins/` packages on `sys.path`. The entry-point variant
   (`[project.entry-points."hydra_plugins"]`) is the correct plumbing for
   in-package plugins shipped via wheel. **Mitigation**: the wheel-from-outside
   smoke (verification step 4) directly exercises the search-path resolution.
   If it fails, fall back to documenting manual `hydra.searchpath:
   [pkg://JacobianODE.jacobians.conf]` in the README.
2. **`wandb.disabled` shim misses a code path that calls `wandb.*`.**
   Multiple files touch wandb (logging.py, trainer.py, wandb_utils.py,
   run_analytics.py). The cfg flag must gate every `wandb.init()` /
   `wandb.Api()` / direct `wandb.log()` call. **Mitigation**: grep for
   `wandb\.` after edits and verify each call is either gated or already
   inside a Lightning logger that's been replaced.
3. **Moving wmtask to extras breaks the maintainer's existing workflow**
   if any wmtask import is now lazy + the package isn't installed locally.
   **Mitigation**: keep `wmtask` in the `dev` dependency-group too so
   the maintainer's `uv sync` keeps installing it; only the
   consumer-facing `[project.dependencies]` loses it.
4. **YAML inclusion in wheel still misses something.** Hatchling's
   `include` is documented but worth manually verifying with `unzip -l`
   that the conf tree is complete (not just config.yaml). The
   verification step covers this.
