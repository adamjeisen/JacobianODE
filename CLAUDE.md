## Working style — caution, clarification, tests

Default to cautious and careful in every implementation. Concretely:

- **Ask on substantive ambiguity — don't ask on routine ops.** The
  clarification bar is for algorithmic / ML-design / mathematical
  decisions where a wrong guess is expensive to undo: tensor shapes and
  conventions, sign / block-ordering choices, which loss term to
  penalise, whether a knob defaults on/off in a way that shapes the
  experiment, loss/metric semantics, which Jacobian block maps where,
  etc. When there are multiple reasonable interpretations at that
  level, list them and stop. For routine operational stuff (file paths,
  tmux session names, whether to use an obvious CLI flag, which exact
  directory to make) just pick the most literal reading and proceed;
  you can always fix a misnamed folder, you can't easily redo a
  swept-over-the-wrong-loss experiment. When genuinely unsure whether
  something is "substantive" or "routine," err on the side of proceeding
  and explaining what you did.
- **Write careful tests.** For any non-trivial logic (tensor reshapes,
  index/block extraction, recursion updates, numerical routines),
  include a focused sanity check before declaring done. Prefer small
  closed-form cases where the expected output is known, plus an
  invariant check (e.g. symmetry, known identity, shape/dtype). When
  touching shared code paths, run whatever existing tests exist; if
  there are none, write one.
- **Verify before asserting.** When a result looks surprisingly good or
  bad, cross-check it (recompute with a different path, compare against
  an untrained / random-init baseline, inspect a hand-picked entry).
  Don't report a result as correct until it survives at least one
  independent check.
- **Prefer narrow changes.** Don't sweep surrounding code into the edit
  unless asked. If you notice a second issue, flag it, don't fix it
  silently.
- **Say what you changed, and what you didn't.** End-of-turn summaries
  should name the specific files/functions modified and call out
  anything you chose not to touch despite being tempted.

## Scientific Skills

Before starting domain-specific work, call `find_helpful_skills` to check for relevant
guidance. The following installed skills are especially relevant to this project — look for
opportunities to apply them:
- **fluidsim** — fluid dynamics simulation
- **scikit-learn** — ML modeling, preprocessing, evaluation
- **optimize-for-gpu** — GPU/CUDA optimization for training and inference
- **aeon** — time series classification, regression, and forecasting

## Python environment (uv)

This project's Python environment is managed by **uv**. All Python commands must go
through uv so they run in the locked project environment:

- Run scripts: `uv run python path/to/script.py` (not bare `python`)
- Run CLIs installed as project deps: `uv run <tool>` (e.g. `uv run pytest`, `uv run jupytext`)
- Inspect installed packages: `uv pip list`
- Install/remove project deps: `uv add <pkg>` / `uv remove <pkg>`

**Warning about `uv add` on this project**: `pyproject.toml` defines conflicting
`cu128` / `cu118` dependency groups for PyTorch, and `uv add` does a full re-resolve
across all groups. That can trigger large unnecessary torch wheel downloads
(~1 GB+) even for unrelated packages. For CLI-only dev tools you don't need to
import from project code, prefer `uv tool install <tool>` or `uv pip install <tool>`
to avoid the re-resolve. Never bypass uv by calling a bare `python` / `pip`.

## Engaging cluster (SLURM job submission)

To submit JacobianODE training jobs on the Engaging cluster, use:

```
engaging-submit experiment=wmtask_identity_encoder_verification
```

This script (located at `~/bin/engaging-submit`):
1. Commits and pushes any local changes on the current branch
2. Pulls latest on engaging from the repo at `/home/eisenaj/code/JacobianODE`
3. Runs `jsweep` (Hydra multirun + submitit) via SSH
4. Polls `squeue` until the SLURM job appears
5. Kills the hanging Hydra process and exits cleanly

Pass any Hydra overrides as arguments. To check job status: `ssh engaging 'squeue -u eisenaj'`

**Important**: The SSH hook exemption in `.claude/hooks/uv-cu118-guard.sh` allows
`ssh` commands containing `uv run` — this is intentional since the guard is for the
local Pascal GPU machine only.

### Sweep automation pipeline (4 stages)

1. **`engaging-submit`** (local → engaging): validates experiment YAML has a
   `metadata:` block, resolves the sweep grid (from `sweep_grid:` YAML field
   and/or CLI), writes `expected.json` to
   `/orcd/data/ekmiller/001/eisenaj/JacobianODE/sweeps/active/<group>.expected.json`,
   then submits via `jsweep`.
2. **Monitor cron on engaging** (`JacobianODE/jacobians/tuning/monitor.py`):
   polls wandb + SLURM, writes `<group>.state.json`, emits `<group>.done.json`
   sentinel into `sweeps/done/` when all runs terminal (and min elapsed time
   passed). Checks `failed/` to prevent ping-pong with auto-analyze.
3. **`engaging-analyze <group>`** (local, manual or via auto-analyze): SSHes
   to engaging, submits an sbatch GPU job running
   `JacobianODE.jacobians.tuning.analyze_sweep`, rsyncs results to
   `~/Documents/jacobian-analyses/<group>/`, renders report.
4. **`engaging-auto-analyze`** (systemd user timer on endeavour): every 15 min
   checks `sweeps/done/` for new sentinels, runs `engaging-analyze`, moves
   sentinel to `processed/` on success or `failed/` on failure.

### Required experiment YAML fields

Every file in `JacobianODE/jacobians/conf/experiment/*.yaml` must have a
top-level `metadata:` block with `description`, `hypothesis`, and
`success_criteria` (list). `prepare_sweep.py` enforces this before submission.

To hardcode a sweep grid in the experiment (so you can launch with just
`engaging-submit experiment=<name>` and no CLI overrides), add:

```yaml
sweep_grid:
  training.lightning.loop_closure_weight: "0,1e-6,1e-4,1e-2,1"
  training.lightning.obs_noise_scale:     "0,0.01,0.05"
```

### Loss functions (`training.lightning.loss_func`)

- `mse` — plain MSE (default).
- `normalized_mse` — per-batch variance normalization; divides MSE by
  `Var(target)` computed in the current batch. Equivalent to `1 - R²` per step.
- `generalized_normalized_mse` — divides MSE by a **precomputed** scalar
  `det(Cov(target))^(1/D)` (generalized variance). Stable across batches and
  across the trajectory/reconstruction/loop-closure terms; preferred when you
  want consistent loss scale regardless of where in the latent you are.

The `latent_criterion` (for latent prediction loss) can use a separate denom
via `gen_variance_mode=adaptive_latent`, which recomputes `det(Cov(z_dyn))^(1/D)`
at the start of each epoch from 50 batches.

## Jupyter Notebooks

Never use the default notebook read/edit/grep tools. Never try to read the raw `.ipynb`
JSON directly. For any non-trivial edit, prefer the **jupytext sidecar** workflow over
the notebook-mcp edit tools, which are slow (multiple minutes per edit on this cluster
filesystem).

### Preferred: jupytext sidecar for editing

Notebooks you expect to edit programmatically should be paired with a `.py:percent`
sidecar via jupytext. This lets Claude use the fast built-in `Edit` / `Read` / `Grep`
tools on the `.py` file instead of paying the notebook-mcp round-trip cost per edit.

One-time pairing (run once per notebook you want to edit this way):
```
uv run jupytext --set-formats ipynb,py:percent "path/to/notebook.ipynb"
```

Editing loop:
1. Read / edit the `.py` sidecar with the normal `Read` / `Edit` / `Grep` tools.
2. Sync back to the `.ipynb` with `uv run jupytext --sync "path/to/notebook.py"`.
3. The `.ipynb` picks up the code changes; outputs stay in the `.ipynb`.

The user's JupyterLab will also auto-sync if both files are open.

### notebook-mcp (structural editing — use when no sidecar)
Use for reading, editing, and organizing notebook structure when a jupytext sidecar
is not set up, or for structural operations (add/delete/move cells, metadata):
- Use `notebook_get_outline` first to understand the structure before editing
- Use `notebook_search` to locate specific cells by keyword
- Use `notebook_edit_cell` for targeted edits
- Use `notebook_add_cell`, `notebook_delete_cell`, `notebook_move_cell` for structure changes

### jupyter-server MCP (execution & debugging)
Use for running code and inspecting results. Requires a running Jupyter server 
(user starts one with `jlab` in terminal, which runs on localhost:8888):
- Use `execute_notebook_code` to run cells and get outputs
- Use `setup_notebook` to connect to a notebook on the server
- Use `query_notebook` to inspect notebook state
- Best for: iterating on code, debugging, checking outputs, running analysis