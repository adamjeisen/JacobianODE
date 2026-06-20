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
- **DO NOT just silently pass through errors.** Don't use try-except or other strategies to silently pass through errors. Let them surface, and actually fix them, rather than catching them so that the code can continue.

## Scientific Skills

Before starting domain-specific work, call `find_helpful_skills` to check for relevant
guidance. The following installed skills are especially relevant to this project — look for
opportunities to apply them:
- **fluidsim** — fluid dynamics simulation
- **scikit-learn** — ML modeling, preprocessing, evaluation
- **optimize-for-gpu** — GPU/CUDA optimization for training and inference
- **aeon** — time series classification, regression, and forecasting

## Metrics, model selection & scientific conventions

These are project-wide invariants — getting them wrong silently corrupts
conclusions, so treat them as hard rules, not preferences.

- **MASE is teacher-forced; trajectory val loss is free-running.** One-step
  MASE and the MASE diagnostics are computed teacher-forced (αTF=1). The
  monitored `trajectory val_loss` is the autoregressive free-running rollout
  loss. They measure different things — never describe one as the other, and
  when both look inconsistent that asymmetry is usually the explanation.
- **Chosen run = `best_traj_loss` subject to the selection criteria.** The
  canonical picker is min trajectory val_loss among runs that pass C1
  (one-step MASE < 1 — prefer `decoder_corrected_one_step_mase` when present,
  else raw), C2 (loop-closure), C3 (fast-eigenvalue fraction). Report side
  and any paper-figure picker must use this same rule so figures match the
  published reports. Derive the run↔cell mapping from an authoritative key
  (run config / swept-grid hash), never from ordering.
- **Never assume access to the true spectrum; baseline empirically.** The
  whole point is learning dynamics from observations. Do not assume a known
  Lyapunov spectrum, and **do not overlay canonical literature values** on
  spectrum plots — compare against the empirical/estimated spectrum only.
- **`obs_noise_scale` ≠ `obs_noise`.** `obs_noise_scale` is the training-time
  injected observation-noise knob; `obs_noise` is the data-generation noise
  level. They are distinct — never conflate them in code, labels, or when
  splitting/sorting results (split by `obs_noise_scale`).

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
j-submit lorenz_full_additive_mse_p30
j-submit lorenz_full_additive_mse_p30 hydra.launcher.partition=ou_bcs_normal
```

`j-submit` (`~/bin/j-submit`) writes a YAML "instruction" file under
`~/Documents/jacobian-analyses/instructions/pending/` and pushes it to the
`jacobian-reports` GitHub repo. The engaging-side cron `engaging-controller`
picks it up within ~5 minutes and runs `jsweep` locally on engaging — no SSH
between endeavour and engaging is involved in routine operation.

### Sweep automation pipeline (git-based, 3 stages)

1. **`j-submit`** (endeavour → github): writes
   `instructions/pending/<ts>-<exp>.yaml` to the local `jacobian-analyses`
   clone of the `jacobian-reports` repo, commits, pushes. Format documented
   in `~/Documents/jacobian-analyses/instructions/README.md`.
2. **`engaging-controller`** (cron on engaging, every 5 min): pulls
   `jacobian-reports` AND `JacobianODE` (so code changes auto-deploy);
   processes each pending instruction (validates via `prepare_sweep`, runs
   `jsweep` locally, captures SLURM array id, moves the file to
   `instructions/claimed/` annotated with the array id); runs a monitor
   cycle (`JacobianODE.jacobians.tuning.monitor`); dispatches analysis
   sbatches for any new sentinels in `/orcd/.../sweeps/done/`; reaps
   completed analyses by rendering the report and copying the analysis dir
   into the repo at `<wandb_project>/<wandb_group>/`; updates
   `status/overview.json`; pushes everything in one commit. The controller
   script lives at `JacobianODE/bin/engaging-controller` so updates ride
   along with normal `git pull`.
3. **`jacobian-discuss`** (systemd user timer on endeavour, every 15 min):
   pulls `jacobian-reports`; for each report missing a fresh `discussion.md`,
   invokes `claude --print` to generate one against the local analysis dir;
   re-renders `report.md` / `report.html` to embed the discussion; commits
   and pushes back. This is GitHub-only, no SSH to engaging required.

To check status: `cd ~/Documents/jacobian-analyses && git pull && cat
status/overview.json`. New reports show up under `<project>/<group>/`.

The previous SSH-based system (`engaging-submit`, `engaging-analyze`,
`engaging-auto-analyze`, `engaging-submit-many` + `engaging-auto-analyze`
systemd timer) is archived to `~/bin/_old/` and disabled. Engaging SSH
master (`engaging-ssh-master.service`) remains installed but disabled by
default — start manually only when interactive debugging on engaging is
needed (then it'll multiplex subsequent `ssh engaging` calls without
re-Duo'ing).

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

### One-off diagnostic figures (shareable via web)

The `jacobian-reports` repo is published at `adamjeisen.com/jacobian-reports/`
(GitHub Pages). To share a one-off diagnostic figure — e.g. for remote viewing
or to link into a conversation — drop the image into
`~/Documents/jacobian-analyses/diagnostics/`, commit, and push:

```
cp figure.png ~/Documents/jacobian-analyses/diagnostics/<name>_<run_id>.png
cd ~/Documents/jacobian-analyses
git add diagnostics/<name>_<run_id>.png
git commit -m "diagnostics: <short description> — run <run_id>"
git push
```

The file is then accessible at
`https://adamjeisen.com/jacobian-reports/diagnostics/<name>_<run_id>.png`.

Pushing: the engaging-controller / jacobian-discuss cron jobs push to this repo
frequently, so a `git pull --rebase` is usually needed before `git push`. If SSH
push fails (no agent), use `gh`'s HTTPS credentials:
`git -c credential.helper='!gh auth git-credential' push https://github.com/adamjeisen/jacobian-reports.git HEAD:main`.

### Operational constraints

- **mit_normal_gpu concurrency cap = 4.** The QOS budget
  (`mit_amf_advanced_gpu`) allows at most 4 concurrent tasks. Seeing >4
  running on mit_normal_gpu (or a flood of unexpected wandb runs) means
  something went wrong — investigate, don't accept it.
- **HDF5 file locking must be disabled for parallel `.mat` reads.** Many
  sweep cells open the same session `.mat` at startup; the default HDF5
  lock fails or serialises them on the networked FS. Set
  `HDF5_USE_FILE_LOCKING=FALSE` for any process (sweep cell, post-hoc
  script) that opens session files.
- **Sweep-1 ≠ two-stage.** A plain scout sweep ("sweep 1", e.g. an
  n_delays scout) runs every cell to the ES criterion or walltime. It is
  NOT the Stage-A-cull-at-20-epochs / Stage-B-continue protocol. Don't
  impose two-stage semantics (epoch cap, cull) on a plain sweep, and
  don't run a two-stage sweep without the cull. See memory:
  two-stage `full_max_epochs`.

### MindControl sweeps + manual report dispatch

`mc-submit` is the MindControl analogue of `j-submit` (same git
message-bus + engaging-controller). The controller auto-dispatches a
report only when every training array task is `COMPLETED` or `TIMEOUT`.
A single `CANCELLED`/`FAILED` cell trips the gate and the report is
**skipped** (`status/overview.json` → `report: skipped, reason:
training array not all-COMPLETED`) — this is intentional, with the
remaining cells still usable.

To dispatch a report on the partial-but-usable data, run on engaging:

```
ssh engaging "cd ~/code/MindControl && \
  /home/eisenaj/.local/bin/uv run --no-sync python -m \
  mindcontrol.cli.mc_report submit --sweep-dir <wandb_group>"
```

Because this is outside the controller flow it will **not** auto-publish.
After it finishes, copy `<sweep>/report/{report.md,report.html,figures,
cells,chosen_extras.pkl}` into
`~/Documents/jacobian-analyses/<wandb_project>/<wandb_group>/`, then
commit + push (rebase-aware; use the `gh` HTTPS credential helper if SSH
push has no agent).

## Report & figure conventions

- **The paper-figure pipeline lives in `MindControl/figures/`** (moved
  out of JacobianODE — it produces MindControl-project paper figures).
  Run it from the MindControl repo (`cd MindControl && uv run --no-sync
  python figures/...`). It still imports `JacobianODE` (for `load_run`
  etc.) via MindControl's installed JacobianODE dependency — same
  coupling the rest of MindControl has, so keep that dep current.
  Output is *not* under `JacobianODE/jacobians/`.
- **Every per-run / per-cell plot must carry the run_id and the cell's
  grid parameters** in the title/legend so a plot is self-identifying.
- **No canonical-literature overlays** on Lyapunov / spectrum figures —
  empirical/estimated values only (see scientific conventions above).
- **Don't substitute a predicted-vs-true scatter** when the request is to
  see the actual values on the y-axis; plot the values directly.
- **When extending report generation, mirror the existing
  `run_analytics` structure exactly.** Use an existing non-neural-data
  group's report as the template rather than inventing layout. Neural-data
  groups have no single ground-truth group: render chosen-run Lyapunov +
  poor-run Lyapunov + gramians **split per condition** (awake / maintenance).
- **Cache expensive analysis intermediates** (per-condition spectra,
  per-pair gramians, trajectory predictions, encoded latents) the first
  time and reuse on plot iterations — persist the full object, not a
  summary, so later questions don't force a recompute.

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
