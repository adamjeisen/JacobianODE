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