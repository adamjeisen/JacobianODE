import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %load_ext autoreload
    # '%autoreload 2' command supported automatically in marimo
    return


@app.cell
def _():
    from JacobianODE.jacobians.run_analytics import run_analytics

    return (run_analytics,)


@app.cell
def _():
    WANDB_ENTITY = "JacobianODE"
    WANDB_PROJECT = "Lorenz_IND0_N100_D1_NormTrue_T3__spline_coupling__JacobianODE"
    run_id = "na2lkyj5"
    SAVE_DIR = "/orcd/data/ekmiller/001/eisenaj/JacobianODE/lightning/latent_jac_runs"
    return SAVE_DIR, WANDB_ENTITY, WANDB_PROJECT, run_id


@app.cell
def _(SAVE_DIR, WANDB_ENTITY, WANDB_PROJECT, run_analytics, run_id):
    run_analytics(
        wandb_entity=WANDB_ENTITY,
        wandb_project=WANDB_PROJECT,
        save_dir=SAVE_DIR,
        run_id=run_id,
        output=["show"],
        epoch=10,
    )
    return


@app.cell
def _(SAVE_DIR, WANDB_ENTITY, WANDB_PROJECT, run_analytics, run_id):
    run_analytics(
        wandb_entity=WANDB_ENTITY,
        wandb_project=WANDB_PROJECT,
        save_dir=SAVE_DIR,
        run_id=run_id,
        output=["show"],
        epoch=20,
    )
    return


@app.cell
def _(SAVE_DIR, WANDB_ENTITY, WANDB_PROJECT, run_analytics, run_id):
    run_analytics(
        wandb_entity=WANDB_ENTITY,
        wandb_project=WANDB_PROJECT,
        save_dir=SAVE_DIR,
        run_id=run_id,
        output=["show"],
        epoch=30,
    )
    return


@app.cell
def _(SAVE_DIR, WANDB_ENTITY, WANDB_PROJECT, run_analytics, run_id):
    run_analytics(
        wandb_entity=WANDB_ENTITY,
        wandb_project=WANDB_PROJECT,
        save_dir=SAVE_DIR,
        run_id=run_id,
        output=["show"],
        epoch=50,
    )
    return


@app.cell
def _(SAVE_DIR, WANDB_ENTITY, WANDB_PROJECT, run_analytics, run_id):
    run_analytics(
        wandb_entity=WANDB_ENTITY,
        wandb_project=WANDB_PROJECT,
        save_dir=SAVE_DIR,
        run_id=run_id,
        output=["show"],
        epoch=70,
    )
    return


@app.cell
def _():
    import datetime
    import subprocess
    import sys
    from pathlib import Path

    # Jupyter / this kernel use the repo's uv-managed environment (e.g. `.venv` here).
    # `sys.executable -m nbconvert` runs nbconvert with that same interpreter, not a stray
    # `jupyter` on PATH. Kernel cwd may still differ from the repo root, so we locate the
    # `.ipynb` explicitly below.

    NB_NAME = "2026-03-27 - Epoch Picking.ipynb"
    # Kernel cwd is often the notebook folder, not repo root — try sensible locations.
    for candidate in (
        Path.cwd() / NB_NAME,
        Path.cwd() / "_jupyter" / "_sandbox" / NB_NAME,
        Path.cwd().parent / "_sandbox" / NB_NAME,
    ):
        if candidate.is_file():
            notebook_path = candidate.resolve()
            break
    else:
        raise FileNotFoundError(
            f"Could not find {NB_NAME}; cwd={Path.cwd()!s}"
        )

    notebook_name = notebook_path.stem

    now = datetime.datetime.now()
    timestamp = now.strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = (notebook_path.parent / "../reports").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # nbconvert --output is the basename only; --output-dir sets the folder.
    output_base = f"{timestamp}_{notebook_name}"
    output_html = output_dir / f"{output_base}.html"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nbconvert",
            "--to",
            "html",
            "--output",
            output_base,
            "--output-dir",
            str(output_dir),
            str(notebook_path),
        ],
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"nbconvert failed with exit code {result.returncode}"
        )
    print(f"Notebook saved as HTML: {output_html}")
    return


if __name__ == "__main__":
    app.run()
