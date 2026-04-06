import marimo

__generated_with = "0.22.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Latent JacobianODE — Experiment Launcher

    Loads an experiment config, checks W&B for already-completed runs, and either
    submits remaining jobs via SLURM **or** runs a single combo locally with a
    live progress bar.

    **Workflow:**
    1. Pick an experiment and set `RUN_LOCALLY` (Section 1)
    2. Review the resolved W&B project / group and sweep grid (Section 2)
    3. Check which combinations already have finished W&B runs (Section 3)
    4. Launch remaining jobs via SLURM — skipped when `RUN_LOCALLY=True` (Section 4)
    5. Run one combo locally with live progress — skipped when `RUN_LOCALLY=False` (Section 5)
    """)
    return


@app.cell
def _():
    import itertools
    import subprocess
    import sys
    from pathlib import Path

    import torch
    import wandb
    from omegaconf import OmegaConf

    import JacobianODE
    from JacobianODE.jacobians import (
        create_dataloaders,
        initialize_config,
        load_config,
        make_trajectories,
        postprocess_data,
        seed_everything,
    )
    from JacobianODE.jacobians.training import make_model, train_model

    torch.set_float32_matmul_precision("high")

    _pkg_dir = Path(JacobianODE.__file__).parent
    EXPERIMENT_DIR = _pkg_dir / "jacobians" / "conf" / "experiment"
    return (
        EXPERIMENT_DIR,
        OmegaConf,
        create_dataloaders,
        initialize_config,
        itertools,
        load_config,
        make_model,
        make_trajectories,
        postprocess_data,
        seed_everything,
        subprocess,
        sys,
        train_model,
        wandb,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Pick an Experiment
    """)
    return


@app.cell
def _(EXPERIMENT_DIR, mo):
    _names = sorted(p.stem for p in EXPERIMENT_DIR.glob("*.yaml"))

    experiment_picker = mo.ui.dropdown(
        options=_names,
        value=_names[1] if _names else None,
        label="Experiment",
    )

    # OFF → submit remaining combos via SLURM (Section 4)
    # ON  → run one combo locally with progress bar (Section 5)
    run_locally_toggle = mo.ui.switch(label="Run locally", value=False)

    mo.vstack([experiment_picker, run_locally_toggle])
    return experiment_picker, run_locally_toggle


@app.cell
def _(run_locally_toggle):
    RUN_LOCALLY = run_locally_toggle.value
    return (RUN_LOCALLY,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Config & Sweep Grid
    """)
    return


@app.cell
def _(
    EXPERIMENT_DIR,
    OmegaConf,
    experiment_picker,
    initialize_config,
    itertools,
    load_config,
):
    EXPERIMENT_NAME = experiment_picker.value

    # ── Parse sweep grid from raw YAML ───────────────────────────────
    # hydra.sweeper.params uses Hydra BasicSweeper CSV format: "a,b,c"
    _raw = OmegaConf.load(EXPERIMENT_DIR / f"{EXPERIMENT_NAME}.yaml")
    _sweeper_params = OmegaConf.select(_raw, "hydra.sweeper.params") or {}

    def _parse_val(s):
        s = s.strip()
        if s == "null":
            return None
        if s in ("true", "True"):
            return True
        if s in ("false", "False"):
            return False
        try:
            return int(s)
        except ValueError:
            pass
        try:
            return float(s)
        except ValueError:
            return s

    SWEEP_PARAMS = {
        k: [_parse_val(v) for v in str(csv).split(",")]
        for k, csv in _sweeper_params.items()
    }

    _sweep_keys = list(SWEEP_PARAMS.keys())
    all_sweep_combos = (
        [
            dict(zip(_sweep_keys, vals))
            for vals in itertools.product(*SWEEP_PARAMS.values())
        ]
        if SWEEP_PARAMS
        else [{}]
    )
    n_sweep_combos = len(all_sweep_combos)

    # ── Resolved config (OmegaConf interpolations filled in) ─────────
    cfg = load_config(overrides=[f"experiment={EXPERIMENT_NAME}"])
    cfg = initialize_config(cfg)

    WANDB_ENTITY = cfg.wandb_entity
    WANDB_PROJECT = cfg.wandb_project
    WANDB_GROUP = cfg.wandb_group
    WANDB_PROJECT_PATH = f"{WANDB_ENTITY}/{WANDB_PROJECT}"

    print(f"Experiment    : {EXPERIMENT_NAME}")
    print(f"W&B entity    : {WANDB_ENTITY}")
    print(f"W&B project   : {WANDB_PROJECT}")
    print(f"W&B group     : {WANDB_GROUP}")
    print(f"\nSweep grid    : {n_sweep_combos} combinations")
    if SWEEP_PARAMS:
        for _k, _vals in SWEEP_PARAMS.items():
            print(f"  {_k}: {_vals}")
    else:
        print("  (no sweep grid — single run)")
    return (
        EXPERIMENT_NAME,
        WANDB_GROUP,
        WANDB_PROJECT_PATH,
        all_sweep_combos,
        n_sweep_combos,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. W&B Status Check
    """)
    return


@app.cell
def _(WANDB_GROUP, WANDB_PROJECT_PATH, all_sweep_combos, wandb):
    def _get_param(run, key):
        val = run.config
        for part in key.split("."):
            if not isinstance(val, dict) or part not in val:
                return None
            val = val[part]
        return val

    def _combo_matches(combo, run):
        for key, target in combo.items():
            rv = _get_param(run, key)
            if rv is None:
                return False
            if isinstance(target, bool):
                if bool(rv) != target:
                    return False
            elif isinstance(target, (int, float)):
                if abs(float(rv) - float(target)) > 1e-10:
                    return False
            elif str(rv) != str(target):
                return False
        return True

    _api = wandb.Api()
    try:
        _filters = {"group": WANDB_GROUP} if WANDB_GROUP else None
        _runs = _api.runs(WANDB_PROJECT_PATH, filters=_filters)
        print(
            f"Found {len(_runs)} run(s) in {WANDB_PROJECT_PATH}"
            + (f"  [group={WANDB_GROUP}]" if WANDB_GROUP else "")
        )
    except Exception as _e:
        print(f"Could not query W&B (project may not exist yet): {_e}")
        _runs = []

    _finished = [r for r in _runs if r.state == "finished"]

    already_done = []
    remaining_combos = []
    for _combo in all_sweep_combos:
        _match = next((r for r in _finished if _combo_matches(_combo, r)), None)
        if _match is not None:
            already_done.append(_combo)
            print(f"  done : {_combo}  (run={_match.id})")
        else:
            remaining_combos.append(_combo)

    print(
        f"\n{len(already_done)} / {len(all_sweep_combos)} combinations already finished"
    )
    if remaining_combos:
        print(f"{len(remaining_combos)} remaining")
    else:
        print("All done — nothing to launch.")
    return (remaining_combos,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Launch Sweep (SLURM)
    """)
    return


@app.cell
def _(EXPERIMENT_NAME, RUN_LOCALLY, n_sweep_combos, remaining_combos, sys):
    _ENTRY = f"{sys.executable} -m JacobianODE.jacobians.run_jacobians"

    def _fmt(v):
        if v is None:
            return "null"
        if isinstance(v, bool):
            return str(v).lower()
        return str(v)

    if RUN_LOCALLY:
        sweep_cmds = []
        print("RUN_LOCALLY=True — skipping SLURM sweep (see Section 5).")
    elif not remaining_combos:
        sweep_cmds = []
        print("Nothing to launch — all combinations already finished.")
    elif len(remaining_combos) == n_sweep_combos:
        # Full grid: let the experiment YAML's hydra.sweeper.params drive the sweep.
        sweep_cmds = [f"{_ENTRY} --multirun experiment={EXPERIMENT_NAME}"]
        print(f"Full grid ({n_sweep_combos} combos): 1 command using experiment sweep grid")
        print(f"\n  {sweep_cmds[0]}")
    else:
        # Subset: one --multirun per remaining combo (each submits one SLURM job).
        sweep_cmds = [
            f"{_ENTRY} --multirun experiment={EXPERIMENT_NAME} "
            + " ".join(f"{k}={_fmt(v)}" for k, v in _combo.items())
            for _combo in remaining_combos
        ]
        print(
            f"Subset ({len(remaining_combos)} / {n_sweep_combos} combos): "
            f"{len(sweep_cmds)} command(s)"
        )
        for _cmd in sweep_cmds[:5]:
            print(f"\n  {_cmd}")
        if len(sweep_cmds) > 5:
            print(f"  ... ({len(sweep_cmds) - 5} more)")
    return (sweep_cmds,)


@app.cell
def _(RUN_LOCALLY, subprocess, sweep_cmds):
    # ── Set LAUNCH = True to submit ──────────────────────────────────
    LAUNCH = True

    if RUN_LOCALLY:
        print("RUN_LOCALLY=True — use Section 5 to train locally.")
    elif LAUNCH:
        if sweep_cmds:
            print(f"Submitting {len(sweep_cmds)} command(s)...")
            _procs = [subprocess.Popen(cmd, shell=True) for cmd in sweep_cmds]
            for _i, _p in enumerate(_procs):
                _p.wait()
                if _p.returncode != 0:
                    print(f"Command {_i + 1} exited with code {_p.returncode}")
            print("Done.")
        else:
            print("Nothing to launch.")
    else:
        print("LAUNCH=False. Set to True and re-run this cell to submit jobs.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Local Run

    Pick one hyperparameter combination and run it in-process with a live progress
    bar. Requires `RUN_LOCALLY = True` (Section 1).
    """)
    return


@app.cell
def _(all_sweep_combos, mo, remaining_combos):
    def _label(c):
        if not c:
            return "(default — no sweep params)"
        return "  ".join(f"{k.split('.')[-1]}={v}" for k, v in c.items())

    combo_options = {_label(c): c for c in all_sweep_combos}
    _default = remaining_combos[0] if remaining_combos else all_sweep_combos[0]

    local_combo_picker = mo.ui.dropdown(
        options=list(combo_options.keys()),
        value=_label(_default),
        label="Combo to run locally",
    )
    local_combo_picker
    return combo_options, local_combo_picker


@app.cell
def _(
    EXPERIMENT_NAME,
    combo_options,
    initialize_config,
    load_config,
    local_combo_picker,
):
    _selected = combo_options[local_combo_picker.value]

    def _fmt(v):
        if v is None:
            return "null"
        if isinstance(v, bool):
            return str(v).lower()
        return str(v)

    cfg_local = load_config(
        overrides=[f"experiment={EXPERIMENT_NAME}"]
        + [f"{k}={_fmt(v)}" for k, v in _selected.items()]
    )
    cfg_local = initialize_config(cfg_local)

    # ── Build a compact, informative run name ────────────────────────
    def _short_val(v):
        """Format a hyperparameter value compactly."""
        if v is None or v == 0:
            return "0"
        if isinstance(v, float):
            # e.g. 0.001 → "1e-3", 0.1 → "1e-1", 10.0 → "10"
            s = f"{v:.2e}"                         # "1.00e-03"
            coeff, exp = s.split("e")
            coeff = coeff.rstrip("0").rstrip(".")  # "1"
            exp_i = int(exp)
            return f"{coeff}e{exp_i:+d}" if coeff != "1" else f"1e{exp_i:+d}"
        return str(v)

    # Encoder: coupling_type field if present, else last part of _target_
    _enc_tgt = cfg_local.model.encoder.get("_target_", "")
    _coupling_type = cfg_local.model.encoder.get("coupling_type", "")
    _enc = _coupling_type or _enc_tgt.split(".")[-1].replace("Encoder", "").lower()

    # Key dimensions
    _t  = cfg_local.model.n_target_dims
    _nd = cfg_local.data.train_test_params.delay_embedding_params.n_delays

    # Swept combo values: use last dotted component as key abbreviation
    _combo_parts = [
        f"{k.split('.')[-1]}={_short_val(v)}"
        for k, v in _selected.items()
    ]

    _name_parts = [_enc, f"T{_t}", f"N{_nd}"] + _combo_parts
    local_run_name = "__".join(_name_parts)

    print(f"Selected combo : {_selected or '(default)'}")
    for _k, _v in _selected.items():
        print(f"  {_k} = {_fmt(_v)}")
    print(f"\nRun name : {local_run_name}")
    return cfg_local, local_run_name


@app.cell
def _(cfg_local, make_trajectories, seed_everything):
    seed_everything(cfg_local.data.flow.random_state)
    _eq_local, sol_local, dt_local = make_trajectories(cfg_local, verbose=True)
    print(f"\nTrajectory shape : {sol_local['values'].shape}")
    return dt_local, sol_local


@app.cell
def _(cfg_local, postprocess_data, sol_local):
    result_local = postprocess_data(cfg_local, sol_local["values"])
    print(f"Postprocessed shape : {result_local.values.shape}")
    print(f"mu={result_local.mu:.4f}  sigma={result_local.sigma:.4f}  noise_scale={result_local.noise_scale_factor:.4f}")
    return (result_local,)


@app.cell
def _(cfg_local, create_dataloaders, result_local):
    train_dl_local, val_dl_local, _test_dl_local, _trajs_local = create_dataloaders(
        cfg_local, result_local.values, verbose=True
    )
    return train_dl_local, val_dl_local


@app.cell
def _(cfg_local, dt_local, make_model, result_local):
    lit_model_local = make_model(
        cfg_local,
        dt=dt_local,
        mu=result_local.mu,
        sigma=result_local.sigma,
        noise_scale_factor=result_local.noise_scale_factor,
        verbose=True,
    )
    _total = sum(p.numel() for p in lit_model_local.parameters())
    print(f"\nTotal parameters : {_total:,}")
    return (lit_model_local,)


@app.cell
def _(
    RUN_LOCALLY,
    cfg_local,
    lit_model_local,
    local_run_name,
    mo,
    train_dl_local,
    train_model,
    val_dl_local,
):
    from JacobianODE.jacobians.training.marimo_utils import EpochProgress

    if RUN_LOCALLY:
        _n_train  = cfg_local.training.trainer_params.limit_train_batches
        _n_val    = cfg_local.training.trainer_params.limit_val_batches
        _n_epochs = cfg_local.training.trainer_params.max_epochs

        train_model(
            cfg_local,
            lit_model_local,
            train_dl_local,
            val_dl_local,
            name=local_run_name,
            extra_callbacks=[EpochProgress(mo, _n_train, _n_val, _n_epochs)],
        )
    else:
        print("RUN_LOCALLY=False. Set to True in Section 1 to run locally.")
    return


if __name__ == "__main__":
    app.run()
