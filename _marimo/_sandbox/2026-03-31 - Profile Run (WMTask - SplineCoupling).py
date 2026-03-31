import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Profile Run — WMTask × SplineCoupling

    Loads the `wmtask_spline_coupling_sweep` experiment config and runs a single
    hyperparameter combination with **W&B step profiling** enabled.

    Fixed values (from the experiment YAML):
    - `loop_closure_weight = 0.001`
    - `kl_dyn_weight = 0.001`

    Per-component timing is logged to W&B as `profile/train/*` and `profile/val/*`
    metrics every `PROFILE_STEPS` steps.
    """)
    return


@app.cell
def _():
    import torch
    from omegaconf import OmegaConf

    from JacobianODE.jacobians import (
        load_config,
        initialize_config,
        seed_everything,
        make_trajectories,
        postprocess_data,
        create_dataloaders,
        make_model,
        train_model,
    )

    torch.set_float32_matmul_precision("high")
    return (
        OmegaConf,
        create_dataloaders,
        initialize_config,
        load_config,
        make_model,
        make_trajectories,
        postprocess_data,
        seed_everything,
        train_model,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Config
    """)
    return


@app.cell
def _(initialize_config, load_config):
    # ----------------------------------------------------------------
    # Profiling settings
    # ----------------------------------------------------------------
    PROFILE_STEPS  = 1   # print timing table every N training steps
    N_EPOCHS       = 3    # short run — just enough to profile

    cfg = load_config(overrides=[
        "experiment=wmtask_spline_coupling_sweep",
        f"model.encoder_warmup_epochs=0",
        f"+training.lightning.profile_output=wandb",
        f"+training.lightning.profile_steps={PROFILE_STEPS}",
        f"training.trainer_params.max_epochs={N_EPOCHS}",
        "+training.trainer_params.enable_progress_bar=false",
    ])
    cfg = initialize_config(cfg)

    print(f"Encoder n_input : {cfg.model.encoder.n_input}")
    print(f"n_target_dims   : {cfg.model.n_target_dims}")
    print(f"MLP input_dim   : {cfg.model.params.input_dim}  →  output_dim: {cfg.model.params.output_dim}")
    print(f"profile_output  : {cfg.training.lightning.profile_output}")
    print(f"profile_steps   : {cfg.training.lightning.profile_steps}")
    return N_EPOCHS, PROFILE_STEPS, cfg


@app.cell(hide_code=True)
def _(OmegaConf, cfg, mo):
    def _yaml(node):
        return mo.plain_text(OmegaConf.to_yaml(node, resolve=True))

    mo.accordion({
        "data":     _yaml(cfg.data),
        "model":    _yaml(cfg.model),
        "training": _yaml(cfg.training),
    })
    return


@app.cell
def _(cfg):
    print(f"wandb_entity  : {cfg.wandb_entity}")
    print(f"wandb_project : {cfg.wandb_project}")
    print(f"wandb_group   : {cfg.wandb_group}")
    print(f"save_dir      : {cfg.training.logger.save_dir}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Data
    """)
    return


@app.cell
def _(cfg, make_trajectories, seed_everything):
    seed_everything(cfg.data.flow.random_state)
    _eq, sol, dt = make_trajectories(cfg, verbose=True)
    print(f"\nTrajectory shape : {sol['values'].shape}")
    print(f"dt               : {dt:.5f}")
    return dt, sol


@app.cell
def _(cfg, postprocess_data, sol):
    result = postprocess_data(cfg, sol["values"])
    print(f"Postprocessed shape : {result.values.shape}")
    print(f"mu={result.mu:.4f}  sigma={result.sigma:.4f}  noise_scale={result.noise_scale_factor:.4f}")
    return (result,)


@app.cell
def _(cfg, create_dataloaders, result):
    train_dl, val_dl, _test_dl, _trajs = create_dataloaders(cfg, result.values, verbose=True)
    return train_dl, val_dl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Model
    """)
    return


@app.cell
def _(cfg, dt, make_model, result):
    lit_model = make_model(
        cfg,
        dt=dt,
        mu=result.mu,
        sigma=result.sigma,
        noise_scale_factor=result.noise_scale_factor,
        verbose=True,
    )

    _enc_params = sum(p.numel() for p in lit_model.encoder.parameters())
    _jac_params = sum(p.numel() for p in lit_model.model.parameters())
    _total      = sum(p.numel() for p in lit_model.parameters())
    print(f"\nParameters")
    print(f"  Encoder       : {_enc_params:>12,}")
    print(f"  Jacobian MLP  : {_jac_params:>12,}")
    print(f"  Total         : {_total:>12,}")
    return (lit_model,)


@app.cell(hide_code=True)
def _(N_EPOCHS, PROFILE_STEPS, mo):
    mo.md(f"""
    ## 4. Profile Run

    Training for **{N_EPOCHS} epochs**. A timing breakdown is printed to stdout
    after every **{PROFILE_STEPS} training steps** and every validation step.
    """)
    return


@app.cell
def _(cfg, lit_model, mo, train_dl, train_model, val_dl):
    from JacobianODE.jacobians.training.marimo_utils import EpochProgress

    _n_train  = cfg.training.trainer_params.limit_train_batches
    _n_val    = cfg.training.trainer_params.limit_val_batches
    _n_epochs = cfg.training.trainer_params.max_epochs

    trainer = train_model(
        cfg,
        lit_model,
        train_dl,
        val_dl,
        name="profiling",
        extra_callbacks=[EpochProgress(mo, _n_train, _n_val, _n_epochs)],
    )
    return


if __name__ == "__main__":
    app.run()
