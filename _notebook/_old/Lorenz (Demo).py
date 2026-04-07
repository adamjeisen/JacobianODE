import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    # '%autoreload 2' command supported automatically in marimo
    import marimo as mo
    from JacobianODE.jacobians import load_config
    from JacobianODE.jacobians.run_jacobians import train_jacobians

    return load_config, mo, train_jacobians


@app.cell
def _():
    # FILL THIS IN FOR YOUR OWN USE
    save_dir = "/orcd/data/ekmiller/001/eisenaj/JacobianODE/lightning"
    wandb_entity = "chaotic-consciousness"
    return save_dir, wandb_entity


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup
    """)
    return


@app.cell
def _(load_config, save_dir, wandb_entity):
    # load the config
    overrides = [
        "data=dysts",
        "data.flow._target_=JacobianODE.dysts_sim.flows.Lorenz",
        "data.postprocessing.obs_noise=0.01",
        "training.lightning.loop_closure_weight=0.001",
        f"training.logger.save_dir={save_dir}",
        f"wandb_entity={wandb_entity}",
    ]
    cfg = load_config(overrides=overrides)
    return (cfg,)


@app.cell
def _(cfg, train_jacobians):
    train_jacobians(cfg)
    return


if __name__ == "__main__":
    app.run()
