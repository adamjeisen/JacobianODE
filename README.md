# JacobianODE

A Python package for learning Jacobian-based ODE models of dynamical systems from time series data. The model learns local linear approximations (Jacobians) at each point in state space, enabling trajectory prediction, Lyapunov exponent estimation, and dynamical systems analysis.

## Installation

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
sh env_create_uv.sh
source .venv/bin/activate
```

This creates a virtual environment, installs the package in editable mode, and registers a `JacobianODE` Jupyter kernel.

## Quick Start

### Train on a built-in dynamical system

```python
from JacobianODE.jacobians import load_config, initialize_config, seed_everything
from JacobianODE.jacobians.run_jacobians import train_jacobians

cfg = load_config(overrides=[
    "data=dysts",
    "data.flow._target_=JacobianODE.dysts_sim.flows.Lorenz",
    "data.postprocessing.obs_noise=0.01",
    "training.lightning.loop_closure_weight=0.01",
])
cfg = initialize_config(cfg)
seed_everything(cfg.data.flow.random_state)
train_jacobians(cfg)
```

### Train on custom data

Data must be a NumPy array with shape `(n_trials, n_timepoints, n_dimensions)`.

```python
import numpy as np
from JacobianODE.jacobians import (
    load_config, initialize_config, seed_everything,
    make_trajectories, postprocess_data, create_dataloaders,
    make_model, train_model, setup_wandb,
)

my_data = np.load("my_data.npy")  # shape: (trials, time, dims)
dt = 0.01

cfg = load_config(overrides=["data=custom", f"data.flow.dim={my_data.shape[-1]}"])
cfg = initialize_config(cfg, data_dim=my_data.shape[-1])
seed_everything(42)

eq, sol, dt = make_trajectories(cfg, data=my_data, dt=dt)
values = postprocess_data(cfg, sol["values"])
train_dl, val_dl, test_dl, trajs = create_dataloaders(cfg, values)

project = "MyProject__JacobianODE"
name, wandb_logger = setup_wandb(cfg, project)
lit_model = make_model(cfg, dt, eq=None, project=project, mu=values.mean(), sigma=values.std())
train_model(cfg, lit_model, train_dl, val_dl, name, project)
```

You can also use `TimeSeriesData` to bundle your data with metadata for reproducibility:

```python
from JacobianODE.jacobians import TimeSeriesData

ts = TimeSeriesData(values=my_data, dt=0.01, metadata={"source": "experiment_1"})
ts.save("my_data.npz")

# Later:
ts = TimeSeriesData.load("my_data.npz")
```

### Command-line interface

Train using Hydra from the command line:

```bash
# Built-in system
python -m JacobianODE.jacobians.run_jacobians \
    data=dysts \
    data.flow._target_=JacobianODE.dysts_sim.flows.Lorenz

# Custom data from file
python -m JacobianODE.jacobians.run_jacobians \
    data=custom \
    data.flow.dim=50 \
    data.dataset_loader._target_=JacobianODE.jacobians.custom_data.load_timeseries_data \
    +data.dataset_loader.file_path=/path/to/data.npz
```

## Hyperparameter Tuning

The key hyperparameter is `loop_closure_weight` (lambda), which balances data fitting against physical consistency. The tuning module implements a physics-informed model selection procedure:

```bash
# Grid sweep over lambda values (submits SLURM jobs with --multirun)
python -m JacobianODE.jacobians.run_jacobians --multirun \
    data=custom \
    training.lightning.loop_closure_weight=0,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10 \
    slurm=default
```

After training, select the best model using three physics-informed criteria:
- **C1**: One-step prediction error must beat a persistence baseline
- **C2**: Loop closure loss must be below `sqrt(n_dims)`
- **C3**: Fraction of fast eigenvalues must be below threshold

```python
from JacobianODE.jacobians.tuning import select_best_model

result = select_best_model(all_diagnostics, persistence_baseline, n_dims)
print(f"Best lambda index: {result.best_index}")
```

See the `_jupyter/Hyperparameter Tuning (demo).ipynb` notebook for a complete walkthrough.

## Configuration

JacobianODE uses [Hydra](https://hydra.cc/) for configuration. YAML config files are in `JacobianODE/jacobians/conf/`:

| Config | File | Key settings |
|--------|------|-------------|
| Data | `data/dysts.yaml`, `data/custom.yaml` | Data source, noise, filtering, sequence length |
| Model | `model/mlp.yaml` | Hidden dims, activation, dropout |
| Training | `training/training.yaml` | Learning rate, teacher forcing, loop closure weight, early stopping |
| SLURM | `slurm/default.yaml` | GPUs, memory, partition, timeout |

Override any setting from the command line or via `load_config(overrides=[...])`.

## SLURM Support

When `slurm=default` is set, Hydra's submitit launcher automatically submits jobs to SLURM. Combine with `--multirun` for grid sweeps that run as parallel SLURM jobs.

## Loading Trained Models

Retrieve trained models from Weights & Biases:

```python
from JacobianODE.jacobians import load_run, load_checkpoint

run, cfg, eq, dt, values, train_dl, val_dl, test_dl, trajs, lit_model = load_run(
    "entity/project", run_id="abc123", save_dir="/path/to/checkpoints"
)
load_checkpoint(run, cfg, lit_model, save_dir="/path/to/checkpoints")
lit_model.eval()
```

## Use as a library

JacobianODE can be installed as a dependency in another project (e.g. via
`pip install JacobianODE @ git+https://github.com/adamjeisen/JacobianODE`).
Consumers bring their own dataset and experiment YAMLs and call
`JacobianODE.train` from a thin Hydra entry script:

```python
# consumer_repo/train.py
import hydra
from JacobianODE import train

@hydra.main(config_path="conf", config_name="train", version_base=None)
def main(cfg):
    train(cfg)

if __name__ == "__main__":
    main()
```

The consumer's `conf/train.yaml` extends JacobianODE-side templates via
`defaults:`. The shipped Hydra `SearchPathPlugin` makes JacobianODE's
config groups discoverable automatically — no `hydra.searchpath` setup
needed:

```yaml
# consumer_repo/conf/train.yaml
defaults:
  - /model: latent_additive_coupling     # JacobianODE-side template
  - /training: training                  # JacobianODE-side template
  - _self_

wandb:
  disabled: true   # opt out of W&B when running locally without a login
```

To override an entry that's already in the JacobianODE base config (e.g.
when extending it via `defaults: [/config, ...]`), use Hydra's `override`
keyword — `override /model: latent_additive_coupling`. When the consumer
config doesn't itself extend the JacobianODE base config, list the
JacobianODE-side defaults directly (without `override`).

The `wmtask` data path is gated behind an optional extra:
`pip install "JacobianODE[wmtask]"`. Consumers that don't use WMTask data
do not need SSH access to the wmtask repo. Sweep automation (`jsweep`,
engaging-controller, jacobian-reports) is intentionally personal infra and
not part of the library surface — consumers run sweeps with Submitit /
sbatch directly.

## Demo Notebooks

The `_jupyter/` directory contains three demo notebooks:

| Notebook | Description |
|----------|-------------|
| `Lorenz (Demo).ipynb` | End-to-end example on the Lorenz attractor |
| `Custom Data (demo).ipynb` | Loading custom time series data and training |
| `Hyperparameter Tuning (demo).ipynb` | Grid sweep and physics-informed model selection |

## License

MIT
