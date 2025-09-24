import hydra
import logging
import torch
import numpy as np
from omegaconf import OmegaConf

from .jacobian_utils import in_ipython, initialize_config, make_trajectories, postprocess_data, normalize_data, create_dataloaders, setup_wandb, make_model, log_training_info, train_model

log = logging.getLogger('JacobianLogger')

@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def train_jacobians(cfg):
    print(in_ipython())
    # check if ipython
    # if in_ipython():
    #     log = None
    log.info("Running in ipython")

    # ----------------------------------------
    # INITIAL SETUP
    # ----------------------------------------
    # initial setup
    torch.set_float32_matmul_precision('high')
    num_gpus = torch.cuda.device_count()
    if log is not None:
        log.info(f"Number of available GPUs: {num_gpus}")
        log.info(OmegaConf.to_yaml(cfg))
    else:
        print("Number of available GPUs: ", torch.cuda.device_count())
        print(OmegaConf.to_yaml(cfg))

    # Initialize configuration
    cfg = initialize_config(cfg)

    # ----------------------------------------
    # GENERATE DATA
    # ----------------------------------------
    np.random.seed(cfg.data.flow.random_state)
    torch.random.manual_seed(cfg.data.flow.random_state)
    eq, sol, dt = make_trajectories(cfg)

    values_raw = sol['values']

    # select which solution to use for noise
    if cfg.data.data_type == 'wmtask' and cfg.data.trajectory_params.model_to_load != 'final':
            temp_cfg = cfg.copy()
            temp_cfg.data.trajectory_params.model_to_load = 'final'
            _, sol_noise, _ = make_trajectories(temp_cfg)
            raw_values_noise = sol_noise['values']
    else:
        raw_values_noise = None

    # ----------------------------------------
    # POSTPROCESS DATA
    # ----------------------------------------
    # Postprocess data
    values = postprocess_data(cfg, values_raw, raw_values_to_use_for_noise=raw_values_noise)
    if cfg.data.postprocessing.normalize:
        values, mu, sigma = normalize_data(values)
    else:
        mu = 0
        sigma = 1

    # ----------------------------------------
    # CREATE DATALOADERS
    # ----------------------------------------
    train_dataloader, val_dataloader, test_dataloader, trajs = create_dataloaders(cfg, values)

    # ----------------------------------------
    # SET UP WANDB
    # ----------------------------------------
    name, project = setup_wandb(cfg, trajs, raw_values_to_use_for_noise=raw_values_noise)

    # ----------------------------------------
    # MAKE MODEL
    # ----------------------------------------
    
    # Make model
    if 'NeuralODE' in cfg.model.params._target_:
        cfg.model.params.dt = float(dt)

    if cfg.training.lightning.use_base_deriv_pt:
        x0 = trajs['train_trajs'].sequence.mean(dim=(0, 1))
    else:
        x0 = None

    torch.random.manual_seed(cfg.data.flow.random_state + cfg.training.run_number)
    if cfg.data.train_test_params.delay_embedding_params.n_delays > 1:
        lit_model = make_model(cfg, dt, eq=None, project=project, mu=mu, sigma=sigma, verbose=True)
    else:
        lit_model = make_model(cfg, dt, eq=eq, project=project, x0=x0, mu=mu, sigma=sigma, verbose=True)

    # Log training information
    log_training_info(train_dataloader, trajs, lit_model, log=log)

    # ----------------------------------------
    # TRAIN MODEL
    # ----------------------------------------
    # train_model(cfg, lit_model, train_dataloader, val_dataloader, name, project)

if __name__ == "__main__":
    train_jacobians()
