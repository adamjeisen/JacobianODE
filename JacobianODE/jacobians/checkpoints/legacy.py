"""Legacy run handling for JacobianODE.

This module handles loading runs created before the new configuration format
was introduced (before January 31, 2025 2pm EST).
"""

from __future__ import annotations

import inspect
import logging
import os
from typing import Any, Dict, Optional

import pandas as pd
import torch
from hydra.utils import instantiate

from ..data.dataloaders import create_dataloaders
from ..data.processing import postprocess_data
from ..data.trajectory import make_trajectories
from ..lightning_base import LitBase
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

# Mapping of run IDs to their optimal epoch numbers for legacy runs
# These runs have issues with their validation loss history and need manual overrides
LEGACY_RUN_EPOCH_OVERRIDES: Dict[str, int] = {
    "oz9rj2ml": 6,
    "8rg44zl8": 4,
    "cf8jaatp": 8,
}


def reverse_wandb_run(
    run: Any,
    return_data: bool = False,
    checkpoint: Optional[str] = None,
    save_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Reverse engineer a W&B run to recreate its configuration and model.

    Used primarily for loading legacy runs, this function reconstructs the training
    setup from a W&B run.

    Args:
        run: W&B run object to reverse engineer.
        return_data: Whether to generate trajectory data. Defaults to False.
        checkpoint: Specific checkpoint to load. Defaults to None.
        save_dir: Directory containing saved data. Defaults to None.

    Returns:
        Dictionary containing all components of the reconstructed run:
            - cfg: Configuration object
            - lit_model: PyTorch Lightning model
            - eq: Equation/model object
            - dt: Time step size
            - values: Processed trajectory values
            - values_orig: Original trajectory values
            - train_dataloader: Training data loader
            - val_dataloader: Validation data loader
            - test_dataloader: Test data loader
            - trajs: Dictionary containing trajectory information

    Example:
        >>> ret = reverse_wandb_run(run, return_data=True)
        >>> lit_model = ret['lit_model']
        >>> cfg = ret['cfg']
    """
    if save_dir is None:
        save_dir = run.config["save_dir"]

    checkpoint_dir = os.path.join(save_dir, run.project, run.id, "checkpoints")
    checkpoint_files = os.listdir(checkpoint_dir)

    if checkpoint is None:
        checkpoint = _find_best_checkpoint(run, checkpoint_files)

    # Reverse engineer configuration
    cfg_rev = reverse_wandb_config(run.config)
    cfg_rev.training.lightning.eq = cfg_rev.data.flow

    # Set seeds
    import numpy as np

    np.random.seed(cfg_rev.data.flow.random_state)
    torch.random.manual_seed(cfg_rev.data.flow.random_state)

    if return_data:
        eq, sol, dt = make_trajectories(cfg_rev, save_dir=save_dir)
        values_orig = sol["values"]
        values = postprocess_data(cfg_rev, sol["values"]).values
        train_dataloader, val_dataloader, test_dataloader, trajs = create_dataloaders(
            cfg_rev, values
        )
    else:
        cfg_temp = cfg_rev.copy()
        if cfg_rev.data.data_type == "dysts":
            cfg_temp.data.trajectory_params.num_ics = 1
            cfg_temp.data.trajectory_params.n_periods = 1

        eq, _, dt = make_trajectories(cfg_temp, save_dir=save_dir)
        values_orig = None
        values = None
        train_dataloader = None
        val_dataloader = None
        test_dataloader = None
        trajs = None

    # Create model
    jac_model = instantiate(cfg_rev.model.params)

    if "use_deriv_net" in cfg_rev.training and cfg_rev.training.use_deriv_net:
        deriv_model = instantiate(cfg_rev.model.deriv_params)
    else:
        deriv_model = None

    # Clean up lightning config to only include valid parameters
    litbase_init_args = list(inspect.signature(LitBase.__init__).parameters.keys())
    keys_to_remove = [
        key
        for key in cfg_rev.training.lightning.keys()
        if key not in litbase_init_args
    ]
    for key in keys_to_remove:
        if key != "_target_":
            del cfg_rev.training.lightning[key]

    lit_model = instantiate(
        cfg_rev.training.lightning,
        model=jac_model,
        deriv_model=deriv_model,
        dt=dt,
        save_dir=run.config["save_dir"],
    )

    # Load checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
    if torch.cuda.is_available():
        lit_model.load_state_dict(
            torch.load(checkpoint_path, weights_only=True)["state_dict"]
        )
    else:
        lit_model.load_state_dict(
            torch.load(checkpoint_path, weights_only=True, map_location="cpu")[
                "state_dict"
            ]
        )

    lit_model.data_type = cfg_rev.data.data_type

    return {
        "cfg": cfg_rev,
        "lit_model": lit_model,
        "eq": eq,
        "dt": dt,
        "values": values,
        "values_orig": values_orig,
        "train_dataloader": train_dataloader,
        "val_dataloader": val_dataloader,
        "test_dataloader": test_dataloader,
        "trajs": trajs,
    }


def _find_best_checkpoint(run: Any, checkpoint_files: list) -> str:
    """Find the best checkpoint based on validation loss or manual overrides."""
    columns = run.history().columns
    history_df = run.scan_history()

    if "random_points val_loss" in columns:
        history_df = pd.DataFrame(
            [
                {"val_loss": row["random_points val_loss"], "epoch": row["epoch"]}
                for row in history_df
                if "random_points val_loss" in row
                and row["random_points val_loss"] is not None
            ]
        )
    elif "trajectory val_loss" in columns:
        history_df = pd.DataFrame(
            [
                {"val_loss": row["trajectory val_loss"], "epoch": row["epoch"]}
                for row in history_df
                if "trajectory val_loss" in row
                and row["trajectory val_loss"] is not None
            ]
        )
    else:
        raise ValueError("No val_loss found in history_df")

    # Remove invalid values
    history_df = history_df[history_df["val_loss"] != "NaN"]
    history_df = history_df[history_df["val_loss"] != "Infinity"]

    # Check for manual epoch overrides for problematic runs
    if run.id in LEGACY_RUN_EPOCH_OVERRIDES:
        opt_epoch = LEGACY_RUN_EPOCH_OVERRIDES[run.id]
        logger.info(f"Using manual epoch override {opt_epoch} for run {run.id}")
    else:
        opt_epoch = history_df.epoch.loc[history_df.val_loss.idxmin()]

    checkpoint = [
        f for f in checkpoint_files if f.split("=")[1].split("-")[0] == str(opt_epoch)
    ][0]

    return checkpoint


def reverse_wandb_config(config):
    data_type = config['data_type'] if 'data_type' in config else 'dysts'
    flow_params = dict(
        random_state=config['random_state'],
    )
    if data_type == 'dysts':
        flow_params['_target_'] = 'ControlJacobians.dysts_sim.flows.' + config['data_cls']
        flow_params['dt'] = config['dt']
        trajectory_params = dict(
            n_periods=config['n_periods'],
            method=config['method'],
            resample=config['resample'],
            pts_per_period=config['pts_per_period'],
            return_times=config['return_times'],
            standardize=config['standardize'],
            noise=config['noise'],
            num_ics=config['num_ics'],
            traj_offset_sd=config['traj_offset_sd'],
            verbose=config['verbose'],
        )
    elif data_type == 'wmtask':
        flow_params['project'] = config['project']
        flow_params['name'] = config['name']
        trajectory_params = dict(
            dataloader_to_use=config['dataloader_to_use'],
            traj_window=config['traj_window'],
            model_to_load=config['model_to_load'] if 'model_to_load' in config else 'final',
        )

    postprocessing = dict(
        obs_noise=config['obs_noise'] if 'obs_noise' in config else False,
        filter_data=config['filter_data'] if 'filter_data' in config else False,
        low_pass=config['low_pass'] if 'low_pass' in config else None,
        high_pass=config['high_pass'] if 'high_pass' in config else None,
    )

    random_points_params = dict(
        n_points=config['n_points'] if 'n_points' in config else 50,
        n_steps=config['n_steps'] if 'n_steps' in config else 1,
    )
    random_points_params['n_validation_steps'] = config['n_validation_steps'] if 'n_validation_steps' in config else random_points_params['n_steps']


    random_points_reverse_params = dict(
        n_points=config['n_points'] if 'n_points' in config else random_points_params['n_points'],
        n_steps=config['n_steps'] if 'n_steps' in config else random_points_params['n_steps'],
        n_validation_steps=config['n_validation_steps'] if 'n_validation_steps' in config else random_points_params['n_steps'],
    )

    train_test_params = dict(
        seq_length=config['seq_length'],
        seq_spacing=config['seq_spacing'],
        train_percent=config['train_percent'],
        test_percent=config['test_percent'],
        split_by=config['split_by'],
        dtype=config['dtype'],
        verbose=config['verbose'],
    )
    delay_embedding_params = dict(
        n_delays=config['n_delays'] if 'n_delays' in config else 1,
        delay_spacing=config['delay_spacing'] if 'delay_spacing' in config else 1,
        observed_indices=config['observed_indices'] if 'observed_indices' in config else 'all',
    )
    train_test_params['delay_embedding_params'] = delay_embedding_params

    train_test_params['reverse_seq_length'] = config['reverse_seq_length'] if 'reverse_seq_length' in config else train_test_params['seq_length']
    train_test_params['reverse_seq_length_validation'] = config['reverse_seq_length_validation'] if 'reverse_seq_length_validation' in config else train_test_params['reverse_seq_length']
    if config['model_cls'] == 'MLP':
        model_params = dict(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            output_dim=config['output_dim'],
            residuals=config['residuals'],
            dropout=config['dropout'] if 'dropout' in config else 0,
            activation=config['activation'],
            use_pre_layer_norm=config['use_pre_layer_norm'] if 'use_pre_layer_norm' in config else False,
            use_mean_and_scale=config['use_mean_and_scale'] if 'use_mean_and_scale' in config else False,
        )

    if config['model_cls'] == 'shPLRNN':
        model_params = dict(
            latent_dim=config['latent_dim'],
            hidden_dim=config['hidden_dim'],
            output_dim=config['output_dim'],
        )

    if config['model_cls'] == 'Transformer':
        model_params = dict(
            input_dim=config['input_dim'],
            output_dim=config['output_dim'],
            d_model=config['d_model'],
            nhead=config['nhead'],
            dim_feedforward=config['dim_feedforward'],
            num_encoder_layers=config['num_encoder_layers'],
            num_decoder_layers=config['num_decoder_layers'],
            positional_embed=config['positional_embed'],
            max_len=config['max_len'],
            dropout=config['dropout'],
            activation=config['activation']
        )

    if config['model_cls'] == 'LSTM':
        model_params = dict(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            output_dim=config['output_dim'],
            residuals=config['residuals'],
            activation=config['activation']
        )

    if config['model_cls'] == 'JacNet':
        model_params = dict(
            input_dim=config['input_dim'],
            embed_dim=config['embed_dim'],
            bottleneck_dim=config['bottleneck_dim'],
            jac_guess_rank=config['jac_guess_rank'],
            embedder=config['embedder'],
            use_resnet=config['use_resnet'],
            num_resnet_channels=config['num_resnet_channels'],
            num_resnet_layers=config['num_resnet_layers'],
            kernel_size=config['kernel_size'],
            activation=config['activation'],
            use_layer_norm=config['use_layer_norm']
        )
        model_params['embedder_kwargs'] = {key[9:]: val for key, val in config.items() if key.startswith('embedder_')}

    if config['model_cls'] == 'MatrixGenerator':
        model_params = dict(
            input_dim=config['input_dim'],
            model_type=config['model_type'],
            embedding_type=config['embedding_type'] if 'embedding_type' in config else 'integer',
        )
        model_params['model_kwargs'] = {key[6:]: val for key, val in config.items() if key.startswith('model_') and key != 'model_cls' and key != 'model_type' and key != 'model_obs_noise_scale'}

    model_params['_target_'] = 'ControlJacobians.models.' + config['model_cls'].lower() + '.' + config['model_cls']

    lightning_params = dict(
        _target_='ControlJacobians.models.' + config['model_cls'].lower() + '.' + config['lightning_cls'],
        direct=config['direct'],
        mode=config['mode'],
        int_method=config['int_method'],
        path=config['path'],
        discretization=config['discretization'] if 'discretization' in config else 'matrix_exp',
        loss_func=config['loss_func'],
        alpha_hal=config['alpha_hal'] if 'alpha_hal' in config else 0,
        loss_func_validation=config['loss_func_validation'] if 'loss_func_validation' in config else config['loss_func'],
        context_length=config['context_length'],
        interp_pts=config['interp_pts'],
        jac_penalty=config['jac_penalty'] if 'jac_penalty' in config else 0,
        jac_penalty_ord=config['jac_penalty_ord'] if 'jac_penalty_ord' in config else 'nuc',
        l2_penalty=config['l2_penalty'] if 'l2_penalty' in config else 0,
        path_point_mode=config['path_point_mode'] if 'path_point_mode' in config else 'interp',
        obs_noise_scale=config['obs_noise_scale'],
        target_smoothing_noise_scale=config['target_smoothing_noise_scale'] if 'target_smoothing_noise_scale' in config else 0,
        y0_noise_scale=config['y0_noise_scale'] if 'y0_noise_scale' in config else 0,
        noise_annealing=config['noise_annealing'] if 'noise_annealing' in config else False,
        # one_step_loss_weight=config['one_step_loss_weight'],
        # multi_step_loss_weight=config['multi_step_loss_weight'],
        # one_step_gen_loss_weight=config['one_step_gen_loss_weight'],
        # multi_step_gen_loss_weight=config['multi_step_gen_loss_weight'],
        # multi_step_spiral_loss_weight=config['multi_step_spiral_loss_weight'] if 'multi_step_spiral_loss_weight' in config else 0,
        # radius=config['radius'] if 'radius' in config else None,
        pre_epochs=config['pre_epochs'],
        model_obs_noise_scale=config['model_obs_noise_scale'],
        log_interval=config['log_interval'] if 'log_interval' in config else 1,
        jac_loss_interval=config['jac_loss_interval'] if 'jac_loss_interval' in config else 1,
        # line_vec=config['line_vec'] if 'line_vec' in config else False,
        # final_step_only=config['final_step_only'] if 'final_step_only' in config else False,
        # validate_all=config['validate_all'] if 'validate_all' in config else False,
        alpha_teacher_forcing=config['alpha_teacher_forcing'] if 'alpha_teacher_forcing' in config else 0,
        alpha_teacher_forcing_reverse=config['alpha_teacher_forcing_reverse'] if 'alpha_teacher_forcing_reverse' in config else 0,
        teacher_forcing_annealing=config['teacher_forcing_annealing'] if 'teacher_forcing_annealing' in config else False,
        gamma_teacher_forcing=config['gamma_teacher_forcing'] if 'gamma_teacher_forcing' in config else 0.999,
        gamma_teacher_forcing_reverse=config['gamma_teacher_forcing_reverse'] if 'gamma_teacher_forcing_reverse' in config else 0.999,
        teacher_forcing_update_interval=config['teacher_forcing_update_interval'] if 'teacher_forcing_update_interval' in config else 5,
        teacher_forcing_steps=config['teacher_forcing_steps'] if 'teacher_forcing_steps' in config else 1,
        min_alpha_teacher_forcing=config['min_alpha_teacher_forcing'] if 'min_alpha_teacher_forcing' in config else 0,
        alpha_validation=config['alpha_validation'] if 'alpha_validation' in config else 0,
        alpha_validation_reverse=config['alpha_validation_reverse'] if 'alpha_validation_reverse' in config else 0,
        random_points_interp_pts_validation=config['random_points_interp_pts_validation'] if 'random_points_interp_pts_validation' in config else 2,
        obs_noise_scale_validation=config['obs_noise_scale_validation'] if 'obs_noise_scale_validation' in config else 0,
        pre_train_epochs=config['pre_train_epochs'] if 'pre_train_epochs' in config else 0,
        reverse_weight=config['reverse_weight'] if 'reverse_weight' in config else 1,
        random_points_weight=config['random_points_weight'] if 'random_points_weight' in config else 1,
    )
    if 'learning_rate' in config:
        lightning_params['learning_rate'] = config['learning_rate']
    lightning_params['random_points_interp_pts'] = config['random_points_interp_pts'] if 'random_points_interp_pts' in config else config['interp_pts']
    lightning_params['max_random_points_interp_pts'] = config['max_random_points_interp_pts'] if 'max_random_points_interp_pts' in config else None
    lightning_params['random_points_interp_pts_validation'] = config['random_points_interp_pts_validation'] if 'random_points_interp_pts_validation' in config else config['random_points_interp_pts']

    logger_params = dict(
        _target_='pytorch_lightning.loggers.' + config['logger_cls'],
        save_dir=config['save_dir'],
        log_model=config['log_model']
    )

    trainer_params = dict(
        max_epochs=config['max_epochs'],
        limit_train_batches=config['limit_train_batches'],
        limit_val_batches=config['limit_val_batches'] if 'limit_val_batches' in config else 1.0,
        accumulate_grad_batches=config['accumulate_grad_batches'] if 'accumulate_grad_batches' in config else 1
    )

    ret_dict = {
        'data': {
            'data_type': data_type,
            'flow': flow_params,
            'trajectory_params': trajectory_params,
            'postprocessing': postprocessing,
            'train_test_params': train_test_params,
            'random_points_params': random_points_params,
            'random_points_reverse_params': random_points_reverse_params
        },
        'model': {
            'params': model_params
        },
        'training': {
            'batch_size': config['batch_size'],
            'save_top_k': config['save_top_k'] if 'save_top_k' in config else 3,
            'use_trajectory': config['use_trajectory'] if 'use_trajectory' in config else True,
            'use_random_points': config['use_random_points'] if 'use_random_points' in config else False,
            'reverse_training': config['reverse_training'] if 'reverse_training' in config else ('inverse_training' if 'inverse_training' in config else False),
            'reverse_validation': config['reverse_validation'] if 'reverse_validation' in config else ('inverse_validation' if 'inverse_validation' in config else False),
            'lightning': lightning_params,
            'logger': logger_params,
            'trainer_params': trainer_params
        }
    }

    return OmegaConf.create(ret_dict)
