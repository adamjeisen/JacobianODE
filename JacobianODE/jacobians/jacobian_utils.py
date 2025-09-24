"""
Utility functions for working with Jacobian ODE models, including configuration setup, trajectory generation,
data processing, model training, and checkpoint management.
"""

from datetime import datetime
from hydra.utils import instantiate
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import hydra
import inspect
import numpy as np
from omegaconf import OmegaConf
import os
import pandas as pd
import pathlib
import pickle
import pytz
import torch
from torch import utils
import wandb

from .data_utils import filter_data, generate_train_and_test_sets
from .lightning_base import LitBase, PercentEarlyStopping
from .lightning_utils import make_run_info, reverse_wandb_config

def in_ipython():
    try:
        get_ipython
        return True
    except NameError:
        return False

def get_config_path():
    """Get the config path as a relative path for Hydra."""
    # Get the package directory
    package_dir = pathlib.Path(__file__).parent
    config_dir = package_dir / "conf"
    
    # Get the current working directory
    current_dir = pathlib.Path.cwd()
    
    # Calculate the relative path from current directory to config directory
    try:
        relative_path = config_dir.relative_to(current_dir)
        return str(relative_path)
    except ValueError:
        # If we can't make it relative, calculate the path differently
        # Count how many levels up we need to go
        current_parts = current_dir.parts
        config_parts = config_dir.parts
        
        # Find common prefix
        common_prefix_len = 0
        for i, (current_part, config_part) in enumerate(zip(current_parts, config_parts)):
            if current_part == config_part:
                common_prefix_len = i + 1
            else:
                break
        
        # Build relative path
        up_levels = len(current_parts) - common_prefix_len
        relative_parts = [".."] * up_levels + list(config_parts[common_prefix_len:])
        return "/".join(relative_parts)

def load_config(config_name="config", overrides=None, custom_dataset_loader=None, custom_dataset_loader_kwargs=None):
    """Load a JacobianODE config with optional overrides."""
    if overrides is None:
        overrides = []
    
    if custom_dataset_loader is not None:
        if "data=custom" not in overrides:
            overrides.append("data=custom")

        overrides.append(f"data.dataset_loader._target_={custom_dataset_loader}")

        if custom_dataset_loader_kwargs is not None:
            for key, value in custom_dataset_loader_kwargs.items():
                if value is not None:
                    overrides.append(f"+data.dataset_loader.{key}={value}")
                else:
                    overrides.append(f"+data.dataset_loader.{key}=null")
    
    # Use the package-relative path
    config_path = get_config_path()
    
    with hydra.initialize(version_base="1.3", config_path=config_path):
        return hydra.compose(config_name=config_name, overrides=overrides)

def initialize_config(cfg):
    """Initialize and complete the configuration setup for the model training.
    
    This function performs several key setup tasks:
    1. Sets up the save directory based on the environment
    2. Configures model dimensions based on data type (dysts or wmtask)
    3. Sets up model parameters including input/output dimensions
    4. Handles special cases for different model types (Transformer, NeuralODE, etc.)
    
    Args:
        cfg (OmegaConf): Configuration object containing model, training, and data parameters
        
    Returns:
        OmegaConf: Updated configuration object with all necessary parameters set
    """
    # ----------------------------------------
    # FINISH SETUP FOR CONFIG
    # ----------------------------------------

    # set the lightning module target
    model_module_components = cfg.model.params._target_.split('.')
    model_module_components[-1] = 'Lit' + model_module_components[-1]
    cfg.training.lightning._target_ = '.'.join(model_module_components)
    cfg.training.lightning.data_type = cfg.data.data_type

    # ----------------------------------------
    # SET DIMENSIONS
    # ----------------------------------------
    # collect the dimension of the data
    if cfg.data.data_type == 'dysts':
        eq = instantiate(cfg.data.flow)
        if cfg.data.train_test_params.delay_embedding_params.observed_indices == 'all':
            dim = eq._load_data()['embedding_dimension']
        else:
            dim = len(cfg.data.train_test_params.delay_embedding_params.observed_indices)*cfg.data.train_test_params.delay_embedding_params.n_delays
    elif cfg.data.data_type == 'custom':
        dim = cfg.data.flow.dim
    else:
        raise ValueError(f"Data type {cfg.data.data_type} not supported")
    
    # set the input dimension
    if 'input_dim' in cfg.model.params:
        cfg.model.params.input_dim = dim

    # set the output dimension
    if 'NeuralODE' not in cfg.model.params._target_:
        if cfg.training.lightning.direct:
            cfg.model.params.output_dim = dim ** 2
        else:  # not direct jacobian estimation
            cfg.model.params.output_dim = dim

    return cfg

def make_trajectories(cfg, save_dir=None, verbose=False):
    """Generate trajectories for training based on the configuration.
    
    Creates trajectories either from dynamical systems (dysts), working memory task (wmtask),
    or custom data sources based on the data type specified in the config.
    
    Args:
        cfg (OmegaConf): Configuration object containing trajectory parameters
        save_dir (str, optional): Directory to save trajectory data. Defaults to None.
        verbose (bool, optional): Whether to print progress information. Defaults to False.
        
    Returns:
        tuple: (eq, sol, dt) where:
            - eq: The equation/model object
            - sol: Dictionary containing trajectory solutions
            - dt: Time step size
    """
    # Make trajectories
    if cfg.data.data_type == 'dysts':
        eq, sol, dt = make_dysts_trajectories(cfg, save_dir=save_dir, verbose=verbose)
    elif cfg.data.data_type == 'custom':
        eq = None
        sol, dt = instantiate(cfg.data.dataset_loader)
    else:
        raise ValueError(f"Unknown data type: {cfg.data.data_type}")
    return eq, sol, dt

def make_dysts_trajectories(cfg, save_dir=None, verbose=False, save_file=True):
    """Generate trajectories for dynamical systems.
    
    Creates and optionally saves trajectories for dynamical systems models.
    If saved data exists, it will be loaded instead of regenerating.
    
    Args:
        cfg (OmegaConf): Configuration object containing dynamical system parameters
        save_dir (str, optional): Directory to save trajectory data. Defaults to None.
        verbose (bool, optional): Whether to print progress information. Defaults to False.
        save_file (bool, optional): Whether to save the generated trajectories. Defaults to True.
        
    Returns:
        tuple: (eq, sol, dt) where:
            - eq: The dynamical system equation object
            - sol: Dictionary containing trajectory solutions
            - dt: Time step size
    """
    # ----------------------------------------
    # MAKE TRAJECTORIES
    # ----------------------------------------
    if save_dir is None:
        save_dir = cfg.training.logger.save_dir
    os.makedirs(save_dir, exist_ok=True)
    data_save_dir = os.path.join(save_dir, 'dysts_data')
    if save_file:
        os.makedirs(data_save_dir, exist_ok=True)
    filename = os.path.join(data_save_dir, f'{cfg.data.flow._target_}_{cfg.data.trajectory_params.n_periods}periods_{cfg.data.trajectory_params.pts_per_period}ptsperperiod_{cfg.data.trajectory_params.method}_noise_{float(cfg.data.trajectory_params.noise):.2f}_random_state_{cfg.data.flow.random_state}.pkl')
    if os.path.exists(filename):
        if verbose:
            print(f"saved data found at {filename}, loading eq")
        ret = pickle.load(open(filename, 'rb'))
        eq = ret['eq']
        sol = ret['sol']
        dt = ret['dt']
    else:
        if verbose:
            print(f"saved data not found at {filename}, instantiating eq")
        eq = instantiate(cfg.data.flow)  # Ensure eq is instantiated here
        cfg.data.trajectory_params.verbose = verbose
        sol = eq.make_trajectory(**cfg.data.trajectory_params)
        dt = sol['dt']
        if save_file:
            pickle.dump({'eq': eq, 'sol': sol, 'dt': dt}, open(filename, 'wb'))
    return eq, sol, dt

def postprocess_data(cfg, raw_values, raw_values_to_use_for_noise=None, scale_noise=True):
    """Post-process trajectory data by adding noise and/or filtering.
    
    Applies observation noise and optional filtering to the trajectory data.
    Noise can be scaled based on the data magnitude.
    
    Args:
        cfg (OmegaConf): Configuration object containing postprocessing parameters
        raw_values (numpy.ndarray or torch.Tensor): Raw trajectory values - must be of shape (n_traj, time_steps, n_dim)
        raw_values_to_use_for_noise (numpy.ndarray or torch.Tensor, optional): Alternative raw values to use for noise scaling. Defaults to None.
        scale_noise (bool, optional): Whether to scale noise based on data magnitude. Defaults to True.
        
    Returns:
        numpy.ndarray or torch.Tensor: Processed trajectory values
    """
    # ----------------------------------------
    # POSTPROCESS
    # ----------------------------------------
    if scale_noise:
        if raw_values_to_use_for_noise is None:
            obs_noise = cfg.data.postprocessing.obs_noise * float(np.linalg.norm(raw_values, axis=-1).mean() / np.sqrt(raw_values.shape[-1]))
        else:
            obs_noise = cfg.data.postprocessing.obs_noise * float(np.linalg.norm(raw_values_to_use_for_noise, axis=-1).mean() / np.sqrt(raw_values_to_use_for_noise.shape[-1]))
    
    values = raw_values.copy()
    if obs_noise > 0:
        if isinstance(values, torch.Tensor):
            values += torch.randn_like(values) * obs_noise
        else:
            values += np.random.normal(0, obs_noise, values.shape)
    if cfg.data.postprocessing.filter_data:
        values_filtered = np.zeros(values.shape)
        for traj_num in range(values.shape[0]):
            values_filtered[traj_num] = filter_data(values[traj_num], low_pass=cfg.data.postprocessing.low_pass, high_pass=cfg.data.postprocessing.high_pass, dt=sol['dt'])
        values = values_filtered
    return values

def normalize_data(values):
    """Normalize data by subtracting mean and dividing by standard deviation.
    
    Args:
        values (numpy.ndarray or torch.Tensor): Input data to normalize
        
    Returns:
        tuple: (normalized_values, mu, sigma) where:
            - normalized_values: The normalized data
            - mu: Mean of the original data
            - sigma: Standard deviation of the original data
    """
    mu = values.mean()
    sigma = values.std()
    values = (values - mu) / sigma
    return values, mu, sigma

def create_dataloaders(cfg, values, verbose=False):
    """Create PyTorch DataLoaders for training, validation, and testing.
    
    Generates DataLoader objects for continuous trajectory data, handling both
    training and validation sets.
    
    Args:
        cfg (OmegaConf): Configuration object containing dataloader parameters
        values (numpy.ndarray or torch.Tensor): Input data to create dataloaders from
        use_test (bool, optional): Whether to create a test dataloader. Defaults to False.
        verbose (bool, optional): Whether to print progress information. Defaults to False.
        
    Returns:
        tuple: (train_dataloader, val_dataloader, test_dataloader, trajs) where:
            - train_dataloader: DataLoader for training data
            - val_dataloader: DataLoader for validation data
            - test_dataloader: DataLoader for test data
            - trajs: Dictionary containing trajectory information
    """
    # ----------------------------------------
    # MAKE TRAIN AND TEST SETS
    # ----------------------------------------
    cfg.data.train_test_params.verbose = verbose
    valid_keys = inspect.signature(generate_train_and_test_sets).parameters.keys()
    del_keys = [key for key in cfg.data.train_test_params.keys() if key not in valid_keys]
    for key in del_keys:
        del cfg.data.train_test_params[key]
    train_dataset, val_dataset, test_dataset, trajs = generate_train_and_test_sets(values, **cfg.data.train_test_params)

    num_workers = 2
    persistent_workers = True
    pin_memory = True

    # CONTINUOUS TRAJECTORY
    train_dataloader_continuous = utils.data.DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=num_workers, persistent_workers=persistent_workers, pin_memory=pin_memory)
    val_dataloader_continuous = utils.data.DataLoader(val_dataset, batch_size=cfg.training.batch_size, shuffle=False, num_workers=num_workers, persistent_workers=persistent_workers, pin_memory=pin_memory)
    test_dataloader_continuous = utils.data.DataLoader(test_dataset, batch_size=cfg.training.batch_size, shuffle=False, num_workers=num_workers, persistent_workers=persistent_workers, pin_memory=pin_memory)

    return train_dataloader_continuous, val_dataloader_continuous, test_dataloader_continuous, trajs

def setup_wandb(cfg, trajs, log=None, raw_values_to_use_for_noise=None, scale_noise=True):
    """Set up Weights & Biases logging for the training run.
    
    Configures W&B logging, including noise scaling and run naming.
    Handles versioning of run names to avoid conflicts.
    
    Args:
        cfg (OmegaConf): Configuration object containing W&B parameters
        trajs (dict): Dictionary containing trajectory information. Must contain 'train_trajs' with sequence of shape (n_traj, time_steps, n_dim)
        log (Logger, optional): Logger object for output. Defaults to None.
        raw_values_to_use_for_noise (numpy.ndarray or torch.Tensor, optional): Alternative raw values to use for noise scaling. Defaults to None.
        scale_noise (bool, optional): Whether to scale noise. Defaults to True.
        
    Returns:
        tuple: (name, project) where:
            - name: The W&B run name
            - project: The W&B project name
    """
    # ----------------------------------------
    # SET UP WANDB
    # ----------------------------------------
    if scale_noise:
        if raw_values_to_use_for_noise is None:
            cfg.training.lightning.obs_noise_scale = cfg.training.lightning.obs_noise_scale * float(torch.linalg.norm(trajs['train_trajs'].sequence, dim=-1).mean() / np.sqrt(trajs['train_trajs'].sequence.shape[-1]))
            cfg.training.lightning.obs_noise_scale_validation = cfg.training.lightning.obs_noise_scale_validation * float(torch.linalg.norm(trajs['train_trajs'].sequence, dim=-1).mean() / np.sqrt(trajs['train_trajs'].sequence.shape[-1]))
            cfg.training.lightning.obs_noise_scale_loop = cfg.training.lightning.obs_noise_scale_loop * float(torch.linalg.norm(trajs['train_trajs'].sequence, dim=-1).mean() / np.sqrt(trajs['train_trajs'].sequence.shape[-1]))
        else:
            cfg.training.lightning.obs_noise_scale = cfg.training.lightning.obs_noise_scale * float(np.linalg.norm(raw_values_to_use_for_noise, axis=-1).mean() / np.sqrt(raw_values_to_use_for_noise.shape[-1]))
            cfg.training.lightning.obs_noise_scale_validation = cfg.training.lightning.obs_noise_scale_validation * float(np.linalg.norm(raw_values_to_use_for_noise, axis=-1).mean() / np.sqrt(raw_values_to_use_for_noise.shape[-1]))
            cfg.training.lightning.obs_noise_scale_loop = cfg.training.lightning.obs_noise_scale_loop * float(np.linalg.norm(raw_values_to_use_for_noise, axis=-1).mean() / np.sqrt(raw_values_to_use_for_noise.shape[-1]))

    name, project = make_run_info(cfg)

    # log.info("Checking for preexisting runs...")
    api = wandb.Api()
    runs = api.runs(project)
    try:
        found_run = True
        version = 1
        base_name = name
        
        while found_run:
            found_run = False
            for run in runs:
                if run.name == name:
                    found_run = True
                    if log is not None:
                        log.info(f"Run {name} already exists, incrementing version")
                    else:
                        print(f"Run {name} already exists, incrementing version")
                    break
            if found_run:
                version += 1
                name = f"{base_name}_v{version}"
    except ValueError:
        print(f"Project {project} does not exist!")

    return name, project

def make_model(cfg, dt, eq=None, project=None, x0=None, save_dir=None, mu=0, sigma=1, verbose=False):
    """Create and initialize the model for training.
    
    Instantiates the main model and optional derivative model, handling various
    model types and pretrained model loading.
    
    Args:
        cfg (OmegaConf): Configuration object containing model parameters
        dt (float): Time step size
        eq (object, optional): Equation/model object. Defaults to None.
        project (str, optional): W&B project name for loading pretrained models. Defaults to None.
        x0 (numpy.ndarray, optional): Initial state to use as base point for model. Defaults to None.
        save_dir (str, optional): Directory to save model checkpoints. Defaults to None.
        mu (float, optional): Mean for normalization. Defaults to 0.
        sigma (float, optional): Standard deviation for normalization. Defaults to 1.
        verbose (bool, optional): Whether to print progress information. Defaults to False.
        
    Returns:
        tuple: (lit_model, jac_model, deriv_model) where:
            - lit_model: The PyTorch Lightning model
            - jac_model: The main Jacobian model
            - deriv_model: The derivative model (if used)
    """
    # ----------------------------------------
    # MAKE MODEL
    # ----------------------------------------
    jac_model = instantiate(cfg.model.params)
    lit_model = instantiate(cfg.training.lightning, model=jac_model, dt=dt, save_dir=cfg.training.logger.save_dir, base_pt_init=x0, mu=mu, sigma=sigma)
    lit_model.eq = eq
    return lit_model

def log_training_info(train_dataloader, trajs, lit_model, log=None):
    """Log information about the training setup.
    
    Prints or logs information about the training data size, model parameters,
    and other relevant training details.
    
    Args:
        train_dataloader (DataLoader): Training data loader
        trajs (dict): Dictionary containing trajectory information
        jac_model (nn.Module): The main Jacobian model
        deriv_model (nn.Module, optional): The derivative model. Defaults to None.
        log (Logger, optional): Logger object for output. Defaults to None.
    """
    num_train_examples = len(train_dataloader.dataset.sequence)
    num_traj_points = trajs['train_trajs'].sequence.shape[0] * trajs['train_trajs'].sequence.shape[1]
    num_train_data_points = num_traj_points * trajs['train_trajs'].sequence.shape[2]
    total_params = sum(p.numel() for p in lit_model.parameters())

    if log is not None:
        log.info(f"Number of training trajectory examples: {num_train_examples / 1000:.3f}k")
        log.info(f"Number of training trajectory points: {num_traj_points / 1000:.3f}k")
        log.info(f"Number of training data points: {num_train_data_points / 1000:.3f}k")
        log.info(f"Total number of model parameters: {total_params / 1000:.3f}k")
    else:
        print(f"Number of training trajectory examples: {num_train_examples / 1000:.3f}k")
        print(f"Number of training trajectory points: {num_traj_points / 1000:.3f}k")
        print(f"Number of training data points: {num_train_data_points / 1000:.3f}k")
        print(f"Total number of model parameters: {total_params / 1000:.3f}k")

def train_model(cfg, lit_model, train_dataloaders, val_dataloaders, name, project):
    """Train the model using PyTorch Lightning.
    
    Sets up the training environment including callbacks, logging, and training
    parameters, then executes the training process.
    
    Args:
        cfg (OmegaConf): Configuration object containing training parameters
        lit_model (LightningModule): The PyTorch Lightning model to train
        train_dataloaders (DataLoader or list): Training data loader(s)
        val_dataloaders (DataLoader or list): Validation data loader(s)
        name (str): Name of the training run
        project (str): W&B project name
    """
    # ----------------------------------------
    # TRAIN MODEL
    # ----------------------------------------

    logger = instantiate(
        cfg.training.logger, 
        name=name, 
        project=project,
    )
    # logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True))

    # Only update the logger config in the main process (rank 0)
    if os.getenv("LOCAL_RANK") == "0" or os.getenv("LOCAL_RANK") is None:
        logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True))

    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.training.model_checkpoint.monitor,
        save_top_k=cfg.training.model_checkpoint.save_top_k,
        mode=cfg.training.model_checkpoint.mode,
    )
    
    if cfg.training.early_stopping.early_stopping_mode == 'percent_thresh':
        early_stopping_callback = PercentEarlyStopping(
            monitor=cfg.training.early_stopping.monitor,
            patience=cfg.training.early_stopping.early_stopping_patience,
            mode=cfg.training.early_stopping.mode,
            percent_thresh=cfg.training.early_stopping.percent_thresh
        )
    else:
        early_stopping_callback = EarlyStopping(
            monitor=cfg.training.early_stopping.monitor,
            patience=cfg.training.early_stopping.early_stopping_patience,
            mode=cfg.training.early_stopping.mode,
        )
    
    # check if interactive environment
    if in_ipython():
        strategy = 'ddp_notebook'
    else:
        strategy = 'ddp'
    
    trainer = L.Trainer(
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=logger,
        log_every_n_steps=10,
        **cfg.training.trainer_params,
        # accelerator='auto',
        # devices=1 if os.path.exists('/home/millerlab-gpu') else 'auto'
        devices='auto',
        strategy=strategy
    )

    trainer.fit(model=lit_model, train_dataloaders=train_dataloaders, val_dataloaders=val_dataloaders)
    wandb.finish()

def load_run(project, run_id=None, run=None, save_dir=None, no_noise=False, generate_data=True, dt=None, verbose=False):
    """Load a previous training run and its associated data.
    
    Handles loading of both recent and legacy runs, including model checkpoints,
    configuration, and trajectory data.
    
    Args:
        project (str): W&B project name
        run_id (str, optional): ID of the run to load. Defaults to None.
        run (wandb.Run, optional): W&B run object. Defaults to None.
        save_dir (str, optional): Directory containing saved data. Defaults to None.
        no_noise (bool, optional): Whether to disable noise in data generation. Defaults to False.
        generate_data (bool, optional): Whether to generate new trajectory data. Defaults to True.
        dt (float, optional): Time step size. Defaults to None.
        verbose (bool, optional): Whether to print progress information. Defaults to False.
        
    Returns:
        tuple: (run, cfg, eq, dt, values, train_dataloader, val_dataloader, test_dataloader, trajs, lit_model)
            containing all components of the loaded run
    """
    # get the run date
    # Convert UTC timestamp to EST

    if run is None:
        api = wandb.Api(timeout=30)
        run = api.run(f"{project}/{run_id}")
    elif run_id is None:
        raise ValueError("run_id and run cannot both be None")
    else:
        api = None

    utc_dt = datetime.strptime(run.created_at, '%Y-%m-%dT%H:%M:%SZ')
    utc_dt = utc_dt.replace(tzinfo=pytz.UTC)
    est_dt = utc_dt.astimezone(pytz.timezone('US/Eastern'))

    # January 31st 2025 at 2pm EST
    target_dt = datetime(2025, 1, 31, 14, 0, tzinfo=pytz.timezone('US/Eastern'))

    if verbose:
        print(f"Run created at {est_dt} EST")
        print(f"Is after Jan 31 2025 2pm EST? {est_dt > target_dt}")

    # save_dir = "/home/millerlab-gpu/data/CommunicationJacobians/lightning"
    if est_dt > target_dt:
        if verbose:
            print("Date is after Jan 31 2025 2pm EST")
        cfg = OmegaConf.create(run.config)
        if 'use_deriv_net' not in cfg.training:
            cfg.training.use_deriv_net = False
        if save_dir is None:
            save_dir = cfg.training.logger.save_dir
        np.random.seed(cfg.data.flow.random_state)
        torch.random.manual_seed(cfg.data.flow.random_state)
        if no_noise:
            cfg.data.postprocessing.obs_noise = 0
        if generate_data:
            if verbose:
                print("Making trajectories")
            eq, sol, dt = make_trajectories(cfg, save_dir=save_dir, verbose=verbose)
            if verbose:
                print("Done making trajectories")

            if verbose:
                # cfg.data.postprocessing.obs_noise = 0
                print(f"obs_noise: {cfg.data.postprocessing.obs_noise}")

                # cfg.data.train_test_params.seq_length = 250
                print(f"seq_length: {cfg.data.train_test_params.seq_length}")
        else:
            eq = None
            sol = None
            dt = None if dt is None else float(dt)
        # for every key in cfg.trainin.lightning that is not an argument to lightning_base, delete it
        del_keys = []
        for key in cfg.training.lightning.keys():
            if key not in LitBase.__init__.__code__.co_varnames and key != '_target_':
                del_keys.append(key)
        for key in del_keys:
            del cfg.training.lightning[key]

        if generate_data:
            # Postprocess data
            values = postprocess_data(cfg, sol)
            
            # Create train and test sets
            # train_dataloaders, val_dataloaders, train_dataloader_names, val_dataloader_names, dataloader_dict = create_dataloaders(cfg, values, use_test=False)
            train_dataloader, val_dataloader, test_dataloader, trajs = create_dataloaders(cfg, values, use_test=False)
        else:
            values = None
            train_dataloader = None
            val_dataloader = None
            test_dataloader = None
            trajs = None

        # Set up WANDB
        # name, project = setup_wandb(cfg, trajs)
        
        if 'params' in cfg.model:
            cfg.model.params = cfg.model.params

        # Make model
        if 'NeuralODE' in cfg.model.params._target_:
            cfg.model.params.dt = float(dt)
        if cfg.data.train_test_params.delay_embedding_params.n_delays > 1:
            lit_model, model, deriv_model = make_model(cfg, dt, eq=None, save_dir=save_dir, verbose=verbose)
        else:
            lit_model, model, deriv_model = make_model(cfg, dt, eq=eq, project=project, save_dir=save_dir, verbose=verbose)
    else:
        if verbose:
            print("Date is before Jan 31 2025 2pm EST")
        ret_dict = reverse_wandb_run(run, return_data=True, save_dir=save_dir, checkpoint=None)
        cfg, lit_model, eq, values_orig, dt, values = ret_dict['cfg'], ret_dict['lit_model'], ret_dict['eq'], ret_dict['values_orig'], ret_dict['dt'], ret_dict['values']
        np.random.seed(cfg.data.flow.random_state)
        torch.random.manual_seed(cfg.data.flow.random_state)
        if generate_data:   
            eq, sol, dt = make_trajectories(cfg)
            train_dataloader, val_dataloader, test_dataloader, trajs = create_dataloaders(cfg, values, use_test=False)
        # # Postprocess data
        # values = postprocess_data(cfg, sol)
    
    return run, cfg, eq, dt, values, train_dataloader, val_dataloader, test_dataloader, trajs, lit_model

def get_all_checkpoints(run, cfg, save_dir=None):
    """Get all available checkpoint files for a run.
    
    Args:
        run (wandb.Run): W&B run object
        cfg (OmegaConf): Configuration object
        save_dir (str, optional): Directory containing checkpoints. Defaults to None.
        
    Returns:
        tuple: (checkpoint_files, checkpoint_dir) where:
            - checkpoint_files: List of checkpoint filenames
            - checkpoint_dir: Directory containing the checkpoints
    """
    if save_dir is None:
        save_dir = run.config['save_dir'] if 'save_dir' in run.config else cfg.training.logger.save_dir
    checkpoint_dir = os.path.join(save_dir, run.project, run.id, 'checkpoints')
    checkpoint_files = os.listdir(checkpoint_dir)
    checkpoint_files = sorted([f for f in checkpoint_files], key=lambda x: int(x.split('=')[1].split('-')[0]))
    return checkpoint_files, checkpoint_dir

def load_checkpoint(run, cfg, lit_model, save_dir=None, epoch=None, loss_key='mean_val_loss', verbose=False):
    """Load a specific checkpoint for a model.
    
    Can load either the best checkpoint (based on validation loss) or a specific epoch.
    
    Args:
        run (wandb.Run): W&B run object
        cfg (OmegaConf): Configuration object
        lit_model (LightningModule): Model to load checkpoint into
        save_dir (str, optional): Directory containing checkpoints. Defaults to None.
        epoch (int, optional): Specific epoch to load. Defaults to None (loads best).
        loss_key (str, optional): Key to use for finding best checkpoint. Defaults to 'mean_val_loss'.
        verbose (bool, optional): Whether to print progress information. Defaults to False.
    """
    # save_dir = "/home/millerlab-gpu/data/CommunicationJacobians/lightning"

    if verbose:
        print(f"Loading checkpoint from {save_dir}")
    checkpoint_files, checkpoint_dir = get_all_checkpoints(run, cfg, save_dir)

    if verbose:
        print(f"Checkpoint epochs: {[int(f.split('=')[1].split('-')[0]) for f in checkpoint_files]}")

    if epoch is None:
        # pick the checkpoint with minimum mean_val_loss
        mean_val_losses = [{'epoch': h['epoch'], 'mean_val_loss': h[loss_key]} for h in run.scan_history() if loss_key in h and h[loss_key] is not None]
        # if mean_val_losses is empty
        if len(mean_val_losses) == 0:
            mean_val_losses = [{'epoch': h['epoch'], 'mean_val_loss': h['mean val loss']} for h in run.scan_history() if 'mean val loss' in h and h['mean val loss'] is not None]
        epoch = mean_val_losses[np.argmin([mean_val_loss['mean_val_loss'] for mean_val_loss in mean_val_losses])]['epoch']
        checkpoint = [f for f in checkpoint_files if f.startswith(f'epoch={epoch}-')][0]
    else:
        # find the checkpoint with the given epoch
        checkpoint = [f for f in checkpoint_files if f.startswith(f'epoch={epoch}-')][0]

    if verbose:
        print(f"Loading checkpoint from epoch {epoch}")

    checkpoint_data = torch.load(
            os.path.join(checkpoint_dir, checkpoint), 
            weights_only=False,
            map_location='cpu',
            mmap=True
            )
    loaded_state = checkpoint_data['state_dict']

    if 'use_uncertainty' in cfg.training.lightning and cfg.training.lightning.use_uncertainty:
        if 'logvar_trajectory' not in loaded_state:
            loaded_state['logvar_trajectory'] = torch.tensor(0)
        if 'logvar_reverse' not in loaded_state:
            loaded_state['logvar_reverse'] = torch.tensor(0)
        if 'logvar_loop_closure' not in loaded_state:
            loaded_state['logvar_loop_closure'] = torch.tensor(0)
        if 'logvar_lipschitz' not in loaded_state and 'lipschitz' in cfg.model.params and cfg.model.params.lipschitz:
            loaded_state['logvar_lipschitz'] = torch.tensor(0)
    lit_model.load_state_dict(loaded_state)
    lit_model.eval()

    del loaded_state  # Delete loaded_state after it's been used
    torch.cuda.empty_cache()

def reverse_wandb_run(run, return_data=False, checkpoint=None, save_dir=None):
    """Reverse engineer a W&B run to recreate its configuration and model.
    
    Used primarily for loading legacy runs, this function reconstructs the training
    setup from a W&B run.
    
    Args:
        run (wandb.Run): W&B run object to reverse engineer
        return_data (bool, optional): Whether to generate trajectory data. Defaults to False.
        checkpoint (str, optional): Specific checkpoint to load. Defaults to None.
        save_dir (str, optional): Directory containing saved data. Defaults to None.
        
    Returns:
        dict: Dictionary containing all components of the reconstructed run:
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
    """
    if save_dir is None:
        save_dir = run.config['save_dir']
    checkpoint_dir = os.path.join(save_dir, run.project, run.id, 'checkpoints')
    checkpoint_files = os.listdir(checkpoint_dir)

    # val_loss_history = run.history()['val_loss']
    if checkpoint is None:
        columns = run.history().columns
        history_df = run.scan_history()
        if 'random_points val_loss' in columns:
            history_df = pd.DataFrame([{'val_loss': row['random_points val_loss'], 'epoch': row['epoch']} for row in history_df if 'random_points val_loss' in row and row['random_points val_loss'] is not None])
        elif 'trajectory val_loss' in columns:
            history_df = pd.DataFrame([{'val_loss': row['trajectory val_loss'], 'epoch': row['epoch']} for row in history_df if 'trajectory val_loss' in row and row['trajectory val_loss'] is not None])
        else:
            raise ValueError("No val_loss found in history_df")
        # remove where val_loss is nan
        history_df = history_df[history_df['val_loss'] != 'NaN']
        history_df = history_df[history_df['val_loss'] != 'Infinity']
        if run.id == 'oz9rj2ml':
            opt_epoch = 6
        elif run.id == '8rg44zl8':
            opt_epoch = 4
        elif run.id == 'cf8jaatp':
            opt_epoch = 8
        else:
            opt_epoch = history_df.epoch.loc[history_df.val_loss.idxmin()]
        checkpoint = [f for f in checkpoint_files if f.split('=')[1].split('-')[0] == str(opt_epoch)][0]


    cfg_rev = reverse_wandb_config(run.config)
    cfg_rev.training.lightning.eq = cfg_rev.data.flow

    np.random.seed(cfg_rev.data.flow.random_state)
    torch.random.manual_seed(cfg_rev.data.flow.random_state)

    if return_data:
        eq, sol, dt = make_trajectories(cfg_rev, save_dir=save_dir)
        values_orig = sol['values']
        values = postprocess_data(cfg_rev, sol)
        train_dataloader, val_dataloader, test_dataloader, trajs = create_dataloaders(cfg_rev, values, use_test=False)
    else:
        cfg_temp = cfg_rev.copy()
        if cfg_rev.data.data_type == 'dysts':
            cfg_temp.data.trajectory_params.num_ics = 1
            cfg_temp.data.trajectory_params.n_periods = 1


        eq, _, dt = make_trajectories(cfg_temp, save_dir=save_dir)
        values_orig = None
        values = None
        train_dataloader = None
        val_dataloader = None
        test_dataloader = None
        trajs = None

    jac_model = instantiate(cfg_rev.model.params)
    if 'use_deriv_net' in cfg_rev.training and cfg_rev.training.use_deriv_net:
        deriv_model = instantiate(cfg_rev.model.deriv_params)
    else:
        deriv_model = None

    litbase_init_args = list(inspect.signature(LitBase.__init__).parameters.keys())
    keys_to_remove = [key for key in cfg_rev.training.lightning.keys() if key not in litbase_init_args]
    for key in keys_to_remove:
        if key != '_target_':
            del cfg_rev.training.lightning[key]
    lit_model = instantiate(cfg_rev.training.lightning, model=jac_model, deriv_model=deriv_model, dt=dt, save_dir=run.config['save_dir'])
    checkpoint_dir = os.path.join(save_dir, run.project, run.id, 'checkpoints')
    if torch.cuda.is_available():
        lit_model.load_state_dict(torch.load(os.path.join(checkpoint_dir, checkpoint), weights_only=True)['state_dict'])
    else:
        lit_model.load_state_dict(torch.load(os.path.join(checkpoint_dir, checkpoint), weights_only=True, map_location='cpu')['state_dict'])

    lit_model.data_type = cfg_rev.data.data_type

    

    return dict(cfg=cfg_rev, lit_model=lit_model, eq=eq, dt=dt, values=values, values_orig=values_orig, train_dataloader=train_dataloader, val_dataloader=val_dataloader, test_dataloader=test_dataloader, trajs=trajs)