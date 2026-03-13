"""Configuration loading and initialization for JacobianODE."""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional, Union

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def load_config(
    config_name: str = "config",
    overrides: Optional[List[str]] = None,
    custom_dataset_loader: Optional[str] = None,
    custom_dataset_loader_kwargs: Optional[Dict[str, Any]] = None,
    data_dim: Optional[int] = None,
) -> DictConfig:
    """Load a JacobianODE config with optional overrides.

    This function initializes Hydra and loads configuration from YAML files
    in the conf directory.

    For file-based custom data, use custom_dataset_loader with file paths.
    For in-memory arrays, use overrides to set up the config, then pass
    the data directly to make_trajectories().

    Args:
        config_name: Name of the config file to load (default: "config").
        overrides: List of Hydra config overrides (e.g., ["training.batch_size=64"]).
        custom_dataset_loader: Path to a custom dataset loader function/class
            (e.g., "JacobianODE.jacobians.custom_data.load_from_numpy").
        custom_dataset_loader_kwargs: Dict of primitive kwargs (strings, numbers)
            to pass to the custom loader. Only use for file paths and simple values.
            For in-memory arrays, pass data directly to make_trajectories() instead.
        data_dim: Dimensionality of the custom data.

    Returns:
        The loaded configuration as an OmegaConf DictConfig.

    Raises:
        hydra.errors.ConfigCompositionException: If config loading fails.
        TypeError: If custom_dataset_loader_kwargs contains non-primitive values.

    Example:
        >>> # Load default config
        >>> cfg = load_config()

        >>> # Load with overrides
        >>> cfg = load_config(overrides=["training.batch_size=128"])

        >>> # Load config for file-based custom data
        >>> cfg = load_config(
        ...     custom_dataset_loader="JacobianODE.jacobians.custom_data.load_from_numpy",
        ...     custom_dataset_loader_kwargs={"file_path": "/path/to/data.npy", "dt": 0.01},
        ...     data_dim=50
        ... )

        >>> # For in-memory arrays, set up config then pass data to make_trajectories:
        >>> cfg = load_config(overrides=["data=custom", "data.flow.dim=3"])
        >>> cfg = initialize_config(cfg)
        >>> eq, sol, dt = make_trajectories(cfg, data=my_array, dt=0.01)
    """
    if overrides is None:
        overrides = []
    else:
        # Make a copy to avoid modifying the input list
        overrides = list(overrides)

    # Handle custom dataset loader
    if custom_dataset_loader is not None:
        if "data=custom" not in overrides:
            overrides.append("data=custom")

        overrides.append(f"data.dataset_loader._target_={custom_dataset_loader}")

        if custom_dataset_loader_kwargs is not None:
            for key, value in custom_dataset_loader_kwargs.items():
                # Only allow primitive types that Hydra can handle
                if value is None:
                    overrides.append(f"+data.dataset_loader.{key}=null")
                elif isinstance(value, (str, int, float, bool)):
                    overrides.append(f"+data.dataset_loader.{key}={value}")
                else:
                    raise TypeError(
                        f"custom_dataset_loader_kwargs['{key}'] has type {type(value).__name__}, "
                        f"but only primitive types (str, int, float, bool) are supported. "
                        f"For in-memory arrays, pass data directly to make_trajectories() instead:\n"
                        f"  cfg = load_config(overrides=['data=custom', 'data.flow.dim=N'])\n"
                        f"  cfg = initialize_config(cfg, data_dim=N)\n"
                        f"  eq, sol, dt = make_trajectories(cfg, data=your_array, dt=your_dt)"
                    )

        # Set data dimension if provided
        if data_dim is not None:
            overrides.append(f"data.flow.dim={data_dim}")

    # Use the package-relative path (relative to this file in core/)
    config_path = "../conf"

    with hydra.initialize(version_base="1.3", config_path=config_path):
        cfg = hydra.compose(config_name=config_name, overrides=overrides)

    logger.debug(f"Loaded config: {config_name} with {len(overrides)} overrides")
    return cfg


def initialize_config(
    cfg: DictConfig,
    data_dim: Optional[int] = None,
) -> DictConfig:
    """Initialize and complete the configuration setup for model training.

    This function performs several key setup tasks:
    1. Sets up the lightning module target based on the model type
    2. Configures model dimensions based on data type (dysts or custom)
    3. Sets up model parameters including input/output dimensions
    4. Handles special cases for different model types (Transformer, NeuralODE, etc.)

    IMPORTANT: This function returns a NEW config via deep copy instead of
    mutating the input. This follows best practices for functional configuration.

    Args:
        cfg: Configuration object containing model, training, and data parameters.
        data_dim: Optional data dimensionality. When provided, skips automatic
            dimension detection (useful when you'll pass data directly to
            make_trajectories later).

    Returns:
        A NEW configuration object with all necessary parameters set.

    Raises:
        ValueError: If data type is not supported or custom data lacks dimension info.

    Example:
        >>> # Standard usage (dimension from config)
        >>> cfg = load_config()
        >>> initialized_cfg = initialize_config(cfg)

        >>> # With in-memory data (dimension provided directly)
        >>> cfg = load_config(overrides=["data=custom"])
        >>> my_data = np.random.randn(32, 1000, 3)  # 3 dimensions
        >>> initialized_cfg = initialize_config(cfg, data_dim=my_data.shape[-1])
    """
    # Create a deep copy to avoid mutation
    cfg = copy.deepcopy(cfg)

    # Set the lightning module target based on model
    if "encoder" in cfg.model:
        # Latent model: use LitLatentJacobianODE
        cfg.training.lightning._target_ = (
            "JacobianODE.models.latent_jacobian.LitLatentJacobianODE"
        )
    else:
        model_module_components = cfg.model.params._target_.split(".")
        model_module_components[-1] = "Lit" + model_module_components[-1]
        cfg.training.lightning._target_ = ".".join(model_module_components)
    cfg.training.lightning.data_type = cfg.data.data_type

    # Collect the dimension of the data
    if data_dim is not None:
        # Dimension provided directly (e.g., for in-memory data)
        dim = data_dim
        # Also update the config so it's consistent
        cfg.data.flow.dim = dim
    elif cfg.data.data_type == "dysts":
        eq = instantiate(cfg.data.flow)
        if cfg.data.train_test_params.delay_embedding_params.observed_indices == "all":
            dim = eq._load_data()["embedding_dimension"]
        else:
            dim = (
                len(cfg.data.train_test_params.delay_embedding_params.observed_indices)
                * cfg.data.train_test_params.delay_embedding_params.n_delays
            )
    elif cfg.data.data_type == "custom":
        if cfg.data.flow.dim is None:
            raise ValueError(
                "For custom data, you must specify 'data.flow.dim' to set the data dimensionality, "
                "or pass data_dim to initialize_config(). "
                "This should match the last axis of your data (trials x time x dim)."
            )
        dim = cfg.data.flow.dim
    else:
        raise ValueError(f"Data type {cfg.data.data_type} not supported")

    # Set model dimensions
    if "encoder" in cfg.model:
        # Latent model: Jacobian MLP operates on n_latent, not data dim
        n_latent = cfg.model.encoder.n_latent
        # Encoder sees delay-embedded observations (dim = n_delays * len(observed_indices))
        cfg.model.encoder.n_input = dim
        if "input_dim" in cfg.model.params:
            cfg.model.params.input_dim = n_latent
        cfg.model.params.output_dim = n_latent ** 2
    else:
        # Standard model: operates on data dim
        if "input_dim" in cfg.model.params:
            cfg.model.params.input_dim = dim
        if "NeuralODE" not in cfg.model.params._target_:
            if cfg.training.lightning.direct:
                cfg.model.params.output_dim = dim**2
            else:  # not direct jacobian estimation
                cfg.model.params.output_dim = dim

    logger.debug(f"Initialized config with dim={dim}")
    return cfg
