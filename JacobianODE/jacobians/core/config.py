"""Configuration loading and initialization for JacobianODE."""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

# Simple arithmetic resolver so experiment YAMLs can derive e.g.
#   seq_length: "${add:${training.lightning.jacobianODEint_kwargs.traj_init_steps},${model.prediction_steps}}"
# Lazy resolution means values stay correct after Hydra sweep overrides.
if not OmegaConf.has_resolver("add"):
    OmegaConf.register_new_resolver(
        "add", lambda *args: sum(int(a) for a in args)
    )


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


def resolve_observed_indices(cfg: DictConfig) -> None:
    """Resolve ``observed_indices='random'`` into a concrete list of indices.

    Two random-sampling modes are supported:

    1. **Flat (single ``n_observed``).** Set ``n_observed`` to an int.
       ``n_observed`` indices are chosen uniformly without replacement from
       ``[0, total_dim)``, then sorted ascending. This is the legacy mode.

    2. **Per-area.** Set ``n_observed_per_area`` (list[int]) and
       ``obs_area_indices`` (list[list[int]]) to specify a partition of the
       observation axis into N areas, each with its own count of observed
       dims. Within each area, indices are sampled without replacement; the
       per-area samples are concatenated **in the order given by
       ``obs_area_indices``** and written back without sorting, so the
       resulting ``observed_indices`` list has area structure
       ``[area0_obs..., area1_obs..., ...]``. Use this when you need the
       partial-observation vector to retain a clean per-area block layout
       (e.g., for ``DirectSumCouplingEncoder.area_indices``, set those to
       ``[[0..n_observed_per_area[0]-1], [n_observed_per_area[0]..., ...]]``
       in the model config so each area's sub-encoder sees its own block).

    The two modes are mutually exclusive. The random seed defaults to
    ``flow.random_state`` but can be overridden with ``partial_obs_seed``.

    For ``'all'`` or an explicit list, this is a no-op.

    Args:
        cfg: Configuration object (mutated in-place).

    Raises:
        ValueError: If neither ``n_observed`` nor ``n_observed_per_area`` is
            set when ``observed_indices='random'``, if both are set, if the
            total data dimensionality cannot be determined, or if the
            per-area config is malformed.
    """
    delay_params = cfg.data.train_test_params.delay_embedding_params
    if delay_params.observed_indices != "random":
        return

    n_observed = delay_params.get("n_observed", None)
    n_observed_per_area = delay_params.get("n_observed_per_area", None)
    obs_area_indices = delay_params.get("obs_area_indices", None)

    has_flat = n_observed is not None
    has_per_area = n_observed_per_area is not None or obs_area_indices is not None
    if has_flat and has_per_area:
        raise ValueError(
            "observed_indices='random' got both `n_observed` and "
            "`n_observed_per_area`/`obs_area_indices`. Pick one mode."
        )
    if not (has_flat or has_per_area):
        raise ValueError(
            "observed_indices='random' requires either "
            "`data.train_test_params.delay_embedding_params.n_observed` "
            "(flat) or `n_observed_per_area` + `obs_area_indices` (per-area)."
        )

    # Determine total number of dimensions (used by the flat path and as a
    # validation upper bound for the per-area path).
    if cfg.data.data_type == "dysts":
        eq = instantiate(cfg.data.flow)
        total_dim = eq._load_data()["embedding_dimension"]
    elif cfg.data.data_type in ("wmtask", "custom"):
        if cfg.data.flow.get("dim", None) is None:
            raise ValueError(
                f"observed_indices='random' with data_type='{cfg.data.data_type}' "
                "requires data.flow.dim to be set."
            )
        total_dim = int(cfg.data.flow.dim)
    else:
        raise ValueError(
            f"Cannot resolve random observed_indices for data_type={cfg.data.data_type}"
        )

    seed = delay_params.get("partial_obs_seed", None)
    if seed is None:
        seed = cfg.data.flow.random_state
    rng = np.random.RandomState(int(seed))

    if has_flat:
        indices = sorted(rng.choice(total_dim, int(n_observed), replace=False).tolist())
        log_msg = (
            f"Resolved random observed_indices (flat): {len(indices)} of "
            f"{total_dim} dims (seed={seed})"
        )
    else:
        # Per-area mode. Validate and sample.
        if n_observed_per_area is None or obs_area_indices is None:
            raise ValueError(
                "Per-area mode requires BOTH `n_observed_per_area` and "
                "`obs_area_indices` to be set."
            )
        n_per = list(OmegaConf.to_container(n_observed_per_area, resolve=True))
        areas = OmegaConf.to_container(obs_area_indices, resolve=True)
        if len(n_per) != len(areas):
            raise ValueError(
                f"len(n_observed_per_area)={len(n_per)} must equal "
                f"len(obs_area_indices)={len(areas)}."
            )
        # Validate area indices: every entry in [0, total_dim), no duplicates.
        flat_areas = [int(i) for a in areas for i in a]
        if any(i < 0 or i >= total_dim for i in flat_areas):
            raise ValueError(
                f"obs_area_indices contains entries outside [0, {total_dim})."
            )
        if len(set(flat_areas)) != len(flat_areas):
            raise ValueError("obs_area_indices contains duplicate dimension indices.")
        # Per-area sampling. We do NOT sort the concatenation: the area block
        # structure is preserved so model.encoder.area_indices can index into
        # the resulting observation vector with simple contiguous slices.
        indices: list[int] = []
        for i, (k, area) in enumerate(zip(n_per, areas)):
            k = int(k)
            area_arr = np.asarray(area, dtype=int)
            if k < 0 or k > len(area_arr):
                raise ValueError(
                    f"n_observed_per_area[{i}]={k} must be in "
                    f"[0, {len(area_arr)}] (size of obs_area_indices[{i}])."
                )
            sampled = rng.choice(area_arr, k, replace=False)
            # Sort within an area so per-area selection is order-stable for
            # the same seed, but keep area order in the concatenation.
            indices.extend(sorted(int(x) for x in sampled))
        log_msg = (
            f"Resolved random observed_indices (per-area): "
            f"{[int(k) for k in n_per]} from areas of sizes "
            f"{[len(a) for a in areas]} (seed={seed}); total={len(indices)}/{total_dim}"
        )

    OmegaConf.update(
        cfg,
        "data.train_test_params.delay_embedding_params.observed_indices",
        indices,
        force_add=True,
    )
    logger.debug(log_msg)


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

    # Resolve observed_indices='random' into a concrete list before anything else
    resolve_observed_indices(cfg)

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
    elif cfg.data.data_type == "wmtask":
        if cfg.data.flow.dim is None:
            raise ValueError(
                "For wmtask data, you must specify 'data.flow.dim' to set the data dimensionality. "
                "This should match the hidden state dimension of the WM RNN (e.g., N1 + N2)."
            )
        delay_params = cfg.data.train_test_params.delay_embedding_params
        if delay_params.observed_indices == "all":
            dim = int(cfg.data.flow.dim) * int(delay_params.n_delays)
        else:
            dim = (
                len(delay_params.observed_indices)
                * int(delay_params.n_delays)
            )
    else:
        raise ValueError(f"Data type {cfg.data.data_type} not supported")

    # Compute per-step observation dimension (number of coords in one delay)
    delay_params = cfg.data.train_test_params.delay_embedding_params
    if delay_params.observed_indices == "all":
        n_recent_dims = dim // int(delay_params.n_delays)
    else:
        n_recent_dims = len(delay_params.observed_indices)

    # Set model dimensions
    if "encoder" in cfg.model:
        # Encoder sees delay-embedded observations (dim = n_delays * len(observed_indices))
        # For standard CouplingEncoder (and similar), write dim into n_input.
        # For DirectSumCouplingEncoder, the partition is user-specified via
        # area_indices — we just validate it covers `dim` distinct input indices.
        if "n_input" in cfg.model.encoder:
            cfg.model.encoder.n_input = dim
        elif "area_indices" in cfg.model.encoder:
            area_indices = OmegaConf.to_container(
                cfg.model.encoder.area_indices, resolve=True
            )
            total = sum(len(a) for a in area_indices)
            if total != dim:
                raise ValueError(
                    f"DirectSumCouplingEncoder area_indices cover {total} "
                    f"input indices, but data dim is {dim}. The partition "
                    f"must match data dimensionality exactly."
                )
        OmegaConf.update(cfg, "model.n_recent_dims", n_recent_dims, force_add=True)
        # n_latent: explicit in config for standard encoders, equals n_input
        # for dimension-preserving encoders (e.g. AffineCouplingEncoder).
        # null → dimension-preserving (resolve to n_input).
        if "n_latent" in cfg.model.encoder and cfg.model.encoder.n_latent is None:
            cfg.model.encoder.n_latent = dim
        n_latent = cfg.model.encoder.get("n_latent", dim)
        # For dimension-preserving encoders with subspace splitting,
        # the MLP Jacobian model operates on n_target_dims, not n_latent.
        n_target_dims = cfg.model.get("n_target_dims", None)
        mlp_dim = n_target_dims if n_target_dims is not None else n_latent
        if "input_dim" in cfg.model.params:
            cfg.model.params.input_dim = mlp_dim
        cfg.model.params.output_dim = mlp_dim ** 2
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
