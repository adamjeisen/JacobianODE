"""Trajectory generation for JacobianODE."""

from __future__ import annotations

import fcntl
import logging
import os
import pickle
import random
import socket
import tempfile
import time
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig

from ..custom_data import validate_data_shape

logger = logging.getLogger(__name__)


def _nfs_safe_makedirs(path: str, jlog: logging.Logger) -> None:
    """Create directories with NFS contention mitigation.

    Skips the syscall entirely if the directory already exists.  When it
    does need to create, a small random jitter prevents many SLURM jobs
    from issuing mkdir RPCs at the exact same instant.
    """
    if os.path.isdir(path):
        return
    # Use a non-seeded RNG so seed_everything() doesn't make all jobs identical
    _rng = random.Random(os.getpid() ^ int(time.time() * 1000))
    time.sleep(_rng.uniform(0, 2.0))
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        # On NFS another job may have created it between our check and
        # our makedirs call — that's fine.
        if not os.path.isdir(path):
            raise


def make_trajectories(
    cfg: DictConfig,
    save_dir: Optional[str] = None,
    verbose: bool = False,
    data: Optional[np.ndarray] = None,
    dt: Optional[float] = None,
) -> Tuple[Optional[Any], Dict[str, Any], float]:
    """Generate trajectories for training based on the configuration.

    Creates trajectories either from dynamical systems (dysts), custom data
    sources based on the config, or directly from provided in-memory data.

    Args:
        cfg: Configuration object containing trajectory parameters.
        save_dir: Directory to save trajectory data. Defaults to None.
        verbose: Whether to print progress information. Defaults to False.
        data: Optional numpy array of shape (trials, time, dimensions) to use
            directly instead of loading from config. When provided, bypasses
            Hydra instantiation of dataset_loader.
        dt: Time step between observations. Required when `data` is provided.

    Returns:
        Tuple of (eq, sol, dt) where:
            - eq: The equation/model object (None for custom data)
            - sol: Dictionary containing trajectory solutions with 'values' key
            - dt: Time step size

    Raises:
        ValueError: If data type is unknown, custom data validation fails,
            or data is provided without dt.

    Example:
        >>> # From config (file-based)
        >>> cfg = load_config()
        >>> cfg = initialize_config(cfg)
        >>> eq, sol, dt = make_trajectories(cfg, verbose=True)

        >>> # From in-memory array (programmatic)
        >>> cfg = load_config(overrides=["data=custom", "data.flow.dim=3"])
        >>> cfg = initialize_config(cfg)
        >>> eq, sol, dt = make_trajectories(cfg, data=my_array, dt=0.01)
    """
    # Handle in-memory data passed directly
    if data is not None:
        if dt is None:
            raise ValueError("When providing `data`, you must also provide `dt` (time step)")

        data = np.asarray(data)
        validate_data_shape(data, "Provided data")

        sol = {"values": data}
        eq = None

        # Validate dimension matches config if specified
        data_dim = data.shape[-1]
        if cfg.data.flow.dim is not None and cfg.data.flow.dim != data_dim:
            raise ValueError(
                f"Data dimension mismatch: config specifies data.flow.dim={cfg.data.flow.dim}, "
                f"but provided data has dimension {data_dim}. "
                f"Please update data.flow.dim to match your data."
            )

        if verbose:
            n_trials, n_time, n_dims = data.shape
            logger.info(
                f"Using provided data: {n_trials} trials, {n_time} timepoints, {n_dims} dimensions"
            )
            logger.info(f"Time step (dt): {dt}")

        return eq, sol, dt

    # Otherwise, load data based on config
    if cfg.data.data_type == "dysts":
        eq, sol, dt_loaded = make_dysts_trajectories(cfg, save_dir=save_dir, verbose=verbose)
        return eq, sol, dt_loaded
    elif cfg.data.data_type == "custom":
        eq = None
        sol, dt_loaded = instantiate(cfg.data.dataset_loader)

        # Validate custom data
        if "values" not in sol:
            raise ValueError(
                "Custom dataset loader must return (sol, dt) where sol is a dict "
                "containing a 'values' key with the trajectory data."
            )

        validate_data_shape(sol["values"], "Custom data")

        # Validate dimension matches config
        data_dim = sol["values"].shape[-1]
        if cfg.data.flow.dim is not None and cfg.data.flow.dim != data_dim:
            raise ValueError(
                f"Data dimension mismatch: config specifies data.flow.dim={cfg.data.flow.dim}, "
                f"but loaded data has dimension {data_dim}. "
                f"Please update data.flow.dim to match your data."
            )

        if verbose:
            n_trials, n_time, n_dims = sol["values"].shape
            logger.info(
                f"Loaded custom data: {n_trials} trials, {n_time} timepoints, {n_dims} dimensions"
            )
            logger.info(f"Time step (dt): {dt_loaded}")

        return eq, sol, dt_loaded
    elif cfg.data.data_type == "wmtask":
        eq = None
        sol, dt_loaded = instantiate(cfg.data.dataset_loader)

        validate_data_shape(sol["values"], "WMTask data")

        data_dim = sol["values"].shape[-1]
        if cfg.data.flow.dim is not None and cfg.data.flow.dim != data_dim:
            raise ValueError(
                f"Data dimension mismatch: config specifies data.flow.dim={cfg.data.flow.dim}, "
                f"but loaded data has dimension {data_dim}. "
                f"Please update data.flow.dim to match your data."
            )

        if verbose:
            n_trials, n_time, n_dims = sol["values"].shape
            logger.info(
                f"Loaded WMTask data: {n_trials} trials, {n_time} timepoints, {n_dims} dimensions"
            )
            logger.info(f"Time step (dt): {dt_loaded}")

        return eq, sol, dt_loaded
    else:
        raise ValueError(f"Unknown data type: {cfg.data.data_type}")


def make_dysts_trajectories(
    cfg: DictConfig,
    save_dir: Optional[str] = None,
    verbose: bool = False,
    save_file: bool = True,
) -> Tuple[Any, Dict[str, Any], float]:
    """Generate trajectories for dynamical systems.

    Creates and optionally saves trajectories for dynamical systems models.
    If saved data exists, it will be loaded instead of regenerating.

    Args:
        cfg: Configuration object containing dynamical system parameters.
        save_dir: Directory to save trajectory data. Defaults to None.
        verbose: Whether to print progress information. Defaults to False.
        save_file: Whether to save the generated trajectories. Defaults to True.

    Returns:
        Tuple of (eq, sol, dt) where:
            - eq: The dynamical system equation object
            - sol: Dictionary containing trajectory solutions
            - dt: Time step size

    Example:
        >>> cfg = load_config()
        >>> eq, sol, dt = make_dysts_trajectories(cfg, verbose=True)
        >>> print(f"dt = {dt}")
    """
    jlog = logging.getLogger("JacobianLogger")
    jlog.info("make_dysts_trajectories: entered")
    if save_dir is None:
        save_dir = cfg.training.logger.save_dir

    jlog.info("make_dysts_trajectories: about to makedirs save_dir=%s", save_dir)
    _nfs_safe_makedirs(save_dir, jlog)
    jlog.info("make_dysts_trajectories: makedirs save_dir done")
    data_save_dir = os.path.join(save_dir, "dysts_data")

    if save_file:
        jlog.info("make_dysts_trajectories: about to makedirs data_save_dir=%s", data_save_dir)
        _nfs_safe_makedirs(data_save_dir, jlog)
        jlog.info("make_dysts_trajectories: makedirs data_save_dir done")

    # Build filename from config parameters
    filename = os.path.join(
        data_save_dir,
        f"{cfg.data.flow._target_}_"
        f"{cfg.data.trajectory_params.n_periods}periods_"
        f"{cfg.data.trajectory_params.pts_per_period}ptsperperiod_"
        f"{cfg.data.trajectory_params.method}_"
        f"noise_{float(cfg.data.trajectory_params.noise):.2f}_"
        f"random_state_{cfg.data.flow.random_state}.pkl",
    )

    host = socket.gethostname()
    lockfile = filename + ".lock"

    # Use file-based locking so only one job generates data; others wait.
    jlog.info("[host=%s] acquiring lock %s", host, lockfile)
    lock_fd = open(lockfile, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        jlog.info("[host=%s] lock acquired", host)

        cache_exists = os.path.exists(filename)
        jlog.info("[host=%s] cache path=%s, exists=%s", host, filename, cache_exists)

        if cache_exists:
            if verbose:
                logger.info(f"Saved data found at {filename}, loading eq")
            jlog.info("make_dysts_trajectories: loading pickle from %s", filename)
            with open(filename, "rb") as f:
                ret = pickle.load(f)
            jlog.info("make_dysts_trajectories: pickle load done")
            eq = ret["eq"]
            sol = ret["sol"]
            dt = ret["dt"]
        else:
            if verbose:
                logger.info(f"Saved data not found at {filename}, instantiating eq")
            eq = instantiate(cfg.data.flow)
            cfg.data.trajectory_params.verbose = verbose
            sol = eq.make_trajectory(**cfg.data.trajectory_params)
            dt = sol["dt"]
            if save_file:
                # Atomic write: write to temp file then rename so readers
                # never see a partially-written pickle.
                tmp_fd, tmp_path = tempfile.mkstemp(
                    dir=data_save_dir, suffix=".pkl.tmp"
                )
                try:
                    with os.fdopen(tmp_fd, "wb") as f:
                        pickle.dump({"eq": eq, "sol": sol, "dt": dt}, f)
                    os.replace(tmp_path, filename)
                except BaseException:
                    os.unlink(tmp_path)
                    raise
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()

    return eq, sol, dt
