"""
Utility functions for FNN time series embedding.

Vendored from https://github.com/williamgilpin/fnn with no modifications
to the core logic. Only import paths have been adjusted for packaging.
"""

import numpy as np
import warnings

from scipy.linalg import hankel
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Loading and featurizing data
# ---------------------------------------------------------------------------

def hankel_matrix(data, q, p=None):
    """
    Find the Hankel matrix dimensionwise for multiple multidimensional
    time series.

    Parameters
    ----------
    data : ndarray
        [N, T, 1] or [N, T, D] collection of time series, or
        [T,] / [T, D] single time series.
    q : int
        The width of the matrix (the number of features / time window).
    p : int, optional
        The height of the matrix (the number of samples).

    Returns
    -------
    ndarray
        Hankel-windowed data.
    """
    if len(data.shape) == 3:
        return np.stack([_hankel_matrix(item, q, p) for item in data])

    if len(data.shape) == 1:
        data = data[:, None]
    return _hankel_matrix(data, q, p)


def _hankel_matrix(data, q, p=None):
    """
    Calculate the hankel matrix of a multivariate timeseries.

    Parameters
    ----------
    data : ndarray
        T x D multidimensional time series.
    q : int
        Window width.
    p : int, optional
        Number of samples.
    """
    if len(data.shape) == 1:
        data = data[:, None]

    if not p:
        p = len(data) - q
    all_hmats = list()
    for row in data.T:
        first, last = row[-(p + q): -p], row[-p - 1:]
        out = hankel(first, last)
        all_hmats.append(out)
    out = np.dstack(all_hmats)
    return np.transpose(out, (1, 0, 2))[:-1]


def resample_dataset(data, n_samples=None, randomize=True, random_state=None):
    """
    Generate random samples from a dataset.

    Parameters
    ----------
    data : ndarray
        [N, ...] collection of datasets.
    n_samples : int, optional
        Number of rows to sample.
    randomize : bool
        Random subsets without replacement.
    random_state : int, optional
        Random seed.

    Returns
    -------
    selection_indices : ndarray
    sampled_data : ndarray
    """
    np.random.seed(random_state)
    if not n_samples:
        n_samples = data.shape[0]
    if randomize:
        selection_indices = np.random.choice(
            np.arange(data.shape[0]), n_samples, replace=False
        )
    else:
        selection_indices = np.arange(n_samples)
    return selection_indices, data[selection_indices]


def train_test(dataset, sample_size, time_window, std=1.0, split=0.5):
    """
    Given a raw 1D time series, perform a standard rescale and find the
    hankel matrix for the train and test partitions.

    Parameters
    ----------
    dataset : ndarray
        1D time series.
    sample_size : int
        Length of the training series.
    time_window : int
        Hankel window width.
    std : float
        Number of standard deviations for rescaling.
    split : float
        Relative split between test/train.
    """
    n = len(dataset)
    n_split = int((split / (1 - split)) * sample_size)

    assert n > sample_size + n_split, "Not enough data to make complete split"

    hm_train = hankel_matrix(dataset, time_window, p=sample_size)
    hm_test = hankel_matrix(
        dataset[:(n_split + time_window)], time_window, p=sample_size
    )

    mn_train, std_train = np.mean(hm_train), np.std(hm_train)

    X_train, X_test = [
        (item - mn_train) / (std * std_train) for item in (hm_train, hm_test)
    ]

    return X_train, X_test


def standardize_ts(a, scale=1.0):
    """
    Standardize a time series along its time dimension.
    For dimensions with zero variance, divide by one instead of zero.

    Parameters
    ----------
    a : ndarray
        Time series, shape (T,), (T, D), or (N, T, D).
    scale : float
        Additional scaling factor.
    """
    if a.ndim == 3:
        stds = np.std(a, axis=1, keepdims=True)   # (N, 1, D)
        stds[stds == 0] = 1
        return (a - np.mean(a, axis=1, keepdims=True)) / (scale * stds)

    stds = np.std(a, axis=0, keepdims=True)
    stds[stds == 0] = 1
    return (a - np.mean(a, axis=0, keepdims=True)) / (scale * stds)


# ---------------------------------------------------------------------------
# Plotting and visualization
# ---------------------------------------------------------------------------

def fixed_aspect_ratio(ratio, ax=None, log=False):
    """Set a fixed aspect ratio on matplotlib plots regardless of axis units."""
    if not ax:
        ax = plt.gca()
    xvals, yvals = ax.axes.get_xlim(), ax.axes.get_ylim()
    xrange = xvals[1] - xvals[0]
    yrange = yvals[1] - yvals[0]
    if log:
        xrange = np.log(xvals[1]) - np.log(xvals[0])
        yrange = np.log(yvals[1]) - np.log(yvals[0])
    ax.set_aspect(ratio * (xrange / yrange), adjustable='box')


def plot_err(y, errs, color=(0, 0, 0), x=[], alpha=.4, linewidth=1, **kwargs):
    """Plot a line with error bands."""
    if len(x) < 1:
        x = np.arange(len(y))

    if len(errs.shape) > 1:
        if errs.shape[1] == 2:
            err_lo, err_hi = errs[:, 0], errs[:, 1]
    else:
        err_lo = errs
        err_hi = errs

    trace_lo, trace_hi = y - err_lo, y + err_hi

    plt.fill_between(x, trace_lo, trace_hi, color=lighter(color), alpha=alpha)
    plt.plot(x, y, color=color, linewidth=linewidth, **kwargs)


def plot3dproj(x, y, z, *args,
               ax=None,
               color=(0, 0, 0),
               shadow_dist=1.0,
               color_proj=None,
               elev_azim=(39, -47),
               show_labels=False,
               aspect_ratio=1.0,
               **kwargs):
    """Create a 3D plot with projections onto 2D coordinate planes."""
    if not ax:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection='3d')
    if not color_proj:
        color_proj = lighter(color, .6)

    if np.isscalar(shadow_dist):
        sdist_x = shadow_dist
        sdist_y = shadow_dist
        sdist_z = shadow_dist
    else:
        sdist_x, sdist_y, sdist_z = shadow_dist

    ax.plot(x, z, *args, zdir='y', zs=sdist_y * np.max(y), color=color_proj, **kwargs)
    ax.plot(y, z, *args, zdir='x', zs=sdist_x * np.min(x), color=color_proj, **kwargs)
    ax.plot(x, y, *args, zdir='z', zs=sdist_z * np.min(z), color=color_proj, **kwargs)
    ax.plot(x, y, z, *args, color=color, **kwargs)

    ax.view_init(elev=elev_azim[0], azim=elev_azim[1])
    ax.set_aspect('auto', adjustable='box')

    if aspect_ratio:
        fixed_aspect_ratio(aspect_ratio)

    if not show_labels:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    return ax


def lighter(clr, f=1 / 3):
    """Brighten an RGB color tuple."""
    gaps = [f * (1 - val) for val in clr]
    return [val + gap for gap, val in zip(gaps, clr)]


def darker(clr, f=1 / 3):
    """Darken an RGB color tuple."""
    gaps = [f * val for val in clr]
    return [val - gap for gap, val in zip(gaps, clr)]

def compute_variances(data, normalize=True):
    """Calculate the variance of a time series."""
    # data has shape (n_samples, n_features)
    # Center the data
    mean_centered = data - np.mean(data, axis=0)
    # Compute Covariance Matrix
    cov_matrix = np.cov(mean_centered, rowvar=False)
    # Compute Eigenvalues (these represent the variance along principal axes)
    eigenvalues, _ = np.linalg.eig(cov_matrix)
    # Sort strictly descending (as per formula A12: SORT(Var(y))) [cite: 572]
    sorted_variances = np.sort(eigenvalues)[::-1]
    # "Normalized variance Var(hm)/Var(h0)" [cite: 222]
    # We normalize by the largest variance (index 0)
    if normalize:
        normalized_variances = sorted_variances / sorted_variances[0]
        return normalized_variances
    else:
        return sorted_variances

def compute_s_dim(var_true, var_est):
    """Compute dimension similarity score S_dim (Eq. A12).
    
    S_dim = 1 - ||SORT(Var(y)) - SORT(Var(y_hat))|| / ||Var(y)||
    """
    v_true = np.sort(var_true)[::-1]
    v_est = np.sort(var_est)[::-1]
    return 1.0 - np.linalg.norm(v_true - v_est) / np.linalg.norm(v_true)