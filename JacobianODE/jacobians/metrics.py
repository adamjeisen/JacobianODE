"""
Metrics for comparing two time series. These metrics are included to faciliate 
benchmarking of the algorithms in this package while reducing dependencies.

For more exhaustive sets of metrics, use the external `tslearn`, `darts`, or `sktime`

Adapted from https://github.com/williamgilpin/dysts
Huge shoutout to William Gilpin for a fantastic repo, check out his work!
"""

import numpy as np

from scipy.stats import spearmanr, pearsonr, kendalltau

from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

import torch

def mase(y_true, y_pred, y_train=None):
    """
    Mean Absolute Scaled Error. If the time series are multivariate, the first axis is
    assumed to be the time dimension.
    """    
    if y_train is None:
        y_train = y_true
    if len(y_true.shape) == 2:
        if isinstance(y_true, np.ndarray):
            return np.mean(np.abs(y_true - y_pred)) / np.mean(np.abs(y_true[1:] - y_train[:-1]))
        else:
            return torch.mean(torch.abs(y_true - y_pred)) / torch.mean(torch.abs(y_true[1:] - y_train[:-1]))
    elif len(y_true.shape) == 3:
        if isinstance(y_true, np.ndarray):
            return np.mean(np.abs(y_true - y_pred)) / np.mean(np.abs(y_true[:, 1:] - y_train[:, :-1]))
        else:
            return torch.mean(torch.abs(y_true - y_pred)) / torch.mean(torch.abs(y_true[:, 1:] - y_train[:, :-1]))
    else:
        raise ValueError("y_true must be 2 or 3 dimensional")


def mse(y_true, y_pred):
    """
    Mean Squared Error
    """
    if isinstance(y_true, np.ndarray):
        return np.mean(np.square(y_true - y_pred))
    else:
        return torch.mean(torch.square(y_true - y_pred))

def mae(y_true, y_pred):
    """
    Mean Absolute Error
    """
    if isinstance(y_true, np.ndarray):
        return np.mean(np.abs(y_true - y_pred))
    else:
        return torch.mean(torch.abs(y_true - y_pred))

def r2_score(y_true, y_pred):
    """
    R2 Score
    """
    if isinstance(y_true, np.ndarray):
        return 1 - np.sum(np.square(y_true - y_pred)) / np.sum(np.square(y_true - np.mean(y_true)))
    else:
        return 1 - torch.sum(torch.square(y_true - y_pred)) / torch.sum(torch.square(y_true - torch.mean(y_true)))


def normalized_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE normalized by the mean variance across dimensions.

    Computes MSE(pred, target) / mean_d(Var_d(target)), making the loss
    scale-invariant without amplifying errors in low-variance dimensions.

    Parameters
    ----------
    pred : torch.Tensor
        Predictions of any shape (..., D).
    target : torch.Tensor
        Targets of the same shape as pred.

    Returns
    -------
    torch.Tensor
        Scalar. A value of 0 is perfect prediction; a value of 1 means the
        MSE equals the mean variance of the target (i.e. no better than
        predicting the global mean).
    """
    D = target.shape[-1]
    pred_flat = pred.reshape(-1, D)
    tgt_flat = target.reshape(-1, D)
    mean_var = tgt_flat.var(dim=0).mean().clamp(min=1e-8)
    mse_total = (pred_flat - tgt_flat).pow(2).mean()
    return mse_total / mean_var


# def mape(y_true, y_pred):
#     """
#     Mean Absolute Percentage Error
#     """
#     return 100 * np.mean(np.abs(y_true - y_pred) / y_true)

def smape(x, y):
    """Symmetric mean absolute percentage error (between 0 and 200)"""
    if isinstance(x, np.ndarray):
        return 100 * np.mean(2 * np.abs(x - y) / (np.abs(x) + np.abs(y)))
    else:
        return 100 * torch.mean(2 * torch.abs(x - y) / (torch.abs(x) + torch.abs(y)))

def spearman(y_true, y_pred):
    """
    Spearman Correlation. Returns dimensionwise mean for multivariate time series of
    shape (T, D)
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_true, y_pred = np.array(y_true).squeeze(), np.array(y_pred).squeeze()
    if y_true.ndim != y_pred.ndim:
        raise ValueError("y_true and y_pred must have the same number of dimensions")
    
    if y_true.ndim == 1:
        return spearmanr(y_true, y_pred)[0]

    else:
        all_vals = []
        for i in range(y_true.shape[1]):
            all_vals.append(spearmanr(y_true[:, i], y_pred[:, i])[0])
        return np.mean(all_vals)