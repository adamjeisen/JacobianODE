"""Jacobian estimation and Lyapunov exponent computation."""

import numpy as np
from scipy.signal import argrelextrema
import torch
from tqdm.auto import tqdm


def get_min_period_lengthscale(x, max_time=None, verbose=False):
    """
    Given a set of points, compute the minimum period lengthscale as defined in the methods of:
    Ahamed, T., Costa, A. C., & Stephens, G. J. (2020). Capturing the continuous complexity of behaviour in Caenorhabditis elegans. Nature Physics, 17(2), 275-283.

    Args:
        x (ndarray, torch.tensor): a set of points
    """
    if max_time is None:
        max_time = int(x.shape[-2]/6)

    num_rs = x.shape[-2] - max_time
    epsilon_vals = torch.zeros(num_rs, max_time).to(x.device)
    for t in tqdm(range(1, max_time + 1), desc='Computing Epsilon Function', disable=not verbose):
        if len(x.shape) == 2:
            epsilon_vals[:, t - 1] = torch.sort(torch.linalg.norm(x[:-t] - x[t:], axis=-1)).values[:num_rs]
        else:
            epsilon_vals[:, t - 1] = torch.sort(torch.linalg.norm(x[:, :-t] - x[:, t:], axis=-1)).values.mean(axis=0)[:num_rs]

    epsilon_mean = epsilon_vals.mean(axis=0).cpu().numpy()
    min_ind = argrelextrema(epsilon_mean, np.less)[0][0]
    return epsilon_mean[min_ind]
    # return epsilon_vals[0, min_ind]

def weighted_jacobian_lstsq(x, lengthscales, train_percent=1.0, iterator=None, verbose=False):
    """
    Compute weighted Jacobian matrices using least squares regression.

    This function computes local linear approximations (Jacobians) of the dynamics
    at each point in the time series, using a weighted least squares approach where
    points closer in state space are given higher weight.

    Parameters
    ----------
    x : torch.Tensor
        Input time series data of shape (n_trajectories, time_steps, n_dims) or
        (time_steps, n_dims)
    lengthscales : torch.Tensor
        Length scales for the weighting function, shape (n_trajectories, time_steps) or
        (time_steps,)
    train_percent : float, optional
        Percentage of points to use for training the Jacobians, by default 1.0
    iterator : tqdm iterator, optional
        Progress bar iterator, by default None
    verbose : bool, optional
        Whether to show progress bar, by default False

    Returns
    -------
    tuple
        (Js, errors) where:
        - Js: Jacobian matrices of shape (n_trajectories, time_steps, n_dims, n_dims) or
              (time_steps, n_dims, n_dims)
        - errors: Mean squared errors for each time step
    """
    # lengthscales is a tensor of shape batch x time so lengthscales can vary by point
    x_src = x[..., :-1, :].reshape(-1, x.shape[-1])
    x_src = torch.cat((x_src, torch.ones(*x_src.shape[:-1], 1).to(x.device)), dim=-1)
    x_tgt = x[..., 1:, :].reshape(-1, x.shape[-1])

    seq_length = x.shape[-2]
    if len(x.shape) == 2:
        Js = torch.zeros(seq_length, x.shape[-1], x.shape[-1]).type(x.dtype).to(x.device)
    else:
        Js = torch.zeros(x.shape[0], seq_length, x.shape[-1], x.shape[-1]).type(x.dtype).to(x.device)

    iterator_passed = True
    if iterator is None:
        iterator = tqdm(total=seq_length, disable = not verbose, desc='Computing Weighted Jacobians')
        iterator_passed = False

    errors = torch.zeros(seq_length).type(x.dtype).to(x.device)
    for i in range(seq_length):
        if len(x.shape) == 2:
            weighting = torch.exp(-torch.linalg.norm(x[i] - x, axis=-1)/lengthscales[i])
            weighting_src = weighting.reshape(-1)
            weighting_tgt = weighting.reshape(-1)
        else:
            weighting = torch.exp(-torch.linalg.norm(x[:, [i]].unsqueeze(0) - x.unsqueeze(1), axis=-1)/lengthscales[:, [i]])
            weighting_src = weighting[..., :-1].reshape(weighting.shape[0], -1)
            weighting_tgt = weighting[..., 1:].reshape(weighting.shape[0], -1)
        if len(x.shape) == 2:
            weighted_x_src = x_src*weighting_src.unsqueeze(-1)
            weighted_x_tgt = x_tgt*weighting_tgt.unsqueeze(-1)
        else:
            weighted_x_src = x_src.unsqueeze(0)*weighting_src.unsqueeze(-1)
            weighted_x_tgt = x_tgt.unsqueeze(0)*weighting_tgt.unsqueeze(-1)

        train_inds = torch.randperm(weighted_x_src.shape[-2])[:int(weighted_x_src.shape[-2]*train_percent)]
        test_inds = torch.randperm(weighted_x_src.shape[-2])[int(weighted_x_src.shape[-2]*train_percent):]

        weighted_x_plus = weighted_x_tgt
        weighted_x_minus = weighted_x_src
        if len(x.shape) == 2:
            output_mat = torch.linalg.lstsq(weighted_x_minus[..., train_inds, :], weighted_x_plus[..., train_inds, :]).solution
            Js[i] = output_mat.transpose(-2, -1)[:, :-1]
            if len(test_inds) > 0:
                errors[i] = ((weighted_x_plus[..., test_inds, :] - torch.matmul(weighted_x_minus[..., test_inds, :], output_mat))**2).mean()
        else:
            output_mat = torch.linalg.lstsq(weighted_x_minus[..., train_inds, :], weighted_x_plus[..., train_inds, :]).solution
            Js[:, i] = output_mat.transpose(-2, -1)[:, :, :-1]
            if len(test_inds) > 0:
                errors[i] = ((weighted_x_plus[..., test_inds, :] - torch.matmul(weighted_x_minus[..., test_inds, :], output_mat))**2).mean()

        iterator.update()

    if not iterator_passed:
        iterator.close()

    return Js, errors

def estimate_weighted_jacobians(x, max_time=None, sweep=False, thetas=None, return_losses=False, device='cpu', discrete=False, dt=None, return_theta=False, verbose=False):
    """
    Estimate Jacobian matrices using weighted least squares with optional lengthscale optimization.

    This function computes local linear approximations of the dynamics, with the option
    to optimize the lengthscale parameter that controls the locality of the approximation.

    Parameters
    ----------
    x : torch.Tensor or np.ndarray
        Input time series data
    max_time : int, optional
        Maximum time lag to consider for lengthscale estimation, by default None
    sweep : bool, optional
        Whether to sweep over different lengthscale parameters, by default False
    thetas : list, optional
        List of lengthscale parameters to try if sweep=True, by default None
    return_losses : bool, optional
        Whether to return the losses for each theta value, by default False
    device : str, optional
        Device to use for computation, by default 'cpu'
    discrete : bool, optional
        Whether the data represents discrete-time dynamics, by default False
    dt : float, optional
        Time step size for continuous-time dynamics, required if discrete=False
    return_theta : bool, optional
        Whether to return the optimal theta value, by default False
    verbose : bool, optional
        Whether to show progress information, by default False

    Returns
    -------
    tuple or torch.Tensor
        If return_losses and sweep:
            (Js, losses, theta) if return_theta else (Js, losses)
        Else:
            (Js, theta) if return_theta else Js
        where Js are the estimated Jacobian matrices

    Raises
    ------
    ValueError
        If dt is not provided for continuous-time dynamics
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    x = x.to(device)

    if not sweep:
        eps = get_min_period_lengthscale(x, max_time=max_time, verbose=verbose)
        eps = torch.ones(x.shape[0], x.shape[1]).to(x.device)*eps
        Js = weighted_jacobian_lstsq(x, eps, verbose=verbose)
    else:
        pairwise_dists = torch.cdist(x, x)
        d_vals = pairwise_dists.mean(axis=-1)
        if thetas is None:
            # thetas = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
            # thetas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5]
            thetas = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10]
        iterator = tqdm(total=x.shape[-2]*len(thetas), disable = not verbose, desc='Computing Weighted Jacobians')
        losses = torch.zeros(len(thetas))
        for theta_ind, theta in enumerate(thetas):
            if theta > 0:
                lengthscales = d_vals/theta
            else:
                lengthscales = torch.ones(d_vals.shape).to(x.device)*torch.inf
            Js, errors = weighted_jacobian_lstsq(x, lengthscales, iterator=iterator,verbose=verbose)

            # losses[theta_ind] = errors.mean()
            preds = torch.zeros(x.shape).type(x.dtype).to(x.device)
            if len(x.shape) == 2:
                preds[:2] = x[:2]
            else:
                preds[:, :2] = x[:, :2]
            for t in range(preds.shape[-2] - 2):
                if len(x.shape) == 2:
                    preds[t + 2] = x[t + 1] + torch.matmul(Js[t], x[t + 1] - x[t])
                else:
                    preds[:, t + 2] = x[:, t + 1] + torch.matmul(Js[:, t], (x[:, t + 1] - x[:, t]).unsqueeze(-1)).squeeze(-1)

            losses[theta_ind] = torch.linalg.norm(preds - x).mean().cpu()
        iterator.close()
        theta = np.array(thetas)[torch.argmin(losses)]
        if theta > 0:
            print(f"Theta: {theta}")
            lengthscales = d_vals/theta
        else:
            lengthscales = torch.ones(d_vals.shape).to(x.device)*torch.inf
        Js, errors = weighted_jacobian_lstsq(x, lengthscales, verbose=verbose)

    if not discrete:
        if dt is None:
            raise ValueError('dt must be provided for continuous data')
        Js = (Js - torch.eye(Js.shape[-1]).type(Js.dtype).to(Js.device))/dt

    if return_losses and sweep:
        if return_theta:
            return Js, losses, theta
        else:
            return Js, losses
    else:
        if return_theta:
            return Js, theta
        else:
            return Js

def compute_lyaps(Js, dt=1, k=None, verbose=False):
    """
    Compute Lyapunov exponents from a sequence of Jacobian matrices.

    This function computes the Lyapunov exponents using the QR decomposition method,
    which tracks the growth rates of perturbations along different directions.

    Parameters
    ----------
    Js : torch.Tensor
        Sequence of Jacobian matrices of shape (n_trajectories, time_steps, n_dims, n_dims)
        or (time_steps, n_dims, n_dims). Must be DISCRETE-TIME Jacobians.
    dt : float, optional
        Time step size, by default 1
    k : int, optional
        Number of Lyapunov exponents to compute, by default None (computes all)
    verbose : bool, optional
        Whether to show progress information, by default False

    Returns
    -------
    torch.Tensor
        Lyapunov exponents sorted in descending order
    """
    squeeze = False
    if len(Js.shape) == 3:
        Js = Js.unsqueeze(0)
        squeeze = True

    T, n, _ = Js.shape[-3], Js.shape[-2], Js.shape[-1]
    old_Q = torch.eye(n, device=Js.device, dtype=Js.dtype)

    if k is None:
        k = n

    old_Q = old_Q[:, :k]
    lexp = torch.zeros(*Js.shape[:-3], k, device=Js.device, dtype=Js.dtype)
    lexp_counts = torch.zeros(*Js.shape[:-3], k, device=Js.device, dtype=Js.dtype)

    for t in tqdm(range(T), disable=not verbose):

        # QR-decomposition of Js[t] * old_Q
        mat_Q, mat_R = torch.linalg.qr(torch.matmul(Js[..., t, :, :], old_Q))

        # force diagonal of R to be positive
        # sign_diag = torch.sign(torch.diag(mat_R))
        diag_R = mat_R.diagonal(dim1=-2, dim2=-1)
        sign_diag = torch.sign(diag_R)
        sign_diag[sign_diag == 0] = 1
        sign_diag = torch.diag_embed(sign_diag)

        mat_Q = mat_Q @ sign_diag
        mat_R = sign_diag @ mat_R
        old_Q = mat_Q

        # Successively build sum for Lyapunov exponents
        diag_R = mat_R.diagonal(dim1=-2, dim2=-1)

        # Filter zeros in mat_R (would lead to -infs)
        idx = diag_R > 0
        lexp_i = torch.zeros_like(diag_R, dtype=Js.dtype, device=Js.device)
        lexp_i[idx] = torch.log(diag_R[idx])
        lexp[idx] += lexp_i[idx]
        lexp_counts[idx] += 1
    if squeeze:
        lexp = lexp.squeeze(0)
        lexp_counts = lexp_counts.squeeze(0)

    return torch.flip(torch.sort((lexp / lexp_counts) * (1 / dt), axis=-1)[0], dims=[-1])
