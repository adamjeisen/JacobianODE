import numpy as np
import scipy.signal as signal
from scipy.signal import argrelextrema
import torch
from tqdm.auto import tqdm


# ----------------------------------------
# Data Organization
# ----------------------------------------

def embed_signal_torch(data, n_delays, delay_interval=1):
    """
    Create a delay embedding from the provided tensor data.

    Parameters
    ----------
    data : torch.tensor
        The data from which to create the delay embedding. Must be either: (1) a
        2-dimensional array/tensor of shape T x N where T is the number
        of time points and N is the number of observed dimensions
        at each time point, or (2) a 3-dimensional array/tensor of shape
        K x T x N where K is the number of "trials" and T and N are
        as defined above.

    n_delays : int
        Parameter that controls the size of the delay embedding. Explicitly,
        the number of delays to include.

    delay_interval : int
        The number of time steps between each delay in the delay embedding. Defaults
        to 1 time step.
    """
    with torch.no_grad():
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        device = data.device

        # initialize the embedding
        if data.ndim == 3:
            embedding = torch.zeros((data.shape[0], data.shape[1] - (n_delays - 1)*delay_interval, data.shape[2]*n_delays)).to(device)
        else:
            embedding = torch.zeros((data.shape[0] - (n_delays - 1)*delay_interval, data.shape[1]*n_delays)).to(device)
        
        for d in range(n_delays):
            index = (n_delays - 1 - d)*delay_interval
            ddelay = d*delay_interval

            if data.ndim == 3:
                ddata = d*data.shape[2]
                embedding[:,:, ddata: ddata + data.shape[2]] = data[:,index:data.shape[1] - ddelay]
            else:
                ddata = d*data.shape[1]
                embedding[:, ddata:ddata + data.shape[1]] = data[index:data.shape[0] - ddelay]
        
        return embedding

def convert_to_trajs_needed(pct):
    """
    Convert a percentage to the number of trajectories needed to satisfy that percentage.

    Parameters
    ----------
    pct : float
        The percentage of trajectories needed (between 0 and 1)

    Returns
    -------
    float
        The number of trajectories needed to satisfy the percentage.
        Returns 0 if pct is 0, otherwise returns 1/pct.
    """
    if pct == 0:
        return 0
    else:
        return 1/pct
    
def get_start_indices(seq_length, seq_spacing, T):
    """
    Generate valid starting indices for sequence extraction from a time series.

    Parameters
    ----------
    seq_length : int
        Length of the sequences to extract
    seq_spacing : int
        Number of time steps between consecutive sequence starts
    T : int
        Total number of time points available

    Returns
    -------
    list
        List of valid starting indices for sequence extraction

    Raises
    ------
    ValueError
        If seq_length is greater than T
        If seq_spacing is 0 when seq_length != T
    """
    if T == 0:
        return []

    if seq_length > T:
            raise ValueError(f'seq_length ({seq_length}) must be less than or equal to the number of time points ({T})')
    if seq_length == T:
        start_indices = [0]
    else:
        if seq_spacing == 0:
            raise ValueError('seq_spacing must be greater than 0 if seq_length != pts.shape[1]')
        start_indices = np.arange(0, T - seq_length, seq_spacing)
    
    return start_indices

def generate_train_and_test_sets(pts, seq_length, seq_spacing=1, train_percent=0.8, test_percent=0.05, split_by='time', dtype='torch.FloatTensor', delay_embedding_params=None, verbose=False):
    """
    Generate training, validation, and test datasets from time series data.

    This function splits the data into training, validation, and test sets either by time
    or by trajectory, and optionally applies delay embedding.

    Parameters
    ----------
    pts : torch.Tensor or np.ndarray
        Input time series data of shape (n_trajectories, time_steps, n_dims) or
        (time_steps, n_dims)
    seq_length : int
        Length of sequences to extract
    seq_spacing : int, optional
        Number of time steps between consecutive sequence starts, by default 1
    train_percent : float, optional
        Percentage of data to use for training, by default 0.8
    test_percent : float, optional
        Percentage of data to use for testing, by default 0.05
    split_by : str, optional
        How to split the data: 'time' or 'trajectory', by default 'time'
    dtype : str, optional
        Data type for the output tensors, by default 'torch.FloatTensor'
    delay_embedding_params : dict, optional
        Parameters for delay embedding:
        - 'observed_indices': indices to use from input data
        - 'n_delays': number of delays to include
        - 'delay_spacing': spacing between delays
    verbose : bool, optional
        Whether to print progress information, by default False

    Returns
    -------
    tuple
        (train_dataset, val_dataset, test_dataset, trajs) where:
        - train_dataset, val_dataset, test_dataset are TimeSeriesDataset objects
        - trajs is a dict containing the full trajectories and indices for each split

    Raises
    ------
    ValueError
        If train_percent + test_percent > 1
        If split_by is 'trajectory' and there aren't enough trajectories
    """
    val_percent = 1 - train_percent - test_percent

    if train_percent + test_percent > 1:
        raise ValueError('train_percent + test_percent must be less than or equal to 1')
    
    if delay_embedding_params is not None:
        if delay_embedding_params['observed_indices'] != 'all':
            pts = pts[:, :, delay_embedding_params['observed_indices']]
        if delay_embedding_params['n_delays'] > 1:
            pts = embed_signal_torch(pts, delay_embedding_params['n_delays'], delay_embedding_params['delay_spacing'])

    if split_by == 'trajectory':
        # select start indices
        start_indices = get_start_indices(seq_length, seq_spacing, pts.shape[1])

        if convert_to_trajs_needed(train_percent) > pts.shape[0]:
            raise ValueError(f'With split_by==trajectory, not enough trajectories ({pts.shape[0]}) to satisfy train_percent ({train_percent:.4f})')
        if convert_to_trajs_needed(test_percent) > pts.shape[0]:
            raise ValueError(f'With split_by==trajectory, not enough trajectories ({pts.shape[0]}) to satisfy test_percent ({test_percent:.4f})')
        if convert_to_trajs_needed(val_percent) > pts.shape[0]:
            raise ValueError(f'With split_by==trajectory, not enough trajectories ({pts.shape[0]}) to satisfy val_percent ({val_percent:.4f})')

        train_inds = np.random.choice(pts.shape[0], int(train_percent*pts.shape[0]), replace=False)
        remaining_inds = np.array([i for i in np.arange(pts.shape[0]) if i not in train_inds])
        test_inds = np.random.choice(remaining_inds, int(test_percent*pts.shape[0]), replace=False)
        val_inds = np.array([i for i in np.arange(pts.shape[0]) if i not in train_inds and i not in test_inds])

        train_trajs = pts[train_inds]
        val_trajs = pts[val_inds]
        test_trajs = pts[test_inds]

        # generate training examples and labels
        n_train = train_trajs.shape[0]
        n_val = val_trajs.shape[0]
        n_test = test_trajs.shape[0]

        train_examples = np.zeros((n_train*len(start_indices), seq_length, train_trajs.shape[2]))
        val_examples = np.zeros((n_val*len(start_indices), seq_length, val_trajs.shape[2]))
        test_examples = np.zeros((n_test*len(start_indices), seq_length, test_trajs.shape[2]))

        for i, start_ind in tqdm(enumerate(start_indices), total=len(start_indices), disable=not verbose, desc='Sequence Indices'):
            train_examples[i*n_train:(i + 1)*n_train] = train_trajs[:, start_ind:start_ind + seq_length]
            val_examples[i*n_val:(i + 1)*n_val] = val_trajs[:, start_ind:start_ind + seq_length]
            test_examples[i*n_test:(i + 1)*n_test] = test_trajs[:, start_ind:start_ind + seq_length]

    # elif split_by == 'random':
    #     all_examples = np.zeros((pts.shape[0]*len(start_indices), seq_length, pts.shape[2]))
    #     for i, start_ind in tqdm(enumerate(start_indices), total=len(start_indices), disable=not verbose, desc='Sequence Indices'):
    #         all_examples[i*pts.shape[0]:(i + 1)*pts.shape[0]] = pts[:, start_ind:start_ind + seq_length]
        
    #     train_inds = np.random.choice(all_examples.shape[0], int(train_percent*all_examples.shape[0]), replace=False)
    #     remaining_inds = np.array([i for i in np.arange(all_examples.shape[0]) if i not in train_inds])
    #     test_inds = np.random.choice(remaining_inds, int(test_percent*all_examples.shape[0]), replace=False)
    #     val_inds = np.array([i for i in np.arange(all_examples.shape[0]) if i not in train_inds and i not in test_inds])

    #     train_examples = all_examples[train_inds]
    #     val_examples = all_examples[val_inds]
    #     test_examples = all_examples[test_inds]

    elif split_by == 'time':
        
        train_trajs = pts[:, np.arange(0, int(train_percent*pts.shape[1]))]
        val_trajs = pts[:, np.arange(int(train_percent*pts.shape[1]), int((train_percent + val_percent)*pts.shape[1]))]
        test_trajs = pts[:, np.arange(int((train_percent + val_percent)*pts.shape[1]), pts.shape[1])]

        # generate examples
        start_indices_train = get_start_indices(seq_length, seq_spacing, train_trajs.shape[1])
        start_indices_val = get_start_indices(seq_length, seq_spacing, val_trajs.shape[1])
        start_indices_test = get_start_indices(seq_length, seq_spacing, test_trajs.shape[1])

        train_inds = start_indices_train
        val_inds = start_indices_val
        test_inds = start_indices_test

        n_trajs = train_trajs.shape[0]

        train_examples = np.zeros((n_trajs*len(start_indices_train), seq_length, train_trajs.shape[2]))
        val_examples = np.zeros((n_trajs*len(start_indices_val), seq_length, val_trajs.shape[2]))
        test_examples = np.zeros((n_trajs*len(start_indices_test), seq_length, test_trajs.shape[2]))

        iterator = tqdm(total=len(start_indices_train) + len(start_indices_val) + len(start_indices_test), disable=not verbose, desc='Sequence Indices')

        for i, start_ind in enumerate(start_indices_train):
            train_examples[i*n_trajs:(i + 1)*n_trajs] = train_trajs[:, start_ind:start_ind + seq_length]
            iterator.update()
        
        for i, start_ind in enumerate(start_indices_val):
            val_examples[i*n_trajs:(i + 1)*n_trajs] = val_trajs[:, start_ind:start_ind + seq_length]
            iterator.update()
        
        for i, start_ind in enumerate(start_indices_test):
            test_examples[i*n_trajs:(i + 1)*n_trajs] = test_trajs[:, start_ind:start_ind + seq_length]
            iterator.update()
        
        iterator.close()

    train_dataset = TimeSeriesDataset(torch.from_numpy(train_examples).type(dtype))
    val_dataset = TimeSeriesDataset(torch.from_numpy(val_examples).type(dtype))
    test_dataset = TimeSeriesDataset(torch.from_numpy(test_examples).type(dtype))

    if isinstance(train_trajs, np.ndarray):
        train_trajs = torch.from_numpy(train_trajs).type(dtype)
    if isinstance(val_trajs, np.ndarray):
        val_trajs = torch.from_numpy(val_trajs).type(dtype)
    if isinstance(test_trajs, np.ndarray):
        test_trajs = torch.from_numpy(test_trajs).type(dtype)

    # if delay_embedding_params is not None:
    #     if delay_embedding_params['observed_indices'] != 'all':
    #         train_trajs = train_trajs[:, :, delay_embedding_params['observed_indices']]
    #         val_trajs = val_trajs[:, :, delay_embedding_params['observed_indices']]
    #         test_trajs = test_trajs[:, :, delay_embedding_params['observed_indices']]
    #     if delay_embedding_params['n_delays'] > 1:
    #         train_trajs = embed_signal_torch(train_trajs, delay_embedding_params['n_delays'], delay_embedding_params['delay_spacing'])
    #         val_trajs = embed_signal_torch(val_trajs, delay_embedding_params['n_delays'], delay_embedding_params['delay_spacing'])
    #         test_trajs = embed_signal_torch(test_trajs, delay_embedding_params['n_delays'], delay_embedding_params['delay_spacing'])

    trajs = dict(
        train_trajs=TimeSeriesDataset(train_trajs),
        val_trajs=TimeSeriesDataset(val_trajs),
        test_trajs=TimeSeriesDataset(test_trajs),
        train_inds=train_inds,
        val_inds=val_inds,
        test_inds=test_inds
    )

    # train_dataset = TimeSeriesDataset(torch.from_numpy(train_examples).type(dtype), torch.from_numpy(train_labels))
    # test_dataset = TimeSeriesDataset(torch.from_numpy(test_examples).type(dtype), torch.from_numpy(test_labels))

    if verbose:
        print(f"Train dataset shape: {train_dataset.sequence.shape}")
        print(f"Validation dataset shape: {val_dataset.sequence.shape}")
        print(f"Test dataset shape: {test_dataset.sequence.shape}")

        print('Train trajectories dataset shape: {}'.format(trajs['train_trajs'].sequence.shape))
        print('Validation trajectories dataset shape: {}'.format(trajs['val_trajs'].sequence.shape))
        print('Test trajectories dataset shape: {}'.format(trajs['test_trajs'].sequence.shape))

    return train_dataset, val_dataset, test_dataset, trajs
    
# Dataset class for time series prediction
class TimeSeriesDataset(torch.utils.data.Dataset):
    """
    Dataset class for time series data.

    This class provides a PyTorch Dataset interface for time series data,
    where each item is a sequence of observations.

    Parameters
    ----------
    sequence : torch.Tensor
        Input time series data of shape (n_sequences, seq_length, n_dims)
    """
    def __init__(self, sequence):
        self.sequence = sequence

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, index):
        return self.sequence[index]

def get_min_period_lengthscale(x, max_time=None, verbose=False):
    """
    Given a set of points, compute the minimum period lengthscale as defined in the methods of:
    Ahamed, T., Costa, A. C., & Stephens, G. J. (2020). Capturing the continuous complexity of behaviour in Caenorhabditis elegans. Nature Physics, 17(2), 275–283.
    
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
        or (time_steps, n_dims, n_dims)
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

import scipy.signal as signal
def butter_highpass(cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    # return b, a
    sos = signal.butter(order, normal_cutoff, btype='high', analog=False, output='sos')
    return sos

def butter_highpass_filter(data, cutoff, fs, order=2, bidirectional=True):
    # b, a = butter_highpass(cutoff, fs, order=order)
    # y = signal.filtfilt(b, a, data)
    sos = butter_highpass(cutoff, fs, order=order)
    if bidirectional:
        y = signal.sosfiltfilt(sos, data)
    else:
        y = signal.sosfilt(sos, data)
    return y

def butter_lowpass(cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    # return b, a
    sos = signal.butter(order, normal_cutoff, btype='low', analog=False, output='sos')
    return sos

def butter_lowpass_filter(data, cutoff, fs, order=2, bidirectional=True):
    # b, a = butter_lowpass(cutoff, fs, order=order)
    # y = signal.filtfilt(b, a, data)
    sos = butter_lowpass(cutoff, fs, order=order)
    if bidirectional:
        y = signal.sosfiltfilt(sos, data)
    else:
        y = signal.sosfilt(sos, data)
    return y

# Define the bandstop filter function
def butter_bandstop_filter(data, lowcut, highcut, fs, order=2, bidirectional=True):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    # b, a = signal.butter(order, [low, high], btype='bandstop')
    # y = signal.filtfilt(b, a, data)
    sos = signal.butter(order, [low, high], btype='bandstop', output='sos')
    if bidirectional:
        y = signal.sosfiltfilt(sos, data)
    else:
        y = signal.sosfilt(sos, data)
    return y

def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    # b, a = signal.butter(order, [low, high], btype='band')
    # return b, a
    sos = signal.butter(order, [low, high], btype='band', output='sos')
    return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2, bidirectional=True):
    # b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    # y = signal.lfilter(b, a, data)
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    if bidirectional:
        y = signal.sosfiltfilt(sos, data)
    else:
        y = signal.sosfilt(sos, data)
    return y

def filter_data(data, low_pass=None, high_pass=None, dt=0.001, order=2, bidirectional=True):
    """
    Apply Butterworth filtering to time series data.

    This function can apply low-pass, high-pass, band-pass, or band-stop filtering
    to the input data using Butterworth filters.

    Parameters
    ----------
    data : np.ndarray
        Input time series data of shape (time_steps, n_dims)
    low_pass : float, optional
        Low-pass cutoff frequency, by default None
    high_pass : float, optional
        High-pass cutoff frequency, by default None
    dt : float, optional
        Time step size, by default 0.001
    order : int, optional
        Filter order, by default 2
    bidirectional : bool, optional
        Whether to apply the filter bidirectionally to avoid phase shifts, by default True

    Returns
    -------
    np.ndarray
        Filtered data of the same shape as the input

    Notes
    -----
    - If both low_pass and high_pass are None, returns original data
    - If only one cutoff is provided, applies either low-pass or high-pass filter
    - If both cutoffs are provided:
        - If low_pass > high_pass: applies band-pass filter
        - If low_pass < high_pass: applies band-stop filter
        - If low_pass == high_pass: returns original data
    """
    if low_pass is None and high_pass is None:
        return data
    elif low_pass is None and high_pass is not None:
        data_filt = np.zeros(data.shape)
        for i in range(data.shape[1]):
            data_filt[:, i] = butter_highpass_filter(data[:, i], high_pass, 1/dt, order=order, bidirectional=bidirectional)
        return data_filt
    elif low_pass is not None and high_pass is None:
        data_filt = np.zeros(data.shape)
        for i in range(data.shape[1]):
            data_filt[:, i] = butter_lowpass_filter(data[:, i], low_pass, 1/dt, order=order, bidirectional=bidirectional)
        return data_filt
    else:
        if low_pass == high_pass:
            return data
        elif low_pass > high_pass:
            data_filt = np.zeros(data.shape)
            for i in range(data.shape[1]):
                data_filt[:, i] = butter_bandpass_filter(data[:, i], high_pass, low_pass, 1/dt, order=order, bidirectional=bidirectional)
            return data_filt
        else: # low_pass < high_pass
            data_filt = np.zeros(data.shape)
            for i in range(data.shape[1]):
                data_filt[:, i] = butter_bandstop_filter(data[:, i], low_pass, high_pass, 1/dt, order=order, bidirectional=bidirectional)
            return data_filt
