"""Data splitting utilities for time series data."""

import numpy as np
import torch
from tqdm.auto import tqdm


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

    Notes
    -----
    Delay blocks are ordered most recent to least recent: columns 0:N contain the
    most recent state, columns N:2N the next delay, and so on through the oldest.
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

def generate_train_and_test_sets(pts, seq_length, seq_spacing=1, train_percent=0.8, test_percent=0.05, split_by='time', dtype='torch.FloatTensor', delay_embedding_params=None, verbose=False, return_full_obs=False, condition=None, split_groups=None):
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
    return_full_obs : bool, optional
        When True and delay_embedding_params filters observed_indices, also
        store the full-dimensional (unfiltered) trajectories for each split
        in the returned trajs dict as ``trajs['train_trajs_full']``,
        ``trajs['val_trajs_full']``, and ``trajs['test_trajs_full']``.
        The same split indices are used, so sequences align with the
        corresponding filtered trajs.  Defaults to False.
    condition : np.ndarray or torch.Tensor, optional
        Per-trajectory condition tensor of shape ``(n_trajectories, condition_dim)``.
        When provided, each split's TimeSeriesDataset is built with the
        per-sequence condition (tiled across start_indices to match the
        sequence layout) so DataLoaders yield ``(batch, c)`` tuples. When
        None (default), datasets carry no condition and DataLoaders yield
        plain tensors. Only meaningful when ``split_by='trajectory'``;
        for ``split_by='time'`` every trajectory appears in every split,
        so the condition naturally propagates without extra wiring.
    split_groups : np.ndarray, optional
        Per-trajectory integer group ID of shape ``(n_trajectories,)``.
        When provided AND ``split_by='trajectory'``, train/val/test
        indices are picked independently within each group and then
        concatenated, so each group's trajectory ratio is preserved
        across splits (= conditions balanced across train/val/test for
        the multi-condition combined-loader use case). When None, the
        full set of trajectory indices is split as before.

    Returns
    -------
    tuple
        (train_dataset, val_dataset, test_dataset, trajs) where:
        - train_dataset, val_dataset, test_dataset are TimeSeriesDataset objects
        - trajs is a dict containing the full trajectories and indices for each
          split; when return_full_obs=True the dict also contains
          ``train_trajs_full``, ``val_trajs_full``, and ``test_trajs_full``
          with all observed dimensions.

    Raises
    ------
    ValueError
        If train_percent + test_percent > 1
        If split_by is 'trajectory' and there aren't enough trajectories
    """
    val_percent = 1 - train_percent - test_percent

    if train_percent + test_percent > 1:
        raise ValueError('train_percent + test_percent must be less than or equal to 1')

    # Keep a reference to the full-dimensional data before any obs filtering.
    pts_full = pts

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

        if split_groups is not None:
            # Balanced split: pick train/test/val indices independently
            # within each group, then concatenate. Each group's
            # train/val/test ratio matches the global train/val/test ratio,
            # so the condition mix is identical across splits.
            split_groups_arr = np.asarray(split_groups)
            if split_groups_arr.shape[0] != pts.shape[0]:
                raise ValueError(
                    f"split_groups length ({split_groups_arr.shape[0]}) must "
                    f"match number of trajectories ({pts.shape[0]})"
                )
            train_parts: list[np.ndarray] = []
            test_parts: list[np.ndarray] = []
            val_parts: list[np.ndarray] = []
            for g in np.unique(split_groups_arr):
                g_inds = np.where(split_groups_arr == g)[0]
                n_g = g_inds.size
                if n_g < 3:
                    raise ValueError(
                        f"split_groups: group {g!r} has only {n_g} "
                        "trajectories; need at least 3 for train/val/test."
                    )
                g_train = np.random.choice(g_inds, int(train_percent * n_g), replace=False)
                g_remaining = np.array([i for i in g_inds if i not in g_train])
                g_test = np.random.choice(g_remaining, int(test_percent * n_g), replace=False)
                g_val = np.array([i for i in g_remaining if i not in g_test])
                train_parts.append(g_train)
                test_parts.append(g_test)
                val_parts.append(g_val)
            train_inds = np.concatenate(train_parts)
            test_inds = np.concatenate(test_parts)
            val_inds = np.concatenate(val_parts)
        else:
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

        if return_full_obs:
            train_trajs_full_raw = pts_full[train_inds]  # (n_train, T, D_full)
            val_trajs_full_raw = pts_full[val_inds]  # (n_val, T, D_full)
            test_trajs_full_raw = pts_full[test_inds]  # (n_test, T, D_full)

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

        if return_full_obs:
            train_trajs_full_raw = pts_full[:, np.arange(0, int(train_percent*pts_full.shape[1]))]  # (n_traj, T_train, D_full)
            val_trajs_full_raw = pts_full[:, np.arange(int(train_percent*pts_full.shape[1]), int((train_percent + val_percent)*pts_full.shape[1]))]  # (n_traj, T_val, D_full)
            test_trajs_full_raw = pts_full[:, np.arange(int((train_percent + val_percent)*pts_full.shape[1]), pts_full.shape[1])]  # (n_traj, T_test, D_full)

    # Per-sequence condition (tiled to match the (block × n_trajs) layout
    # of the *_examples arrays). Mirrors the inner loops above:
    #   trajectory split: examples[i*n_split : (i+1)*n_split] holds
    #     trajectory order [0, 1, ..., n_split-1] at start_indices[i].
    #     So per-sequence cond = tile(cond[split_inds], len(start_indices)).
    #   time split: examples[i*n_trajs : (i+1)*n_trajs] holds the same
    #     trajectory set at start_indices_split[i].
    #     So per-sequence cond = tile(cond, len(start_indices_split)).
    train_cond_t = val_cond_t = test_cond_t = None
    if condition is not None:
        cond_arr = condition.detach().cpu().numpy() if isinstance(condition, torch.Tensor) else np.asarray(condition)
        if cond_arr.ndim != 2 or cond_arr.shape[0] != pts.shape[0]:
            raise ValueError(
                f"condition must have shape (n_trajectories={pts.shape[0]}, "
                f"condition_dim), got {cond_arr.shape}"
            )
        if split_by == 'trajectory':
            train_cond_per_seq = np.tile(cond_arr[train_inds], (len(start_indices), 1))
            val_cond_per_seq = np.tile(cond_arr[val_inds], (len(start_indices), 1))
            test_cond_per_seq = np.tile(cond_arr[test_inds], (len(start_indices), 1))
        elif split_by == 'time':
            train_cond_per_seq = np.tile(cond_arr, (len(start_indices_train), 1))
            val_cond_per_seq = np.tile(cond_arr, (len(start_indices_val), 1))
            test_cond_per_seq = np.tile(cond_arr, (len(start_indices_test), 1))
        else:
            raise ValueError(f"condition not supported for split_by={split_by!r}")
        train_cond_t = torch.from_numpy(train_cond_per_seq).type(dtype)
        val_cond_t = torch.from_numpy(val_cond_per_seq).type(dtype)
        test_cond_t = torch.from_numpy(test_cond_per_seq).type(dtype)

    train_dataset = TimeSeriesDataset(torch.from_numpy(train_examples).type(dtype), condition=train_cond_t)
    val_dataset = TimeSeriesDataset(torch.from_numpy(val_examples).type(dtype), condition=val_cond_t)
    test_dataset = TimeSeriesDataset(torch.from_numpy(test_examples).type(dtype), condition=test_cond_t)

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
        test_inds=test_inds,
    )

    # Per-trajectory condition slices for downstream analytics. These are
    # PER-TRAJECTORY (NOT per-sequence-window like the dataset condition
    # above) — same first-axis length as trajs['*_trajs'].sequence. The
    # analytics pipeline (run_analytics) uses these to compute per-condition
    # Lyapunov spectra for the chosen run.
    if condition is not None:
        cond_arr_for_trajs = condition.detach().cpu().numpy() if isinstance(condition, torch.Tensor) else np.asarray(condition)
        if split_by == 'trajectory':
            trajs['train_condition'] = cond_arr_for_trajs[train_inds]
            trajs['val_condition'] = cond_arr_for_trajs[val_inds]
            trajs['test_condition'] = cond_arr_for_trajs[test_inds]
        elif split_by == 'time':
            # All trajectories appear in every split.
            trajs['train_condition'] = cond_arr_for_trajs
            trajs['val_condition'] = cond_arr_for_trajs
            trajs['test_condition'] = cond_arr_for_trajs

    if return_full_obs:
        for key, raw in [
            ('train_trajs_full', train_trajs_full_raw),
            ('val_trajs_full', val_trajs_full_raw),
            ('test_trajs_full', test_trajs_full_raw),
        ]:
            if isinstance(raw, np.ndarray):
                raw = torch.from_numpy(raw).type(dtype)
            trajs[key] = TimeSeriesDataset(raw)

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
    Dataset class for time series data with optional per-sample condition.

    Parameters
    ----------
    sequence : torch.Tensor
        Input time series data of shape ``(n_sequences, seq_length, n_dims)``.
    condition : torch.Tensor or None, optional
        Per-sample condition of shape ``(n_sequences, condition_dim)``. When
        provided, ``__getitem__`` returns a ``(traj, c)`` tuple; when None
        (default) it returns the trajectory tensor alone (back-compat).
    """

    def __init__(self, sequence, condition=None):
        self.sequence = sequence
        if condition is not None and len(condition) != len(sequence):
            raise ValueError(
                f"condition length ({len(condition)}) must match sequence "
                f"length ({len(sequence)})."
            )
        self.condition = condition

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, index):
        if self.condition is None:
            return self.sequence[index]
        return self.sequence[index], self.condition[index]


def collate_with_optional_condition(samples):
    """Collate that handles both ``Tensor`` and ``(Tensor, c)`` items.

    - If items are plain tensors → return a single stacked batch tensor
      (identical to PyTorch's default collate for tensor-only datasets).
    - If items are ``(traj, c)`` tuples → return ``(batch, c)`` where both
      sides are stacked. Use this collate on a DataLoader whose dataset
      may or may not carry per-sample conditions.
    """
    first = samples[0]
    if isinstance(first, (tuple, list)) and len(first) == 2:
        trajs, conds = zip(*samples)
        return torch.stack(trajs, dim=0), torch.stack(conds, dim=0)
    return torch.stack(samples, dim=0)
