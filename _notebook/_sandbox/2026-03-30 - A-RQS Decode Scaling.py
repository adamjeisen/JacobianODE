import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # A-RQS Decode Speed Scaling Analysis

    **Date:** 2026-03-30

    Measures how decode (inverse) time scales with embedding dimension D for the
    `SplineAutoregressiveEncoder` (A-RQS) vs `CouplingEncoder` with spline coupling (C-RQS).

    The autoregressive inverse is O(D) sequential MADE passes per layer, while coupling
    is O(1) per layer. This notebook quantifies the actual wall-clock cost.
    """)
    return


@app.cell
def _():
    import torch
    import time
    import numpy as np
    import matplotlib.pyplot as plt

    from JacobianODE.fnn.coupling_flows import SplineAutoregressiveEncoder, CouplingEncoder

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    return (
        CouplingEncoder,
        SplineAutoregressiveEncoder,
        device,
        np,
        plt,
        time,
        torch,
    )


@app.cell
def _():
    # --- Configuration ---
    DIMS = [5, 10, 15, 20, 30, 40, 50, 75, 100]
    BATCH_SIZE = 32
    SEQ_LEN = 50
    N_LAYERS = 4
    HIDDEN_DIM = 128
    N_HIDDEN_LAYERS = 2
    NUM_BINS = 8
    WARMUP_ITERS = 3
    TIMING_ITERS = 10
    return (
        BATCH_SIZE,
        DIMS,
        HIDDEN_DIM,
        NUM_BINS,
        N_HIDDEN_LAYERS,
        N_LAYERS,
        SEQ_LEN,
        TIMING_ITERS,
        WARMUP_ITERS,
    )


@app.cell
def _(TIMING_ITERS, WARMUP_ITERS, device, np, time, torch):
    def time_fn(fn, warmup=WARMUP_ITERS, iters=TIMING_ITERS):
        """Time a function with CUDA synchronization."""
        for _ in range(warmup):
            fn()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        return np.mean(times), np.std(times)

    return (time_fn,)


@app.cell
def _(
    BATCH_SIZE,
    CouplingEncoder,
    DIMS,
    HIDDEN_DIM,
    NUM_BINS,
    N_HIDDEN_LAYERS,
    N_LAYERS,
    SEQ_LEN,
    SplineAutoregressiveEncoder,
    device,
    time_fn,
    torch,
):
    results = {'dims': [], 'ar_encode_mean': [], 'ar_encode_std': [], 'ar_decode_mean': [], 'ar_decode_std': [], 'cp_encode_mean': [], 'cp_encode_std': [], 'cp_decode_mean': [], 'cp_decode_std': []}
    for _D in DIMS:
        print(f'D={_D:3d} ... ', end='', flush=True)
        _ar_enc = SplineAutoregressiveEncoder(n_input=_D, n_layers=N_LAYERS, hidden_dim=HIDDEN_DIM, n_hidden_layers=N_HIDDEN_LAYERS, num_bins=NUM_BINS, use_actnorm=False, zero_init=True).to(device).eval()
        _cp_enc = CouplingEncoder(n_input=_D, n_coupling_layers=N_LAYERS, coupling_type='spline', hidden_dim=HIDDEN_DIM, n_hidden_layers=N_HIDDEN_LAYERS, num_bins=NUM_BINS, use_actnorm=False, zero_init=True).to(device).eval()
        x = torch.randn(BATCH_SIZE, SEQ_LEN, _D, device=device)
        with torch.no_grad():
            z_ar = _ar_enc.encode(x)
            z_cp = _cp_enc.encode(x)
            ar_enc_t = time_fn(lambda: _ar_enc.encode(x))
            cp_enc_t = time_fn(lambda: _cp_enc.encode(x))
            ar_dec_t = time_fn(lambda: _ar_enc.decode(z_ar))  # Build encoders
            cp_dec_t = time_fn(lambda: _cp_enc.decode(z_cp))
        results['dims'].append(_D)
        results['ar_encode_mean'].append(ar_enc_t[0])
        results['ar_encode_std'].append(ar_enc_t[1])
        results['ar_decode_mean'].append(ar_dec_t[0])
        results['ar_decode_std'].append(ar_dec_t[1])
        results['cp_encode_mean'].append(cp_enc_t[0])
        results['cp_encode_std'].append(cp_enc_t[1])
        results['cp_decode_mean'].append(cp_dec_t[0])
        results['cp_decode_std'].append(cp_dec_t[1])
        print(f'AR enc={ar_enc_t[0] * 1000.0:.1f}ms  dec={ar_dec_t[0] * 1000.0:.1f}ms | CP enc={cp_enc_t[0] * 1000.0:.1f}ms  dec={cp_dec_t[0] * 1000.0:.1f}ms | decode ratio={ar_dec_t[0] / cp_dec_t[0]:.1f}x')
        del _ar_enc, _cp_enc, x, z_ar, z_cp
        torch.cuda.empty_cache() if device.type == 'cuda' else None
    print('\nDone.')  # Time encode  # Time decode
    return (results,)


@app.cell
def _(BATCH_SIZE, N_LAYERS, SEQ_LEN, np, plt, results):
    dims = np.array(results['dims'])
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ax = axes[0]
    ax.errorbar(dims, np.array(results['ar_encode_mean']) * 1000.0, yerr=np.array(results['ar_encode_std']) * 1000.0, marker='o', label='A-RQS encode', color='C0', ls='--')
    # --- Panel 1: Absolute times ---
    ax.errorbar(dims, np.array(results['ar_decode_mean']) * 1000.0, yerr=np.array(results['ar_decode_std']) * 1000.0, marker='s', label='A-RQS decode', color='C0')
    ax.errorbar(dims, np.array(results['cp_encode_mean']) * 1000.0, yerr=np.array(results['cp_encode_std']) * 1000.0, marker='o', label='C-RQS encode', color='C1', ls='--')
    ax.errorbar(dims, np.array(results['cp_decode_mean']) * 1000.0, yerr=np.array(results['cp_decode_std']) * 1000.0, marker='s', label='C-RQS decode', color='C1')
    ax.set_xlabel('Embedding dimension D')
    ax.set_ylabel('Time (ms)')
    ax.set_title(f'Absolute timing (B={BATCH_SIZE}, T={SEQ_LEN}, L={N_LAYERS})')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax = axes[1]
    _ratio = np.array(results['ar_decode_mean']) / np.array(results['cp_decode_mean'])
    ax.plot(dims, _ratio, 'ko-', markersize=6)
    ax.axhline(1, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('Embedding dimension D')
    ax.set_ylabel('A-RQS / C-RQS decode time')
    ax.set_title('Decode slowdown factor')
    ax.grid(True, alpha=0.3)
    ax = axes[2]
    ax.loglog(dims, np.array(results['ar_decode_mean']) * 1000.0, 'C0s-', label='A-RQS decode')
    ax.loglog(dims, np.array(results['cp_decode_mean']) * 1000.0, 'C1s-', label='C-RQS decode')
    # --- Panel 2: Decode ratio ---
    d0, t0 = (dims[0], results['ar_decode_mean'][0] * 1000.0)
    ax.loglog(dims, t0 * (dims / d0) ** 1, 'k:', alpha=0.4, label='O(D)')
    ax.loglog(dims, t0 * (dims / d0) ** 2, 'k--', alpha=0.4, label='O(D$^2$)')
    ax.set_xlabel('Embedding dimension D')
    ax.set_ylabel('Decode time (ms)')
    ax.set_title('Scaling (log-log)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    # --- Panel 3: Log-log to check scaling ---
    plt.savefig('_sandbox/arqs_decode_scaling.png', dpi=150, bbox_inches='tight')
    plt.show()
    # Reference lines
    print('Saved: _sandbox/arqs_decode_scaling.png')
    return


@app.cell
def _(results):
    # --- Summary table ---
    print(f'{'D':>5}  {'AR enc':>10}  {'AR dec':>10}  {'CP enc':>10}  {'CP dec':>10}  {'dec ratio':>10}')
    print('-' * 65)
    for i, _D in enumerate(results['dims']):
        _ar_enc = results['ar_encode_mean'][i] * 1000.0
        ar_dec = results['ar_decode_mean'][i] * 1000.0
        _cp_enc = results['cp_encode_mean'][i] * 1000.0
        cp_dec = results['cp_decode_mean'][i] * 1000.0
        _ratio = ar_dec / cp_dec
        print(f'{_D:5d}  {_ar_enc:8.2f}ms  {ar_dec:8.2f}ms  {_cp_enc:8.2f}ms  {cp_dec:8.2f}ms  {_ratio:8.1f}x')
    return


if __name__ == "__main__":
    app.run()
