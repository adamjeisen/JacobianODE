# %% [markdown]
# # Diagnostic: noise amplification σ(T) vs prediction horizon, for many n_delays
#
# Uses the same quantity the `loss_amplification` regularizer already
# computes — the ratio of neighbor-variance T steps ahead to the
# neighbor pairwise scatter in embedding space:
#
#     σ(T) = E_k(T) / ε_k
#
# For a chaotic time series, σ(T) grows roughly exponentially with T
# (rate ≈ largest Lyapunov exponent) and plateaus at the attractor
# variance divided by the squared initial neighborhood size. The plateau
# marks the horizon past which prediction degenerates — i.e. a
# data-driven ``prediction_steps`` upper bound.
#
# This diagnostic sweeps `n_delays ∈ {5, 10, 15, ..., 50}`, holding k=10
# neighbors and the source time series fixed. We can then see:
#   - which n_delays gives the cleanest exponential ramp,
#   - where σ(T) plateaus at each n_delays (predictability horizon),
#   - whether higher n_delays at fixed noise delays the plateau.

# %%
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

from JacobianODE.jacobians.checkpoints.loader import load_run
from JacobianODE.jacobians import create_dataloaders

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def amplification_curve(
    embedding: torch.Tensor,   # (N, emb_dim)
    data: torch.Tensor,        # (N, D)  — "ahead" data; 1D obs = (N, 1)
    n_neighbors: int = 10,
    max_T: int = 300,
    temporal_exclusion: int = 10,
    epsilon: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Returns dict with:
        E_k:    (max_T,) neighbor forecast-residual variance at T ahead (mean over refs)
        eps_k:  scalar, mean neighbor pairwise-squared-distance at T=0 (over refs)
        sigma:  (max_T,) = E_k / eps_k, the original loss_amplification quantity
    """
    K = n_neighbors
    N_total = embedding.shape[0]
    assert max_T < N_total, f"need max_T < N_total={N_total}, got max_T={max_T}"

    N = N_total - max_T                    # reference set: those with max_T ahead
    emb_ref = embedding[:N]
    data_full = data                       # (N_total, D) — used at shifted indices

    with torch.no_grad():
        dists = torch.cdist(emb_ref, emb_ref)
        t_idx = torch.arange(N, device=dists.device)
        mask = (t_idx[:, None] - t_idx[None, :]).abs() < temporal_exclusion
        dists = dists.masked_fill(mask, float("inf"))
        _, indices = torch.topk(dists, K, largest=False)  # (N, K)

    neighbors = emb_ref[indices]                            # (N, K, emb_dim)
    diff = neighbors.unsqueeze(2) - neighbors.unsqueeze(1)  # (N, K, K, emb_dim)
    sq_pair = diff.pow(2).sum(dim=-1)
    eps_k_per = sq_pair.sum(dim=(1, 2)) / (K * (K - 1) * embedding.shape[-1])  # (N,)
    eps_k_scalar = eps_k_per.mean()

    E_k = torch.zeros(max_T)
    sigma = torch.zeros(max_T)
    for t in range(max_T):
        shifted = indices + t                                # (N, K)
        nbr_data_t = data_full[shifted]                      # (N, K, D)
        mu = nbr_data_t.mean(dim=1, keepdim=True)
        E_kT_per = (nbr_data_t - mu).pow(2).mean(dim=1).sum(dim=-1)  # (N,)
        E_k[t] = E_kT_per.mean()
        sigma[t] = (E_kT_per / (eps_k_per + epsilon)).mean()
    return dict(E_k=E_k, eps_k=eps_k_scalar, sigma=sigma)


def one_d_from_test_traj(trajs):
    """Extract the 1D scalar observation time series from the test set.

    test_trajs.sequence has shape (n_traj, T_emb, n_delays); index 0 in
    the last axis is "most recent" per the delay-embedding convention.
    Concatenate across trajectories for more data; return a 1-D array.
    """
    seq = torch.as_tensor(trajs["test_trajs"].sequence).float()
    # (n_traj, T_emb, n_delays) -> take the most-recent channel -> (n_traj, T_emb)
    x1d = seq[..., 0]
    # Flatten into one continuous time series for NN stats. Don't
    # concatenate across trajs because time boundaries are fake; just
    # use the first trajectory.
    return x1d[0].numpy()


def build_delay_embedding(x1d: np.ndarray, n_delays: int) -> np.ndarray:
    """Return (T_emb, n_delays) with row t = (x_t, x_{t-1}, ..., x_{t-n_delays+1})."""
    T = len(x1d) - n_delays + 1
    emb = np.empty((T, n_delays), dtype=np.float32)
    for i in range(n_delays):
        emb[:, i] = x1d[n_delays - 1 - i : n_delays - 1 - i + T]
    return emb


# %% ── Load a representative dataset at obs_noise=0.05 ──────────────────────
# Any run in the obsnoise005 ndelays sweep works — we only want the
# observation data, not the model.
PROJECT = "JacobianODE/Lorenz_INDpartial_NDsweep_D1_NormTrue__JacobianODE"
RUN_ID = "16s8uli1"  # n_delays=50 at obs_noise=0.05
print(f"loading data via run {RUN_ID} (for its test trajectories, not the model)…")
run, cfg, eq, dt, values, _, _, _, _, _ = load_run(
    PROJECT, run_id=RUN_ID, generate_data=True, verbose=False,
)
_, _, _, trajs = create_dataloaders(
    cfg, values, verbose=False, return_full_obs=True,
)
x1d_005 = one_d_from_test_traj(trajs)
print(f"obs_noise={cfg.data.postprocessing.obs_noise}, "
      f"1D series length: {len(x1d_005)}")

# And the same at obs_noise=0.01 for comparison.
PROJECT_001 = "JacobianODE/Lorenz_INDpartial_N25_D1_NormTrue_T3__JacobianODE"
RUN_001 = "39prin7v"
print(f"\nloading obsnoise001 data via run {RUN_001}…")
run001, cfg001, _, dt001, values001, _, _, _, _, _ = load_run(
    PROJECT_001, run_id=RUN_001, generate_data=True, verbose=False,
)
_, _, _, trajs001 = create_dataloaders(
    cfg001, values001, verbose=False, return_full_obs=True,
)
x1d_001 = one_d_from_test_traj(trajs001)
print(f"obs_noise={cfg001.data.postprocessing.obs_noise}, "
      f"1D series length: {len(x1d_001)}")

# %% ── Sweep over n_delays and compute σ(T) curve for each ─────────────────
delays = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
MAX_T = 300
K = 10
TEMPORAL_EXCLUSION = 10

def sweep(x1d, label, obs_noise):
    print(f"\n=== {label}: length={len(x1d)} ===")
    var_x = float(np.var(x1d))
    noise_floor = obs_noise ** 2 / var_x   # relative to Var(x)
    print(f"  Var(x_1d)={var_x:.4g}, obs_noise²/Var(x)={noise_floor:.4g}")
    results = {"var_x": var_x, "noise_floor_rel": noise_floor, "curves": {}}
    for n_d in delays:
        emb = build_delay_embedding(x1d, n_d)
        emb_t = torch.from_numpy(emb).float().to(DEVICE)
        data_t = torch.from_numpy(emb[:, [0]]).float().to(DEVICE)  # (T_emb, 1)
        out = amplification_curve(
            emb_t, data_t,
            n_neighbors=K, max_T=MAX_T,
            temporal_exclusion=TEMPORAL_EXCLUSION,
        )
        out = {k: v.cpu().numpy() for k, v in out.items()}
        results["curves"][n_d] = out
        nmse = out["E_k"] / var_x                           # normalized forecast variance
        sigma = out["sigma"]
        print(f"  n_d={n_d:>2}  E_k(1)/Var={nmse[1]:.3g}  E_k(∞)/Var≈{nmse[-20:].mean():.3g}  "
              f"σ(1)={sigma[1]:.3g}  σ(∞)≈{sigma[-20:].mean():.3g}")
    return results


data_005 = sweep(x1d_005, "obs_noise=0.05", 0.05)
data_001 = sweep(x1d_001, "obs_noise=0.01", 0.01)

# %% ── Plot ────────────────────────────────────────────────────────────────
# Two rows × two columns.
#
#   Row 0 — E_k(T) / Var(x_1d): absolute normalized forecast-residual variance.
#           This is the quantity MSE-loss training actually sees in the limit
#           of perfectly-learned dynamics (it's the Bayes-optimal prediction
#           error given the initial-state observability). Ranges from the
#           noise floor (σ_noise² / Var(x)) at T=0 to ≈1 at saturation (no
#           information beyond the mean). Natural "training useful" band:
#                 noise_floor  <  E_k/Var  <  ~0.3–0.5.
#
#   Row 1 — σ(T) = E_k(T) / ε_k: the original FNN-regularizer amplification.
#           Useful for Lyapunov-shape diagnostics (exponential rate of
#           information loss) but NOT for picking prediction_steps since ε_k
#           itself scales with obs noise — "σ=1" is a moving target.
#
# Columns: noise=0.01 | noise=0.05.
fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
cmap = plt.cm.viridis

panels = [(data_001, "obs_noise = 0.01"), (data_005, "obs_noise = 0.05")]

# Horizon thresholds on the E_k/Var axis:
UPPER_BAND = 0.3      # "signal still worth learning": E_k(T)/Var(x) < 0.3
LOWER_BAND = None     # set per-panel to noise_floor

for col, (d, label) in enumerate(panels):
    ax_top = axes[0, col]
    ax_bot = axes[1, col]
    var_x = d["var_x"]
    noise_floor = d["noise_floor_rel"]

    for i, n_d in enumerate(delays):
        color = cmap(i / (len(delays) - 1))
        E_k = d["curves"][n_d]["E_k"]
        sigma = d["curves"][n_d]["sigma"]

        nmse = E_k / var_x
        ax_top.plot(np.arange(MAX_T), nmse, color=color, lw=1.4,
                    label=f"n_delays={n_d}")
        # Marker where E_k/Var first exceeds UPPER_BAND
        if (nmse >= UPPER_BAND).any():
            T_upper = int(np.argmax(nmse >= UPPER_BAND))
            ax_top.plot([T_upper], [nmse[T_upper]], 'o', color=color, ms=5)

        ax_bot.plot(np.arange(MAX_T), sigma, color=color, lw=1.4,
                    label=f"n_delays={n_d}")

    # Row 0 formatting: absolute normalized MSE
    ax_top.axhline(noise_floor, color="b", ls="--", lw=1.0, alpha=0.7,
                   label=f"noise floor = $\\sigma_{{noise}}^2 / $Var$(x)$ = {noise_floor:.4g}")
    ax_top.axhline(UPPER_BAND, color="r", ls="--", lw=1.0, alpha=0.7,
                   label=f"upper band = {UPPER_BAND}")
    ax_top.axhline(1.0, color="k", ls=":", lw=0.8, alpha=0.5,
                   label="useless (= Var(x), pred = mean)")
    ax_top.set_yscale("log")
    ax_top.set_ylim(max(noise_floor * 0.3, 1e-5), 2.0)
    ax_top.set_title(f"{label}  —  $E_k(T) / $Var$(x_{{1d}})$  (MSE-relevant)")
    ax_top.set_ylabel(r"$E_k(T)\,/\,$Var$(x_{1d})$")
    ax_top.grid(True, which="both", alpha=0.3)

    # Row 1 formatting: σ
    ax_bot.axhline(1.0, color="k", ls="--", lw=0.6, alpha=0.5,
                   label=r"$\sigma = 1$")
    ax_bot.set_yscale("log")
    ax_bot.set_title(f"{label}  —  $\\sigma(T) = E_k(T) / \\epsilon_k$  (amplification)")
    ax_bot.set_ylabel(r"$\sigma(T)$")
    ax_bot.set_xlabel("Prediction horizon T (obs steps)")
    ax_bot.grid(True, which="both", alpha=0.3)

axes[0, 0].legend(loc="lower right", fontsize=7, ncol=2)
axes[0, 1].legend(loc="lower right", fontsize=7, ncol=2)

fig.suptitle(
    "Forecast-residual variance $E_k(T)$ vs prediction horizon, across delay-embedding sizes.\n"
    "Top row: normalized by Var($x_{1d}$) — the quantity MSE sees. Useful training band: between noise floor (blue) and upper band (red).\n"
    "Bottom row: the amplification $\\sigma = E_k/\\epsilon_k$ — Lyapunov-rate diagnostic; \"$\\sigma=1$\" is a moving target with noise.",
    y=1.00,
)
fig.tight_layout()

out = Path("/home/adameisen/Documents/code/JacobianODE/_notebook/_sandbox/"
           "amplification_vs_delays.png")
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"Saved {out}")

# Horizon table in E_k(T)/Var(x) units — the MSE-relevant axis.
#
#   τ_10% = first T with E_k(T)/Var(x) ≥ 0.10  (forecast error starts to matter)
#   τ_30% = first T with E_k(T)/Var(x) ≥ 0.30  (recommended upper bound for training)
#   τ_50% = first T with E_k(T)/Var(x) ≥ 0.50  (still learnable but marginal)
def first_cross(sig, thresh):
    idx = np.argmax(sig >= thresh)
    return int(idx) if (sig >= thresh).any() else -1

print("\nHorizon table: prediction step T at which E_k(T)/Var(x) first crosses threshold.")
print(f"  {'n_delays':<8}"
      f"{'noise_floor01':<16}{'τ_10%':<8}{'τ_30%':<8}{'τ_50%':<8}"
      f"  |  "
      f"{'noise_floor05':<16}{'τ_10%':<8}{'τ_30%':<8}{'τ_50%':<8}")
for n_d in delays:
    nmse01 = data_001["curves"][n_d]["E_k"] / data_001["var_x"]
    nmse05 = data_005["curves"][n_d]["E_k"] / data_005["var_x"]
    t01_10 = first_cross(nmse01, 0.10)
    t01_30 = first_cross(nmse01, 0.30)
    t01_50 = first_cross(nmse01, 0.50)
    t05_10 = first_cross(nmse05, 0.10)
    t05_30 = first_cross(nmse05, 0.30)
    t05_50 = first_cross(nmse05, 0.50)
    nf01 = f"{nmse01[0]:.4g}"
    nf05 = f"{nmse05[0]:.4g}"
    print(f"  {n_d:<8}"
          f"{nf01:<16}{t01_10:<8}{t01_30:<8}{t01_50:<8}"
          f"  |  "
          f"{nf05:<16}{t05_10:<8}{t05_30:<8}{t05_50:<8}")
