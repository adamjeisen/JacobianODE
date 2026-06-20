"""Regenerate ``stability_and_control_damping_vs_coupling.{png,pdf}`` in
this directory.

Four panels, all plotting log trace of the reach / ctrl Gramians
(``log tr W``):

1. **Damping sweep.** 8-D 2-area damped oscillator, ``B = C = g·I_4``;
   vary target-area damping ``b`` with coupling fixed. As ``b → 0``
   (target less stable), reach log-tr ↑ while ctrl log-tr ↓.

2. **Coupling sweep.** Same 8-D system; vary ``g`` with damping fixed.
   Both reach and ctrl scale as ``2 log g + const`` (because
   ``BB^T = g^2 I``), giving the characteristic concave curve.

3. **Coupling intrinsic dimension PR(σ²_B), two paths.** Simpler 4-D
   diagonal target ``A_used = -diag(d_1,…,d_4)`` with leading eigenvector
   ``e_1``. ``B`` is diagonal with ``σ_1^2 = 0.5`` held fixed (so the
   projection ``‖v_1^T B‖^2`` on the leading eigenvector is fixed at
   0.5) and ``‖B‖_F^2 = 1`` held fixed. Two sweeps both span PR≈2 →
   PR≈3 and meet at the uniform-minor endpoint, but approach it from
   opposite ends of damping-space:

   * **slow path (solid)** — rank-2 mass on σ_2 (slowest minor, d=0.6)
     at PR≈2 → uniform at PR≈3. PR-increase moves mass *into* faster-
     damped minors: reach ↓, ctrl ↑.
   * **fast path (dashed)** — rank-2 mass on σ_4 (fastest minor, d=1.5)
     at PR≈2 → same uniform endpoint at PR≈3. PR-increase moves mass
     *out of* the fastest minor: reach ↑, ctrl ↓.

   The fan between the two paths is the point: PR is permutation-
   invariant over modes, so it does not by itself determine R/C —
   *which* mode is gaining mass does.

4. **Coupling projection on v_1.** Same 4-D diagonal target; ``B`` is
   diagonal with ``σ_2 = σ_3 = σ_4`` (uniform minor) and varies
   ``σ_1^2 = p^2`` from ~0 to ~1 with ``‖B‖_F^2 = 1`` held fixed.
   Reach log-tr ↑ as the projection energy on the slowest mode grows.
   PR co-varies along this sweep (1.1 → 3.3) since holding it exactly
   constant would shrink the sweep range substantially.

Run::

    cd <JacobianODE-repo>
    uv run --no-sync python _notebook/render_damping_vs_coupling.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))
from JacobianODE.control.gramians import compute_all_gramians      # noqa: E402

OUT_DIR = HERE       # write next to the script (= the _notebook dir)

# ── shared 2-area toy (panels 1 & 2) ───────────────────────────────────
T, dt = 1000, 0.01
w1, w2 = 1.0, 2.3
a_src = 0.8                       # source-area damping (fixed)
I4 = np.eye(4)


def _area_block(damp: float) -> np.ndarray:
    o = lambda d, w: np.array([[-d, w], [-w, -d]])
    Z = np.zeros((2, 2))
    return np.block([[o(damp, w1), Z], [Z, o(damp, w2)]])


def _full_A8(b: float, g: float) -> np.ndarray:
    """8×8 two-area system; ``b`` = target damping, ``g`` = inter-area
    coupling (B = C = g·I₄ in the pipeline-block convention)."""
    return np.block([[_area_block(a_src), g * I4],
                     [g * I4,             _area_block(b)]])


def _pair_A_to_B(A8: np.ndarray):
    sA, sB = slice(0, 4), slice(4, 8)
    return A8[sB, sB], A8[sB, sA], A8[sA, sB]            # A_used, B, C


def _logtrace(A_used: np.ndarray, B: np.ndarray,
              C: np.ndarray) -> tuple[float, float]:
    seq = lambda M: torch.tensor(M, dtype=torch.float64).reshape(
        1, 1, 4, 4).repeat(1, T, 1, 1)
    (_), (sr, sc, _) = compute_all_gramians(
        seq(A_used), seq(B), seq(C), dt=dt,
        return_sequences=False, return_spectrums=True, rescale=True)
    return (float(torch.logsumexp(sr, dim=-1).item()),
            float(torch.logsumexp(sc, dim=-1).item()))


# ── simpler 4-D target for panels 3 & 4 ────────────────────────────────
# Diagonal damping with distinct rates → leading eigenvector e_1 (slowest).
A_DIAG = -np.diag([0.3, 0.6, 1.0, 1.5])


def _B_from_sigmasq(sigsq: list[float]) -> np.ndarray:
    """B as a diagonal matrix with prescribed squared singular values.
    PR and projection-on-e_1 are direct functions of sigsq."""
    return np.diag(np.sqrt(np.maximum(sigsq, 0.0)))


def _PR(sigsq: list[float]) -> float:
    s = np.asarray(sigsq, float)
    return float(s.sum() ** 2 / (s ** 2).sum())


def main() -> None:
    # ── Panel 1: damping sweep (coupling fixed at g_cpl) ────────────────
    g_cpl = 0.25
    b_vals = np.geomspace(0.8, 0.02, 40)
    R_lt_d, C_lt_d = [], []
    for b in b_vals:
        r, c = _logtrace(*_pair_A_to_B(_full_A8(b, g_cpl)))
        R_lt_d.append(r); C_lt_d.append(c)
    R_lt_d, C_lt_d = np.asarray(R_lt_d), np.asarray(C_lt_d)

    # ── Panel 2: coupling sweep (damping fixed at b_fixed) ──────────────
    b_fixed = 0.4
    g_vals = np.linspace(0.02, 1.5, 50)
    R_lt_g, C_lt_g = [], []
    for g in g_vals:
        r, c = _logtrace(*_pair_A_to_B(_full_A8(b_fixed, g)))
        R_lt_g.append(r); C_lt_g.append(c)
    R_lt_g, C_lt_g = np.asarray(R_lt_g), np.asarray(C_lt_g)

    # ── Panel 3: PR(σ²_B) sweep, fix Frobenius and projection-on-v_1 ───
    # σ_1² = 0.5 (= ‖v_1^T B‖² with v_1 = e_1) held constant.
    # Two PR=2→PR=3 paths to demonstrate the mirror effect — PR alone
    # is permutation-invariant over minors, so different paths through
    # B-space at the *same* PR can drive R/C in opposite directions:
    #
    #   "slow path" — rank-2 mass on σ_2 (slowest minor, d=0.6) at s=1
    #                  → uniform across (σ_2,σ_3,σ_4) at s=0.
    #                  PR-increase pushes mass slow→fast.
    #   "fast path" — rank-2 mass on σ_4 (fastest minor, d=1.5) at s=1
    #                  → same uniform endpoint at s=0.
    #                  PR-increase pushes mass fast→slow.
    p2_p3 = 0.5
    s_vals = np.linspace(0.0, 1.0, 30)
    PR_p3, R_p3, C_p3 = [], [], []           # slow path
    PR_p3m, R_p3m, C_p3m = [], [], []        # fast path (mirror)
    for s in s_vals:
        rest = 1.0 - p2_p3                         # mass off the leading mode
        load = rest * (1.0 / 3.0 + 2.0 * s / 3.0)  # rank-2 mode at s=1
        spread = rest * ((1.0 - s) / 3.0)          # the other two minors
        # slow path: rank-2 mass on σ_2.
        sigsq_slow = [p2_p3, load, spread, spread]
        Bm_slow = _B_from_sigmasq(sigsq_slow)
        r, c = _logtrace(A_DIAG, Bm_slow, Bm_slow)
        PR_p3.append(_PR(sigsq_slow)); R_p3.append(r); C_p3.append(c)
        # fast path (mirror): rank-2 mass on σ_4 (permute σ_2 ↔ σ_4).
        sigsq_fast = [p2_p3, spread, spread, load]
        Bm_fast = _B_from_sigmasq(sigsq_fast)
        r, c = _logtrace(A_DIAG, Bm_fast, Bm_fast)
        PR_p3m.append(_PR(sigsq_fast)); R_p3m.append(r); C_p3m.append(c)
    PR_p3, R_p3, C_p3 = (np.asarray(x) for x in (PR_p3, R_p3, C_p3))
    PR_p3m, R_p3m, C_p3m = (np.asarray(x) for x in (PR_p3m, R_p3m, C_p3m))

    # ── Panel 4: projection-on-v_1 sweep, fix Frobenius (PR co-varies) ─
    # σ_2² = σ_3² = σ_4² = (1-p²)/3 (uniform minor); vary p² = σ_1².
    p2_vals = np.linspace(0.05, 0.95, 30)
    proj_p4, R_p4, C_p4, PR_p4 = [], [], [], []
    for p2 in p2_vals:
        rest = (1.0 - p2) / 3.0
        sigsq = [p2, rest, rest, rest]
        Bm = _B_from_sigmasq(sigsq)
        r, c = _logtrace(A_DIAG, Bm, Bm)
        proj_p4.append(p2); R_p4.append(r); C_p4.append(c)
        PR_p4.append(_PR(sigsq))
    proj_p4, R_p4, C_p4, PR_p4 = (np.asarray(x)
                                  for x in (proj_p4, R_p4, C_p4, PR_p4))

    # ── render 2×2 ──────────────────────────────────────────────────────
    C_REACH = "#D67037"     # orange
    C_CTRL = "#42627D"      # blue

    fig, ax = plt.subplots(2, 2, figsize=(9.0, 7.5))
    LABEL_FS = 15
    TICK_FS = 13
    LEG_FS = 13

    # Panel 1 — damping, twin y for the two scales.
    axR = ax[0, 0]
    axC = axR.twinx()
    axR.plot(b_vals, R_lt_d, "-o", ms=4, color=C_REACH, label="reach")
    axC.plot(b_vals, C_lt_d, "-s", ms=4, color=C_CTRL, label="ctrl")
    axR.set_xlabel("Target area damping  ($\\rightarrow$ 0 = less stable)",
                   fontsize=LABEL_FS)
    axR.set_ylabel(r"Reach $\log\,\mathrm{tr}\,\mathbf{W}_r$", color=C_REACH,
                   fontsize=LABEL_FS)
    axC.set_ylabel(r"Ctrl $\log\,\mathrm{tr}\,\mathbf{W}_c$", color=C_CTRL,
                   fontsize=LABEL_FS)
    axR.tick_params(axis="y", colors=C_REACH, labelsize=TICK_FS)
    axC.tick_params(axis="y", colors=C_CTRL, labelsize=TICK_FS)
    axR.tick_params(axis="x", labelsize=TICK_FS)
    axR.spines["top"].set_visible(False)
    axC.spines["top"].set_visible(False)
    axR.invert_xaxis()

    def _style_twin(axR_, axC_):
        """Common styling for a twin-y axis pair: green-tinted left y,
        blue-tinted right y, hide top spines, bumped tick labels."""
        axR_.set_ylabel(r"Reach $\log\,\mathrm{tr}\,\mathbf{W}_r$",
                        color=C_REACH, fontsize=LABEL_FS)
        axC_.set_ylabel(r"Ctrl $\log\,\mathrm{tr}\,\mathbf{W}_c$",
                        color=C_CTRL, fontsize=LABEL_FS)
        axR_.tick_params(axis="y", colors=C_REACH, labelsize=TICK_FS)
        axC_.tick_params(axis="y", colors=C_CTRL, labelsize=TICK_FS)
        axR_.tick_params(axis="x", labelsize=TICK_FS)
        axR_.spines["top"].set_visible(False)
        axC_.spines["top"].set_visible(False)

    # Panel 2 — coupling sweep, twin y. Reach + ctrl trace the same
    # 2*log(g)+const shape (because BB^T = g^2 I), so when both are
    # plotted at their natural scales the markers stack on top of each
    # other; bump reach's marker size so it shows through ctrl's.
    axR2 = ax[0, 1]
    axC2 = axR2.twinx()
    axR2.plot(g_vals, R_lt_g, "-o", ms=8, color=C_REACH, label="reach",
              zorder=2)
    axC2.plot(g_vals, C_lt_g, "-s", ms=4, color=C_CTRL, label="ctrl",
              zorder=3)
    axR2.set_xlabel("Inter-area coupling strength  $g$", fontsize=LABEL_FS)
    _style_twin(axR2, axC2)

    # Panel 3 — PR(σ²_B) sweep, twin y. Two paths to the same PR=3
    # endpoint: slow-loaded (σ_2, solid) and fast-loaded (σ_4, dashed).
    # The fan between the two solid/dashed pairs is the point — PR
    # alone doesn't determine R/C; the direction of mass movement does.
    axR3 = ax[1, 1]      # bottom-right (swapped with the projection panel)
    axC3 = axR3.twinx()
    axR3.plot(PR_p3, R_p3, "-o", ms=4, color=C_REACH)
    axC3.plot(PR_p3, C_p3, "-s", ms=4, color=C_CTRL)
    axR3.plot(PR_p3m, R_p3m, "--o", ms=4, color=C_REACH, mfc="none")
    axC3.plot(PR_p3m, C_p3m, "--s", ms=4, color=C_CTRL, mfc="none")
    axR3.set_xlabel(r"Coupling intrinsic dim.  PR$(\sigma^2_{\mathbf{B}})$",
                    fontsize=LABEL_FS)
    _style_twin(axR3, axC3)
    from matplotlib.lines import Line2D
    _path_proxies = [
        Line2D([0], [0], color="0.35", linestyle="-", marker="o", ms=4,
               label="incr. proj. to fast modes"),
        Line2D([0], [0], color="0.35", linestyle="--", marker="o", ms=4,
               mfc="none", label="incr. proj. to slow modes"),
    ]
    axR3.legend(handles=_path_proxies, loc="center left", fontsize=LEG_FS - 2,
                frameon=False, handlelength=2.4)

    # Panel 4 — projection-on-v_1 sweep, twin y.
    axR4 = ax[1, 0]      # bottom-left (swapped with the PR panel)
    axC4 = axR4.twinx()
    axR4.plot(proj_p4, R_p4, "-o", ms=4, color=C_REACH, label="reach")
    axC4.plot(proj_p4, C_p4, "-s", ms=4, color=C_CTRL, label="ctrl")
    axR4.set_xlabel(r"Projection on leading eigvec.  $\|\mathbf{v}_1^\top \mathbf{B}\|^2$",
                    fontsize=LABEL_FS)
    _style_twin(axR4, axC4)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "stability_and_control_damping_vs_coupling.png",
                dpi=200, bbox_inches="tight")
    fig.savefig(OUT_DIR / "stability_and_control_damping_vs_coupling.pdf",
                bbox_inches="tight")
    print(f"wrote {OUT_DIR / 'stability_and_control_damping_vs_coupling.png'}")
    print(f"wrote {OUT_DIR / 'stability_and_control_damping_vs_coupling.pdf'}")


if __name__ == "__main__":
    main()
