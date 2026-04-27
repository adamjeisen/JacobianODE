# Phase 1 — hypotheses for faster JacobianODE training

**Read first**: this document is *evidence-based hypothesis generation*, not
validation. Each hypothesis below is supported by W&B archaeology and (where
noted) sandbox per-batch micro-pilots, but **none has been validated end-to-end
on the question that actually matters** ("does it pick the same sweep winners
on engaging with similar Jacobian / Lyapunov accuracy?"). Validation requires
a real engaging sweep — it is Phase 2 work, not done here.

## Source data

Three reference groups, all under entity `JacobianODE`:

| slug | project | group | n_runs | era |
|---|---|---|---|---|
| `lorenz_current` | `Lorenz_INDpartial_NDInitSweep_autodim_D1_NormTrue__JacobianODE` | `lorenz_partial_additive_splitmode_p30_obsnoise005_top3nd_init15_autodim__lc_sweep` | 21 | latent (current) |
| `lorenz_current_obsnoise001` | same | `lorenz_partial_additive_splitmode_p30_obsnoise001_top3nd_init15_autodim__lc_sweep` | 21 | latent (current) |
| `wmtask_vanilla_old` | `WMTask_identity_encoder_verification` | `wmtask_vanilla_mse_p30__lc_sweep` | 9 | vanilla (old paper era) |

All histories cached at
`JacobianODE/perf-experiment/analysis/cache/<slug>__{configs,history}.pkl.gz`.
All analysis scripts at `JacobianODE/perf-experiment/analysis/q*.py`.
Plots at `JacobianODE/perf-experiment/analysis/figures/`.

## Per-question findings

### Q1: When do training-loss terms saturate?

For `lorenz_current` (n=21, median 109 epochs/run):
- `train/recon_loss` reaches within 5 % of its run-final value at epoch **52** (median)
- `train/latent_pred_loss` at epoch **27**
- `train/loop_closure_loss` at epoch **28**
- `train/trajectory_loss` at epoch **25**

Train losses look "saturated" by epoch 25–50, but **val loss keeps improving** —
median epoch at which `trajectory val_loss` is within 5 % of final-best is
**98 of 109 (90 %)**. So early-epoch train-loss flatness is misleading; the model
*is* still improving. (See `analysis/q1q2_summary.md`.)

### Q2: How much walltime is "past best"?

Median walltime spent **after** best `trajectory val_loss`:
- `lorenz_current`: **3.4 %**
- `lorenz_current_obsnoise001`: **1.9 %**
- `wmtask_vanilla_old`: **8.2 %**

ES with patience=5 + percent_thresh=1 % is already lean. **There is essentially
no "wasted training at the end" to recover.** The slowness is "long, smooth val
descent across many epochs," not "trains too long after plateau."

### Q2′: Cost / quality curve (this is the real story)

If we stopped at fraction *f* of total runtime, here's how much worse the
best-so-far val loss is vs run-final, median across runs:

| f | lorenz_current | lorenz_current_obsnoise001 | wmtask_vanilla_old |
|---|---|---|---|
| 0.25 | +155 % | +963 % | +435 % |
| 0.50 | +50 % | +131 % | +91 % |
| 0.75 | +19 % | +43 % | +43 % |
| 0.90 | +7 % | +16 % | +21 % |

Walltime to first reach within tol of run-final-best (median, in units of
total-runtime fraction):

| tol | lorenz_current | lorenz_current_obsnoise001 |
|---|---|---|
| 5 % | 91 % of runtime | 98 % of runtime |
| 10 % | 88 % | 96 % |
| 20 % | 73 % | 87 % |

**Takeaway**: descent is smooth and long-tailed. **There is no early "knee"
to exploit per single run.** Walltime savings have to come from
(a) cheaper per-epoch cost, (b) cheaper sweeps via early winner-selection, or
(c) accepting slightly worse final loss (at quantified cost).

### Q3: Are sweep winners separable early?

Spearman rank correlation between best-so-far val at epoch *K* and run-final
best val:

| group | spearman ≥ 0.7 first reached | spearman ≥ 0.9 first reached | top-3 stable |
|---|---|---|---|
| `lorenz_current` (obs_noise=0.05) | epoch 10 (9 % of runtime) | epoch 58 (52 %) | 1 of 3 by epoch 11; never reaches 3/3 |
| `lorenz_current_obsnoise001` (obs_noise=0.01) | epoch 13 (12 %) | **epoch 15 (13 %)** | **3 of 3 stable by epoch 11** |
| `wmtask_vanilla_old` | epoch 0 | epoch 0 | 3 of 3 by epoch 1 |

This is the **biggest structural opportunity** in the data: in the obs_noise=0.01
sweep, the top-3 configurations are 100 % stable by epoch 11 (out of ~110).
In obs_noise=0.05 the rank order is roughly stable but the very top
reshuffles slightly, so a top-1 selection at epoch 30 isn't perfectly safe —
but a top-half cull at epoch 30 would be.

(See `analysis/q3_winner_sep_*.csv`. Files exist for all 3 groups.)

### Q4: Latent vs vanilla curve shape

Both eras show smooth, monotone, log-linear-ish descent
(see `figures/q4_val_curves_by_era.png`).
- `wmtask_vanilla_old`: ~12 epochs total, ~215 s/epoch, ~43 min total
- `lorenz_current`: ~109 epochs total, ~94 s/epoch, ~2.9 h total
- `lorenz_current_obsnoise001`: ~105 epochs total, ~99 s/epoch, ~2.9 h total

The new latent runs aren't *slower* per epoch — they're ~2× faster than
old runs per epoch. They're slower *because they go ~9× more epochs*. The
question is whether those extra epochs are buying real quality.

Per-optimizer-step speed (engaging cluster, not sandbox):
- old wmtask_vanilla: ~1.7 s/optimizer-step (215 s × 1/(500 batches × 1/4 accum))
- new lorenz_current: ~0.47 s/optimizer-step (94 s × 1/200 batches)
The new step is faster (smaller models, fewer batches per epoch), but the
schedule asks for much more.

### Q5: Loss terms that may not be earning their cost

**This is the headline finding.**

In every single sweep group examined, runs with **`loop_closure_weight = 0`**
(or 1e-6, effectively zero) tie or beat all other LC weights at run-final
val loss:

`lorenz_current` (obs_noise=0.05) — n=3 per bin, all seed=42:

| `loop_closure_weight` | median final best val |
|---|---|
| **0** | **0.00565** |
| 1e-6 | 0.00550 |
| 1e-5 | 0.00562 |
| 1e-4 | 0.00648 |
| 1e-3 | 0.00596 |
| 1e-2 | 0.00782 |
| 1e-1 | 0.00755 |

`lorenz_current_obsnoise001` (obs_noise=0.01):

| `loop_closure_weight` | median final best val |
|---|---|
| **0** | **0.00059** |
| 1e-6 | 0.00067 |
| 1e-5 | 0.00066 |
| 1e-4 | 0.00063 |
| 1e-3 | 0.00080 |
| 1e-2 | 0.00117 |
| 1e-1 | 0.00221 |

`wmtask_vanilla_old`: same shape — `loop_closure_weight ∈ {0, 1e-6, 1e-5}`
all give ~0.0098–0.0102, while higher weights monotonically degrade up to 0.118
at weight=10.

In `lorenz_current_obsnoise001` the between-bin / within-bin std ratio is
**5.06×** — i.e. the sweep IS picking up real signal — but the signal is
"loop_closure weight should be ≤ 1e-4." The lower-LC runs *also descend
faster* per epoch (see `figures/q5b_lc_curves_lorenz_current_obsnoise001.png`):
high-weight (≥ 1e-2) runs visibly lag the low-weight curves at every epoch.

**For non-NeuralODE models (the latent-coupling case), `loop_closure_model_step`
runs in `training_step` regardless of `loop_closure_weight`** (lightning_base.py:670:
the gating is by `loop_closure_training`, not by weight). With weight=0 the term
contributes zero gradient but pays full forward-pass cost — pure waste.

### Q6: Other things worth flagging

**alpha_teacher_forcing**: anneals from ~0.97 → 0.05 over training, bottoming
out at epoch ~108 (median). This means the model only fully experiences hard
free-running rollouts in the last ~5–10 epochs of training. Any aggressive
epoch-cap proposal needs to also re-scale this schedule (currently tied to
training length implicitly via `gamma_teacher_forcing` decay) or risk training
a model that's never seen its eventual evaluation regime. (`alpha_validation`
is set to 0 by default, so val *evaluates* with no teacher forcing — this is
the behavioural mismatch I'm worried about.)

**Cosine LR schedule**: configured for `T_max = trainer.max_epochs = 200` even
though runs end at ~109. So LR is roughly halfway through the cosine when ES
fires. If we cap max_epochs lower, LR will be more aggressively annealed at
the end — possibly compensating for shorter training. Possibly making it worse.

**`encoder_warmup_epochs` / `dynamics_warmup_epochs`**: not in the cfg in any
of the runs sampled. Either both default to 0 or both are unset.

## Hypotheses

Ranked by *expected impact × confidence in evidence × cheapness to test*.

### H1 — Disable `loop_closure_training` entirely (or ship `loop_closure_weight=0` as default and **gate the forward pass on weight > 0**)

**Confidence**: HIGH. The W&B sweeps directly answer this question with 21+
runs each across two obs-noise levels. lc_w=0 ties or beats every other value
on final val loss in all sweeps. The supposed regularization benefit isn't
showing up empirically.

**Wall-clock impact (sandbox-measured per-batch)**:
- lorenz: **−13.4 %** per batch
  (1.040 s → 0.901 s, baseline run trace
  `runs/baseline-lorenz-N80-seed42.json` vs `runs/H1-no-lc-lorenz.json`)
- wmtask: **−13.3 %** per batch
  (0.967 s → 0.838 s, `runs/H1-no-lc-wmtask.json`)

This is the per-batch saving on the sandbox GPU; engaging numbers will
be in the same ballpark since the saving comes from skipping forward+backward
through the n_loop_pts=45 inner-integration loop, not from kernel-launch
overhead specifically.

**Risk to the 3 quality axes**:
- (1) trajectory val_loss: zero — already shown to be *better* without LC.
- (2) Jacobian / Lyapunov: **need to check.** Loop closure was originally
  motivated by enforcing path-independence of the Jacobian field. The
  sweep shows it doesn't help val loss, but val loss doesn't directly
  measure Jacobian-field consistency. There may be a *Jacobian R²* or
  *Lyapunov spectrum* metric where LC is the only thing keeping the
  Jacobians honest. **W&B has `train/jac_r2_score` and `val/...` metrics
  in the older era; the newer era has `train/jac_norm` but I didn't find
  a per-run jac_r2_score in this group's history.** I'd want to compare
  Jacobian-quality-at-final between LC=0 and LC>0 runs before greenlighting.
- (3) Sweep winner selection: trivially preserved (we'd be running with
  LC=0 always, no sweep over LC needed).

**Cheap test plan (to fully validate before changing defaults)**:

1. Pull *all* completed runs in `lorenz_current` and `lorenz_current_obsnoise001`,
   compute Jacobian R² and Lyapunov-spectrum-distance from the saved
   checkpoints (the analysis pipeline in `JacobianODE/jacobians/tuning/...`
   already does this when it generates the `report.md`). Compare distribution
   for `loop_closure_weight=0` vs `1e-2` (or some similarly differentiated value).
   This is **engaging-side** work — I can describe it but not run it.
2. If Jacobian quality survives: ship `loop_closure_training: false` as the
   default in the experiment YAML (or gate the forward pass on
   `self.loop_closure_weight > 0` in lightning_base.py:670 to make weight=0
   actually skip the work).

**Why this is the strongest hypothesis**: the W&B data already shows it doesn't
hurt val loss across 51 runs; the only remaining unknown is Jacobian quality;
the implementation is one config flip; the wall-clock saving is measured.

### H2 — Two-stage sweep: short-prefix shortlist + long survivors

**Confidence**: MEDIUM-HIGH. Top-3 sweep winners are 3/3 stable by epoch 11 in
the obs_noise=0.01 sweep. In obs_noise=0.05 the picture is murkier (top-3
overlap is 1/3 even at epoch 99 — there's seed-noise reshuffling among
runs with very similar final losses).

**Concrete change**: replace the current single-pass sweep with a two-stage
protocol:

- Stage A: train *all* configs with `max_epochs=30` and `early_stopping_patience=2`
  (or just a hard `max_steps` cap). On engaging this is ~15 min/config.
- Stage B: keep the top half by Stage-A best val loss, restart-or-continue
  them to convergence with the normal schedule.

For a 21-config sweep, Stage A costs 21 × 15 min = 5.25 h. Stage B costs
10 × 3 h = 30 h. Total ~35 h. Single-pass sweep currently costs 21 × 3 h = 63 h.

**Wall-clock impact**: ~45 % sweep wall-clock reduction at the cost of
slightly increased risk of dropping a slow-burner.

**Risk**: (3) sweep-winner selection — in obs_noise=0.05, the bottom *half*
at epoch 30 is unlikely to contain the eventual top-1 (rank correlation
already > 0.6 by epoch 27), but the *very top* slot could flip. Mitigation:
keep top 2/3 instead of top 1/2, larger budget but lower risk.

**Cheap test plan**:
- For each of the existing 21-run sweeps, simulate the two-stage protocol
  on the W&B data: which configs would Stage A keep at epoch 30? What's their
  rank distribution at run-final? **This is purely retrospective W&B analysis,
  no training needed.** I have the raw data cached and could do this in
  another hour if you want.

### H3 — Cap max_epochs and re-scale the cosine LR + alpha_teacher_forcing schedules

**Confidence**: MEDIUM. The Q2 cost-quality table directly quantifies the
tradeoff: if you stop at 75 % of current runtime you accept ~20–43 % worse
final val loss, depending on group.

**Concrete change**: `max_epochs: 60` (or 80) instead of 200 (current ES
typically fires at ~109). Re-scale `gamma_teacher_forcing` so that alpha
reaches its `min_alpha_teacher_forcing` by the new epoch budget — otherwise
the model is evaluated on free-running rollouts it never saw in training.
The cosine LR scheduler's `T_max=trainer.max_epochs` will auto-rescale.

**Wall-clock impact**: 60/109 ≈ −45 % per run.

**Risk**:
- (1) trajectory val_loss: directly tracked by Q2 table; with new schedule
  this could be smaller or bigger than the naive cost-quality estimate
  (more aggressive LR decay might compensate).
- (2) Jacobian quality: **highest-risk axis.** alpha_teacher_forcing only
  reaches its minimum at epoch 108 currently — that's exactly when val
  loss reaches near-final and presumably when Jacobians are tuned most
  carefully. Cutting epochs without re-scaling is dangerous.
- (3) sweep winners: this change is orthogonal to the within-sweep ranking
  question, applies uniformly so should preserve relative ranking.

**Cheap test plan**: this is the one I'd actually pilot-train. Recommend a
single 30-min sandbox run with `max_epochs=15`, `gamma_teacher_forcing`
re-scaled, and check val loss + alpha trajectory at end vs the corresponding
W&B runs at epoch 15. If alpha is at its minimum by epoch 15 *and* val loss
is comparable to baseline-at-epoch-15, the schedule re-scaling worked. If
it's not, the re-scaling needs more thought.

I did **not** run this pilot autonomously — it requires picking a re-scaled
gamma value and changing the lightning_base behaviour, which is exactly the
kind of design decision your CLAUDE.md says I should ask about first.

### H4 — Use the old-era `jac_r2_score` as a Jacobian-quality monitor + ES condition

**Confidence**: LOW (speculative).

The current configs log `train/jac_norm` and `val/jac_loss` (sometimes), but
the *quality* of the learned Jacobians is logged less directly than in the
old era's `train jac r2_score`. The old era ran sweeps where this metric was
the actual signal of "have we learned the right Jacobians."

If a sweep's *eventual best by jac_r2_score* is reached **earlier** than
*eventual best by trajectory val_loss*, you could ES on jac_r2_score and
stop earlier without harming the quality axis you actually care about for
downstream Lyapunov work.

**Cheap test plan**: re-enable `train jac r2_score` logging in the current
configs (it requires `eq` to be available — for the obs_noise=0.05 / 0.01
lorenz partial setup, `eq` is None because `n_delays > 1`, so the eq-aware
Jacobian comparison branch is skipped). This is partly a code change to make
true Jacobians accessible for the partial-obs case. Bigger lift than H1–H3.

### H5 — Mid-impact, low-risk: reduce `limit_val_batches` and val frequency

**Confidence**: LOW (W&B doesn't tell me which fraction of per-epoch wallclock
is val vs train).

Running validation every epoch on all 10 limit_val_batches has a cost. If
val takes 10 % of epoch wallclock, validating every other epoch saves ~5 %.
Marginal but free if it doesn't break ES.

**Cheap test plan**: configure `check_val_every_n_epoch=2` and re-run.

## Recommendations (read me)

1. **First check** (no training needed, ~1 h of analysis): pull the
   `report.md` Lyapunov-spectrum-distance numbers for the lc_w=0 vs
   lc_w=1e-2 runs in `lorenz_current_obsnoise001` from
   `~/Documents/jacobian-analyses/...` and confirm whether **H1's
   Jacobian-quality risk is real**. If LC=0 runs have similar Lyapunov
   accuracy to LC=1e-4 runs, **H1 is a free 13 % wall-clock win plus a
   simpler training loop** — ship it.

2. **Then** (also no training, retrospective analysis on the cached data):
   run the H2 simulation on the obs_noise=0.05 sweep to see whether a
   30-epoch top-half cull would have preserved the top-1. I can do this
   in an hour if you want; just say yes.

3. **Then if you still want more** (training): pilot H3 with a re-scaled
   alpha schedule. This needs design input from you on what
   `gamma_teacher_forcing` should be re-scaled to.

## What I'm explicitly NOT recommending

- "Just turn off loop_closure for everyone right now": I want the Jacobian-quality
  cross-check first. The user's quality definition explicitly includes Lyapunov
  spectrum, which is exactly what loop_closure was originally designed to support.
- "Cap epochs aggressively without thinking about the alpha schedule":
  alpha_teacher_forcing minimum-reach happens at epoch ~108. Cutting epochs
  without re-scaling the anneal would mean training on a different problem
  than evaluation tests.
- "torch.compile" or related low-level wins: see the prior REPORT.md — those
  are blocked on the sandbox GPU, and they are the wrong scope here anyway.

## Status

- **All Phase-1 analysis ran inside the sandbox** with `wandb.Api()` only;
  no training was launched on engaging.
- **Per-batch wall-clock pilot** for H1 *was* run in the sandbox (5 min) — it
  confirms the −13 % saving across both target configs.
- **No model-quality validation done.** That's Phase 2.
