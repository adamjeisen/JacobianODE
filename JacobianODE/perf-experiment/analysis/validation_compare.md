# Step 1 validation: NEW (interval=1, max_ep=80) vs BASELINE (interval=5, max_ep=200)

baseline group: lorenz_partial_additive_splitmode_p30_obsnoise005_top3nd_init15_autodim__lc_sweep
new group: lorenz_partial_additive_splitmode_p30_obsnoise005_top3nd_init15_autodim__lc_sweep__step1_int1_maxep80

**Median epochs to ES/end**: baseline=109, new=80
  → with the same per-epoch cost, **new is ≈ 1.36× faster** wall-clock

## Axis 1: trajectory val_loss


### BASELINE (interval=5, max_ep=200, ~109 ran)
```
                   count   median      min      max
n_delays lc_w                                      
45       0.000000      1  0.00565  0.00565  0.00565
         0.000001      1  0.00550  0.00550  0.00550
         0.000010      1  0.00557  0.00557  0.00557
         0.000100      1  0.00745  0.00745  0.00745
         0.001000      1  0.00562  0.00562  0.00562
         0.010000      1  0.00609  0.00609  0.00609
         0.100000      1  0.00755  0.00755  0.00755
80       0.000000      1  0.00622  0.00622  0.00622
         0.000001      1  0.00602  0.00602  0.00602
         0.000010      1  0.00618  0.00618  0.00618
         0.000100      1  0.00648  0.00648  0.00648
         0.001000      1  0.00727  0.00727  0.00727
         0.010000      1  0.01016  0.01016  0.01016
         0.100000      1  0.00950  0.00950  0.00950
85       0.000000      1  0.00517  0.00517  0.00517
         0.000001      1  0.00535  0.00535  0.00535
         0.000010      1  0.00562  0.00562  0.00562
         0.000100      1  0.00553  0.00553  0.00553
         0.001000      1  0.00596  0.00596  0.00596
         0.010000      1  0.00782  0.00782  0.00782
         0.100000      1  0.00754  0.00754  0.00754
```

### NEW (interval=1, max_ep=80, ~79 ran)
```
                   count   median      min      max
n_delays lc_w                                      
45       0.000000      1  0.00548  0.00548  0.00548
         0.000001      1  0.00552  0.00552  0.00552
         0.000010      1  0.00563  0.00563  0.00563
         0.000100      1  0.00580  0.00580  0.00580
         0.001000      1  0.00628  0.00628  0.00628
         0.010000      1  0.00741  0.00741  0.00741
         0.100000      1  0.01175  0.01175  0.01175
80       0.000000      1  0.04316  0.04316  0.04316
         0.000001      1  0.00741  0.00741  0.00741
         0.000010      1  0.00703  0.00703  0.00703
         0.000100      1  0.00723  0.00723  0.00723
         0.001000      1  0.00780  0.00780  0.00780
         0.010000      1  0.01223  0.01223  0.01223
         0.100000      1  0.01080  0.01080  0.01080
85       0.000000      1  0.00720  0.00720  0.00720
         0.000001      1  0.00751  0.00751  0.00751
         0.000010      1  0.00557  0.00557  0.00557
         0.000100      1  0.00538  0.00538  0.00538
         0.001000      1  0.00555  0.00555  0.00555
         0.010000      1  0.00613  0.00613  0.00613
         0.100000      1  0.00845  0.00845  0.00845
```

### Per-bin comparison (median across 3 replicate seeds)
```
                   baseline_median  new_median    delta  rel_delta
n_delays lc_w                                                     
45       0.000000          0.00565     0.00548 -0.00017   -0.02930
         0.000001          0.00550     0.00552  0.00002    0.00404
         0.000010          0.00557     0.00563  0.00006    0.01029
         0.000100          0.00745     0.00580 -0.00165   -0.22158
         0.001000          0.00562     0.00628  0.00067    0.11863
         0.010000          0.00609     0.00741  0.00132    0.21683
         0.100000          0.00755     0.01175  0.00420    0.55629
80       0.000000          0.00622     0.04316  0.03695    5.94303
         0.000001          0.00602     0.00741  0.00139    0.23095
         0.000010          0.00618     0.00703  0.00085    0.13697
         0.000100          0.00648     0.00723  0.00075    0.11585
         0.001000          0.00727     0.00780  0.00053    0.07296
         0.010000          0.01016     0.01223  0.00207    0.20408
         0.100000          0.00950     0.01080  0.00130    0.13659
85       0.000000          0.00517     0.00720  0.00203    0.39346
         0.000001          0.00535     0.00751  0.00215    0.40182
         0.000010          0.00562     0.00557 -0.00005   -0.00874
         0.000100          0.00553     0.00538 -0.00014   -0.02587
         0.001000          0.00596     0.00555 -0.00041   -0.06895
         0.010000          0.00782     0.00613 -0.00169   -0.21609
         0.100000          0.00754     0.00845  0.00091    0.12076
```

**Median rel_delta across all 21 bins**: +11.9%
**Worst rel_delta**: +594.3% (at (np.int64(80), np.float64(0.0)))
**Best rel_delta**: -22.2% (at (np.int64(45), np.float64(0.0001)))

## Axis 2: Jacobian / Lyapunov quality


### Per-run Lyapunov spectrum (chosen run + others)


**Empirical (true) Lyapunov spectrum:**
  baseline: None
  new:      None

**Predicted Lyapunov (chosen run, by best_traj_loss):**

**per_run_lyapunov_error (RMSE vs true) — baseline:** None
**per_run_lyapunov_error (RMSE vs true) — new:**      None

### R² at best trajectory loss (proxy for Jacobian quality)
```
  baseline: median=0.9836, min=0.9725, max=0.9864
  new:      median=0.9809, min=0.8857, max=0.9859
```

## Axis 3: Sweep winner stability


**Baseline top-1**: lc_w=0, n_delays=85, traj_loss=0.00517
**New top-1**:      lc_w=0.0001, n_delays=85, traj_loss=0.00538

### Per-n_delays bin winner (lc_w that minimizes median traj_loss)
```
  n_delays=45: baseline winner lc_w=1e-06  vs  new winner lc_w=0  ✗
  n_delays=80: baseline winner lc_w=1e-06  vs  new winner lc_w=1e-05  ✗
  n_delays=85: baseline winner lc_w=0  vs  new winner lc_w=0.0001  ✗
```

**Top-3 (n_delays, lc_w) matches**:
  baseline: [(85.0, 0.0), (85.0, 1e-06), (45.0, 1e-06)]
  new:      [(85.0, 0.0001), (45.0, 0.0), (45.0, 1e-06)]
  overlap: 1/3 — {(45.0, 1e-06)}

**Spearman rank correlation across all 21 (n_delays, lc_w) bins**: 0.577