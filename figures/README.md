# JacobianODE paper figures

Each figure is a separate subdirectory (e.g. `lyapunov/`, `rotation_distance/`)
with two scripts:

| script   | role                                              | runs on               |
|----------|---------------------------------------------------|------------------------|
| `eval.py`| Heavy: pulls checkpoints, computes raw arrays, saves `.npz` | GPU box (engaging or endeavour) |
| `plot.py`| Light: reads `.npz`, applies paper style, writes PNG + PDF | anywhere |

**Why the split?** Plotting is iterative — fonts, colors, axis limits. The
eval is expensive — model loading, Lyapunov computation, full-trajectory
Jacobians. Splitting them means we re-run eval once per sweep, then iterate
on plot style as many times as we like off the cached `.npz`.

## Adding a new figure

1. `mkdir figures/<name>/`
2. Write `eval.py`: CLI args for `--group`, `--run` (default `"best"`, picked
   by `select_best_model(ranking_method="best_traj_loss")`), `--out`. Emit
   a single `<out>/<name>.npz` with named arrays. Document what each key is.
3. Write `plot.py`: CLI args for `--npz`, `--out`. Use `figures.style` for
   rcParams + color palette. Save PNG (raster, for talks) and PDF (vector,
   for Illustrator-stitching into multi-panel figures).
4. Add an entry to this README.

## Output location

By convention, scripts write to:

```
~/Documents/paper-figures/<group>/<figure_name>/
  ├── <figure_name>.npz       # raw data from eval
  ├── <figure_name>.png       # raster
  └── <figure_name>.pdf       # vector
```

Keeps paper-quality outputs separate from the auto-generated reports in
`~/Documents/jacobian-analyses/`.

## Figures

- `lyapunov/` – Predicted (batch + burn-in) vs Predicted (full trajectory)
  vs Empirical (true J) Lyapunov spectrum for the best run of a sweep group.
