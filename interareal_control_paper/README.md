# Interareal control — standalone preprint

A self-contained `article`-class preprint built from the "Disruption of
interareal control during propofol anesthesia" chapter and appendix of the
MIT thesis.

## Build

```
latexmk -pdf interareal_control_paper.tex
```

Requires a TeX distribution with `biblatex` + `biber`. The rendered
`interareal_control_paper.pdf` is committed for convenience.

## Layout

- `interareal_control_paper.tex` — driver (preamble + main text + supplement).
- `interareal-control.tex` — main text (Abstract, Introduction, Methods,
  Results, Discussion).
- `interareal-control-appendix.tex` — supplementary information (proofs,
  modeling/experimental details, statistical tables).
- `figures/interareal-control/` — figures.
- `paperpile.bib` — bibliography (full library; only cited entries are printed).

## Source of truth

The prose is maintained in the `mit-thesis` repository
(`MIT-thesis/sections/interareal-control{,-appendix}.tex`). The copies here
are kept self-contained so this folder builds on its own; substantive edits
should be made in the thesis and re-synced. The only deviation from the thesis
source is the noise-filtering paragraph's cross-reference, which points to a
thesis chapter (`\ref{chap:similar}`) there and is rendered here as a direct
citation to the corresponding paper.
