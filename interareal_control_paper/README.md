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

## Google Doc / Word export

`paper.docx` is a pandoc conversion for collaborative editing in Word or Google
Docs (upload to Drive, then "Open with Google Docs"). Regenerate with:

```
latexpand pandoc_src.tex > flat.tex
pandoc flat.tex -o paper.docx --citeproc --bibliography=paperpile.bib \
  --resource-path=.:figures/interareal-control
```

`pandoc_src.tex` is a pandoc-only wrapper that defines the custom macros
(`\vb`, `\insettitle`, theorem environments) in a form pandoc can parse, so
equations convert to editable Word equations rather than raw TeX. It is *not*
used for the real PDF build (`interareal_control_paper.tex`). Figures embed and
equations become native (editable) Word/OMML equations; citations are resolved
to an author-date reference list (pass `--csl <style.csl>` for a different
citation style, e.g. a numeric/superscript one to match the PDF).

## Source of truth

The prose is maintained in the `mit-thesis` repository
(`MIT-thesis/sections/interareal-control{,-appendix}.tex`). The copies here
are kept self-contained so this folder builds on its own; substantive edits
should be made in the thesis and re-synced. The only deviation from the thesis
source is the noise-filtering paragraph's cross-reference, which points to a
thesis chapter (`\ref{chap:similar}`) there and is rendered here as a direct
citation to the corresponding paper.
