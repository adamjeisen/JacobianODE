"""Render a sweep analysis directory into a human-readable report.

Given an analysis directory produced by ``analyze_sweep.py`` (containing
``context.json``, ``metrics.json``, and ``figures/``), this script produces:

- ``report.md``  — Markdown report with tables, figures, and success-criteria
  verdicts. Structured sections are auto-populated; a "Discussion" section is
  left as a placeholder for a human / Claude to fill in.
- ``report.html`` — Rendered HTML (via the ``markdown`` library if available,
  otherwise a minimal passthrough).
- ``report.pdf``  — Optional. Rendered from HTML via ``weasyprint`` if
  installed; skipped with a warning otherwise.

Usage:

    python -m JacobianODE.jacobians.tuning.render_report <analysis_dir>

The report is self-contained — figures are referenced via relative paths,
so the whole analysis directory can be shared as a unit.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger("JacobianODE.render_report")


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing: {path}")
    return json.loads(path.read_text())


def _fmt(v, fmt: str = ".4f") -> str:
    if v is None:
        return "—"
    try:
        return format(v, fmt)
    except (TypeError, ValueError):
        return str(v)


def _fmt_float(v, digits=4):
    return _fmt(v, f".{digits}f")


def build_header(ctx: dict) -> list[str]:
    lines: list[str] = []
    lines.append(f"# Sweep Analysis: `{ctx.get('group')}`")
    lines.append("")
    wandb = ctx.get("wandb", {})
    lines.append(
        f"**Project**: [{wandb.get('project')}]"
        f"(https://wandb.ai/{wandb.get('entity')}/{wandb.get('project')}"
        f"/groups/{ctx.get('group')})  "
    )
    lines.append(f"**Launched**: {ctx.get('launched_at')}  ")
    lines.append(f"**Completed**: {ctx.get('completed_at')}  ")
    lines.append(f"**Outcome**: `{ctx.get('outcome')}`  ")
    git = ctx.get("git", {}) or {}
    lines.append(f"**Git**: `{git.get('branch')}` @ `{git.get('commit')}`  ")
    lines.append(f"**Expected runs**: {ctx.get('expected_run_count')}")
    lines.append("")
    return lines


def build_experiment_section(ctx: dict) -> list[str]:
    lines: list[str] = ["## Experiment Context", ""]
    emd = ctx.get("experiment_metadata") or {}
    if not emd:
        lines.append("_(no metadata)_")
        lines.append("")
        return lines
    for exp_name, meta in emd.items():
        lines.append(f"### `{exp_name}`")
        lines.append("")
        for field in ("description", "hypothesis"):
            v = meta.get(field)
            if v:
                lines.append(f"**{field.capitalize()}**")
                lines.append("")
                lines.append(str(v).strip())
                lines.append("")
        sc = meta.get("success_criteria") or []
        if sc:
            lines.append("**Success criteria**")
            lines.append("")
            for c in sc:
                lines.append(f"- {c}")
            lines.append("")
    return lines


def build_results_section(metrics_doc: dict) -> list[str]:
    lines: list[str] = ["## Results", ""]
    summary = metrics_doc.get("metrics_summary") or {}
    best = summary.get("overall_best_mase") or {}
    lines.append(
        f"**Overall best MASE**: {_fmt_float(best.get('best_mase'))} "
        f"(LC weight = {_fmt(best.get('lc_weight'), '.1e')}, "
        f"obs_noise_scale = {_fmt_float(best.get('obs_noise_scale'), 2)})"
    )
    lines.append(
        f"**Overall best traj loss**: {_fmt_float(best.get('best_traj_loss'), 5)} "
        f"at epoch {best.get('best_traj_loss_epoch')}"
    )
    lines.append(f"**Runs analyzed**: {summary.get('n_runs')}")
    lines.append("")

    # Best per obs_noise_scale table
    by_ons = summary.get("best_by_obs_noise_scale") or {}
    if by_ons:
        lines.append("### Best run per `obs_noise_scale`")
        lines.append("")
        lines.append("| obs_noise_scale | Best LC weight | Best traj loss | MASE at best | R² | LC loss | epoch |")
        lines.append("|---|---|---|---|---|---|---|")
        for ons, row in sorted(by_ons.items(), key=lambda x: float(x[0]) if x[0] not in (None, "None") else -1):
            lines.append(
                f"| {ons} | {_fmt(row.get('lc_weight'), '.1e')} "
                f"| {_fmt_float(row.get('best_traj_loss'), 5)} "
                f"| {_fmt_float(row.get('best_mase'))} "
                f"| {_fmt_float(row.get('r2_at_best_tl'))} "
                f"| {_fmt_float(row.get('lc_loss_at_best_tl'), 3)} "
                f"| {row.get('best_traj_loss_epoch')} |"
            )
        lines.append("")
    return lines


def build_verdicts_section(metrics_doc: dict) -> list[str]:
    lines: list[str] = ["## Success-criteria verdicts (automated)", ""]
    verdicts = metrics_doc.get("success_criteria_verdicts") or []
    if not verdicts:
        lines.append("_(none)_")
        lines.append("")
        return lines
    lines.append("| Criterion | Verdict | Note |")
    lines.append("|---|---|---|")
    for v in verdicts:
        lines.append(f"| {v.get('criterion')} | **{v.get('verdict')}** | {v.get('note')} |")
    lines.append("")
    lines.append(
        "_Automated verdicts use simple numeric-threshold parsing and may mis-classify "
        "qualitative criteria. The Discussion section below takes precedence._"
    )
    lines.append("")
    return lines


def build_figures_section(metrics_doc: dict, analysis_dir: Path) -> list[str]:
    lines: list[str] = ["## Figures", ""]
    figures = metrics_doc.get("figures") or {}
    if not figures:
        lines.append("_(no figures produced — analytics may have failed)_")
        if metrics_doc.get("analytics_error"):
            lines.append("")
            lines.append(f"```\n{metrics_doc['analytics_error']}\n```")
        lines.append("")
        return lines
    # Preferred display order: sweep overview first, then prediction / mase,
    # then Lyapunov plots. Unknown names go last in whatever order.
    preferred = [
        "sweep_overview", "sweep_pareto",
        "prediction_windows", "mase",
        "lyapunov", "lyapunov_top10",
        "per_run_lyapunov",
        "per_run_lyapunov_vs_true",
        "lyapunov_spectrum_mse_vs_val_loss",
    ]
    seen = set()
    ordered = []
    for key in preferred:
        if key in figures:
            ordered.append((key, figures[key]))
            seen.add(key)
    for key, path in figures.items():
        if key not in seen:
            ordered.append((key, path))

    for section, path in ordered:
        p = Path(path)
        try:
            rel = p.relative_to(analysis_dir)
        except ValueError:
            rel = Path("figures") / p.name
        lines.append(f"### {section}")
        lines.append("")
        lines.append(f"![{section}]({rel.as_posix()})")
        lines.append("")
    return lines


def build_analytics_log_section(metrics_doc: dict, analysis_dir: Path) -> list[str]:
    """Embed run_analytics's stdout as a collapsible <details> block."""
    log_name = metrics_doc.get("analytics_log_file")
    if not log_name:
        return []
    log_path = analysis_dir / log_name
    if not log_path.is_file():
        return []
    content = log_path.read_text().strip()
    if not content:
        return []
    # Truncate pathological logs to keep the report browsable
    if len(content) > 60_000:
        content = content[:60_000] + "\n\n... (log truncated — see run_analytics.log for full output)"
    return [
        "## `run_analytics` stdout",
        "",
        "<details><summary>Click to expand — full diagnostic output from <code>run_analytics</code></summary>",
        "",
        "```",
        content,
        "```",
        "",
        "</details>",
        "",
    ]


def build_discussion(analysis_dir: Path) -> list[str]:
    """Emit the Discussion section.

    If ``discussion.md`` exists in the analysis dir, use its contents verbatim
    (written by Claude or a human). Otherwise emit a placeholder stub that
    invites authorship.
    """
    discussion_path = analysis_dir / "discussion.md"
    if discussion_path.is_file():
        body = discussion_path.read_text().strip()
        if body:
            return ["## Discussion", "", body, ""]
    return [
        "## Discussion",
        "",
        "<!--",
        "This section is intentionally left as a placeholder. A human reviewer",
        "or Claude Code agent should fill it in based on the tables and figures",
        "above, explicitly addressing each success criterion and comparing the",
        "outcome to the stated hypothesis. Write the Discussion to",
        "`discussion.md` in this directory and re-run `render_report`.",
        "-->",
        "",
        "_(to be written)_",
        "",
    ]


def build_markdown(analysis_dir: Path) -> str:
    ctx = load_json(analysis_dir / "context.json")
    metrics_doc = load_json(analysis_dir / "metrics.json")
    lines: list[str] = []
    lines += build_header(ctx)
    lines += build_experiment_section(ctx)
    lines += build_results_section(metrics_doc)
    lines += build_verdicts_section(metrics_doc)
    lines += build_figures_section(metrics_doc, analysis_dir)
    lines += build_discussion(analysis_dir)
    lines += build_analytics_log_section(metrics_doc, analysis_dir)
    return "\n".join(lines)


def render_html(md_text: str) -> str:
    """Convert markdown to HTML via the ``markdown`` lib if available,
    otherwise a minimal passthrough that at least honours line breaks.
    """
    try:
        import markdown  # type: ignore
        body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])
    except ImportError:
        logger.warning(
            "`markdown` library not installed; emitting raw text as <pre>. "
            "Install with: uv pip install markdown"
        )
        body = f"<pre>{md_text}</pre>"
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Sweep Analysis Report</title>
<style>
body {{ font-family: -apple-system, sans-serif; max-width: 900px; margin: 2em auto; padding: 0 1em; line-height: 1.5; }}
table {{ border-collapse: collapse; margin: 1em 0; }}
th, td {{ border: 1px solid #ccc; padding: 6px 12px; }}
th {{ background: #f5f5f5; }}
img {{ max-width: 100%; height: auto; }}
code {{ background: #f5f5f5; padding: 2px 4px; }}
pre {{ white-space: pre-wrap; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def try_render_pdf(html_path: Path, pdf_path: Path) -> bool:
    """Attempt to produce a PDF via weasyprint. Returns True on success."""
    try:
        from weasyprint import HTML  # type: ignore
    except ImportError:
        logger.warning(
            "`weasyprint` not installed; skipping PDF. "
            "Install with: uv pip install weasyprint"
        )
        return False
    try:
        HTML(filename=str(html_path), base_url=str(html_path.parent)).write_pdf(
            str(pdf_path)
        )
        return True
    except Exception as e:
        logger.error(f"weasyprint failed: {e}")
        return False


def render(analysis_dir: Path) -> None:
    md_text = build_markdown(analysis_dir)
    (analysis_dir / "report.md").write_text(md_text)
    logger.info(f"Wrote {analysis_dir / 'report.md'}")

    html_text = render_html(md_text)
    (analysis_dir / "report.html").write_text(html_text)
    logger.info(f"Wrote {analysis_dir / 'report.html'}")

    if try_render_pdf(analysis_dir / "report.html", analysis_dir / "report.pdf"):
        logger.info(f"Wrote {analysis_dir / 'report.pdf'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "analysis_dir",
        help="Path to the analysis directory (contains context.json, metrics.json, figures/)",
    )
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    analysis_dir = Path(args.analysis_dir).resolve()
    if not analysis_dir.is_dir():
        print(f"ERROR: {analysis_dir} is not a directory", file=sys.stderr)
        return 2
    render(analysis_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
