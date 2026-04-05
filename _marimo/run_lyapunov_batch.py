"""Batch runner for Lyapunov Spectrum Analysis notebook.

Usage:
    uv run python _marimo/run_lyapunov_batch.py
"""
import importlib.util
from pathlib import Path


def main():
    spec = importlib.util.spec_from_file_location(
        "lyapunov_nb",
        Path(__file__).parent / "Lyapunov Spectrum Analysis.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    result = mod.app.run()
    # app.run() may return a coroutine or a tuple depending on marimo version
    import asyncio
    if asyncio.iscoroutine(result):
        outputs, defs = asyncio.run(result)
    else:
        outputs, defs = result

    print("\n=== Done ===")
    if "summary_df" in defs:
        print(defs["summary_df"].to_string())
    if "results_dir" in defs:
        print(f"\nResults cached in: {defs['results_dir']}")


if __name__ == "__main__":
    main()
