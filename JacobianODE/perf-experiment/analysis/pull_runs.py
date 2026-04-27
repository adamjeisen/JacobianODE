"""Pull W&B run histories + configs for the two reference groups and cache locally.

Subsequent analysis scripts read the cache (parquet) instead of re-hitting the
API. Re-run with --refresh to force-fetch.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import wandb


CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


GROUPS = {
    # Primary: current latent / autodim / split-mode lorenz LC sweep.
    "lorenz_current": {
        "project": "Lorenz_INDpartial_NDInitSweep_autodim_D1_NormTrue__JacobianODE",
        "group": "lorenz_partial_additive_splitmode_p30_obsnoise005_top3nd_init15_autodim__lc_sweep",
        # New-era metric names (train/* namespace).
        "era": "latent",
    },
    # Sibling: current latent at obs_noise=0.01 — same sweep at the other obs noise.
    "lorenz_current_obsnoise001": {
        "project": "Lorenz_INDpartial_NDInitSweep_autodim_D1_NormTrue__JacobianODE",
        "group": "lorenz_partial_additive_splitmode_p30_obsnoise001_top3nd_init15_autodim__lc_sweep",
        "era": "latent",
    },
    # Reference: old fast-era vanilla wmtask LC sweep (paper-era runs).
    "wmtask_vanilla_old": {
        "project": "WMTask_identity_encoder_verification",
        "group": "wmtask_vanilla_mse_p30__lc_sweep",
        "era": "vanilla",
    },
}


def pull_group(slug: str, project: str, group: str, era: str, refresh: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pull all runs in a group; return (configs_df, history_df)."""
    cfg_path = CACHE_DIR / f"{slug}__configs.pkl.gz"
    hist_path = CACHE_DIR / f"{slug}__history.pkl.gz"

    if not refresh and cfg_path.exists() and hist_path.exists():
        print(f"  [cache hit] {slug}")
        return pd.read_pickle(cfg_path), pd.read_pickle(hist_path)

    print(f"  fetching {slug} ...")
    api = wandb.Api()
    runs = list(api.runs(f"JacobianODE/{project}", filters={"group": group}, per_page=200))
    print(f"    {len(runs)} runs")
    if not runs:
        return pd.DataFrame(), pd.DataFrame()

    cfg_rows = []
    hist_frames = []
    for i, r in enumerate(runs):
        # Flatten config — keep only top-level scalars + nested ones we care about.
        flat_cfg = {
            "run_id": r.id,
            "run_name": r.name,
            "state": r.state,
            "created_at": r.created_at,
            "group": r.group,
            "project": project,
            "era": era,
            "tags": json.dumps(list(r.tags)),
        }
        cfg = r.config
        # Expose useful parameters explicitly.
        for path in [
            "data.flow.random_state",
            "training.batch_size",
            "training.lightning.loop_closure_weight",
            "training.lightning.obs_noise_scale",
            "training.lightning.reconstruction_loss_weight",
            "training.lightning.latent_prediction_loss_weight",
            "training.lightning.encoder_warmup_epochs",
            "training.lightning.dynamics_warmup_epochs",
            "training.lightning.prediction_steps",
            "training.lightning.alpha_teacher_forcing",
            "training.lightning.teacher_forcing_annealing",
            "training.optimizer_kwargs.lr",
            "training.early_stopping.early_stopping_patience",
            "training.early_stopping.early_stopping_mode",
            "training.early_stopping.percent_thresh",
            "training.trainer_params.max_epochs",
            "training.trainer_params.max_steps",
            "training.trainer_params.limit_train_batches",
            "training.trainer_params.limit_val_batches",
            "training.trainer_params.accumulate_grad_batches",
            "model.n_target_dims",
            "model.params.input_dim",
            "data.train_test_params.delay_embedding_params.n_delays",
        ]:
            cur = cfg
            for part in path.split("."):
                if isinstance(cur, dict) and part in cur:
                    cur = cur[part]
                else:
                    cur = None
                    break
            flat_cfg[path] = cur
        # Final summary metrics
        for k, v in r.summary.items():
            if k.startswith("_"):
                continue
            try:
                # Skip dict-valued summary entries (e.g. histograms)
                if isinstance(v, (int, float, str, bool)) or v is None:
                    flat_cfg[f"final/{k}"] = v
            except Exception:
                pass
        cfg_rows.append(flat_cfg)

        # History: pull ALL rows. Use scan_history to avoid the sampled API.
        hist = []
        try:
            for row in r.scan_history():
                hist.append(row)
        except Exception as e:
            print(f"    [warn] history fetch failed for {r.id}: {e}")
            continue
        if hist:
            df = pd.DataFrame(hist)
            df["run_id"] = r.id
            df["era"] = era
            hist_frames.append(df)
        if (i + 1) % 5 == 0:
            print(f"    pulled {i+1}/{len(runs)}")

    cfg_df = pd.DataFrame(cfg_rows)
    hist_df = pd.concat(hist_frames, ignore_index=True) if hist_frames else pd.DataFrame()
    cfg_df.to_pickle(cfg_path)
    hist_df.to_pickle(hist_path)
    print(f"    cached → {cfg_path.name} ({len(cfg_df)} rows), {hist_path.name} ({len(hist_df)} rows)")
    return cfg_df, hist_df


def main(refresh: bool = False) -> int:
    for slug, spec in GROUPS.items():
        print(f"=== {slug} ({spec['project']} / {spec['group']}) ===")
        pull_group(slug, spec["project"], spec["group"], spec["era"], refresh)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    sys.exit(main(refresh=args.refresh))
