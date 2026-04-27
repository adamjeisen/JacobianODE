"""Pull the Step 1 validation sweep + the cached baseline and compare on the
three quality axes."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import wandb

CACHE = Path(__file__).parent / "cache"
CACHE.mkdir(exist_ok=True)


PROJECT = "Lorenz_INDpartial_NDInitSweep_autodim_D1_NormTrue__JacobianODE"
NEW_GROUP = "lorenz_partial_additive_splitmode_p30_obsnoise005_top3nd_init15_autodim__lc_sweep__step1_int1_maxep80"
BASELINE_GROUP = "lorenz_partial_additive_splitmode_p30_obsnoise005_top3nd_init15_autodim__lc_sweep"


def pull(slug: str, group: str, refresh: bool = False):
    cfg_path = CACHE / f"{slug}__configs.pkl.gz"
    hist_path = CACHE / f"{slug}__history.pkl.gz"
    if not refresh and cfg_path.exists() and hist_path.exists():
        print(f"  [cache hit] {slug}")
        return pd.read_pickle(cfg_path), pd.read_pickle(hist_path)
    api = wandb.Api()
    runs = list(api.runs(f"JacobianODE/{PROJECT}", filters={"group": group}, per_page=200))
    print(f"  fetching {slug}: {len(runs)} runs")
    cfg_rows, hist_frames = [], []
    for i, r in enumerate(runs):
        flat = {"run_id": r.id, "run_name": r.name, "state": r.state, "group": r.group, "created_at": r.created_at}
        cfg = r.config
        for path in [
            "training.lightning.loop_closure_weight",
            "training.lightning.teacher_forcing_update_interval",
            "training.lightning.gamma_teacher_forcing",
            "training.trainer_params.max_epochs",
            "training.trainer_params.limit_train_batches",
            "data.train_test_params.delay_embedding_params.n_delays",
            "data.flow.random_state",
            "model.n_target_dims",
        ]:
            cur = cfg
            for part in path.split("."):
                if isinstance(cur, dict) and part in cur:
                    cur = cur[part]
                else:
                    cur = None
                    break
            flat[path] = cur
        for k, v in r.summary.items():
            if k.startswith("_"):
                continue
            if isinstance(v, (int, float, str, bool)) or v is None:
                flat[f"final/{k}"] = v
        cfg_rows.append(flat)
        try:
            hist = list(r.scan_history())
        except Exception as e:
            print(f"    [warn] history fail {r.id}: {e}")
            continue
        if hist:
            df = pd.DataFrame(hist)
            df["run_id"] = r.id
            hist_frames.append(df)
        if (i + 1) % 5 == 0:
            print(f"    pulled {i+1}/{len(runs)}")
    cfg_df = pd.DataFrame(cfg_rows)
    hist_df = pd.concat(hist_frames, ignore_index=True) if hist_frames else pd.DataFrame()
    cfg_df.to_pickle(cfg_path)
    hist_df.to_pickle(hist_path)
    print(f"  cached → cfg={len(cfg_df)} rows, hist={len(hist_df)} rows")
    return cfg_df, hist_df


def main():
    print("=== validation sweep (Step 1, max_ep=80) ===")
    pull("lorenz_step1_maxep80", NEW_GROUP)
    print("\n=== baseline (cached, may already be present) ===")
    pull("lorenz_current", BASELINE_GROUP)


if __name__ == "__main__":
    main()
