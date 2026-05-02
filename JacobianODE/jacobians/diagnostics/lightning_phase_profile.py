"""Per-phase profile of validation_step run INSIDE Lightning's Trainer.fit.

The sync_profile_val script timed an isolated validation_step replication
and got DS=MO=441ms. The production_profile measured DS=445ms vs MO=7.3sec
when run via Trainer.fit. Same code, different result. We want to localize
WHICH phase of validation_step takes 7 sec in production.

This script monkey-patches the model's `validation_step` to wrap each
phase (encode, trajectory, loop_closure, recon, one_step, eigvals,
log_metrics) with cuda.synchronize + perf_counter. Then runs the actual
Lightning Trainer.fit. Per-phase wall time is collected on a side dict
and reported at the end.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wandb-entity", required=True)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--groups", required=True)
    parser.add_argument("--run-ids", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-batches", type=int, default=5)
    parser.add_argument("--val-batches", type=int, default=8)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    import lightning.pytorch as L
    import torch
    import torch.nn.functional as F

    from JacobianODE.jacobians.checkpoints.loader import load_run

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_dir = Path(args.save_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    groups = [g.strip() for g in args.groups.split(",")]
    run_ids = [s.strip() for s in args.run_ids.split(",")]
    summary = {}

    for group, rid in zip(groups, run_ids):
        logger.info(f"=== {group}  rid={rid} ===")
        loaded = load_run(
            f"{args.wandb_entity}/{args.wandb_project}",
            run_id=rid, save_dir=str(save_dir),
            generate_data=True, verbose=False, return_full_obs=False,
        )
        run_obj, cfg, eq, dt, values, train_dl, val_dl, test_dl, trajs, lit_model = loaded

        sink: dict[str, list[float]] = defaultdict(list)

        def _sync():
            if device.type == "cuda":
                torch.cuda.synchronize()

        def patched_validation_step(self, batch, batch_idx=0, dataloader_idx=0,
                                     log_metrics=True):
            """Re-implements validation_step from latent_jacobian.py:1775
            with explicit syncs around each phase. KEEP IN SYNC if upstream
            changes."""
            _sync(); t0 = time.perf_counter()
            kw = dict(
                alpha_teacher_forcing=getattr(self, "alpha_validation", 1),
                obs_noise_scale=0,

                reconstruction_mode="most_recent",
            )
            val_rets = {}
            val_rets["trajectory"] = self.trajectory_model_step(
                batch, batch_idx, dataloader_idx, **kw
            )
            _sync(); t1 = time.perf_counter()
            sink["1.trajectory_step"].append(t1 - t0)

            z_full = self.encode_trajectory(batch)
            mu_dyn, z_null = self._split_latent(z_full)
            z_dyn, log_var, kl_mu = self._vae_reparameterize(mu_dyn)
            _sync(); t2 = time.perf_counter()
            sink["2.encode"].append(t2 - t1)

            val_loop_closure = self.loop_closure_model_step(
                z_dyn, batch_idx, dataloader_idx
            )
            _sync(); t3 = time.perf_counter()
            sink["3.loop_closure"].append(t3 - t2)

            val_recon_loss = None
            if self.reconstruction_loss_weight > 0:
                with torch.no_grad():
                    val_recon_loss = self._reconstruction_loss(batch, z_full=z_full)
            _sync(); t4 = time.perf_counter()
            sink["4.recon_loss"].append(t4 - t3)

            val_kl_null_loss = None
            val_kl_dyn_loss = None
            if self.n_target_dims is not None:
                with torch.no_grad():
                    if z_null is not None and z_null.numel() > 0:
                        val_kl_null_loss = F.mse_loss(z_null, torch.zeros_like(z_null))
                    else:
                        val_kl_null_loss = torch.tensor(0.0, device=batch.device)
                    val_kl_dyn_loss = torch.tensor(0.0, device=batch.device)

            val_diffeo_loss = None

            val_latent_pred_loss = val_rets["trajectory"]["metric_vals"].get(
                "latent_pred_loss"
            )

            with torch.no_grad():
                one_step_ret = self.trajectory_model_step(
                    batch, batch_idx, dataloader_idx,
                    alpha_teacher_forcing=1, obs_noise_scale=0,
                )
            _sync(); t5 = time.perf_counter()
            sink["5.one_step_traj"].append(t5 - t4)

            if not hasattr(self, "_val_one_step_model_maes"):
                self._val_one_step_model_maes = []
                self._val_one_step_persistence_maes = []
            self._val_one_step_model_maes.append(
                one_step_ret["metric_vals"]["model_mae"].float().item()
            )
            self._val_one_step_persistence_maes.append(
                one_step_ret["metric_vals"]["persistence_mae"].float().item()
            )

            with torch.no_grad():
                pred_jacs = self.compute_jacobians(z_dyn)
                B, T, D, _ = pred_jacs.shape
                jacs_flat = pred_jacs.reshape(B * T, D, D)
                if self.n_eigval_jacobians is not None and self.n_eigval_jacobians < B * T:
                    idx = torch.randperm(B * T, device=jacs_flat.device)[:self.n_eigval_jacobians]
                    jacs_flat = jacs_flat[idx]
                finite_mask = torch.isfinite(jacs_flat).all(dim=-1).all(dim=-1)
                jacs_finite = jacs_flat[finite_mask]
                if jacs_finite.numel() > 0:
                    eigs_real = torch.linalg.eigvals(jacs_finite).real.flatten()
                    threshold = -1.0 / self.dt
                    n_too_fast = torch.sum(eigs_real <= threshold).float().item()
                    n_total = len(eigs_real)
                else:
                    n_too_fast = 0.0
                    n_total = 0
            _sync(); t6 = time.perf_counter()
            sink["6.eigvals"].append(t6 - t5)

            if not hasattr(self, "_val_eig_too_fast"):
                self._val_eig_too_fast = []
                self._val_eig_total = []
            self._val_eig_too_fast.append(n_too_fast)
            self._val_eig_total.append(n_total)

            if log_metrics:
                self.log_validation_metrics(
                    val_rets=val_rets,
                    batch=batch,
                    sync_dist=True,
                    val_loop_closure=val_loop_closure,
                    val_recon_loss=val_recon_loss,
                    val_latent_pred_loss=val_latent_pred_loss,
                    val_kl_null_loss=val_kl_null_loss,
                    val_kl_dyn_loss=val_kl_dyn_loss,
                    val_one_step_loss=one_step_ret["loss"],
                    val_diffeo_loss=val_diffeo_loss,
                )
            _sync(); t7 = time.perf_counter()
            sink["7.log_metrics"].append(t7 - t6)

            total_loss = sum(val_rets[pt]["loss"] for pt in val_rets)
            if val_recon_loss is not None:
                total_loss = total_loss + self.reconstruction_loss_weight * val_recon_loss
            if val_latent_pred_loss is not None:
                total_loss = total_loss + self.latent_prediction_loss_weight * val_latent_pred_loss
            mean_val_loss = torch.stack(
                [val_rets[pt]["loss"] for pt in val_rets]
            ).mean()
            if not hasattr(self, "current_epoch_val_losses"):
                self.current_epoch_val_losses = []
            self.current_epoch_val_losses.append(mean_val_loss.item())
            _sync(); t8 = time.perf_counter()
            sink["8.totals"].append(t8 - t7)
            sink["TOTAL"].append(t8 - t0)
            return total_loss

        # Bind to the instance
        import types
        lit_model.validation_step = types.MethodType(patched_validation_step, lit_model)

        accum = int(cfg.training.trainer_params.get("accumulate_grad_batches", 1))
        trainer = L.Trainer(
            max_epochs=1,
            limit_train_batches=args.train_batches,
            limit_val_batches=args.val_batches,
            accumulate_grad_batches=accum,
            accelerator="gpu", devices=1,
            logger=False, enable_checkpointing=False,
            enable_progress_bar=False, num_sanity_val_steps=0,
        )
        trainer.fit(lit_model, train_dataloaders=train_dl, val_dataloaders=val_dl)

        # Aggregate
        import numpy as np
        stats = {}
        for k, vs in sink.items():
            if not vs:
                continue
            arr = np.asarray(vs)
            stats[k] = {
                "n_calls": int(arr.size),
                "mean_ms": float(1e3 * arr.mean()),
                "median_ms": float(1e3 * np.median(arr)),
                "p95_ms": float(1e3 * np.percentile(arr, 95)),
                "max_ms": float(1e3 * arr.max()),
            }
        total_mean = stats.get("TOTAL", {}).get("mean_ms", 0)
        logger.info(f"  total mean: {total_mean:.1f} ms/val_iter")
        for k in sorted(stats.keys()):
            if k == "TOTAL":
                continue
            s = stats[k]
            pct = 100 * s["mean_ms"] / max(total_mean, 1e-9)
            logger.info(
                f"    {k:>22}: {s['mean_ms']:>7.1f} ms (p95 {s['p95_ms']:>7.1f}, "
                f"max {s['max_ms']:>7.1f}) {pct:>5.1f}%"
            )

        summary[group] = {"run_id": rid, "stats": stats,
                          "total_mean_ms": total_mean}
        del lit_model, trainer
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    # Cross-group ratio
    if len(summary) >= 2:
        keys = list(summary.keys())
        s1, s2 = summary[keys[0]]["stats"], summary[keys[1]]["stats"]
        logger.info(f"\n=== ratio {keys[1][:30]} / {keys[0][:30]} ===")
        for k in sorted(set(s1.keys()) | set(s2.keys())):
            if k == "TOTAL":
                continue
            t1 = s1.get(k, {}).get("mean_ms", 0)
            t2 = s2.get(k, {}).get("mean_ms", 0)
            r = t2 / t1 if t1 > 0 else float("inf")
            logger.info(f"  {k:>22}: {t1:>7.1f} vs {t2:>7.1f} ms  ratio={r:.2f}x")
        t1 = s1.get("TOTAL", {}).get("mean_ms", 0)
        t2 = s2.get("TOTAL", {}).get("mean_ms", 0)
        r = t2 / t1 if t1 > 0 else float("inf")
        logger.info(f"  {'TOTAL':>22}: {t1:>7.1f} vs {t2:>7.1f} ms  ratio={r:.2f}x")

    out = output_dir / "lightning_phase_profile.json"
    out.write_text(json.dumps(summary, indent=2))
    logger.info(f"Saved → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
