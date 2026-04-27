"""Lightning callback recording per-step (step, loss, walltime) into a JSON sidecar.

Designed for the perf-experiment regression-test gate: same-seed runs must
produce per-step training losses that match within 1e-4. The walltime field
is per-step wall-clock seconds (CUDA-synced) so that timing comparisons
across optimization attempts are fair.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import lightning as L
import torch


class PerStepTimingCallback(L.Callback):
    def __init__(self, output_path: str, sync_cuda: bool = True) -> None:
        self.output_path = Path(output_path)
        self.sync_cuda = sync_cuda
        self._step_start: Optional[float] = None
        self.records: list[dict] = []

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        self._step_start = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        if self._step_start is None:
            return
        dt = t1 - self._step_start
        # outputs may be a tensor (loss) or dict
        loss_val: Optional[float] = None
        if isinstance(outputs, torch.Tensor):
            loss_val = float(outputs.detach().item())
        elif isinstance(outputs, dict) and "loss" in outputs:
            loss_val = float(outputs["loss"].detach().item())
        # Capture alpha_teacher_forcing if the model exposes it (LitBase
        # stores it as a python float on self.alpha_teacher_forcing).
        alpha = getattr(pl_module, "alpha_teacher_forcing", None)
        try:
            alpha_val = float(alpha) if alpha is not None else None
        except (TypeError, ValueError):
            alpha_val = None
        self.records.append(
            {
                "step": int(trainer.global_step),
                "batch_idx": int(batch_idx),
                "epoch": int(trainer.current_epoch),
                "loss": loss_val,
                "walltime_seconds": dt,
                "alpha_teacher_forcing": alpha_val,
            }
        )
        self._flush()

    def _flush(self) -> None:
        # Write incrementally so a crash mid-run still leaves a sidecar.
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_path, "w") as f:
                json.dump(self.records, f)
        except Exception:
            pass

    def on_train_end(self, trainer, pl_module) -> None:
        self._flush()
