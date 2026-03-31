"""Shared utilities for JacobianODE Marimo notebooks."""

from __future__ import annotations

import lightning as L


class EpochProgress(L.Callback):
    """Marimo progress bar per training phase.

    Shows one progress bar for each training or validation phase, labelled
    with the current epoch.  The Lightning sanity-check validation (2 batches
    before epoch 1) is displayed separately so the real epoch count is never
    off by one.

    Usage::

        from marimo_utils import EpochProgress

        trainer = train_model(
            cfg, lit_model, train_dl, val_dl,
            name="my-run",
            extra_callbacks=[EpochProgress(mo, n_train, n_val, n_epochs)],
        )

    Args:
        mo: The ``marimo`` module imported inside the calling cell.
        n_train: Number of training batches per epoch
            (``cfg.training.trainer_params.limit_train_batches``).
        n_val: Number of validation batches per epoch
            (``cfg.training.trainer_params.limit_val_batches``).
        n_epochs: Total number of training epochs
            (``cfg.training.trainer_params.max_epochs``).
    """

    def __init__(self, mo, n_train: int, n_val: int, n_epochs: int) -> None:
        self._mo = mo
        self._n_train = n_train
        self._n_val = n_val
        self._n_epochs = n_epochs
        self._bar = None
        self._ctx = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open(self, title: str, total: int) -> None:
        self._ctx = self._mo.status.progress_bar(total=total, title=title)
        self._bar = self._ctx.__enter__()

    def _close(self) -> None:
        if self._ctx is not None:
            self._ctx.__exit__(None, None, None)
            self._bar = self._ctx = None

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        ep = trainer.current_epoch + 1
        self._open(f"Train — epoch {ep}/{self._n_epochs}", self._n_train)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if self._bar is not None:
            self._bar.update()

    def on_validation_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._close()
        if trainer.sanity_checking:
            n_sanity = trainer.num_sanity_val_steps
            self._open(f"Sanity check ({n_sanity} batches)", n_sanity)
        else:
            ep = trainer.current_epoch + 1
            self._open(f"Val   — epoch {ep}/{self._n_epochs}", self._n_val)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0) -> None:
        if self._bar is not None:
            self._bar.update()

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._close()
