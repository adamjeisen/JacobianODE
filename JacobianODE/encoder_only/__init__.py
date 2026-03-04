"""Encoder-only representation learning module.

Train a causal sequence encoder (Transformer / SSM / TCN) without the
downstream JacobianODE, using reconstruction and/or prediction objectives
plus optional geometric regularisation.

Quick start
-----------
From the repo root::

    python -m JacobianODE.encoder_only.run_encoder model=transformer

    # Sweep over regularisation weights
    python -m JacobianODE.encoder_only.run_encoder --multirun \\
        ++training.lightning.fnn_weight=0,0.001,0.01,0.1

See Also
--------
JacobianODE/encoder_only/conf/  – Hydra configuration files
_jupyter/Encoder-Only*/         – Example and sweep notebooks
"""

from .model import LitEncoderDecoder
from .pretrained import (
    PretrainedEncoderAdapter,
    PretrainedJacRunResult,
    load_pretrained_encoder,
    load_pretrained_jac_run,
)

__all__ = [
    "LitEncoderDecoder",
    "PretrainedEncoderAdapter",
    "PretrainedJacRunResult",
    "load_pretrained_encoder",
    "load_pretrained_jac_run",
]
