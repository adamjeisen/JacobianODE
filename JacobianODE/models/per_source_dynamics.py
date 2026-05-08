"""Per-source dynamics MLP wrapper.

Holds N parallel copies of a dynamics MLP and routes each sample to one
of them based on its condition value. Routing is hard nearest-neighbor
in ``section_condition_values`` space — no learned routing.

Motivation
----------
When training data spans multiple distinct regimes (e.g. awake vs.
propofol-anesthesia LFP), training a SHARED dynamics MLP over all of
them biases the fitted operator toward whichever regime contributes the
most gradient signal. For burst-suppression-style data, the active /
high-variance regime dominates while the suppressed regime gets
trivially predicted by persistence — the model never really learns the
suppressed-regime operator.

Splitting the dynamics MLP per regime lets each sub-MLP fit its own
data distribution while still sharing the encoder. The encoder is the
shared representation; the dynamics is regime-specific.

Usage at the cfg level
----------------------
Set ``cfg.model.per_source_dynamics=True`` and provide
``cfg.model.section_condition_values`` (a list of floats matching the
dataset's ``section_condition_values``). ``make_model`` builds
``len(section_condition_values)`` copies of the dynamics MLP and
wraps them.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PerSourceDynamicsMLP(nn.Module):
    """Drop-in replacement for a single dynamics MLP that hard-routes
    samples to one of ``n_sources`` parallel sub-MLPs based on the
    condition tensor.

    Routing: for each sample's ``c[..., 0]``, find the nearest value in
    ``section_condition_values``. That index selects which sub-MLP runs.

    All sub-MLPs are constructed identically (same architecture,
    independent parameters) and have identical signatures to the
    underlying ``jac_model``, so this wrapper is a drop-in replacement
    in :meth:`LitLatentJacobianODE.compute_jacobians`.

    Parameters
    ----------
    jac_models : list[nn.Module]
        N parallel copies of the dynamics MLP, constructed externally
        (e.g. via repeated ``hydra.utils.instantiate(cfg.model.params)``).
    section_condition_values : list[float]
        Routing centers — one per sub-MLP. A sample with ``c[..., 0]``
        closest to ``section_condition_values[s]`` dispatches to
        ``jac_models[s]``.
    """

    def __init__(
        self,
        jac_models: list[nn.Module],
        section_condition_values: list[float],
    ) -> None:
        super().__init__()
        if len(jac_models) != len(section_condition_values):
            raise ValueError(
                f"PerSourceDynamicsMLP: got {len(jac_models)} sub-MLPs but "
                f"{len(section_condition_values)} section_condition_values "
                f"({list(section_condition_values)}). They must agree in length."
            )
        if len(jac_models) < 2:
            raise ValueError(
                "PerSourceDynamicsMLP needs ≥2 sub-MLPs; for n=1, use the bare "
                "MLP instead — the wrapper has no value-add at n=1."
            )
        self.mlps = nn.ModuleList(jac_models)
        self.register_buffer(
            "section_values",
            torch.tensor(list(section_condition_values), dtype=torch.float32),
        )

    @property
    def n_sources(self) -> int:
        return len(self.mlps)

    def _route(self, c: torch.Tensor) -> torch.Tensor:
        """Map per-sample condition to integer sub-MLP index.

        Routing rule: nearest neighbor of ``c[..., 0]`` in
        ``self.section_values``. Using only the first condition dim keeps
        the routing decoupled from any other per-sample condition info
        (e.g. continuous covariates) the model might also use.

        Returns
        -------
        torch.Tensor of shape ``c.shape[:-1]`` (long dtype)
            Per-sample routing indices in ``[0, n_sources)``.
        """
        if c is None:
            raise ValueError(
                "PerSourceDynamicsMLP requires a non-None condition tensor "
                "for routing."
            )
        c0 = c[..., 0]
        d = (c0.unsqueeze(-1) - self.section_values.to(c0.dtype)).abs()
        return d.argmin(dim=-1)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Per-sample dispatch.

        Parameters
        ----------
        x : torch.Tensor of shape ``(B, ..., D_in)``
            Leading batch dim followed by any number of intermediate
            dims (e.g. time), then the feature dim.
        c : torch.Tensor of shape ``(B, condition_dim)``
            Fixed-per-sample condition. Used for routing AND passed
            through to the selected sub-MLP.

        Returns
        -------
        torch.Tensor of shape ``(B, ..., D_out)``
            Same leading dims as ``x``; last dim is whatever each
            sub-MLP emits (assumed identical across sub-MLPs since they
            have the same architecture).
        """
        source_idx = self._route(c)            # (B,)
        out: torch.Tensor | None = None
        for s in range(self.n_sources):
            mask = (source_idx == s)
            if not mask.any():
                continue
            y_s = self.mlps[s](x[mask], c[mask])
            if out is None:
                out = x.new_empty((x.shape[0],) + y_s.shape[1:])
            out[mask] = y_s
        if out is None:
            raise RuntimeError(
                "PerSourceDynamicsMLP: no samples routed to any sub-MLP. "
                f"section_values={self.section_values.tolist()}, "
                f"c sample={c[0].tolist()}"
            )
        return out
