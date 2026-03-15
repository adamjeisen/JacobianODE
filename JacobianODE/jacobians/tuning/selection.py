"""Model selection algorithm with relaxation rules.

Implements the physics-informed selection procedure from Appendix D.8.8.
Operates purely on sequences of DiagnosticMetrics — no model or GPU needed.

Reference: checkpoints/wandb_utils.py lines 276-323.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .criteria import (
    DiagnosticMetrics,
    fails_eigenvalue_criterion,
    fails_loop_closure_criterion,
    fails_one_step_criterion,
)


@dataclass
class SelectionResult:
    """Result of the model selection procedure.

    Attributes:
        best_index: Index of the selected model in the input sequence,
            or None if no candidates were provided.
        best_metrics: DiagnosticMetrics of the selected model, or None.
        surviving_indices: Indices of all models that passed the (possibly
            relaxed) criteria.
        criteria_applied: Which criteria were actually enforced after
            relaxation (subset of {'C1', 'C2', 'C3'}).
        exclusion_details: Per-criterion dict mapping criterion name to the
            list of indices excluded by that criterion (before relaxation).
    """

    best_index: Optional[int]
    best_metrics: Optional[DiagnosticMetrics]
    surviving_indices: List[int]
    criteria_applied: List[str]
    exclusion_details: Dict[str, List[int]]


def select_best_model(
    candidates: Sequence[DiagnosticMetrics],
    n_dims: int,
    eigenvalue_threshold: float = 0.001,
    use_loop_closure: bool = True,
    loop_closure_n_dims: Optional[int] = None,
) -> SelectionResult:
    """Select the best model from a sequence of diagnostic metrics.

    Algorithm (mirrors wandb_utils.py lines 276-323):
      1. Evaluate all three criteria per candidate.
      2. Relaxation:
         - If no C2-passer also passes C1 -> discard C2.
         - If no C3-passer also passes C1 -> discard C1 (high noise regime).
      3. Return the candidate with lowest ``trajectory_val_loss`` among survivors.

    Args:
        candidates: Sequence of DiagnosticMetrics, one per model.
        n_dims: State-space dimensionality (used for C2 when loop_closure_n_dims
            is None; for latent models, pass loop_closure_n_dims=n_latent instead).
        eigenvalue_threshold: Threshold for C3 (default 0.001).
        use_loop_closure: Whether to apply C2 (False for NeuralODE).
        loop_closure_n_dims: Dimension for C2 threshold sqrt(n). For latent models
            (LitLatentJacobianODE), use n_latent. When None, defaults to n_dims.

    Returns:
        SelectionResult with selection outcome and diagnostics.
    """
    n = len(candidates)
    if n == 0:
        return SelectionResult(
            best_index=None,
            best_metrics=None,
            surviving_indices=[],
            criteria_applied=[],
            exclusion_details={"C1": [], "C2": [], "C3": []},
        )

    # Step 1: evaluate each criterion independently
    fails_c1 = [
        fails_one_step_criterion(m) for m in candidates
    ]
    lc_dims = loop_closure_n_dims if loop_closure_n_dims is not None else n_dims
    fails_c2 = [
        fails_loop_closure_criterion(m, lc_dims) if use_loop_closure else False
        for m in candidates
    ]
    fails_c3 = [
        fails_eigenvalue_criterion(m, eigenvalue_threshold) for m in candidates
    ]

    excluded_by_c1 = [i for i in range(n) if fails_c1[i]]
    excluded_by_c2 = [i for i in range(n) if fails_c2[i]]
    excluded_by_c3 = [i for i in range(n) if fails_c3[i]]

    exclusion_details = {"C1": excluded_by_c1, "C2": excluded_by_c2, "C3": excluded_by_c3}

    # Which criteria to enforce (may be relaxed below)
    apply_c1 = True
    apply_c2 = use_loop_closure
    apply_c3 = True

    # Step 2: relaxation rules
    # C1 passes
    passes_c1 = [i for i in range(n) if not fails_c1[i]]

    if apply_c2:
        # If no model that passes C2 also passes C1 -> discard C2
        passes_c2_and_c1 = [
            i for i in range(n) if not fails_c2[i] and not fails_c1[i]
        ]
        if len(passes_c2_and_c1) == 0:
            apply_c2 = False

    # If no model that passes C3 also passes C1 -> discard C1
    passes_c3_and_c1 = [
        i for i in range(n) if not fails_c3[i] and not fails_c1[i]
    ]
    if len(passes_c3_and_c1) == 0:
        apply_c1 = False

    # Step 3: compute survivors
    survivors = []
    for i in range(n):
        if apply_c1 and fails_c1[i]:
            continue
        if apply_c2 and fails_c2[i]:
            continue
        if apply_c3 and fails_c3[i]:
            continue
        survivors.append(i)

    # If all models are eliminated (shouldn't happen after relaxation, but
    # handle gracefully), fall back to all candidates
    if not survivors:
        survivors = list(range(n))

    criteria_applied = []
    if apply_c1:
        criteria_applied.append("C1")
    if apply_c2:
        criteria_applied.append("C2")
    if apply_c3:
        criteria_applied.append("C3")

    # Pick lowest trajectory_val_loss among survivors.
    # Exclude NaN/inf: min() treats NaN as "smaller" than everything (nothing
    # ever replaces it), so models with invalid traj loss would be wrongly selected.
    valid_survivors = [
        i for i in survivors if math.isfinite(candidates[i].trajectory_val_loss)
    ]
    if valid_survivors:
        best_idx = min(valid_survivors, key=lambda i: candidates[i].trajectory_val_loss)
    else:
        best_idx = survivors[0]  # fallback if all have NaN/inf

    return SelectionResult(
        best_index=best_idx,
        best_metrics=candidates[best_idx],
        surviving_indices=survivors,
        criteria_applied=criteria_applied,
        exclusion_details=exclusion_details,
    )
