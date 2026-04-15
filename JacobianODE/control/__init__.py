"""Control-theoretic analysis tools for JacobianODE.

Primary export: :func:`compute_all_gramians`, a batched square-root QR
implementation of the reachability, controllability, and observability
Gramians for LTV systems discretised via zero-order hold along a trajectory.
"""
from .gramians import compute_all_gramians

__all__ = ["compute_all_gramians"]
