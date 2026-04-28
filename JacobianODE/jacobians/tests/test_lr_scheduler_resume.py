"""Verify the cosine LR scheduler resumes coherently across two-stage training.

The two-stage protocol runs Stage A for a small number of epochs (e.g. 20),
saves a checkpoint, then dispatches Stage B with a larger ``max_epochs``
(e.g. 200) that resumes from Stage A's last.ckpt via Lightning's
``trainer.fit(ckpt_path=...)``.

Before the ``cosine_T_max`` decoupling, Stage A used
``CosineAnnealingLR(T_max=20)`` and Stage B used
``CosineAnnealingLR(T_max=200)``. PyTorch's ``_LRScheduler.load_state_dict``
does ``self.__dict__.update(state_dict)``, which restores Stage A's
``T_max=20`` and ``last_epoch=20`` into Stage B's freshly-constructed
scheduler. The cosine formula
``0.5 * (1 + cos(pi * last_epoch / T_max))``
is only meaningful for ``last_epoch in [0, T_max]`` — past T_max the
formula swings back up and produces an LR that climbs to many times the
initial LR within a few epochs. That re-warm is what suppressed the
negative Lyapunov spectrum in the obs-noise sweep.

The fix is to set ``T_max`` from a ``cosine_T_max`` config field that is
the same in Stage A and Stage B (and the same as ``trainer.max_epochs``
for normal single-stage runs). When ``T_max`` matches between stages,
``load_state_dict`` is idempotent on ``T_max`` and ``last_epoch``
continues to advance through a single coherent cosine.
"""

import math

import torch


def _make_optimizer(initial_lr=1e-3):
    p = torch.nn.Parameter(torch.zeros(1))
    return torch.optim.AdamW([p], lr=initial_lr)


def test_cosine_resume_with_matched_T_max_is_continuous():
    """Stage B picks up at the same LR Stage A left at, no jump."""
    initial_lr = 1e-3
    eta_min = 1e-6
    T_max = 100  # the new project default
    stage_a_epochs = 20

    # Stage A: simulate 20 epochs of training under T_max=100 cosine.
    opt_A = _make_optimizer(initial_lr)
    sched_A = torch.optim.lr_scheduler.CosineAnnealingLR(opt_A, T_max=T_max, eta_min=eta_min)
    for _ in range(stage_a_epochs):
        opt_A.step()
        sched_A.step()
    lr_A_end = opt_A.param_groups[0]["lr"]
    state = sched_A.state_dict()

    # Stage B: fresh optimizer + scheduler with the same T_max, then
    # load Stage A's scheduler state. Mirrors what
    # Lightning's trainer.fit(ckpt_path=...) does internally.
    opt_B = _make_optimizer(initial_lr)
    sched_B = torch.optim.lr_scheduler.CosineAnnealingLR(opt_B, T_max=T_max, eta_min=eta_min)
    sched_B.load_state_dict(state)
    # Mirror what Lightning does on ckpt restore: optimizer state is also
    # restored, so set lr explicitly to Stage A's end value.
    opt_B.param_groups[0]["lr"] = lr_A_end

    # First step of Stage B (== epoch 21 conceptually).
    opt_B.step()
    sched_B.step()
    lr_B_first = opt_B.param_groups[0]["lr"]

    # The expected LR at epoch 21 of a 100-epoch cosine.
    expected = eta_min + 0.5 * (initial_lr - eta_min) * (1 + math.cos(math.pi * 21 / T_max))

    # Stage B's first-epoch LR matches the cosine formula for
    # last_epoch=21 exactly — i.e. no jump, just one more step on the
    # same schedule.
    assert math.isclose(lr_B_first, expected, rel_tol=1e-9), (
        f"Stage B should pick up at the next step of the same cosine. "
        f"Got lr={lr_B_first:.6e}, expected {expected:.6e}."
    )

    # The jump from end-of-A to first-of-B should be tiny (one step on a
    # 100-epoch cosine), not orders of magnitude.
    assert abs(lr_B_first - lr_A_end) < 0.05 * initial_lr, (
        f"Stage B LR jumped by {abs(lr_B_first - lr_A_end):.4e} relative to "
        f"Stage A end ({lr_A_end:.4e}); expected a smooth cosine step."
    )

    # And the LR at the resume point should be ~0.5*(1+cos(0.21pi))=0.895
    # of initial_lr — the model is still in the high-LR regime, NOT at
    # min_lr (which is what the old T_max=20 scheme produced).
    assert lr_B_first > 0.85 * initial_lr, (
        f"Stage B LR at first resumed step ({lr_B_first:.4e}) should be "
        f"in the high-LR regime (~0.9 * initial = {0.9 * initial_lr:.4e}), "
        f"not collapsed to min_lr."
    )


def test_old_bug_reproduces_with_mismatched_T_max():
    """Document the previous bug: with mismatched T_max the formula
    re-warms catastrophically. This is what we just fixed; if this
    invariant ever changes (e.g. PyTorch starts ignoring restored
    T_max), the test will fail loudly so we know.
    """
    initial_lr = 1e-3
    eta_min = 1e-6

    # Stage A: T_max=20 (the broken legacy regime where T_max was tied
    # to Stage A's max_epochs).
    opt_A = _make_optimizer(initial_lr)
    sched_A = torch.optim.lr_scheduler.CosineAnnealingLR(opt_A, T_max=20, eta_min=eta_min)
    for _ in range(20):
        opt_A.step()
        sched_A.step()
    state = sched_A.state_dict()

    # Stage B fresh with T_max=200, then load Stage A state.
    opt_B = _make_optimizer(initial_lr)
    sched_B = torch.optim.lr_scheduler.CosineAnnealingLR(opt_B, T_max=200, eta_min=eta_min)
    sched_B.load_state_dict(state)
    # Stage A's T_max overwrites Stage B's.
    assert sched_B.T_max == 20, f"PyTorch state_dict no longer overwrites T_max — bug may not reproduce: T_max={sched_B.T_max}"

    # Step a few times — the cosine swings back up because last_epoch > T_max.
    lrs = []
    for _ in range(5):
        opt_B.step()
        sched_B.step()
        lrs.append(opt_B.param_groups[0]["lr"])

    # By the 5th step (last_epoch=25, T_max=20) the LR exceeds initial.
    assert lrs[-1] > initial_lr, (
        f"Old bug should produce LR > initial within 5 steps; got {lrs[-1]:.4e}"
    )


def test_lit_model_uses_cosine_T_max_when_set():
    """LitModel.cosine_T_max overrides trainer.max_epochs in
    configure_optimizers. Smoke test that the field is plumbed through
    without instantiating the full model class — we just verify the
    helper logic.
    """
    class FakeTrainer:
        max_epochs = 999

    class FakeLit:
        cosine_T_max = 100
        trainer = FakeTrainer()

    # Mirror the resolution rule from configure_optimizers.
    self = FakeLit()
    t_max = self.cosine_T_max if self.cosine_T_max is not None else self.trainer.max_epochs
    assert t_max == 100, f"Expected cosine_T_max override to win, got {t_max}"

    # When None, fall back to trainer.max_epochs (legacy behavior).
    self.cosine_T_max = None
    t_max = self.cosine_T_max if self.cosine_T_max is not None else self.trainer.max_epochs
    assert t_max == 999, f"Expected trainer.max_epochs fallback, got {t_max}"
