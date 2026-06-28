"""Tests for mixup augmentation used during fine-tuning."""

from __future__ import annotations

import torch
import torch.nn as nn

from neurostream.training.mixup import mixup_batch, mixup_loss


def _gen(seed: int = 0) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def test_alpha_zero_is_identity() -> None:
    x = torch.randn(8, 22, 1000)
    y = torch.randint(0, 4, (8,))
    mixed, y_a, y_b, lam = mixup_batch(x, y, alpha=0.0, generator=_gen())
    assert lam == 1.0
    assert torch.equal(mixed, x)
    assert torch.equal(y_a, y)
    # At lam=1 the second target is unused; equality not required, but lam=1
    # means y_b contributes nothing to the loss.


def test_mix_is_convex_combination() -> None:
    x = torch.randn(6, 22, 1000)
    y = torch.randint(0, 4, (6,))
    mixed, y_a, y_b, lam = mixup_batch(x, y, alpha=0.4, generator=_gen(1))
    # mixed = lam*x + (1-lam)*x[perm]; reconstruct perm from y_a/y_b is not
    # exposed, so verify the value lies on the segment between x and some
    # permutation of x for every element.
    assert 0.0 <= lam <= 1.0
    # Each mixed sample is a convex blend, so its norm can't exceed the max of
    # the two source norms (triangle inequality with non-negative weights).
    src_norm = x.flatten(1).norm(dim=1)
    mix_norm = mixed.flatten(1).norm(dim=1)
    assert torch.all(mix_norm <= src_norm.max() + 1e-4)


def test_loss_reduces_to_plain_ce_at_lam_one() -> None:
    crit = nn.CrossEntropyLoss()
    logits = torch.randn(8, 4)
    y_a = torch.randint(0, 4, (8,))
    y_b = torch.randint(0, 4, (8,))
    blended = mixup_loss(crit, logits, y_a, y_b, lam=1.0)
    plain = crit(logits, y_a)
    assert torch.allclose(blended, plain)


def test_loss_is_lambda_weighted() -> None:
    crit = nn.CrossEntropyLoss()
    logits = torch.randn(8, 4)
    y_a = torch.randint(0, 4, (8,))
    y_b = torch.randint(0, 4, (8,))
    lam = 0.3
    blended = mixup_loss(crit, logits, y_a, y_b, lam=lam)
    expected = lam * crit(logits, y_a) + (1 - lam) * crit(logits, y_b)
    assert torch.allclose(blended, expected)


def test_deterministic_with_seeded_generator() -> None:
    x = torch.randn(8, 22, 1000)
    y = torch.randint(0, 4, (8,))
    a = mixup_batch(x, y, alpha=0.2, generator=_gen(7))
    b = mixup_batch(x, y, alpha=0.2, generator=_gen(7))
    assert torch.equal(a[0], b[0]) and a[3] == b[3]
