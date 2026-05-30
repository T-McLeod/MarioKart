"""Tests for src.utils — global seeding."""
import random
import numpy as np
import torch
import pytest

from src.utils import seed_everything


def test_same_seed_produces_identical_random_outputs():
    seed_everything(42)
    r1, n1, t1 = random.random(), np.random.rand(), torch.rand(1).item()

    seed_everything(42)
    r2, n2, t2 = random.random(), np.random.rand(), torch.rand(1).item()

    assert r1 == r2, "random.random() not deterministic across identical seeds"
    assert n1 == n2, "np.random.rand() not deterministic across identical seeds"
    assert t1 == t2, "torch.rand() not deterministic across identical seeds"


def test_different_seeds_produce_different_outputs():
    seed_everything(1)
    t1 = torch.rand(16)
    seed_everything(2)
    t2 = torch.rand(16)
    assert not torch.allclose(t1, t2), "Different seeds produced identical torch samples"


def test_seed_affects_numpy():
    seed_everything(7)
    a = np.random.rand(100)
    seed_everything(7)
    b = np.random.rand(100)
    np.testing.assert_array_equal(a, b)


def test_seed_affects_stdlib_random():
    seed_everything(99)
    a = [random.randint(0, 10_000) for _ in range(50)]
    seed_everything(99)
    b = [random.randint(0, 10_000) for _ in range(50)]
    assert a == b


def test_seed_zero_is_valid():
    seed_everything(0)
    # Should not raise; sample to confirm RNG state is set
    _ = torch.rand(1)
    _ = np.random.rand()
    _ = random.random()
