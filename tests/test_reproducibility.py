from __future__ import annotations

import os
import random

import numpy as np
import pytest
import torch

from mabe.reproducibility import seed_everything


def test_seed_everything_reproduces_python_numpy_and_torch() -> None:
    state = seed_everything(1234, deterministic_torch=False)
    first = (random.random(), np.random.random(), torch.rand(3))
    seed_everything(1234, deterministic_torch=False)
    second = (random.random(), np.random.random(), torch.rand(3))

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])
    assert state.xgboost_params == {"random_state": 1234, "seed": 1234}
    assert os.environ["PYTHONHASHSEED"] == "1234"


@pytest.mark.parametrize("seed", [True, 1.5, "1", None])
def test_seed_rejects_non_integer_values(seed: object) -> None:
    with pytest.raises(TypeError):
        seed_everything(seed)  # type: ignore[arg-type]


@pytest.mark.parametrize("seed", [-1, 2**32])
def test_seed_rejects_out_of_range_values(seed: int) -> None:
    with pytest.raises(ValueError):
        seed_everything(seed)
