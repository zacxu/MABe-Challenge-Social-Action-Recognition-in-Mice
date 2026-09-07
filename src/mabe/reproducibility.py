"""Cross-library random seed controls."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class SeedState:
    """Applied seed and explicit parameters needed by estimator libraries."""

    seed: int
    xgboost_params: dict[str, int]


def seed_everything(seed: int, *, deterministic_torch: bool = True) -> SeedState:
    """Seed Python, NumPy, XGBoost parameters, and PyTorch.

    XGBoost has no process-wide seed setter, so the returned parameters must be
    merged into every estimator configuration.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be between 0 and 2**32 - 1")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():  # type: ignore[no-untyped-call]
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    return SeedState(seed=seed, xgboost_params={"random_state": seed, "seed": seed})
