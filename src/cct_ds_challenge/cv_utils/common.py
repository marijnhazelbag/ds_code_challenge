from __future__ import annotations
import random
import time
from dataclasses import asdict, dataclass
from typing import Dict
import numpy as np
import pandas as pd
import torch


# ============================================================
# SHARED UTILITIES
# ============================================================


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Timer:
    """Simple named timer used for runtime reporting."""

    def __init__(self) -> None:
        self._times: Dict[str, float] = {}
        self._start_times: Dict[str, float] = {}

    def start(self, name: str) -> None:
        self._start_times[name] = time.perf_counter()

    def stop(self, name: str) -> None:
        if name not in self._start_times:
            raise KeyError(f"Timer '{name}' was not started")
        self._times[name] = time.perf_counter() - self._start_times[name]

    def summary_df(self) -> pd.DataFrame:
        """Return timer summary as a DataFrame."""
        return pd.DataFrame(
            [{"step": step_name, "seconds": seconds, "minutes": seconds / 60.0} for step_name, seconds in self._times.items()]
        ).sort_values("seconds", ascending=False)

    def to_dict(self) -> Dict[str, float]:
        """Return timer measurements as a dictionary."""
        return dict(self._times)


class EarlyStopping:
    """Track validation improvements and trigger early stopping."""

    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = -np.inf
        self.counter = 0
        self.should_stop = False

    def step(self, score: float) -> bool:
        """Update stopping state from a new validation score."""
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop

