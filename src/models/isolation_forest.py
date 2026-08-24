"""Isolation Forest wrapper: tree-based unsupervised anomaly detector.

Captures multivariate outliers (unusual combinations of features) without
assuming any particular distribution, which complements the Autoencoder's
reconstruction-error view in the hybrid score.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.config import IsolationForestConfig


class IsolationForestModel:
    def __init__(self, config: IsolationForestConfig, random_state: int = 42):
        self.model = IsolationForest(
            n_estimators=config.n_estimators,
            contamination=config.contamination,
            max_samples=config.max_samples,
            random_state=random_state,
            n_jobs=config.n_jobs,
        )
        self._fitted = False

    def fit(self, X: pd.DataFrame | np.ndarray) -> IsolationForestModel:
        self.model.fit(X)
        self._fitted = True
        return self

    def score(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Higher = more anomalous (sklearn's decision_function is
        inverted: lower/negative means anomalous, so we flip the sign)."""
        if not self._fitted:
            raise RuntimeError("IsolationForestModel must be fit before scoring")
        return -self.model.decision_function(X)

    def save(self, path: str | Path) -> None:
        joblib.dump(self.model, path)

    @classmethod
    def load(
        cls, path: str | Path, config: IsolationForestConfig, random_state: int = 42
    ) -> IsolationForestModel:
        instance = cls(config, random_state)
        instance.model = joblib.load(path)
        instance._fitted = True
        return instance
