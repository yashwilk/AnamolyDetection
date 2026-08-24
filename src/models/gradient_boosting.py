"""Supervised gradient-boosted classifiers trained on weak failure labels.

Isolation Forest and the Autoencoder are unsupervised — trained only on
presumed-normal operation, with no notion of what a *known* failure
looked like. XGBoost and LightGBM here are trained directly on the
pre-failure labels from ``src/data/labels.py`` (1 = inside a known
failure's lead-up window), adding a genuinely different signal to the
hybrid score: pattern-matched risk against the fleet's actual failure
history, not just deviation from a machine's own baseline.

They train on *unscaled* features — tree ensembles split on raw
thresholds and don't benefit from standardization the way the
Autoencoder does — and their ``predict_proba`` output is already a
well-formed [0, 1] score, so no separate score normalizer is persisted
for them (unlike Isolation Forest / Autoencoder, see
``src/inference/predictor.py``).

Persistence uses each library's native format (XGBoost's UBJ, LightGBM's
text format) rather than pickle/joblib: gradient-boosting library
internals changing pickle layout across versions is a well-known
production pain point that native formats sidestep.
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from src.config import GradientBoostingConfig

# Anomalies are rare by construction (a handful of pre-failure windows out
# of months of normal operation). Without class weighting, both libraries
# would learn to just predict "normal" for everything and still score
# well on accuracy while being useless for detection.
DEFAULT_SCALE_POS_WEIGHT = 1.0


def compute_scale_pos_weight(y: pd.Series | np.ndarray) -> float:
    """Ratio of negative to positive examples, the standard scale_pos_weight
    heuristic for both XGBoost and LightGBM under class imbalance."""
    y = np.asarray(y)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0:
        return DEFAULT_SCALE_POS_WEIGHT
    return n_neg / n_pos


class XGBoostAnomalyModel:
    def __init__(self, config: GradientBoostingConfig, random_state: int = 42):
        self.config = config
        self.model = XGBClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            learning_rate=config.learning_rate,
            subsample=config.subsample,
            colsample_bytree=config.colsample_bytree,
            scale_pos_weight=config.scale_pos_weight or DEFAULT_SCALE_POS_WEIGHT,
            eval_metric="aucpr",
            random_state=random_state,
            n_jobs=-1,
        )
        self._fitted = False

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> XGBoostAnomalyModel:
        self.model.fit(X, y)
        self._fitted = True
        return self

    def score(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Probability of the positive (anomalous) class — already in [0, 1]."""
        if not self._fitted:
            raise RuntimeError("XGBoostAnomalyModel must be fit before scoring")
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str | Path) -> None:
        self.model.save_model(str(path))

    @classmethod
    def load(
        cls, path: str | Path, config: GradientBoostingConfig, random_state: int = 42
    ) -> XGBoostAnomalyModel:
        instance = cls(config, random_state)
        instance.model.load_model(str(path))
        instance._fitted = True
        return instance


class LightGBMAnomalyModel:
    def __init__(self, config: GradientBoostingConfig, random_state: int = 42):
        self.config = config
        self.random_state = random_state
        self.model = LGBMClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            learning_rate=config.learning_rate,
            subsample=config.subsample,
            colsample_bytree=config.colsample_bytree,
            scale_pos_weight=config.scale_pos_weight or DEFAULT_SCALE_POS_WEIGHT,
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )
        self._booster: lgb.Booster | None = None

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> LightGBMAnomalyModel:
        self.model.fit(X, y)
        self._booster = self.model.booster_
        return self

    def score(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Probability of the positive (anomalous) class — already in [0, 1].

        Uses the raw ``Booster`` (rather than the sklearn wrapper) so
        scoring works identically whether the model was just fit or
        loaded from a native-format checkpoint.
        """
        if self._booster is None:
            raise RuntimeError("LightGBMAnomalyModel must be fit before scoring")
        return np.asarray(self._booster.predict(X))

    def save(self, path: str | Path) -> None:
        if self._booster is None:
            raise RuntimeError("LightGBMAnomalyModel must be fit before saving")
        self._booster.save_model(str(path))

    @classmethod
    def load(
        cls, path: str | Path, config: GradientBoostingConfig, random_state: int = 42
    ) -> LightGBMAnomalyModel:
        instance = cls(config, random_state)
        instance._booster = lgb.Booster(model_file=str(path))
        return instance
