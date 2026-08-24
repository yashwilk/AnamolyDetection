"""Offline SHAP attribution for the Isolation Forest.

This is a *validation* tool, not part of the live prediction path — it
runs against a batch (e.g. the test set, or a specific incident window)
to check that the rule-based explanations in ``src/inference/explain.py``
actually track what the tree ensemble keys off, and to produce the
global feature-importance summary plot for model documentation / audits.

Requires the optional ``shap`` dependency (see requirements.txt).
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def compute_shap_values(model, X: pd.DataFrame):
    """Return per-feature SHAP values for an Isolation Forest's raw
    sklearn estimator (``IsolationForestModel.model``) over dataframe X."""
    import shap  # local import: optional, heavier dependency

    explainer = shap.TreeExplainer(model)
    return explainer.shap_values(X)


def plot_summary(model, X: pd.DataFrame, show: bool = True):
    import matplotlib.pyplot as plt
    import shap

    shap_values = compute_shap_values(model, X)
    shap.summary_plot(shap_values, X, show=show)
    return plt.gcf()


def top_global_features(model, X: pd.DataFrame, top_n: int = 10) -> pd.Series:
    """Mean absolute SHAP value per feature, ranked descending — the
    global counterpart to the per-row explanations served by the API."""
    import numpy as np

    shap_values = np.asarray(compute_shap_values(model, X))
    importance = pd.Series(data=np.abs(shap_values).mean(axis=0), index=X.columns)
    return importance.sort_values(ascending=False).head(top_n)
