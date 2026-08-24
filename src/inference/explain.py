"""Human-facing interpretation of a raw hybrid anomaly score: severity
tier, a best-guess failure mode, and a plain-language explanation.

This is deliberately rule-based rather than a second ML model — it needs
to be fast (runs on every prediction, every 5 minutes, per machine),
deterministic, and auditable by the operations team it's built for.
Offline, richer SHAP-based attribution (src/models/shap_explainability.py)
is used to validate that these rules track what the models actually key
off.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import SeverityThresholdsConfig

# Below this magnitude, a sensor's z-score is treated as within normal
# scatter rather than a meaningful contributor to an anomaly explanation.
CONTRIBUTION_THRESHOLD = 1.5

# Minimum "confidence" score (see classify_anomaly_type) required to name
# a specific failure mode instead of falling back to "unspecified_anomaly".
FAILURE_TYPE_CONFIDENCE_THRESHOLD = 2.0

STATISTICAL_SCORE_CAP = 6.0


def severity_from_score(score: float, thresholds: SeverityThresholdsConfig) -> str:
    if score < thresholds.low:
        return "LOW"
    if score < thresholds.medium:
        return "MEDIUM"
    if score < thresholds.high:
        return "HIGH"
    return "CRITICAL"


def compute_statistical_score(row: pd.Series, zscore_columns: list[str]) -> float:
    """A 0-1 statistical anomaly indicator: mean absolute z-score across
    sensors, clipped and rescaled. This is the third leg of the hybrid
    score, giving a transparent, model-free sanity signal alongside the
    Isolation Forest and Autoencoder outputs."""
    values = []
    for column in zscore_columns:
        value = row.get(column)
        if value is not None and not pd.isna(value):
            values.append(min(abs(float(value)), STATISTICAL_SCORE_CAP))
    if not values:
        return 0.0
    return float(np.mean(values) / STATISTICAL_SCORE_CAP)


def classify_anomaly_type(row: pd.Series) -> str:
    """Best-guess failure mode from which sensors are most out of line,
    matching the three failure types tracked in ``failures.csv``:
    bearing_failure, overheating, pump_blockage."""

    def z(column: str) -> float:
        value = row.get(column)
        return float(value) if value is not None and not pd.isna(value) else 0.0

    vibration_z = z("vibration_zscore")
    temperature_z = z("temperature_zscore")
    flow_z = z("flow_rate_zscore")
    pressure_z = z("pressure_zscore")
    current_z = z("current_zscore")

    candidates = {
        "bearing_failure": abs(vibration_z),
        "overheating": max(temperature_z, 0.0),
        "pump_blockage": max(-flow_z, 0.0) + 0.5 * max(current_z, 0.0) + 0.5 * max(pressure_z, 0.0),
    }

    best_type = max(candidates, key=lambda name: candidates[name])
    if candidates[best_type] < FAILURE_TYPE_CONFIDENCE_THRESHOLD:
        return "unspecified_anomaly"
    return best_type


def build_explanation(row: pd.Series, zscore_columns: list[str], top_n: int = 3) -> str:
    """Plain-language summary of the top contributing sensors, e.g.
    'vibration is 4.2 std devs above its recent baseline; temperature is
    3.1 std devs above its recent baseline'."""
    contributions = []
    for column in zscore_columns:
        value = row.get(column)
        if value is None or pd.isna(value):
            continue
        value = float(value)
        if abs(value) < CONTRIBUTION_THRESHOLD:
            continue
        sensor_name = column.replace("_zscore", "").replace("_", " ")
        direction = "above" if value > 0 else "below"
        text = f"{sensor_name} is {abs(value):.1f} std devs {direction} its recent baseline"
        contributions.append((abs(value), text))

    if not contributions:
        return (
            "Anomaly score elevated without a single dominant sensor deviation "
            "— likely a subtle multivariate pattern."
        )

    contributions.sort(key=lambda pair: pair[0], reverse=True)
    top = [text for _, text in contributions[:top_n]]
    return "; ".join(top) + "."
