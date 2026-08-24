"""Post-deployment monitoring: data drift (PSI) and prediction-level
health signals. This is the "ML Monitoring" box in the architecture
diagram feeding the retraining / champion-challenger decision.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def population_stability_index(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """PSI between a reference (training-time) distribution and a current
    one. Rule of thumb: < 0.1 no significant shift, 0.1-0.25 moderate
    shift worth watching, > 0.25 significant shift — retraining candidate.
    """
    expected = expected.dropna()
    actual = actual.dropna()

    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 2:
        return 0.0

    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)

    expected_pct = expected_counts / len(expected) + 1e-6
    actual_pct = actual_counts / len(actual) + 1e-6

    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


@dataclass
class DriftReport:
    feature_psi: dict[str, float]
    warnings: list[str]
    critical: list[str]

    @property
    def has_critical_drift(self) -> bool:
        return len(self.critical) > 0


def compute_feature_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    columns: list[str],
    psi_warning: float = 0.1,
    psi_critical: float = 0.25,
    bins: int = 10,
) -> DriftReport:
    feature_psi: dict[str, float] = {}
    warnings: list[str] = []
    critical: list[str] = []

    for column in columns:
        if column not in reference_df.columns or column not in current_df.columns:
            continue
        psi = population_stability_index(reference_df[column], current_df[column], bins=bins)
        feature_psi[column] = psi
        if psi >= psi_critical:
            critical.append(column)
        elif psi >= psi_warning:
            warnings.append(column)

    return DriftReport(feature_psi=feature_psi, warnings=warnings, critical=critical)


def daily_anomaly_rate(predictions: pd.DataFrame) -> pd.Series:
    """Fraction of readings flagged anomalous per day — a sudden jump
    usually means prediction drift (or a real fleet-wide issue) rather
    than routine per-machine anomalies, and is worth a human look."""
    return predictions.groupby(predictions["timestamp"].dt.date)["anomaly"].mean()


def false_positive_rate(predictions: pd.DataFrame, confirmed_column: str = "confirmed_failure") -> float:
    """Fraction of raised alerts that were *not* subsequently confirmed
    by an operator/maintenance record. Requires the alert-confirmation
    feedback loop shown in the architecture diagram to populate
    ``confirmed_column``."""
    alerted = predictions[predictions["anomaly"]]
    if len(alerted) == 0:
        return 0.0
    return float((~alerted[confirmed_column].astype(bool)).mean())
