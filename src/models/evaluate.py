"""Offline evaluation: threshold selection, ranking metrics, and the two
business-facing metrics that matter more than raw F1 for this use case —
alert aggregation and Mean Time To Detection (MTTD) ahead of a failure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def ranking_metrics(y_true: pd.Series, scores: pd.Series) -> dict[str, float]:
    """PR-AUC and ROC-AUC — evaluated on the raw continuous score rather
    than a thresholded prediction, since threshold choice is a separate,
    business-driven decision (see ``best_threshold_by_f1``)."""
    return {
        "pr_auc": average_precision_score(y_true, scores),
        "roc_auc": roc_auc_score(y_true, scores),
    }


def best_threshold_by_f1(y_true: pd.Series, scores: pd.Series) -> tuple[float, float]:
    """Sweep the precision-recall curve and return the score threshold
    that maximizes F1, plus that F1 value. Selected on the *validation*
    split, then applied unchanged to test/production."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    best_idx = int(np.argmax(f1))
    best_idx_clamped = min(best_idx, len(thresholds) - 1)
    return float(thresholds[best_idx_clamped]), float(f1[best_idx])


def aggregate_alerts(df: pd.DataFrame, score_threshold: float) -> pd.DataFrame:
    """Collapse raw per-reading scores into one alert record per machine
    per contiguous high-score period's outer bounds — what an operator
    actually wants to see, not a flood of individual 5-minute triggers."""
    df = df.copy()
    df["alert"] = df["anomaly_score"] > score_threshold

    alerts = []
    for machine_id, group in df.groupby("machine_id"):
        group = group.sort_values("timestamp")
        active = group[group["alert"]]
        if len(active) == 0:
            continue
        alerts.append(
            {
                "machine_id": machine_id,
                "first_detection": active["timestamp"].min(),
                "last_detection": active["timestamp"].max(),
                "max_score": active["anomaly_score"].max(),
                "number_of_alerts": len(active),
            }
        )
    return pd.DataFrame(alerts)


def calculate_mttd(alerts: pd.DataFrame, failures: pd.DataFrame) -> pd.DataFrame:
    """Lead time between first alert and the actual recorded failure —
    the metric that translates directly to avoided emergency-repair and
    downtime cost."""
    merged = alerts.merge(failures, on="machine_id", how="inner")
    merged["lead_time"] = merged["failure_timestamp"] - merged["first_detection"]
    return merged


def mttd_summary(mttd_df: pd.DataFrame) -> dict[str, float]:
    if len(mttd_df) == 0:
        return {"mean_lead_time_hours": float("nan"), "n_failures_detected": 0}
    lead_hours = mttd_df["lead_time"].dt.total_seconds() / 3600
    detected_early = lead_hours[lead_hours > 0]
    return {
        "mean_lead_time_hours": float(detected_early.mean()) if len(detected_early) else 0.0,
        "n_failures_detected": int(len(detected_early)),
        "n_failures_total": int(len(mttd_df)),
    }
