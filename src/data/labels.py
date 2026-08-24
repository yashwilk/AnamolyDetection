"""Ground-truth label construction from recorded failure events.

A reading is labeled anomalous if it falls in the pre-failure window
(default 24h) leading up to a known failure on that machine. This is a
weak-label heuristic — real degradation onset time is unknown — used
only for offline evaluation (PR-AUC, ROC-AUC, MTTD), never as a training
target for the unsupervised models (Isolation Forest / Autoencoder both
train only on presumed-normal data).
"""

from __future__ import annotations

import pandas as pd


def create_labels(
    sensor_df: pd.DataFrame,
    failures_df: pd.DataFrame,
    pre_failure_window_hours: int = 24,
) -> pd.DataFrame:
    sensor_df = sensor_df.copy()
    sensor_df["anomaly"] = 0
    sensor_df["failure_type"] = pd.array([None] * len(sensor_df), dtype="object")

    windows = failures_df.copy()
    windows["window_start"] = windows["failure_timestamp"] - pd.Timedelta(hours=pre_failure_window_hours)

    for _, failure in windows.iterrows():
        mask = (
            (sensor_df["machine_id"] == failure["machine_id"])
            & (sensor_df["timestamp"] >= failure["window_start"])
            & (sensor_df["timestamp"] < failure["failure_timestamp"])
        )
        sensor_df.loc[mask, "anomaly"] = 1
        sensor_df.loc[mask, "failure_type"] = failure["failure_type"]

    return sensor_df
