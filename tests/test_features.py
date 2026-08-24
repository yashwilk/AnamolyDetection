from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.feature_engineering import create_features, feature_column_names


def test_feature_generation_adds_expected_columns():
    df = pd.DataFrame(
        {
            "machine_id": ["M001", "M001"],
            "timestamp": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 00:15"]),
            "temperature": [70, 71],
            "pressure": [4.8, 4.9],
            "vibration": [0.1, 0.2],
            "current": [13, 14],
            "rpm": [1450, 1460],
            "flow_rate": [80, 81],
            "power": [5, 5.2],
        }
    )
    result = create_features(df)

    assert "temperature_delta" in result.columns
    assert "vibration_delta" in result.columns
    assert "power_per_flow" in result.columns
    assert "vibration_zscore" in result.columns


def test_all_declared_feature_columns_are_produced(sample_sensor_df):
    result = create_features(sample_sensor_df)
    expected = feature_column_names()
    missing = [c for c in expected if c not in result.columns]
    assert missing == []


def test_delta_is_first_difference_per_machine():
    df = pd.DataFrame(
        {
            "machine_id": ["M001", "M001", "M001"],
            "timestamp": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 00:15", "2026-01-01 00:30"]),
            "temperature": [70.0, 71.0, 73.0],
            "pressure": [4.8, 4.8, 4.8],
            "vibration": [0.1, 0.1, 0.1],
            "current": [13.0, 13.0, 13.0],
            "rpm": [1450.0, 1450.0, 1450.0],
            "flow_rate": [80.0, 80.0, 80.0],
            "power": [5.0, 5.0, 5.0],
        }
    )
    result = create_features(df)
    assert result["temperature_delta"].isna().iloc[0]
    assert result["temperature_delta"].iloc[1] == 1.0
    assert result["temperature_delta"].iloc[2] == 2.0


def test_rolling_stats_do_not_leak_across_machines():
    df = pd.DataFrame(
        {
            "machine_id": ["A"] * 3 + ["B"] * 3,
            "timestamp": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 00:15", "2026-01-01 00:30"] * 2),
            "temperature": [70.0, 70.0, 70.0, 200.0, 200.0, 200.0],
            "pressure": [4.8] * 6,
            "vibration": [0.1] * 6,
            "current": [13.0] * 6,
            "rpm": [1450.0] * 6,
            "flow_rate": [80.0] * 6,
            "power": [5.0] * 6,
        }
    )
    result = create_features(df, rolling_window=3, rolling_min_periods=1)
    machine_a_mean = result.loc[result["machine_id"] == "A", "temperature_rolling_mean"]
    assert (machine_a_mean == 70.0).all()


def test_missing_flag_set_for_nan_sensor_reading():
    df = pd.DataFrame(
        {
            "machine_id": ["M001", "M001"],
            "timestamp": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 00:15"]),
            "temperature": [70.0, np.nan],
            "pressure": [4.8, 4.8],
            "vibration": [0.1, 0.1],
            "current": [13.0, 13.0],
            "rpm": [1450.0, 1450.0],
            "flow_rate": [80.0, 80.0],
            "power": [5.0, 5.0],
        }
    )
    result = create_features(df)
    assert result["temperature_missing"].tolist() == [0, 1]
