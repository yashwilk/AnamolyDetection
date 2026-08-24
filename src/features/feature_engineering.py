"""Feature engineering for pump sensor time series.

Every feature here is computed **causally** (only using current and past
readings, grouped per machine) so the exact same function can run in
training (over a static dataframe) and in the live API (over a rolling
per-machine buffer) without a train/serve skew.

Feature families:

- Rolling statistics: delta, rolling mean, rolling std per sensor
- Cross-sensor ratios: physically-motivated efficiency indicators
- Rolling z-scores: how far the current reading is from this machine's
  own recent baseline, in units of its own recent standard deviation
"""

from __future__ import annotations

import pandas as pd

from src.data.schema import SENSOR_COLUMNS

RATIO_EPSILON = 1e-6


def add_rolling_statistics(
    df: pd.DataFrame,
    sensor_columns: list[str] = SENSOR_COLUMNS,
    window: int = 8,
    min_periods: int = 3,
) -> pd.DataFrame:
    df = df.sort_values(["machine_id", "timestamp"])
    grouped = df.groupby("machine_id")

    for column in sensor_columns:
        df[f"{column}_delta"] = grouped[column].diff()
        df[f"{column}_rolling_mean"] = grouped[column].transform(
            lambda s: s.rolling(window, min_periods=min_periods).mean()
        )
        df[f"{column}_rolling_std"] = grouped[column].transform(
            lambda s: s.rolling(window, min_periods=min_periods).std()
        )
    return df


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    df["power_per_flow"] = df["power"] / (df["flow_rate"] + RATIO_EPSILON)
    df["current_per_rpm"] = df["current"] / (df["rpm"] + RATIO_EPSILON)
    df["pressure_per_flow"] = df["pressure"] / (df["flow_rate"] + RATIO_EPSILON)
    df["vibration_per_rpm"] = df["vibration"] / (df["rpm"] + RATIO_EPSILON)
    return df


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    return (series - rolling_mean) / (rolling_std + RATIO_EPSILON)


def add_rolling_zscores(
    df: pd.DataFrame,
    sensor_columns: list[str] = SENSOR_COLUMNS,
    window: int = 96,
) -> pd.DataFrame:
    grouped = df.groupby("machine_id")
    for column in sensor_columns:
        df[f"{column}_zscore"] = grouped[column].transform(lambda s: _rolling_zscore(s, window))
    return df


def add_missing_flags(df: pd.DataFrame, sensor_columns: list[str] = SENSOR_COLUMNS) -> pd.DataFrame:
    for column in sensor_columns:
        df[f"{column}_missing"] = df[column].isna().astype(int)
    return df


def interpolate_sensors(
    df: pd.DataFrame, sensor_columns: list[str] = SENSOR_COLUMNS, limit: int = 4
) -> pd.DataFrame:
    """Fill short gaps (<= limit consecutive points) per machine via linear
    interpolation. Longer gaps are left null and surface via the
    ``*_missing`` flags rather than being silently fabricated."""
    df[sensor_columns] = df.groupby("machine_id")[sensor_columns].transform(
        lambda s: s.interpolate(limit=limit)
    )
    return df


def create_features(
    df: pd.DataFrame,
    sensor_columns: list[str] = SENSOR_COLUMNS,
    rolling_window: int = 8,
    rolling_min_periods: int = 3,
    zscore_window: int = 96,
    interpolation_limit: int = 4,
) -> pd.DataFrame:
    """Full feature pipeline: interpolate small gaps, flag missing values,
    then derive rolling stats, ratios, and rolling z-scores.

    Column order/grouping is preserved from the input except for the new
    derived columns appended at the end.
    """
    df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)
    df = add_missing_flags(df, sensor_columns)
    df = interpolate_sensors(df, sensor_columns, limit=interpolation_limit)
    df = add_rolling_statistics(df, sensor_columns, rolling_window, rolling_min_periods)
    df = add_ratio_features(df)
    df = add_rolling_zscores(df, sensor_columns, zscore_window)
    return df


def feature_column_names(sensor_columns: list[str] = SENSOR_COLUMNS) -> list[str]:
    """The full model-input feature list produced by ``create_features``,
    in a stable order. Persisted alongside trained models so inference
    always selects columns identically to training."""
    derived = []
    for column in sensor_columns:
        derived.append(f"{column}_delta")
        derived.append(f"{column}_rolling_mean")
        derived.append(f"{column}_rolling_std")
    ratios = ["power_per_flow", "current_per_rpm", "pressure_per_flow", "vibration_per_rpm"]
    zscores = [f"{column}_zscore" for column in sensor_columns]
    return list(sensor_columns) + derived + ratios + zscores
