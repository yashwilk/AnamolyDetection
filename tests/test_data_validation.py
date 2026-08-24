from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.validation import clean_sensor_data, validate_sensor_data


def test_valid_data_passes(sample_sensor_df):
    report = validate_sensor_data(sample_sensor_df)
    assert report.is_valid
    assert report.errors == []


def test_missing_required_columns_is_an_error(sample_sensor_df):
    df = sample_sensor_df.drop(columns=["vibration"])
    report = validate_sensor_data(df)
    assert not report.is_valid
    assert any("missing required columns" in e for e in report.errors)


def test_negative_rpm_is_an_error(sample_sensor_df):
    df = sample_sensor_df.copy()
    df.loc[0, "rpm"] = -100
    report = validate_sensor_data(df)
    assert not report.is_valid
    assert any("rpm" in e for e in report.errors)


def test_null_machine_id_is_an_error(sample_sensor_df):
    df = sample_sensor_df.copy()
    df.loc[0, "machine_id"] = None
    report = validate_sensor_data(df)
    assert not report.is_valid
    assert any("machine_id" in e for e in report.errors)


def test_high_null_fraction_is_an_error(sample_sensor_df):
    df = sample_sensor_df.copy()
    df.loc[df.sample(frac=0.6, random_state=0).index, "temperature"] = np.nan
    report = validate_sensor_data(df)
    assert not report.is_valid
    assert any("temperature" in e for e in report.errors)


def test_clean_sensor_data_drops_duplicates_and_sorts(sample_sensor_df):
    duplicated = pd.concat([sample_sensor_df, sample_sensor_df.iloc[:5]], ignore_index=True)
    cleaned = clean_sensor_data(duplicated)

    assert len(cleaned) == len(sample_sensor_df)
    assert cleaned.duplicated(subset=["machine_id", "timestamp"]).sum() == 0
    for _, group in cleaned.groupby("machine_id"):
        assert group["timestamp"].is_monotonic_increasing


def test_clean_sensor_data_flags_invalid_readings(sample_sensor_df):
    df = sample_sensor_df.copy()
    df.loc[0, "flow_rate"] = -5
    cleaned = clean_sensor_data(df)
    assert cleaned.loc[0, "data_quality_issue"] == 1
    assert pd.isna(cleaned.loc[0, "flow_rate"])
