"""Data quality validation for incoming sensor data.

Used both offline (checking a freshly generated/ingested batch before it
enters the feature pipeline) and online (the API rejects individual
readings that fail physical-plausibility checks before scoring them).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.data.schema import NON_NEGATIVE_COLUMNS, REQUIRED_RAW_COLUMNS


@dataclass
class DataQualityReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    n_rows: int = 0
    n_duplicates: int = 0
    null_fraction_by_column: dict[str, float] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0


def validate_sensor_data(df: pd.DataFrame) -> DataQualityReport:
    """Structural and physical-plausibility checks on a raw sensor batch.

    Errors indicate the batch should be quarantined rather than fed
    downstream (see architecture: Data Validation -> Quarantine).
    Warnings are informational and do not block processing.
    """
    report = DataQualityReport(n_rows=len(df))

    missing_columns = [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing_columns:
        report.errors.append(f"missing required columns: {missing_columns}")
        return report

    if df["machine_id"].isna().any():
        report.errors.append("machine_id contains nulls")
    if df["timestamp"].isna().any():
        report.errors.append("timestamp contains nulls")

    for column in NON_NEGATIVE_COLUMNS:
        if (df[column] < 0).any():
            report.errors.append(f"negative values detected in '{column}'")

    report.n_duplicates = int(df.duplicated(subset=["machine_id", "timestamp"]).sum())
    if report.n_duplicates > 0:
        report.warnings.append(f"{report.n_duplicates} duplicate (machine_id, timestamp) rows")

    sensor_columns = [c for c in REQUIRED_RAW_COLUMNS if c not in ("machine_id", "timestamp")]
    null_fractions = df[sensor_columns].isna().mean().to_dict()
    report.null_fraction_by_column = {k: float(v) for k, v in null_fractions.items()}
    for column, fraction in report.null_fraction_by_column.items():
        if fraction > 0.5:
            report.errors.append(f"'{column}' is more than 50% null ({fraction:.1%}) — likely a dead sensor")
        elif fraction > 0.05:
            report.warnings.append(f"'{column}' is {fraction:.1%} null")

    return report


def clean_sensor_data(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate, sort, and flag physically-invalid readings.

    Invalid readings (negative rpm/pressure/flow/current/power) are kept
    in the frame but flagged via ``data_quality_issue`` rather than
    dropped, so downstream interpolation can fill them from neighboring
    valid readings instead of losing the timestamp entirely.
    """
    df = df.drop_duplicates(subset=["machine_id", "timestamp"]).copy()
    df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)

    invalid_mask = pd.Series(False, index=df.index)
    for column in NON_NEGATIVE_COLUMNS:
        invalid_mask |= df[column] < 0
    df["data_quality_issue"] = invalid_mask.astype(int)

    non_negative_present = [c for c in NON_NEGATIVE_COLUMNS if c in df.columns]
    df.loc[invalid_mask, non_negative_present] = pd.NA

    return df
