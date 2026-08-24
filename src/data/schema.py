"""Canonical column names and physical constraints for sensor data.

Kept separate from validation.py so both the generator and the validator
agree on one definition instead of drifting apart.
"""

from __future__ import annotations

SENSOR_COLUMNS: list[str] = [
    "temperature",
    "pressure",
    "vibration",
    "current",
    "rpm",
    "flow_rate",
    "power",
]

ID_COLUMNS: list[str] = ["machine_id", "timestamp"]

REQUIRED_RAW_COLUMNS: list[str] = ID_COLUMNS + SENSOR_COLUMNS

# Sensors that are physically non-negative. A negative reading indicates a
# sensor/transmission fault rather than a real anomaly and is flagged as a
# data-quality issue rather than fed to the model as-is.
NON_NEGATIVE_COLUMNS: list[str] = ["rpm", "pressure", "flow_rate", "current", "power"]

MACHINE_COLUMNS: list[str] = [
    "machine_id",
    "machine_type",
    "site",
    "age_years",
    "manufacturer",
]
