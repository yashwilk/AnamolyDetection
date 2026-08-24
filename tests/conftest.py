from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.schema import SENSOR_COLUMNS
from src.logging_config import configure_logging

configure_logging()


@pytest.fixture
def sensor_columns() -> list[str]:
    return list(SENSOR_COLUMNS)


@pytest.fixture
def sample_sensor_df() -> pd.DataFrame:
    """Two machines, 120 clean readings each at 15-minute cadence — enough
    history for the rolling_window=8 and a partial zscore_window to be
    exercised without needing the full 96-point window everywhere."""
    rng = np.random.default_rng(0)
    n_per_machine = 120
    timestamps = pd.date_range("2026-01-01", periods=n_per_machine, freq="15min")

    frames = []
    for machine_id in ["M0001", "M0002"]:
        frames.append(
            pd.DataFrame(
                {
                    "machine_id": machine_id,
                    "timestamp": timestamps,
                    "temperature": rng.normal(70, 1.5, n_per_machine),
                    "pressure": rng.normal(4.8, 0.08, n_per_machine),
                    "vibration": rng.normal(0.15, 0.02, n_per_machine),
                    "current": rng.normal(13, 0.3, n_per_machine),
                    "rpm": rng.normal(1450, 20, n_per_machine),
                    "flow_rate": rng.normal(82, 1.0, n_per_machine),
                    "power": rng.normal(5.2, 0.15, n_per_machine),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)
