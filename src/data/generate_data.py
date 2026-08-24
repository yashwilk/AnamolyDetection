"""Synthetic industrial pump fleet data generator.

Produces three artifacts under ``data/raw``:

- ``machines.csv``     — static fleet metadata (type, site, age, manufacturer)
- ``sensor_data.parquet`` — 15-minute sensor readings per machine, with
  injected point anomalies, sensor drift, a stuck-sensor failure, a
  multivariate anomaly, and pre-failure degradation trends
- ``failures.csv``     — ground-truth failure events used to derive labels

This is a *simulator*, not production ingestion — real deployments read
from the Kafka / API path in the architecture diagram (see README). It
exists so the rest of the pipeline (features, models, evaluation, API)
can be built and tested against realistic, labeled data.

Run: ``python -m src.data.generate_data``
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from src.config import AppConfig, load_config
from src.logging_config import configure_logging

logger = logging.getLogger(__name__)


def generate_machines(config: AppConfig, rng: np.random.Generator) -> pd.DataFrame:
    gen = config.data_generation
    machines = [
        {
            "machine_id": f"M{i:04d}",
            "machine_type": rng.choice(gen.machine_types),
            "site": rng.choice(gen.sites),
            "age_years": round(float(rng.uniform(0.5, 12)), 2),
            "manufacturer": rng.choice(gen.manufacturers),
        }
        for i in range(gen.n_machines)
    ]
    return pd.DataFrame(machines)


def generate_sensor_readings(
    config: AppConfig, machines: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """Simulate physically-coupled sensor readings for each active machine.

    RPM drives every other sensor (temperature, pressure, vibration,
    current, flow, power) through simple linear couplings plus noise, so
    the synthetic data has the kind of cross-sensor correlation structure
    a real pump exhibits — which is what makes multivariate anomalies
    (e.g. current up + flow down + vibration up together) detectable.
    """
    gen = config.data_generation
    active_machines = machines.head(gen.n_active_machines)
    timestamps = pd.date_range(start=gen.start_date, end=gen.end_date, freq=gen.freq)

    rows: list[dict] = []
    for _, machine in active_machines.iterrows():
        machine_id = machine["machine_id"]
        age = machine["age_years"]
        machine_type = machine["machine_type"]
        rpm_base = gen.rpm_base[machine_type]

        hours = timestamps.hour.to_numpy()
        operating_factor = 1 + 0.15 * np.sin(2 * np.pi * hours / 24)
        n = len(timestamps)

        rpm = rpm_base * operating_factor + rng.normal(0, 20, size=n)
        rpm_delta = rpm - rpm_base

        temperature = 65 + 0.015 * age + 0.01 * rpm_delta + rng.normal(0, 1.5, size=n)
        pressure = 4.8 + 0.0005 * rpm_delta + rng.normal(0, 0.08, size=n)
        vibration = 0.10 + 0.01 * age + 0.0001 * np.abs(rpm_delta) + rng.normal(0, 0.015, size=n)
        current = 13 + 0.003 * rpm_delta + rng.normal(0, 0.3, size=n)
        flow_rate = 82 + 0.02 * rpm_delta + rng.normal(0, 1, size=n)
        power = current * 0.4 + rng.normal(0, 0.15, size=n)

        machine_df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "machine_id": machine_id,
                "temperature": temperature,
                "pressure": pressure,
                "vibration": vibration,
                "current": current,
                "rpm": rpm,
                "flow_rate": flow_rate,
                "power": power,
            }
        )
        rows.append(machine_df)

    return pd.concat(rows, ignore_index=True)


def inject_point_anomalies(df: pd.DataFrame, rng: np.random.Generator, fraction: float) -> pd.DataFrame:
    """Sudden, isolated spikes — e.g. a transient sensor glitch."""
    n = int(len(df) * fraction)
    if n == 0:
        return df
    indices = rng.choice(df.index, size=n, replace=False)
    df.loc[indices, "vibration"] *= rng.uniform(4, 8, size=n)
    return df


def inject_sensor_drift(df: pd.DataFrame, rng: np.random.Generator, n_machines: int) -> pd.DataFrame:
    """Gradual calibration drift on a subset of machines' vibration sensor."""
    machine_ids = rng.choice(df["machine_id"].unique(), size=n_machines, replace=False)
    for machine_id in machine_ids:
        indices = df.index[df["machine_id"] == machine_id]
        drift = np.linspace(0, 1.5, len(indices))
        df.loc[indices, "vibration"] += drift
    return df


def inject_sensor_failure(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """A stuck temperature sensor: reports a frozen value for 100 readings."""
    machine_id = rng.choice(df["machine_id"].unique())
    indices = df.index[df["machine_id"] == machine_id]
    start = len(indices) // 2
    stuck_value = df.loc[indices[start], "temperature"]
    df.loc[indices[start : start + 100], "temperature"] = stuck_value
    return df


def inject_multivariate_anomaly(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """A blockage-like signature: current up, flow down, vibration up together."""
    machine_id = rng.choice(df["machine_id"].unique())
    indices = df.index[df["machine_id"] == machine_id]
    selected = indices[:100]
    df.loc[selected, "current"] *= 1.35
    df.loc[selected, "flow_rate"] *= 0.65
    df.loc[selected, "vibration"] *= 2.5
    return df


def generate_failures(config: AppConfig, sensor_df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Spread failures across each machine's *entire* time range.

    Sampling only from the tail of the range (as the original prototype
    did, via ``.iloc[-1000:]``) clusters every failure into the last few
    days of the dataset — which then falls entirely inside one time-based
    split. A validation split with zero positive examples makes threshold
    selection on it meaningless (F1/PR-AUC collapse to 0), and a training
    split with zero positives means the supervised models
    (XGBoost/LightGBM) have nothing to learn from at all. Sampling from
    the full range (minus enough headroom for a complete pre-failure
    window) gives every split a realistic mix of failures.
    """
    gen = config.data_generation
    min_offset_periods = int(pd.Timedelta(hours=gen.pre_failure_window_hours) / pd.Timedelta(gen.freq)) + 1

    failures = []
    candidate_ids = rng.choice(sensor_df["machine_id"].unique(), size=gen.n_failures, replace=False)
    for machine_id in candidate_ids:
        machine_timestamps = sensor_df.loc[sensor_df["machine_id"] == machine_id, "timestamp"]
        eligible_timestamps = machine_timestamps.iloc[min_offset_periods:]
        failure_time = rng.choice(eligible_timestamps.to_numpy())
        failure_type = rng.choice(gen.failure_types)
        failures.append(
            {
                "machine_id": machine_id,
                "failure_timestamp": pd.Timestamp(failure_time),
                "failure_type": failure_type,
            }
        )
    return pd.DataFrame(failures)


def inject_pre_failure_degradation(
    sensor_df: pd.DataFrame, failures_df: pd.DataFrame, window_hours: int
) -> pd.DataFrame:
    """Ramp vibration/temperature up and flow down in the lead-up to each
    failure, so the model has a learnable degradation signature rather than
    failures appearing out of nowhere."""
    for _, failure in failures_df.iterrows():
        machine = failure["machine_id"]
        failure_time = failure["failure_timestamp"]
        start_time = failure_time - pd.Timedelta(hours=window_hours)

        mask = (
            (sensor_df["machine_id"] == machine)
            & (sensor_df["timestamp"] >= start_time)
            & (sensor_df["timestamp"] < failure_time)
        )
        n = int(mask.sum())
        if n == 0:
            continue

        degradation = np.linspace(0, 1, n)
        sensor_df.loc[mask, "vibration"] += degradation * 0.5
        sensor_df.loc[mask, "temperature"] += degradation * 10
        sensor_df.loc[mask, "flow_rate"] -= degradation * 15
    return sensor_df


def generate_all(config: AppConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(config.random_state)

    logger.info("Generating machine fleet metadata (n=%d)", config.data_generation.n_machines)
    machines_df = generate_machines(config, rng)

    logger.info(
        "Simulating sensor readings for %d active machines",
        config.data_generation.n_active_machines,
    )
    sensor_df = generate_sensor_readings(config, machines_df, rng)

    logger.info("Injecting anomalies and failures")
    gen = config.data_generation
    sensor_df = inject_point_anomalies(sensor_df, rng, gen.point_anomaly_fraction)
    sensor_df = inject_sensor_drift(sensor_df, rng, gen.n_drift_machines)
    sensor_df = inject_sensor_failure(sensor_df, rng)
    sensor_df = inject_multivariate_anomaly(sensor_df, rng)

    failures_df = generate_failures(config, sensor_df, rng)
    sensor_df = inject_pre_failure_degradation(sensor_df, failures_df, gen.pre_failure_window_hours)

    return machines_df, sensor_df, failures_df


def save_all(
    config: AppConfig,
    machines_df: pd.DataFrame,
    sensor_df: pd.DataFrame,
    failures_df: pd.DataFrame,
) -> None:
    config.paths.raw_dir.mkdir(parents=True, exist_ok=True)
    machines_df.to_csv(config.paths.machines_file, index=False)
    sensor_df.to_parquet(config.paths.sensor_file, index=False)
    failures_df.to_csv(config.paths.failures_file, index=False)
    logger.info(
        "Wrote %s (%d rows), %s (%d rows), %s (%d rows)",
        config.paths.machines_file,
        len(machines_df),
        config.paths.sensor_file,
        len(sensor_df),
        config.paths.failures_file,
        len(failures_df),
    )


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    machines_df, sensor_df, failures_df = generate_all(config)
    save_all(config, machines_df, sensor_df, failures_df)


if __name__ == "__main__":
    main()
