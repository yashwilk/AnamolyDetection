"""Single entry point that chains the offline pipeline stages together:
generate (synthetic data only) -> validate -> train -> evaluate report.

Each stage is also runnable standalone as its own module (see README),
this just wires them for a one-command local run / CI smoke test.

Usage:
    python scripts/run_pipeline.py --stages generate validate train
    python scripts/run_pipeline.py --stages train        # reuse existing data/raw
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from src.config import load_config
from src.data.generate_data import generate_all, save_all
from src.data.validation import validate_sensor_data
from src.logging_config import configure_logging
from src.models.train import run_training

logger = logging.getLogger(__name__)

STAGES = ["generate", "validate", "train"]


def stage_generate(config) -> None:
    machines_df, sensor_df, failures_df = generate_all(config)
    save_all(config, machines_df, sensor_df, failures_df)


def stage_validate(config) -> None:
    sensor_df = pd.read_parquet(config.paths.sensor_file)
    report = validate_sensor_data(sensor_df)
    for warning in report.warnings:
        logger.warning(warning)
    for error in report.errors:
        logger.error(error)
    if not report.is_valid:
        raise SystemExit(f"Data quality validation failed with {len(report.errors)} error(s); see log above.")
    logger.info("Data quality validation passed (%d rows checked)", report.n_rows)


def stage_train(config) -> None:
    metadata = run_training(config)
    logger.info(
        "Training complete. model_version=%s threshold=%.4f",
        metadata["model_version"],
        metadata["threshold"],
    )


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=STAGES)
    args = parser.parse_args()

    config = load_config(args.config)
    dispatch = {"generate": stage_generate, "validate": stage_validate, "train": stage_train}

    for stage in args.stages:
        logger.info("=== Stage: %s ===", stage)
        dispatch[stage](config)


if __name__ == "__main__":
    main()
