"""Typed access to configs/config.yaml.

Every module in the project should read runtime parameters through
``load_config()`` rather than hardcoding values, so behavior (feature
windows, model hyperparameters, thresholds) can change from one file
without touching code.
"""

from __future__ import annotations

from dataclasses import astuple, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


@dataclass(frozen=True)
class PathsConfig:
    raw_dir: Path
    processed_dir: Path
    features_dir: Path
    models_dir: Path
    machines_file: Path
    sensor_file: Path
    failures_file: Path
    features_file: Path


@dataclass(frozen=True)
class DataGenerationConfig:
    n_machines: int
    n_active_machines: int
    machine_types: list[str]
    sites: list[str]
    manufacturers: list[str]
    start_date: str
    end_date: str
    freq: str
    rpm_base: dict[str, float]
    n_failures: int
    n_drift_machines: int
    point_anomaly_fraction: float
    failure_types: list[str]
    pre_failure_window_hours: int


@dataclass(frozen=True)
class FeatureEngineeringConfig:
    rolling_window: int
    rolling_min_periods: int
    zscore_window: int
    interpolation_limit: int


@dataclass(frozen=True)
class SplitConfig:
    train_end: str
    validation_end: str


@dataclass(frozen=True)
class IsolationForestConfig:
    n_estimators: int
    contamination: Any
    max_samples: Any
    n_jobs: int


@dataclass(frozen=True)
class AutoencoderConfig:
    hidden_dims: list[int]
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    early_stopping_patience: int


@dataclass(frozen=True)
class GradientBoostingConfig:
    n_estimators: int
    max_depth: int
    learning_rate: float
    subsample: float
    colsample_bytree: float
    scale_pos_weight: float | None = None


@dataclass(frozen=True)
class HybridWeightsConfig:
    autoencoder: float
    isolation_forest: float
    statistical: float
    xgboost: float
    lightgbm: float

    def total(self) -> float:
        return sum(astuple(self))


@dataclass(frozen=True)
class SeverityThresholdsConfig:
    low: float
    medium: float
    high: float


@dataclass(frozen=True)
class MonitoringConfig:
    psi_warning: float
    psi_critical: float
    psi_bins: int
    performance_window_days: int


@dataclass(frozen=True)
class MlflowConfig:
    experiment_name: str
    tracking_uri: str


@dataclass(frozen=True)
class ApiConfig:
    title: str
    version: str
    history_window: int


@dataclass(frozen=True)
class AppConfig:
    random_state: int
    sensor_columns: list[str]
    paths: PathsConfig
    data_generation: DataGenerationConfig
    feature_engineering: FeatureEngineeringConfig
    labeling: dict[str, Any]
    split: SplitConfig
    isolation_forest: IsolationForestConfig
    autoencoder: AutoencoderConfig
    xgboost: GradientBoostingConfig
    lightgbm: GradientBoostingConfig
    hybrid_weights: HybridWeightsConfig
    severity_thresholds: SeverityThresholdsConfig
    monitoring: MonitoringConfig
    mlflow: MlflowConfig
    api: ApiConfig
    raw: dict[str, Any] = field(repr=False, default_factory=dict)


def _validated_hybrid_weights(weights: HybridWeightsConfig) -> HybridWeightsConfig:
    total = weights.total()
    if not (0.99 <= total <= 1.01):
        raise ValueError(
            f"hybrid_weights must sum to 1.0 (got {total:.4f}) — a silently mis-weighted "
            "hybrid score would misrank every prediction, so this fails fast at config load."
        )
    return weights


def _resolve(root: Path, relative: str) -> Path:
    path = Path(relative)
    return path if path.is_absolute() else root / path


def _build_paths(raw: dict[str, Any], root: Path) -> PathsConfig:
    p = raw["paths"]
    return PathsConfig(
        raw_dir=_resolve(root, p["raw_dir"]),
        processed_dir=_resolve(root, p["processed_dir"]),
        features_dir=_resolve(root, p["features_dir"]),
        models_dir=_resolve(root, p["models_dir"]),
        machines_file=_resolve(root, p["machines_file"]),
        sensor_file=_resolve(root, p["sensor_file"]),
        failures_file=_resolve(root, p["failures_file"]),
        features_file=_resolve(root, p["features_file"]),
    )


@lru_cache(maxsize=8)
def load_config(config_path: str | Path | None = None) -> AppConfig:
    """Load and cache the application configuration.

    Cached per distinct ``config_path`` so tests can load an alternate
    file without polluting the default singleton.
    """
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    root = path.resolve().parent.parent

    return AppConfig(
        random_state=raw["random_state"],
        sensor_columns=raw["sensor_columns"],
        paths=_build_paths(raw, root),
        data_generation=DataGenerationConfig(**raw["data_generation"]),
        feature_engineering=FeatureEngineeringConfig(**raw["feature_engineering"]),
        labeling=raw["labeling"],
        split=SplitConfig(**raw["split"]),
        isolation_forest=IsolationForestConfig(**raw["isolation_forest"]),
        autoencoder=AutoencoderConfig(**raw["autoencoder"]),
        xgboost=GradientBoostingConfig(**raw["xgboost"]),
        lightgbm=GradientBoostingConfig(**raw["lightgbm"]),
        hybrid_weights=_validated_hybrid_weights(HybridWeightsConfig(**raw["hybrid_weights"])),
        severity_thresholds=SeverityThresholdsConfig(**raw["severity_thresholds"]),
        monitoring=MonitoringConfig(**raw["monitoring"]),
        mlflow=MlflowConfig(**raw["mlflow"]),
        api=ApiConfig(**raw["api"]),
        raw=raw,
    )
