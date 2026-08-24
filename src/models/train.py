"""End-to-end training pipeline: load engineered features, time-based
split, fit the scaler + both models + score normalizers, select an
operating threshold on validation, evaluate on test, and persist every
artifact the API needs as one versioned bundle under ``models/``.

Run: ``python -m src.models.train``
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from datetime import UTC, datetime

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.config import AppConfig, HybridWeightsConfig, load_config
from src.data.labels import create_labels
from src.features.feature_engineering import create_features, feature_column_names
from src.inference.explain import compute_statistical_score
from src.logging_config import configure_logging
from src.models.autoencoder import AutoencoderTrainer
from src.models.evaluate import (
    aggregate_alerts,
    best_threshold_by_f1,
    calculate_mttd,
    classification_metrics,
    mttd_summary,
    ranking_metrics,
)
from src.models.gradient_boosting import (
    LightGBMAnomalyModel,
    XGBoostAnomalyModel,
    compute_scale_pos_weight,
)
from src.models.isolation_forest import IsolationForestModel

logger = logging.getLogger(__name__)


def load_labeled_features(config: AppConfig) -> pd.DataFrame:
    sensor_df = pd.read_parquet(config.paths.sensor_file)
    failures_df = pd.read_csv(config.paths.failures_file, parse_dates=["failure_timestamp"])

    fe = config.feature_engineering
    features_df = create_features(
        sensor_df,
        sensor_columns=config.sensor_columns,
        rolling_window=fe.rolling_window,
        rolling_min_periods=fe.rolling_min_periods,
        zscore_window=fe.zscore_window,
        interpolation_limit=fe.interpolation_limit,
    )
    labeled_df = create_labels(
        features_df, failures_df, pre_failure_window_hours=config.labeling["pre_failure_window_hours"]
    )
    return labeled_df, failures_df


def time_split(df: pd.DataFrame, config: AppConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_end = pd.Timestamp(config.split.train_end)
    validation_end = pd.Timestamp(config.split.validation_end)

    train = df[df["timestamp"] <= train_end]
    validation = df[(df["timestamp"] > train_end) & (df["timestamp"] <= validation_end)]
    test = df[df["timestamp"] > validation_end]
    return train, validation, test


def fit_score_normalizer(raw_scores: np.ndarray) -> MinMaxScaler:
    scaler = MinMaxScaler()
    scaler.fit(raw_scores.reshape(-1, 1))
    return scaler


def compute_hybrid_score(
    ae_norm: np.ndarray,
    if_norm: np.ndarray,
    statistical: np.ndarray,
    xgb_score: np.ndarray,
    lgbm_score: np.ndarray,
    weights: HybridWeightsConfig,
) -> np.ndarray:
    return (
        weights.autoencoder * ae_norm
        + weights.isolation_forest * if_norm
        + weights.statistical * statistical
        + weights.xgboost * xgb_score
        + weights.lightgbm * lgbm_score
    )


def run_training(config: AppConfig) -> dict:
    logger.info("Loading and labeling features")
    labeled_df, failures_df = load_labeled_features(config)

    feature_columns = feature_column_names(config.sensor_columns)
    zscore_columns = [f"{c}_zscore" for c in config.sensor_columns]

    train, validation, test = time_split(labeled_df, config)
    train_model = train.dropna(subset=feature_columns)
    validation_model = validation.dropna(subset=feature_columns).copy()
    test_model = test.dropna(subset=feature_columns).copy()

    logger.info(
        "Split sizes — train: %d, validation: %d, test: %d",
        len(train_model),
        len(validation_model),
        len(test_model),
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_model[feature_columns])
    X_val = scaler.transform(validation_model[feature_columns])
    X_test = scaler.transform(test_model[feature_columns])

    logger.info("Training Isolation Forest")
    isolation_forest = IsolationForestModel(config.isolation_forest, config.random_state)
    isolation_forest.fit(X_train)

    logger.info("Training Autoencoder")
    autoencoder = AutoencoderTrainer(config.autoencoder, input_dim=len(feature_columns))
    autoencoder.fit(X_train, X_val)

    # XGBoost / LightGBM are supervised on the weak pre-failure label and
    # trained on *unscaled* features — tree ensembles don't need or
    # benefit from standardization.
    pos_weight = compute_scale_pos_weight(train_model["anomaly"])
    logger.info("Computed scale_pos_weight=%.2f from training class balance", pos_weight)
    xgb_config = replace(config.xgboost, scale_pos_weight=config.xgboost.scale_pos_weight or pos_weight)
    lgbm_config = replace(config.lightgbm, scale_pos_weight=config.lightgbm.scale_pos_weight or pos_weight)

    logger.info("Training XGBoost")
    xgboost_model = XGBoostAnomalyModel(xgb_config, config.random_state)
    xgboost_model.fit(train_model[feature_columns], train_model["anomaly"])

    logger.info("Training LightGBM")
    lightgbm_model = LightGBMAnomalyModel(lgbm_config, config.random_state)
    lightgbm_model.fit(train_model[feature_columns], train_model["anomaly"])

    if_raw_val = isolation_forest.score(X_val)
    ae_raw_val = autoencoder.score(X_val)

    if_score_scaler = fit_score_normalizer(if_raw_val)
    ae_score_scaler = fit_score_normalizer(ae_raw_val)

    if_norm_val = np.clip(if_score_scaler.transform(if_raw_val.reshape(-1, 1)).flatten(), 0, 1)
    ae_norm_val = np.clip(ae_score_scaler.transform(ae_raw_val.reshape(-1, 1)).flatten(), 0, 1)
    statistical_val = np.array(
        [compute_statistical_score(row, zscore_columns) for _, row in validation_model.iterrows()]
    )
    xgb_val = np.clip(xgboost_model.score(validation_model[feature_columns]), 0, 1)
    lgbm_val = np.clip(lightgbm_model.score(validation_model[feature_columns]), 0, 1)

    submodel_validation_metrics = {
        "isolation_forest": ranking_metrics(validation_model["anomaly"], if_norm_val),
        "autoencoder": ranking_metrics(validation_model["anomaly"], ae_norm_val),
        "statistical": ranking_metrics(validation_model["anomaly"], statistical_val),
        "xgboost": ranking_metrics(validation_model["anomaly"], xgb_val),
        "lightgbm": ranking_metrics(validation_model["anomaly"], lgbm_val),
    }
    logger.info("Per-submodel validation ranking metrics: %s", submodel_validation_metrics)

    validation_model["anomaly_score"] = compute_hybrid_score(
        ae_norm_val, if_norm_val, statistical_val, xgb_val, lgbm_val, config.hybrid_weights
    )

    threshold, val_f1 = best_threshold_by_f1(validation_model["anomaly"], validation_model["anomaly_score"])
    logger.info("Selected threshold=%.4f (validation F1=%.4f)", threshold, val_f1)

    val_ranking = ranking_metrics(validation_model["anomaly"], validation_model["anomaly_score"])
    logger.info("Validation ranking metrics: %s", val_ranking)

    # --- Test set evaluation ---
    if_raw_test = isolation_forest.score(X_test)
    ae_raw_test = autoencoder.score(X_test)
    if_norm_test = np.clip(if_score_scaler.transform(if_raw_test.reshape(-1, 1)).flatten(), 0, 1)
    ae_norm_test = np.clip(ae_score_scaler.transform(ae_raw_test.reshape(-1, 1)).flatten(), 0, 1)
    statistical_test = np.array(
        [compute_statistical_score(row, zscore_columns) for _, row in test_model.iterrows()]
    )
    xgb_test = np.clip(xgboost_model.score(test_model[feature_columns]), 0, 1)
    lgbm_test = np.clip(lightgbm_model.score(test_model[feature_columns]), 0, 1)

    test_model["anomaly_score"] = compute_hybrid_score(
        ae_norm_test, if_norm_test, statistical_test, xgb_test, lgbm_test, config.hybrid_weights
    )
    test_model["predicted_anomaly"] = (test_model["anomaly_score"] > threshold).astype(int)

    test_ranking = ranking_metrics(test_model["anomaly"], test_model["anomaly_score"])
    test_classification = classification_metrics(test_model["anomaly"], test_model["predicted_anomaly"])
    logger.info("Test ranking metrics: %s", test_ranking)
    logger.info("Test classification metrics: %s", test_classification)

    # MTTD must only be evaluated against failures that actually fall
    # inside the test window — calculate_mttd joins on machine_id alone,
    # so passing the full (all-time) failures_df would let a January
    # failure spuriously pair with an unrelated June test-period alert on
    # the same machine and corrupt the reported lead time.
    validation_end = pd.Timestamp(config.split.validation_end)
    test_failures_df = failures_df[failures_df["failure_timestamp"] > validation_end]
    alerts = aggregate_alerts(test_model, threshold)
    mttd_df = calculate_mttd(alerts, test_failures_df)
    mttd = mttd_summary(mttd_df)
    logger.info("MTTD summary: %s", mttd)

    # --- Persist artifacts ---
    models_dir = config.paths.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(scaler, models_dir / "scaler.pkl")
    joblib.dump(if_score_scaler, models_dir / "if_score_scaler.pkl")
    joblib.dump(ae_score_scaler, models_dir / "ae_score_scaler.pkl")
    isolation_forest.save(models_dir / "isolation_forest.pkl")
    autoencoder.save(models_dir / "autoencoder.pt")
    xgboost_model.save(models_dir / "xgboost.ubj")
    lightgbm_model.save(models_dir / "lightgbm.txt")

    model_version = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    metadata = {
        "model_version": model_version,
        "feature_columns": feature_columns,
        "sensor_columns": config.sensor_columns,
        "zscore_columns": zscore_columns,
        "threshold": threshold,
        "validation_f1": val_f1,
        "validation_metrics": val_ranking,
        "test_metrics": {**test_ranking, **test_classification},
        "submodel_validation_metrics": submodel_validation_metrics,
        "scale_pos_weight": pos_weight,
        "mttd": mttd,
    }
    with open(models_dir / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, default=str)

    logger.info("Artifacts written to %s (model_version=%s)", models_dir, model_version)

    try:
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
        mlflow.set_experiment(config.mlflow.experiment_name)
        with mlflow.start_run(run_name=model_version):
            mlflow.log_param("isolation_forest_n_estimators", config.isolation_forest.n_estimators)
            mlflow.log_param("autoencoder_hidden_dims", config.autoencoder.hidden_dims)
            mlflow.log_param("xgboost_n_estimators", xgb_config.n_estimators)
            mlflow.log_param("lightgbm_n_estimators", lgbm_config.n_estimators)
            mlflow.log_param("scale_pos_weight", pos_weight)
            mlflow.log_param("threshold", threshold)
            mlflow.log_metrics({f"val_{k}": v for k, v in val_ranking.items()})
            mlflow.log_metrics({f"test_{k}": v for k, v in test_ranking.items()})
            mlflow.log_metrics({f"test_{k}": v for k, v in test_classification.items()})
            for submodel, metrics in submodel_validation_metrics.items():
                mlflow.log_metrics({f"val_{submodel}_{k}": v for k, v in metrics.items()})
            if mttd["n_failures_detected"] > 0:
                mlflow.log_metric("mttd_mean_lead_time_hours", mttd["mean_lead_time_hours"])
    except Exception:
        logger.warning("MLflow logging failed — continuing without experiment tracking", exc_info=True)

    return metadata


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    run_training(config)


if __name__ == "__main__":
    main()
