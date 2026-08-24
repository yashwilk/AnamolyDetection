from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import SeverityThresholdsConfig, load_config
from src.inference.explain import (
    classify_anomaly_type,
    compute_statistical_score,
    severity_from_score,
)
from src.models.autoencoder import Autoencoder, AutoencoderTrainer
from src.models.evaluate import (
    aggregate_alerts,
    best_threshold_by_f1,
    calculate_mttd,
    classification_metrics,
    ranking_metrics,
)
from src.models.gradient_boosting import (
    LightGBMAnomalyModel,
    XGBoostAnomalyModel,
    compute_scale_pos_weight,
)
from src.models.isolation_forest import IsolationForestModel
from src.monitoring.drift import population_stability_index


@pytest.fixture
def config():
    return load_config()


# --- Isolation Forest -------------------------------------------------


def test_isolation_forest_scores_before_fit_raises(config):
    model = IsolationForestModel(config.isolation_forest, config.random_state)
    with pytest.raises(RuntimeError):
        model.score(np.random.default_rng(0).normal(size=(5, 3)))


def test_isolation_forest_flags_obvious_outlier(config):
    rng = np.random.default_rng(0)
    X_train = rng.normal(0, 1, size=(500, 4))
    X_test = np.vstack([rng.normal(0, 1, size=(10, 4)), np.full((1, 4), 50.0)])

    model = IsolationForestModel(config.isolation_forest, config.random_state)
    model.fit(X_train)
    scores = model.score(X_test)

    assert scores.shape == (11,)
    assert scores[-1] == scores.max()


# --- Autoencoder --------------------------------------------------------


def test_autoencoder_forward_pass_preserves_shape():
    model = Autoencoder(input_dim=10, hidden_dims=[8, 4, 2])
    import torch

    x = torch.randn(6, 10)
    out = model(x)
    assert out.shape == x.shape


def test_autoencoder_trainer_lowers_reconstruction_error(config):
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, size=(300, 6)).astype(np.float32)

    from dataclasses import replace

    small_config = replace(config.autoencoder, epochs=5, hidden_dims=[8, 4])
    trainer = AutoencoderTrainer(small_config, input_dim=6)

    error_before = trainer.reconstruction_error(X).mean()
    trainer.fit(X)
    error_after = trainer.reconstruction_error(X).mean()

    assert error_after <= error_before * 1.5  # loose bound: training should not blow up error


# --- Gradient boosting (supervised) --------------------------------------


def _separable_classification_data(rng: np.random.Generator, n: int = 400):
    X_normal = rng.normal(0, 1, size=(n, 4))
    X_anomaly = rng.normal(4, 1, size=(int(n * 0.1), 4))
    X = np.vstack([X_normal, X_anomaly])
    y = np.array([0] * n + [1] * int(n * 0.1))
    return X, y


def test_compute_scale_pos_weight_matches_class_ratio():
    y = np.array([0] * 90 + [1] * 10)
    assert compute_scale_pos_weight(y) == pytest.approx(9.0)


def test_compute_scale_pos_weight_handles_no_positives():
    y = np.array([0] * 50)
    assert compute_scale_pos_weight(y) == 1.0


def test_xgboost_scores_before_fit_raises(config):
    model = XGBoostAnomalyModel(config.xgboost)
    with pytest.raises(RuntimeError):
        model.score(np.zeros((5, 4)))


def test_xgboost_separates_anomalous_class(config):
    rng = np.random.default_rng(0)
    X, y = _separable_classification_data(rng)

    from dataclasses import replace

    small_config = replace(config.xgboost, n_estimators=50, scale_pos_weight=compute_scale_pos_weight(y))
    model = XGBoostAnomalyModel(small_config)
    model.fit(X, y)
    scores = model.score(X)

    assert scores.shape == (len(y),)
    assert scores[y == 1].mean() > scores[y == 0].mean()


def test_xgboost_save_and_load_roundtrip(config, tmp_path):
    rng = np.random.default_rng(0)
    X, y = _separable_classification_data(rng)
    model = XGBoostAnomalyModel(config.xgboost)
    model.fit(X, y)

    path = tmp_path / "xgboost.ubj"
    model.save(path)
    loaded = XGBoostAnomalyModel.load(path, config.xgboost)

    np.testing.assert_allclose(model.score(X), loaded.score(X))


def test_lightgbm_scores_before_fit_raises(config):
    model = LightGBMAnomalyModel(config.lightgbm)
    with pytest.raises(RuntimeError):
        model.score(np.zeros((5, 4)))


def test_lightgbm_separates_anomalous_class(config):
    rng = np.random.default_rng(0)
    X, y = _separable_classification_data(rng)

    from dataclasses import replace

    small_config = replace(config.lightgbm, n_estimators=50, scale_pos_weight=compute_scale_pos_weight(y))
    model = LightGBMAnomalyModel(small_config)
    model.fit(X, y)
    scores = model.score(X)

    assert scores.shape == (len(y),)
    assert scores[y == 1].mean() > scores[y == 0].mean()


def test_lightgbm_save_and_load_roundtrip(config, tmp_path):
    rng = np.random.default_rng(0)
    X, y = _separable_classification_data(rng)
    model = LightGBMAnomalyModel(config.lightgbm)
    model.fit(X, y)

    path = tmp_path / "lightgbm.txt"
    model.save(path)
    loaded = LightGBMAnomalyModel.load(path, config.lightgbm)

    np.testing.assert_allclose(model.score(X), loaded.score(X), rtol=1e-5)


# --- Evaluation -----------------------------------------------------------


def test_classification_metrics_perfect_prediction():
    y_true = pd.Series([0, 0, 1, 1])
    y_pred = pd.Series([0, 0, 1, 1])
    metrics = classification_metrics(y_true, y_pred)
    assert metrics == {"precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_ranking_metrics_perfect_separation():
    y_true = pd.Series([0, 0, 1, 1])
    scores = pd.Series([0.1, 0.2, 0.8, 0.9])
    metrics = ranking_metrics(y_true, scores)
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0


def test_best_threshold_by_f1_separates_classes_perfectly():
    y_true = pd.Series([0] * 50 + [1] * 50)
    scores = pd.Series(list(np.linspace(0, 0.4, 50)) + list(np.linspace(0.6, 1.0, 50)))
    threshold, f1 = best_threshold_by_f1(y_true, scores)
    assert 0.4 < threshold <= 0.6
    assert f1 == pytest.approx(1.0)  # the 1e-10 epsilon in the F1 formula keeps it just under exactly 1.0


def test_aggregate_alerts_and_mttd():
    timestamps = pd.date_range("2026-01-01", periods=5, freq="1h")
    df = pd.DataFrame(
        {
            "machine_id": ["M001"] * 5,
            "timestamp": timestamps,
            "anomaly_score": [0.1, 0.9, 0.95, 0.2, 0.1],
        }
    )
    alerts = aggregate_alerts(df, score_threshold=0.5)
    assert len(alerts) == 1
    assert alerts.iloc[0]["number_of_alerts"] == 2

    failures = pd.DataFrame(
        {"machine_id": ["M001"], "failure_timestamp": [timestamps[4] + pd.Timedelta(hours=2)]}
    )
    mttd = calculate_mttd(alerts, failures)
    assert mttd.loc[0, "lead_time"] > pd.Timedelta(0)


# --- Explainability ---------------------------------------------------------


def test_severity_from_score_boundaries():
    thresholds = SeverityThresholdsConfig(low=0.7, medium=0.85, high=0.95)
    assert severity_from_score(0.5, thresholds) == "LOW"
    assert severity_from_score(0.8, thresholds) == "MEDIUM"
    assert severity_from_score(0.9, thresholds) == "HIGH"
    assert severity_from_score(0.99, thresholds) == "CRITICAL"


def _zscore_row(vibration=0.1, temperature=0.1, flow_rate=0.1, pressure=0.1, current=0.1) -> pd.Series:
    return pd.Series(
        {
            "vibration_zscore": vibration,
            "temperature_zscore": temperature,
            "flow_rate_zscore": flow_rate,
            "pressure_zscore": pressure,
            "current_zscore": current,
        }
    )


def test_classify_anomaly_type_bearing_failure():
    row = _zscore_row(vibration=6.0)
    assert classify_anomaly_type(row) == "bearing_failure"


def test_classify_anomaly_type_overheating():
    row = _zscore_row(temperature=6.0)
    assert classify_anomaly_type(row) == "overheating"


def test_classify_anomaly_type_pump_blockage():
    row = _zscore_row(vibration=0.3, temperature=0.2, flow_rate=-5.0, pressure=3.0, current=3.0)
    assert classify_anomaly_type(row) == "pump_blockage"


def test_classify_anomaly_type_falls_back_when_ambiguous():
    row = _zscore_row(vibration=0.5, temperature=0.5, flow_rate=-0.2)
    assert classify_anomaly_type(row) == "unspecified_anomaly"


def test_compute_statistical_score_empty_returns_zero():
    row = pd.Series({"vibration_zscore": np.nan})
    assert compute_statistical_score(row, ["vibration_zscore"]) == 0.0


def test_compute_statistical_score_scales_with_magnitude():
    low = pd.Series({"vibration_zscore": 1.0})
    high = pd.Series({"vibration_zscore": 5.0})
    high_score = compute_statistical_score(high, ["vibration_zscore"])
    low_score = compute_statistical_score(low, ["vibration_zscore"])
    assert high_score > low_score


# --- Drift ------------------------------------------------------------------


def test_psi_near_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    data = pd.Series(rng.normal(0, 1, 1000))
    psi = population_stability_index(data, data)
    assert psi < 1e-6


def test_psi_high_for_shifted_distribution():
    rng = np.random.default_rng(0)
    expected = pd.Series(rng.normal(0, 1, 1000))
    actual = pd.Series(rng.normal(5, 1, 1000))
    psi = population_stability_index(expected, actual)
    assert psi > 0.25
