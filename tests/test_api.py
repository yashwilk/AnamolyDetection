from __future__ import annotations

import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient

from app.main import app, get_predictor, get_state_store
from src.config import DEFAULT_CONFIG_PATH
from src.inference.predictor import PredictionResult
from src.inference.state_store import FeatureStateStore


class FakeAnomalyPredictor:
    """Stub used in place of a real trained model so API tests are fast,
    deterministic, and don't require artifacts from `python -m src.models.train`."""

    model_version = "test-fixture-v1"
    threshold = 0.82
    feature_columns = ["temperature", "pressure", "vibration"]
    sensor_columns = ["temperature", "pressure", "vibration", "current", "rpm", "flow_rate", "power"]
    metadata = {"validation_metrics": {"pr_auc": 0.9}, "test_metrics": {"pr_auc": 0.85}}

    def predict_streaming(self, machine_id, timestamp, sensor_values, state_store):
        is_anomaly = sensor_values["vibration"] > 1.0
        component_value = 0.9 if is_anomaly else 0.1
        return PredictionResult(
            machine_id=machine_id,
            timestamp=pd.Timestamp(timestamp),
            anomaly=is_anomaly,
            anomaly_score=0.95 if is_anomaly else 0.1,
            severity="CRITICAL" if is_anomaly else "LOW",
            anomaly_type="bearing_failure" if is_anomaly else "normal",
            explanation=(
                "vibration is 6.0 std devs above its recent baseline."
                if is_anomaly
                else "No significant deviation from baseline."
            ),
            threshold=self.threshold,
            component_scores=dict.fromkeys(
                ["isolation_forest", "autoencoder", "statistical", "xgboost", "lightgbm"], component_value
            ),
        )


VALID_READING = {
    "machine_id": "M0001",
    "timestamp": "2026-01-01T00:00:00",
    "temperature": 70.0,
    "pressure": 4.8,
    "vibration": 0.15,
    "current": 13.0,
    "rpm": 1450.0,
    "flow_rate": 82.0,
    "power": 5.2,
}


@pytest.fixture
def client():
    app.dependency_overrides[get_predictor] = lambda: FakeAnomalyPredictor()
    app.dependency_overrides[get_state_store] = lambda: FeatureStateStore()
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_health_reports_model_loaded_with_override(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_health_reports_degraded_without_model(tmp_path, monkeypatch):
    # Point at a config whose models_dir is guaranteed empty, rather than
    # relying on the real models/ directory happening to be untrained —
    # this test should pass regardless of ambient repo state.
    with open(DEFAULT_CONFIG_PATH, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    raw["paths"]["models_dir"] = str(tmp_path / "empty_models")
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(raw, fh)
    monkeypatch.setenv("ANOMALY_CONFIG_PATH", str(config_path))

    app.dependency_overrides.pop(get_predictor, None)
    with TestClient(app) as test_client:
        # No trained model artifacts exist at this models_dir, so the
        # lifespan should leave app.state.predictor as None rather than crash.
        response = test_client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["model_loaded"] is False
        assert body["status"] == "degraded"


def test_predict_normal_reading(client):
    response = client.post("/predict", json=VALID_READING)
    assert response.status_code == 200
    body = response.json()
    assert body["anomaly"] is False
    assert body["severity"] == "LOW"
    assert body["machine_id"] == "M0001"


def test_predict_anomalous_reading(client):
    reading = {**VALID_READING, "vibration": 5.0}
    response = client.post("/predict", json=reading)
    assert response.status_code == 200
    body = response.json()
    assert body["anomaly"] is True
    assert body["severity"] == "CRITICAL"
    assert body["anomaly_type"] == "bearing_failure"


def test_predict_rejects_negative_rpm(client):
    reading = {**VALID_READING, "rpm": -100}
    response = client.post("/predict", json=reading)
    assert response.status_code == 422


def test_predict_rejects_blank_machine_id(client):
    reading = {**VALID_READING, "machine_id": "  "}
    response = client.post("/predict", json=reading)
    assert response.status_code == 422


def test_predict_batch(client):
    payload = {"readings": [VALID_READING, {**VALID_READING, "machine_id": "M0002", "vibration": 5.0}]}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert len(predictions) == 2
    assert predictions[0]["anomaly"] is False
    assert predictions[1]["anomaly"] is True


def test_model_info(client):
    response = client.get("/model/info")
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "test-fixture-v1"
    assert body["threshold"] == pytest.approx(0.82)
