"""FastAPI service: scores one sensor reading (or a batch) per request and
returns anomaly score + type + severity + explanation — the contract the
business asked for, served every 5 minutes per pump.

Run: ``uvicorn app.main:app --host 0.0.0.0 --port 8000``
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request

from app.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    SensorReading,
)
from src.config import load_config
from src.inference.predictor import AnomalyPredictor, ModelArtifactsNotFoundError
from src.inference.state_store import FeatureStateStore
from src.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Overridable per-environment (dev/staging/prod) without a code change
    # or editing the checked-in default — set before starting uvicorn.
    config = load_config(os.environ.get("ANOMALY_CONFIG_PATH"))
    app.state.config = config
    app.state.state_store = FeatureStateStore(
        history_window=config.api.history_window, sensor_columns=config.sensor_columns
    )
    try:
        app.state.predictor = AnomalyPredictor(config)
    except ModelArtifactsNotFoundError as exc:
        logger.error("Model artifacts not found at startup — /predict will return 503 until trained: %s", exc)
        app.state.predictor = None
    yield


app = FastAPI(
    title="Industrial Anomaly Detection API",
    version="1.0.0",
    description="Anomaly score, type, severity, and explanation for pump fleet sensor readings.",
    lifespan=lifespan,
)


def get_predictor(request: Request) -> AnomalyPredictor:
    predictor = request.app.state.predictor
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train a model with `python -m src.models.train`.",
        )
    return predictor


def get_state_store(request: Request) -> FeatureStateStore:
    return request.app.state.state_store


def _score_reading(
    reading: SensorReading, predictor: AnomalyPredictor, state_store: FeatureStateStore
) -> PredictionResponse:
    sensor_values = reading.model_dump(exclude={"machine_id", "timestamp"})
    try:
        result = predictor.predict_streaming(
            reading.machine_id, pd.Timestamp(reading.timestamp), sensor_values, state_store
        )
    except Exception as exc:  # noqa: BLE001 — surfaced to the caller as a 500
        logger.exception("Prediction failed for machine_id=%s", reading.machine_id)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
    return PredictionResponse(**result.to_dict())


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    predictor = request.app.state.predictor
    return HealthResponse(
        status="healthy" if predictor is not None else "degraded",
        model_loaded=predictor is not None,
        model_version=predictor.model_version if predictor is not None else None,
    )


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info(predictor: AnomalyPredictor = Depends(get_predictor)) -> ModelInfoResponse:
    return ModelInfoResponse(
        model_version=predictor.model_version,
        threshold=predictor.threshold,
        feature_count=len(predictor.feature_columns),
        sensor_columns=predictor.sensor_columns,
        validation_metrics=predictor.metadata.get("validation_metrics", {}),
        test_metrics=predictor.metadata.get("test_metrics", {}),
        submodel_validation_metrics=predictor.metadata.get("submodel_validation_metrics", {}),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(
    reading: SensorReading,
    predictor: AnomalyPredictor = Depends(get_predictor),
    state_store: FeatureStateStore = Depends(get_state_store),
) -> PredictionResponse:
    return _score_reading(reading, predictor, state_store)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(
    payload: BatchPredictionRequest,
    predictor: AnomalyPredictor = Depends(get_predictor),
    state_store: FeatureStateStore = Depends(get_state_store),
) -> BatchPredictionResponse:
    predictions = [_score_reading(reading, predictor, state_store) for reading in payload.readings]
    return BatchPredictionResponse(predictions=predictions)
