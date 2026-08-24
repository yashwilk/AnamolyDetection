"""Pydantic request/response contracts for the anomaly detection API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class SensorReading(BaseModel):
    machine_id: str = Field(..., min_length=1, examples=["M0001"])
    timestamp: datetime
    temperature: float
    pressure: float = Field(..., ge=0)
    vibration: float = Field(..., ge=0)
    current: float = Field(..., ge=0)
    rpm: float = Field(..., ge=0)
    flow_rate: float = Field(..., ge=0)
    power: float = Field(..., ge=0)

    @field_validator("machine_id")
    @classmethod
    def machine_id_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("machine_id must not be blank")
        return value


class BatchPredictionRequest(BaseModel):
    readings: list[SensorReading] = Field(..., min_length=1)


class PredictionResponse(BaseModel):
    machine_id: str
    timestamp: str
    anomaly: bool
    anomaly_score: float
    severity: str
    anomaly_type: str
    explanation: str
    threshold: float
    component_scores: dict[str, float]


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None


class ModelInfoResponse(BaseModel):
    model_version: str
    threshold: float
    feature_count: int
    sensor_columns: list[str]
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    submodel_validation_metrics: dict[str, dict[str, float]]
