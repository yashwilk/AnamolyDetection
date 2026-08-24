"""Production inference entry point: hybrid Isolation Forest + Autoencoder
+ statistical scoring, with severity classification and explanation.

Design notes (fixing issues present in the original prototype snippets
this project was built from):

- The Isolation Forest / Autoencoder raw scores are rescaled to [0, 1]
  using ``MinMaxScaler`` instances *fit once during training* and
  persisted (``if_score_scaler.pkl`` / ``ae_score_scaler.pkl``), not
  re-fit on whatever batch happens to arrive at inference time. Refitting
  per-batch would make the same sensor reading produce a different score
  depending on what else was in the batch — unacceptable for a system
  that scores one pump every 5 minutes.
- All artifacts (scaler, models, score normalizers, feature list,
  threshold, hybrid weights) are versioned together in one
  ``metadata.json`` so training and serving can never silently drift
  apart on which features / threshold to use.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd

from src.config import AppConfig, load_config
from src.features.feature_engineering import create_features
from src.inference.explain import (
    build_explanation,
    classify_anomaly_type,
    compute_statistical_score,
    severity_from_score,
)
from src.inference.state_store import FeatureStateStore
from src.models.autoencoder import AutoencoderTrainer
from src.models.gradient_boosting import LightGBMAnomalyModel, XGBoostAnomalyModel
from src.models.isolation_forest import IsolationForestModel

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    machine_id: str
    timestamp: pd.Timestamp
    anomaly: bool
    anomaly_score: float
    severity: str
    anomaly_type: str
    explanation: str
    threshold: float
    component_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        timestamp = self.timestamp.isoformat() if isinstance(self.timestamp, pd.Timestamp) else self.timestamp
        return {
            "machine_id": self.machine_id,
            "timestamp": timestamp,
            "anomaly": self.anomaly,
            "anomaly_score": round(self.anomaly_score, 4),
            "severity": self.severity,
            "anomaly_type": self.anomaly_type,
            "explanation": self.explanation,
            "threshold": self.threshold,
            "component_scores": {k: round(v, 4) for k, v in self.component_scores.items()},
        }


class ModelArtifactsNotFoundError(RuntimeError):
    pass


class AnomalyPredictor:
    def __init__(self, config: AppConfig | None = None):
        self.config = config or load_config()
        models_dir = self.config.paths.models_dir

        metadata_path = models_dir / "metadata.json"
        if not metadata_path.exists():
            raise ModelArtifactsNotFoundError(
                f"No trained model metadata at {metadata_path}. Run `python -m src.models.train` first."
            )
        with open(metadata_path, encoding="utf-8") as fh:
            self.metadata = json.load(fh)

        self.feature_columns: list[str] = self.metadata["feature_columns"]
        self.zscore_columns: list[str] = self.metadata["zscore_columns"]
        self.sensor_columns: list[str] = self.metadata["sensor_columns"]
        self.threshold: float = self.metadata["threshold"]
        self.model_version: str = self.metadata.get("model_version", "unknown")

        self.scaler = joblib.load(models_dir / "scaler.pkl")
        self.if_score_scaler = joblib.load(models_dir / "if_score_scaler.pkl")
        self.ae_score_scaler = joblib.load(models_dir / "ae_score_scaler.pkl")

        self.isolation_forest = IsolationForestModel.load(
            models_dir / "isolation_forest.pkl", self.config.isolation_forest, self.config.random_state
        )
        self.autoencoder = AutoencoderTrainer.load(
            models_dir / "autoencoder.pt",
            self.config.autoencoder,
            input_dim=len(self.feature_columns),
        )
        self.xgboost_model = XGBoostAnomalyModel.load(
            models_dir / "xgboost.ubj", self.config.xgboost, self.config.random_state
        )
        self.lightgbm_model = LightGBMAnomalyModel.load(
            models_dir / "lightgbm.txt", self.config.lightgbm, self.config.random_state
        )

        self.weights = self.config.hybrid_weights
        self.severity_thresholds = self.config.severity_thresholds

        logger.info(
            "AnomalyPredictor loaded (model_version=%s, threshold=%.4f)", self.model_version, self.threshold
        )

    def _missing_feature_columns(self, df: pd.DataFrame) -> list[str]:
        return [c for c in self.feature_columns if c not in df.columns]

    def predict_dataframe(self, df: pd.DataFrame) -> list[PredictionResult]:
        """Score every row of an already feature-engineered dataframe.

        Rows missing required features (e.g. not enough history yet to
        compute a rolling z-score) are returned as low-confidence,
        non-anomalous results rather than raising — a cold-start machine
        should not crash the pipeline.
        """
        missing = self._missing_feature_columns(df)
        if missing:
            raise ValueError(f"Input is missing required feature columns: {missing}")

        results: list[PredictionResult] = []
        valid_mask = df[self.feature_columns].notna().all(axis=1)

        if valid_mask.any():
            valid_df = df.loc[valid_mask]
            # Keep X a DataFrame (not .to_numpy()) so column names match
            # what the scaler was fit on during training — passing a bare
            # array here is otherwise harmless but makes sklearn warn on
            # every single prediction.
            X = valid_df[self.feature_columns]
            X_scaled = self.scaler.transform(X)

            if_raw = self.isolation_forest.score(X_scaled)
            ae_raw = self.autoencoder.score(X_scaled)

            if_norm = np.clip(self.if_score_scaler.transform(if_raw.reshape(-1, 1)).flatten(), 0.0, 1.0)
            ae_norm = np.clip(self.ae_score_scaler.transform(ae_raw.reshape(-1, 1)).flatten(), 0.0, 1.0)

            # XGBoost/LightGBM score unscaled features and already output a
            # [0, 1] probability — no learned normalizer needed for them.
            xgb_norm = np.clip(self.xgboost_model.score(X), 0.0, 1.0)
            lgbm_norm = np.clip(self.lightgbm_model.score(X), 0.0, 1.0)

            statistical = np.array(
                [compute_statistical_score(row, self.zscore_columns) for _, row in valid_df.iterrows()]
            )

            hybrid = (
                self.weights.autoencoder * ae_norm
                + self.weights.isolation_forest * if_norm
                + self.weights.statistical * statistical
                + self.weights.xgboost * xgb_norm
                + self.weights.lightgbm * lgbm_norm
            )

            for i, (_, row) in enumerate(valid_df.iterrows()):
                score = float(hybrid[i])
                is_anomaly = score > self.threshold
                results.append(
                    PredictionResult(
                        machine_id=row["machine_id"],
                        timestamp=row["timestamp"],
                        anomaly=bool(is_anomaly),
                        anomaly_score=score,
                        severity=severity_from_score(score, self.severity_thresholds),
                        anomaly_type=classify_anomaly_type(row) if is_anomaly else "normal",
                        explanation=(
                            build_explanation(row, self.zscore_columns)
                            if is_anomaly
                            else "No significant deviation from baseline."
                        ),
                        threshold=self.threshold,
                        component_scores={
                            "isolation_forest": float(if_norm[i]),
                            "autoencoder": float(ae_norm[i]),
                            "statistical": float(statistical[i]),
                            "xgboost": float(xgb_norm[i]),
                            "lightgbm": float(lgbm_norm[i]),
                        },
                    )
                )

        if (~valid_mask).any():
            for _, row in df.loc[~valid_mask].iterrows():
                results.append(
                    PredictionResult(
                        machine_id=row["machine_id"],
                        timestamp=row["timestamp"],
                        anomaly=False,
                        anomaly_score=0.0,
                        severity="UNKNOWN",
                        anomaly_type="insufficient_data",
                        explanation="Not enough recent history to compute reliable features yet.",
                        threshold=self.threshold,
                        component_scores={},
                    )
                )

        results.sort(key=lambda r: (str(r.machine_id), pd.Timestamp(r.timestamp)))
        return results

    def predict_streaming(
        self,
        machine_id: str,
        timestamp: pd.Timestamp,
        sensor_values: dict[str, float],
        state_store: FeatureStateStore,
    ) -> PredictionResult:
        """Score a single live reading using a machine's rolling history.

        The reading is appended to the state store first (so it
        contributes to its own rolling window, matching how it will be
        seen historically once persisted), features are recomputed over
        the whole buffered window, and only the latest row is scored.
        """
        record = {"machine_id": machine_id, "timestamp": timestamp, **sensor_values}
        state_store.append(machine_id, record)

        history = state_store.get_history(machine_id)
        featured = create_features(
            history,
            sensor_columns=self.sensor_columns,
            rolling_window=self.config.feature_engineering.rolling_window,
            rolling_min_periods=self.config.feature_engineering.rolling_min_periods,
            zscore_window=self.config.feature_engineering.zscore_window,
            interpolation_limit=self.config.feature_engineering.interpolation_limit,
        )
        latest_row = featured.iloc[[-1]]
        return self.predict_dataframe(latest_row)[0]
