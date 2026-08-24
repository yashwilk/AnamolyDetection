# Industrial Pump Anomaly Detection

A production ML system that scores every pump in a fleet every 5 minutes and
outputs **anomaly score + anomaly type + severity + explanation**, so
operations can act on early degradation signals before a failure happens.

## Why this exists

Each pump continuously streams temperature, pressure, vibration, current,
RPM, flow rate, and power consumption. Left undetected, a failure costs:

- **£8,000+** in emergency repair
- **£15,000+** in production downtime
- potentially **several days** of lost production

The system is designed to catch degradation during the pre-failure window
(hours before a failure, not just at the point of failure) and report it in
a form an operator can act on immediately, not a raw anomaly score with no
context.

## Architecture

```
   IoT Sensors
        |
        v
   Kafka / API
        |
        v
 +-------------------+
 |  Data Validation  |
 +---------+---------+
           |
   +-------+-------+
   v               v
Raw Data       Quarantine
   |
   v
Feature Pipeline
   |
   v
Feature Store  (state_store.py locally; Redis/feature store in production)
   |
   +-------+-------+
   v               v
Isolation Forest   Autoencoder
   |               |
   +-------+-------+
           v
      Hybrid Score
           |
           v
     Threshold Engine
           |
           v
     Alert Aggregator
           |
   +-------+-------+
   v               v
Database        Alert System --> Operations Team

+---------------------------------+
|          ML Monitoring          |
|  Data Drift / Prediction Drift  |
|  Model Performance / FPR / MTTD |
+----------------+----------------+
                 v
            Retraining
                 v
        Champion / Challenger
                 v
             Deployment
```

## Project layout

```
data/               raw / processed / feature parquet & csv outputs (gitignored)
models/             trained model artifacts (gitignored, produced by training)
notebooks/          thin exploratory wrappers around src/ — no logic lives here
src/
  config.py         typed loader for configs/config.yaml
  logging_config.py structured logging setup
  data/             synthetic data generation, schema, validation, labeling
  features/         causal feature engineering (rolling stats, ratios, z-scores)
  models/           Isolation Forest, Autoencoder, training pipeline, evaluation, SHAP
  inference/        hybrid predictor, severity/type/explanation, streaming state store
  monitoring/       PSI drift detection, prediction-health monitoring
app/                FastAPI service (main.py) + request/response schemas
tests/              pytest suite mirroring src/ and app/
configs/config.yaml single source of truth for all runtime parameters
scripts/run_pipeline.py  chains generate -> validate -> train for local/CI runs
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements-dev.txt
```

> **Windows note:** create the venv under a short path (e.g. `C:\venvs\anomaly`)
> rather than deep inside a long, spaced `OneDrive` folder. `torch`'s wheel
> unpacks files with very long nested paths, and combined with a long
> project path this can exceed Windows' `MAX_PATH` and abort the install
> partway through — `pip` may still exit 0 having only installed part of
> `requirements.txt`. If a package is unexpectedly missing after install,
> this is the first thing to check.

Common tasks are also wired up in the `Makefile`: `make install`, `make
lint`, `make format`, `make typecheck`, `make test`, `make pipeline`, `make
report` (regenerates [RESULTS.md](RESULTS.md)'s charts), `make serve`,
`make docker-build`.

## Running the pipeline

```bash
# 1. Generate synthetic fleet + sensor data with injected anomalies/failures
python -m src.data.generate_data

# 2. Validate data quality (raises if a batch should be quarantined)
python scripts/run_pipeline.py --stages validate

# 3. Train: fits scaler + all 5 models + score normalizers, selects an
#    operating threshold on validation, evaluates on test, writes
#    everything to models/ as one versioned bundle
python -m src.models.train

# ...or run all three stages in one command:
python scripts/run_pipeline.py --stages generate validate train
```

Training writes to `models/`:

| File                     | Contents                                                    |
|--------------------------|---------------------------------------------------------------|
| `scaler.pkl`             | `StandardScaler` fit on training features (IF/AE input)        |
| `isolation_forest.pkl`   | trained `IsolationForest` (unsupervised)                        |
| `autoencoder.pt`         | trained autoencoder state dict (unsupervised)                   |
| `xgboost.ubj`            | trained XGBoost classifier, native UBJ format (supervised)      |
| `lightgbm.txt`           | trained LightGBM booster, native text format (supervised)       |
| `if_score_scaler.pkl`    | `MinMaxScaler` normalizing raw IF scores to [0,1]               |
| `ae_score_scaler.pkl`    | `MinMaxScaler` normalizing raw AE scores to [0,1]               |
| `metadata.json`          | feature list, threshold, hybrid weights, per-submodel + overall eval metrics |

The IF/AE normalizer scalers are fit **once**, on the validation split,
during training — not refit per inference batch. That guarantees the same
sensor reading always produces the same score, independent of what else
happens to be in the batch it arrives with. XGBoost/LightGBM need no such
normalizer: `predict_proba` already returns a well-formed [0, 1] score.

Gradient boosting models are saved in their **native formats** (UBJ,
LightGBM text), not pickled — pickle layout for these libraries has
historically broken across version upgrades, which native formats avoid.

## Serving

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

or via Docker:

```bash
docker build -t anomaly-detection .
docker run -p 8000:8000 anomaly-detection
```

Set `ANOMALY_CONFIG_PATH` to point the running service at a non-default
`config.yaml` (e.g. a staging config with different thresholds) without
rebuilding the image or touching the checked-in default:

```bash
ANOMALY_CONFIG_PATH=/etc/anomaly-detection/staging.yaml uvicorn app.main:app
```

### API

- `GET /health` — `{"status", "model_loaded", "model_version"}`. Returns
  `degraded` (not a 500) if no trained model is present yet, so the
  container can still start and be inspected.
- `GET /model/info` — active model version, threshold, and validation/test
  metrics.
- `POST /predict` — score one sensor reading:

  ```json
  {
    "machine_id": "M0001",
    "timestamp": "2026-08-24T10:00:00",
    "temperature": 78.2, "pressure": 4.9, "vibration": 0.42,
    "current": 13.6, "rpm": 1452.0, "flow_rate": 79.1, "power": 5.4
  }
  ```

  returns:

  ```json
  {
    "machine_id": "M0001", "timestamp": "2026-08-24T10:00:00",
    "anomaly": true, "anomaly_score": 0.91, "severity": "HIGH",
    "anomaly_type": "bearing_failure",
    "explanation": "vibration is 4.2 std devs above its recent baseline.",
    "threshold": 0.82,
    "component_scores": {
      "isolation_forest": 0.88, "autoencoder": 0.93, "statistical": 0.71,
      "xgboost": 0.95, "lightgbm": 0.90
    }
  }
  ```

- `POST /predict/batch` — same shape, `{"readings": [...]}` in, `{"predictions": [...]}` out.

The API is **stateful per machine**: rolling/z-score features need recent
history, so each reading is appended to an in-memory `FeatureStateStore`
keyed by `machine_id` before scoring. A cold-start machine (fewer than a
few readings seen) returns `severity: "UNKNOWN"` / `anomaly_type:
"insufficient_data"` rather than a misleading score. In a multi-replica
deployment, swap `FeatureStateStore` for the Redis/feature-store backend
implied in the architecture diagram — `AnomalyPredictor` only depends on
its `append` / `get_history` interface.

## How scoring works

Five models feed one hybrid score, split across two training regimes:

**Unsupervised** (trained only on presumed-normal operation, no failure
label used):

1. **Isolation Forest** — multivariate structural outlier detection.
2. **Autoencoder** — reconstruction error against learned normal operating
   patterns; catches subtler multivariate drift IF's axis-aligned splits
   can miss.
3. **Statistical score** — mean clipped |z-score| across sensors, a
   transparent, model-free sanity signal.

**Supervised** (trained on the weak pre-failure labels in
`src/data/labels.py` — did this reading fall in the 24h lead-up to a
*known* historical failure):

4. **XGBoost** — gradient-boosted trees, `scale_pos_weight`-corrected for
   the severe class imbalance (failures are rare).
5. **LightGBM** — same idea, different boosting implementation; ensembling
   two independently-trained GBMs reduces variance from either one
   overfitting the sparse failure record.

The two regimes catch different things: the unsupervised models flag "this
doesn't look like this machine's own normal operation" without needing any
failure to have happened before; the supervised models flag "this looks
like the lead-up to a failure we've seen elsewhere in the fleet" — and can
only learn that pattern where a labeled failure exists at all.

Combined into one **hybrid score** (`configs/config.yaml: hybrid_weights`,
must sum to 1.0 — enforced at config load, default `0.25 / 0.25 / 0.1 /
0.2 / 0.2` for autoencoder / isolation_forest / statistical / xgboost /
lightgbm), thresholded (`best_threshold_by_f1` on the validation split)
into `anomaly: true/false`, then mapped to:

- **severity**: LOW / MEDIUM / HIGH / CRITICAL, from configurable score bands
- **anomaly_type**: `bearing_failure` / `overheating` / `pump_blockage` /
  `unspecified_anomaly`, from which sensor z-scores dominate
- **explanation**: plain-language sentence naming the top contributing
  sensors

## Evaluation

**See [RESULTS.md](RESULTS.md) for a full evaluation report with real
metrics and charts from a full-scale training run** — per-model
performance, PR/ROC curves, score separation, and an MTTD case study
walking through an actual detected failure.

`src/models/evaluate.py` provides both ML metrics (precision/recall/F1,
PR-AUC, ROC-AUC) and the two metrics that translate to business value:

- **Alert aggregation** — collapses raw 5-minute-interval scores into one
  alert per contiguous high-score episode, per machine.
- **MTTD (Mean Time To Detection)** — lead time between first alert and
  the actual recorded failure. This is the number that maps to avoided
  emergency-repair and downtime cost.

`src/models/shap_explainability.py` provides offline SHAP attribution for
the Isolation Forest, used to sanity-check that the rule-based
`anomaly_type`/`explanation` logic tracks what the model actually keys off
— not to explain individual live predictions (too slow for a 5-minute
cadence at fleet scale).

## Monitoring & retraining

`src/monitoring/drift.py`:

- `population_stability_index` / `compute_feature_drift` — PSI per
  feature, train vs. current window. `< 0.1` no action, `0.1-0.25` watch,
  `>= 0.25` retraining candidate (thresholds in `configs/config.yaml:
  monitoring`).
- `daily_anomaly_rate` — a sudden fleet-wide jump signals prediction
  drift or a systemic issue, not routine per-machine anomalies.
- `false_positive_rate` — requires operator alert-confirmation feedback
  (the `Database` -> `Alert System` -> `Operations Team` loop in the
  architecture diagram) to populate.

Recommended cadence: compute drift + FPR weekly; retrain when critical PSI
drift is flagged or FPR/MTTD regress past agreed bounds; validate a
challenger against the current champion on the same test window before
promoting (champion/challenger, per the architecture diagram) — this
repo's training pipeline (`src.models.train.run_training`) is deliberately
a pure function of `AppConfig` so a challenger run is just training again
against a new data window or config and diffing `metadata.json`.

## Testing

```bash
pytest                                                  # full suite
pytest --cov=src --cov=app --cov-report=term-missing    # with coverage
```

The API test suite (`tests/test_api.py`) uses FastAPI dependency
injection (`app.dependency_overrides`) with a stub predictor, so it runs
without needing trained model artifacts. `tests/test_model.py` and
`tests/test_features.py` exercise the real Isolation Forest, Autoencoder,
XGBoost, LightGBM, and feature-engineering code directly — including a
save/load round-trip test for each model's persistence format.
`tests/test_config.py` asserts the config loader fails fast if
`hybrid_weights` doesn't sum to 1.0.

## Code quality & CI/CD

```bash
ruff check .            # lint
ruff check --fix .      # lint, auto-fixing what's safe to fix
ruff format .           # format (black-compatible)
mypy src app            # type check (advisory — see below)
```

`.github/workflows/ci.yml` runs on every push to `main` and every PR:

| Job         | What it gates                                            |
|-------------|-----------------------------------------------------------|
| `lint`      | `ruff check` + `ruff format --check` — **required**        |
| `typecheck` | `mypy src app` — advisory (`continue-on-error`), promote once the codebase's type coverage is clean |
| `test`      | `pytest` with coverage, uploaded as a build artifact — **required**, depends on `lint` |
| `docker`    | `docker build .` — validates the Dockerfile builds, depends on `test` |

`.github/dependabot.yml` opens weekly update PRs for pip, Docker base
images, and GitHub Actions versions.

## Configuration

Everything runtime-tunable — data generation volume, feature windows,
model hyperparameters (including XGBoost/LightGBM), hybrid weights,
severity thresholds, drift thresholds — lives in `configs/config.yaml`,
loaded via `src.config.load_config()`. Change behavior there, not by
editing code. `hybrid_weights` is validated to sum to 1.0 at load time —
a typo there would otherwise silently mis-rank every prediction.
