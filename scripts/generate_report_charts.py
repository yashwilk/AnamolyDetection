"""Regenerate the charts embedded in RESULTS.md from the currently trained
artifacts in ``models/`` and the currently generated data in ``data/raw/``.

Recomputes scores directly from the saved model artifacts (vectorized —
not through the per-row `AnomalyPredictor.predict_dataframe` streaming
path, which is fine for one machine at a time but far too slow to
bulk-score hundreds of thousands of historical rows for a report).

Run after `python -m src.models.train`:
    python scripts/generate_report_charts.py

Colors follow the validated categorical/status palette documented in the
`dataviz` skill (see RESULTS.md's "How these charts were made" note).
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.metrics import precision_recall_curve, roc_curve

from src.config import load_config
from src.models.autoencoder import AutoencoderTrainer
from src.models.evaluate import aggregate_alerts, calculate_mttd
from src.models.gradient_boosting import LightGBMAnomalyModel, XGBoostAnomalyModel
from src.models.isolation_forest import IsolationForestModel
from src.models.train import compute_hybrid_score, load_labeled_features, time_split

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"

STATUS_GOOD = "#0ca30c"
STATUS_WARNING = "#fab219"
STATUS_CRITICAL = "#d03b3b"

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

CATEGORICAL_3 = [BLUE, ORANGE, AQUA]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "docs" / "charts"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.size": 11,
        "text.color": INK_PRIMARY,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_SECONDARY,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "axes.grid": True,
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)


def savefig(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT_DIR / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def load_trained_artifacts(config, metadata):
    models_dir = config.paths.models_dir
    feature_columns = metadata["feature_columns"]
    return {
        "scaler": joblib.load(models_dir / "scaler.pkl"),
        "if_score_scaler": joblib.load(models_dir / "if_score_scaler.pkl"),
        "ae_score_scaler": joblib.load(models_dir / "ae_score_scaler.pkl"),
        "isolation_forest": IsolationForestModel.load(
            models_dir / "isolation_forest.pkl", config.isolation_forest, config.random_state
        ),
        "autoencoder": AutoencoderTrainer.load(
            models_dir / "autoencoder.pt", config.autoencoder, input_dim=len(feature_columns)
        ),
        "xgboost": XGBoostAnomalyModel.load(models_dir / "xgboost.ubj", config.xgboost, config.random_state),
        "lightgbm": LightGBMAnomalyModel.load(
            models_dir / "lightgbm.txt", config.lightgbm, config.random_state
        ),
    }


def score_split(df: pd.DataFrame, artifacts: dict, feature_columns, zscore_columns, weights) -> pd.DataFrame:
    X_scaled = artifacts["scaler"].transform(df[feature_columns])
    if_raw = artifacts["isolation_forest"].score(X_scaled)
    ae_raw = artifacts["autoencoder"].score(X_scaled)
    if_norm = np.clip(artifacts["if_score_scaler"].transform(if_raw.reshape(-1, 1)).flatten(), 0, 1)
    ae_norm = np.clip(artifacts["ae_score_scaler"].transform(ae_raw.reshape(-1, 1)).flatten(), 0, 1)

    z = df[zscore_columns].abs().clip(upper=6.0)
    statistical = (z.mean(axis=1) / 6.0).to_numpy()

    xgb = np.clip(artifacts["xgboost"].score(df[feature_columns]), 0, 1)
    lgbm = np.clip(artifacts["lightgbm"].score(df[feature_columns]), 0, 1)
    hybrid = compute_hybrid_score(ae_norm, if_norm, statistical, xgb, lgbm, weights)

    out = df.copy()
    out["anomaly_score"] = hybrid
    return out


def chart_dataset_volume(train_model, validation_model, test_model) -> None:
    splits = ["Train", "Validation", "Test"]
    counts = [len(train_model), len(validation_model), len(test_model)]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(splits, counts, color=CATEGORICAL_3, width=0.55)
    for b, c in zip(bars, counts, strict=False):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"{c:,}",
            ha="center",
            va="bottom",
            color=INK_PRIMARY,
        )
    ax.set_ylabel("Sensor readings")
    ax.set_title("Dataset volume per time-based split", color=INK_PRIMARY, fontweight="bold")
    ax.grid(axis="x", visible=False)
    savefig(fig, "01_dataset_volume.png")


def chart_failures_per_split(failures_df, train_end, validation_end) -> tuple[int, int, int]:
    splits = ["Train", "Validation", "Test"]
    train_failures = int((failures_df["failure_timestamp"] <= train_end).sum())
    val_failures = int(
        (
            (failures_df["failure_timestamp"] > train_end)
            & (failures_df["failure_timestamp"] <= validation_end)
        ).sum()
    )
    test_failures = int((failures_df["failure_timestamp"] > validation_end).sum())
    failure_counts = [train_failures, val_failures, test_failures]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(splits, failure_counts, color=CATEGORICAL_3, width=0.55)
    for b, c in zip(bars, failure_counts, strict=False):
        ax.text(
            b.get_x() + b.get_width() / 2, b.get_height(), str(c), ha="center", va="bottom", color=INK_PRIMARY
        )
    ax.set_ylabel("Failure events")
    ax.set_title(f"Failures per split (n={len(failures_df)} total)", color=INK_PRIMARY, fontweight="bold")
    ax.grid(axis="x", visible=False)
    savefig(fig, "02_failures_per_split.png")
    return train_failures, val_failures, test_failures


def chart_submodel_performance(metadata) -> None:
    submodels = metadata["submodel_validation_metrics"]
    model_order = ["xgboost", "isolation_forest", "lightgbm", "autoencoder", "statistical"]
    labels = ["XGBoost", "Isolation\nForest", "LightGBM", "Autoencoder", "Statistical"]
    pr_aucs = [submodels[m]["pr_auc"] for m in model_order]
    roc_aucs = [submodels[m]["roc_auc"] for m in model_order]

    x = np.arange(len(model_order))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2, pr_aucs, width, label="PR-AUC", color=BLUE)
    ax.bar(x + width / 2, roc_aucs, width, label="ROC-AUC", color=ORANGE)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (validation split)")
    ax.set_title("Per-model standalone performance", color=INK_PRIMARY, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    savefig(fig, "03_submodel_performance.png")


def chart_score_distribution(test_scored, threshold) -> None:
    normal_scores = test_scored.loc[test_scored["anomaly"] == 0, "anomaly_score"]
    anomaly_scores = test_scored.loc[test_scored["anomaly"] == 1, "anomaly_score"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0, 1, 60)
    ax.hist(
        normal_scores,
        bins=bins,
        color=STATUS_GOOD,
        alpha=0.55,
        label=f"Normal (n={len(normal_scores):,})",
        density=True,
    )
    ax.hist(
        anomaly_scores,
        bins=bins,
        color=STATUS_CRITICAL,
        alpha=0.65,
        label=f"Anomaly (n={len(anomaly_scores):,})",
        density=True,
    )
    ax.axvline(threshold, color=INK_PRIMARY, linestyle="--", linewidth=1.5)
    ax.text(
        threshold - 0.01,
        ax.get_ylim()[1] * 0.5,
        f"threshold={threshold:.3f}",
        va="center",
        ha="right",
        fontsize=9,
        color=INK_PRIMARY,
        rotation=90,
    )
    ax.set_xlabel("Hybrid anomaly score")
    ax.set_ylabel("Density")
    ax.set_title(
        "Score separation: normal vs. true anomaly (test split)", color=INK_PRIMARY, fontweight="bold"
    )
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    savefig(fig, "04_score_distribution.png")


def chart_pr_roc_curves(test_scored, metadata) -> None:
    precision, recall, _ = precision_recall_curve(test_scored["anomaly"], test_scored["anomaly_score"])
    fpr, tpr, _ = roc_curve(test_scored["anomaly"], test_scored["anomaly_score"])

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    axes[0].plot(recall, precision, color=BLUE, linewidth=2)
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_title(f"Precision-Recall (AUC={metadata['test_metrics']['pr_auc']:.3f})", fontsize=11)
    axes[0].set_xlim(0, 1.02)
    axes[0].set_ylim(0, 1.02)

    axes[1].plot(fpr, tpr, color=BLUE, linewidth=2)
    axes[1].plot([0, 1], [0, 1], color=BASELINE, linestyle="--", linewidth=1)
    axes[1].set_xlabel("False positive rate")
    axes[1].set_ylabel("True positive rate")
    axes[1].set_title(f"ROC (AUC={metadata['test_metrics']['roc_auc']:.3f})", fontsize=11)
    axes[1].set_xlim(0, 1.02)
    axes[1].set_ylim(0, 1.02)
    fig.suptitle("Hybrid score ranking quality — test split", fontweight="bold", color=INK_PRIMARY)
    savefig(fig, "05_pr_roc_curves.png")


def chart_mttd(mttd_df, test_failures_count) -> None:
    mttd_hours = (mttd_df["lead_time"].dt.total_seconds() / 3600).to_numpy()
    machine_labels = [
        f"{m}\n({t})" for m, t in zip(mttd_df["machine_id"], mttd_df["failure_type"], strict=False)
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.barh(machine_labels, mttd_hours, color=BLUE, height=0.55)
    for b, v in zip(bars, mttd_hours, strict=False):
        ax.text(v, b.get_y() + b.get_height() / 2, f"  {v:.1f}h", va="center", ha="left", color=INK_PRIMARY)

    top_y = len(mttd_hours) - 1
    ax.axvline(mttd_hours.mean(), color=INK_MUTED, linestyle="--", linewidth=1)
    ax.text(
        mttd_hours.mean(),
        top_y + 0.55,
        f"mean = {mttd_hours.mean():.1f}h",
        ha="center",
        va="bottom",
        fontsize=9,
        color=INK_SECONDARY,
    )
    ax.set_ylim(-0.6, top_y + 0.9)
    ax.set_xlim(0, max(mttd_hours) * 1.35)
    ax.set_xlabel("Lead time ahead of failure (hours)")
    ax.set_title(
        f"MTTD — {len(mttd_df)}/{test_failures_count} test-period failures detected early",
        color=INK_PRIMARY,
        fontweight="bold",
    )
    ax.grid(axis="y", visible=False)
    savefig(fig, "06_mttd_lead_times.png")


def chart_example_failure_trace(mttd_df, validation_scored, test_scored, threshold) -> None:
    # The failure with the longest lead time is used as the example — an
    # earlier test-period failure can sit right at the val/test boundary,
    # truncating the 48h lookback window this chart shows.
    example_row = mttd_df.loc[mttd_df["lead_time"].idxmax()]
    example_machine = example_row["machine_id"]
    example_failure_time = example_row["failure_timestamp"]
    first_detection = example_row["first_detection"]
    window_start = example_failure_time - pd.Timedelta(hours=48)
    window_end = example_failure_time + pd.Timedelta(hours=6)

    trace_source = pd.concat([validation_scored, test_scored], ignore_index=True)
    trace = trace_source[
        (trace_source["machine_id"] == example_machine)
        & (trace_source["timestamp"] >= window_start)
        & (trace_source["timestamp"] <= window_end)
    ].sort_values("timestamp")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True, height_ratios=[2, 1.2])
    ax1.plot(trace["timestamp"], trace["vibration"], color=BLUE, linewidth=1.5)
    ax1.set_ylabel("Vibration")
    ax1.axvspan(
        example_failure_time - pd.Timedelta(hours=24), example_failure_time, color=STATUS_WARNING, alpha=0.12
    )
    ax1.axvline(example_failure_time, color=STATUS_CRITICAL, linewidth=1.5)
    ax1.text(example_failure_time, ax1.get_ylim()[1], " failure", color=STATUS_CRITICAL, va="top", fontsize=9)

    ax2.plot(trace["timestamp"], trace["anomaly_score"], color=ORANGE, linewidth=1.5)
    ax2.axhline(threshold, color=INK_MUTED, linestyle="--", linewidth=1)
    ax2.axvline(first_detection, color=STATUS_GOOD, linewidth=1.5)
    ax2.text(first_detection, threshold, " first detected", color=STATUS_GOOD, va="bottom", fontsize=9)
    ax2.axvline(example_failure_time, color=STATUS_CRITICAL, linewidth=1.5)
    ax2.set_ylabel("Hybrid score")
    ax2.set_ylim(0, 1.05)
    fig.autofmt_xdate()

    lead_hours = (example_failure_time - first_detection).total_seconds() / 3600
    fig.suptitle(
        f"{example_machine} — detected {lead_hours:.1f}h before failure", fontweight="bold", color=INK_PRIMARY
    )
    legend_elements = [
        Line2D([0], [0], color=STATUS_WARNING, lw=8, alpha=0.3, label="24h pre-failure window"),
        Line2D([0], [0], color=STATUS_CRITICAL, lw=2, label="failure"),
        Line2D([0], [0], color=STATUS_GOOD, lw=2, label="first detection"),
    ]
    ax1.legend(handles=legend_elements, loc="upper left", fontsize=8)
    savefig(fig, "07_example_failure_trace.png")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config()
    models_dir = config.paths.models_dir

    with open(models_dir / "metadata.json", encoding="utf-8") as fh:
        metadata = json.load(fh)

    feature_columns = metadata["feature_columns"]
    zscore_columns = metadata["zscore_columns"]
    threshold = metadata["threshold"]

    print("Loading + labeling features (this re-derives the full dataset — can take ~15s)...")
    labeled_df, failures_df = load_labeled_features(config)
    train, validation, test = time_split(labeled_df, config)

    train_model = train.dropna(subset=feature_columns)
    validation_model = validation.dropna(subset=feature_columns).copy()
    test_model = test.dropna(subset=feature_columns).copy()

    print("Loading trained artifacts and scoring validation + test splits...")
    artifacts = load_trained_artifacts(config, metadata)
    weights = config.hybrid_weights
    test_scored = score_split(test_model, artifacts, feature_columns, zscore_columns, weights)
    validation_scored = score_split(validation_model, artifacts, feature_columns, zscore_columns, weights)

    train_end = pd.Timestamp(config.split.train_end)
    validation_end = pd.Timestamp(config.split.validation_end)
    test_failures_df = failures_df[failures_df["failure_timestamp"] > validation_end]

    alerts = aggregate_alerts(test_scored, threshold)
    mttd_df = calculate_mttd(alerts, test_failures_df).sort_values("failure_timestamp")

    print("Rendering charts...")
    chart_dataset_volume(train_model, validation_model, test_model)
    chart_failures_per_split(failures_df, train_end, validation_end)
    chart_submodel_performance(metadata)
    chart_score_distribution(test_scored, threshold)
    chart_pr_roc_curves(test_scored, metadata)
    chart_mttd(mttd_df, len(test_failures_df))
    chart_example_failure_trace(mttd_df, validation_scored, test_scored, threshold)

    print(f"Done — charts written to {OUT_DIR}")


if __name__ == "__main__":
    main()
