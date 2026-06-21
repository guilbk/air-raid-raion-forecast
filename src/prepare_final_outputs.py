import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

DATA_PATH = Path("data/processed/model_features_v2_15min_purged.csv")
MODEL_PATH = Path("data/processed/logistic_v3_final_model.joblib")

OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FINAL_METRICS_PATH = OUT_DIR / "final_model_metrics.csv"
MODEL_COMPARISON_PATH = OUT_DIR / "final_model_comparison.csv"
LATEST_SNAPSHOT_PATH = OUT_DIR / "latest_risk_snapshot.csv"
RISK_BY_OBLAST_PATH = OUT_DIR / "risk_by_oblast.csv"
RISK_BY_HOUR_PATH = OUT_DIR / "risk_by_hour.csv"
PRED_SAMPLE_PATH = OUT_DIR / "logistic_v3_test_predictions_sample.csv"

TARGET = "target_start_60m"
RANDOM_STATE = 42


def evaluate_predictions(y_true, proba, model_name, split_name):
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba).astype(float)
    proba = np.clip(proba, 0, 1)

    y_pred_05 = (proba >= 0.5).astype(int)

    positive_rows = int(y_true.sum())
    total_rows = len(y_true)

    k = max(1, int(total_rows * 0.10))
    top_idx = np.argpartition(proba, -k)[-k:]
    top_true = y_true[top_idx]

    return {
        "model": model_name,
        "split": split_name,
        "rows": total_rows,
        "positive_rows": positive_rows,
        "positive_rate": positive_rows / total_rows if total_rows else 0,
        "pr_auc": average_precision_score(y_true, proba),
        "roc_auc": roc_auc_score(y_true, proba),
        "brier": brier_score_loss(y_true, proba),
        "accuracy_at_0_5": accuracy_score(y_true, y_pred_05),
        "precision_at_0_5": precision_score(y_true, y_pred_05, zero_division=0),
        "recall_at_0_5": recall_score(y_true, y_pred_05, zero_division=0),
        "f1_at_0_5": f1_score(y_true, y_pred_05, zero_division=0),
        "precision_top_10_percent": float(top_true.sum() / k),
        "recall_top_10_percent": float(top_true.sum() / positive_rows) if positive_rows else 0,
    }


print("Reading model...")
saved = joblib.load(MODEL_PATH)
model = saved["model"]
feature_cols = saved["feature_cols"]

print("Final model info:")
print("best_search_model:", saved["best_search_model"])
print("feature_set_name:", saved["feature_set_name"])
print("c_value:", saved["c_value"])
print("feature_count:", len(feature_cols))
print()

print("Reading dataset...")
df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

test = df[df["split"] == "test"].copy()

print("Test rows:", len(test))
print("Test target rate:", test[TARGET].mean())
print()

print("Predicting test...")
test["pred_logistic_v3"] = model.predict_proba(test[feature_cols])[:, 1]

metrics = pd.DataFrame(
    [
        evaluate_predictions(
            y_true=test[TARGET],
            proba=test["pred_logistic_v3"],
            model_name="logistic_v3_final",
            split_name="test",
        )
    ]
)

metrics["best_search_model"] = saved["best_search_model"]
metrics["feature_set_name"] = saved["feature_set_name"]
metrics["c_value"] = saved["c_value"]
metrics["feature_count"] = len(feature_cols)

metrics.to_csv(FINAL_METRICS_PATH, index=False)

print("Creating model comparison...")

metric_files = [
    OUT_DIR / "baseline_metrics.csv",
    OUT_DIR / "catboost_metrics.csv",
    OUT_DIR / "model_v2_metrics.csv",
    OUT_DIR / "logistic_v3_search_metrics.csv",
    OUT_DIR / "logistic_v3_final_metrics.csv",
    FINAL_METRICS_PATH,
]

frames = []

for path in metric_files:
    if path.exists():
        part = pd.read_csv(path)
        part["source_file"] = path.name
        frames.append(part)

comparison = pd.concat(frames, ignore_index=True)
comparison = comparison.sort_values(["split", "pr_auc"], ascending=[True, False])
comparison.to_csv(MODEL_COMPARISON_PATH, index=False)

print("Creating latest risk snapshot...")

latest_ts = test["timestamp"].max()
latest = test[test["timestamp"] == latest_ts].copy()

latest["risk_rank"] = latest["pred_logistic_v3"].rank(
    ascending=False,
    method="first",
).astype(int)

latest["risk_percentile"] = latest["pred_logistic_v3"].rank(
    pct=True,
    method="average",
).astype(float)

latest["risk_level"] = "low"
latest.loc[latest["risk_percentile"] >= 0.50, "risk_level"] = "medium"
latest.loc[latest["risk_percentile"] >= 0.75, "risk_level"] = "high"
latest.loc[latest["risk_percentile"] >= 0.90, "risk_level"] = "very_high"

latest_out = latest[
    [
        "timestamp",
        "oblast",
        "raion",
        "pred_logistic_v3",
        "risk_rank",
        "risk_percentile",
        "risk_level",
        TARGET,
    ]
].sort_values("risk_rank")

latest_out.to_csv(LATEST_SNAPSHOT_PATH, index=False)

print("Creating risk summaries...")

risk_by_oblast = (
    test.groupby("oblast", as_index=False)
    .agg(
        rows=("pred_logistic_v3", "size"),
        actual_positive_rate=(TARGET, "mean"),
        avg_predicted_risk=("pred_logistic_v3", "mean"),
        max_predicted_risk=("pred_logistic_v3", "max"),
    )
    .sort_values("avg_predicted_risk", ascending=False)
)

risk_by_oblast.to_csv(RISK_BY_OBLAST_PATH, index=False)

risk_by_hour = (
    test.groupby("hour", as_index=False)
    .agg(
        rows=("pred_logistic_v3", "size"),
        actual_positive_rate=(TARGET, "mean"),
        avg_predicted_risk=("pred_logistic_v3", "mean"),
    )
    .sort_values("hour")
)

risk_by_hour.to_csv(RISK_BY_HOUR_PATH, index=False)

print("Creating prediction sample...")

sample_size = min(150_000, len(test))
sample = test.sample(n=sample_size, random_state=RANDOM_STATE).copy()

sample_out = sample[
    [
        "timestamp",
        "oblast",
        "raion",
        TARGET,
        "pred_logistic_v3",
        "hour",
        "day_of_week",
        "oblast_active_share_now",
        "country_active_share_now",
    ]
].copy()

sample_out.to_csv(PRED_SAMPLE_PATH, index=False)

print("Saved:")
print(FINAL_METRICS_PATH)
print(MODEL_COMPARISON_PATH)
print(LATEST_SNAPSHOT_PATH)
print(RISK_BY_OBLAST_PATH)
print(RISK_BY_HOUR_PATH)
print(PRED_SAMPLE_PATH)
print()

print("FINAL METRICS")
print(metrics.to_string(index=False))
print()

print("TOP 15 LATEST RISKS")
print(latest_out.head(15).to_string(index=False))
print()

print("TOP 10 OBLASTS BY AVG RISK")
print(risk_by_oblast.head(10).to_string(index=False))