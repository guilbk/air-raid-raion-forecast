import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.preprocessing import OneHotEncoder
    one_hot = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
except TypeError:
    from sklearn.preprocessing import OneHotEncoder
    one_hot = OneHotEncoder(handle_unknown="ignore", sparse=True)


IN_PATH = Path("data/processed/model_features_15min_purged.csv")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS_PATH = OUT_DIR / "baseline_metrics.csv"
PRED_SAMPLE_PATH = OUT_DIR / "baseline_predictions_sample.csv"

TARGET = "target_start_60m"
RANDOM_STATE = 42
MAX_LR_TRAIN_ROWS = 600_000

feature_cols = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_night",
    "starts_last_1h",
    "starts_last_3h",
    "starts_last_24h",
    "active_minutes_last_1h",
    "active_minutes_last_3h",
    "active_minutes_last_24h",
    "oblast_active_raions_now",
    "oblast_active_share_now",
]

categorical_features = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_night",
]

numeric_features = [
    "starts_last_1h",
    "starts_last_3h",
    "starts_last_24h",
    "active_minutes_last_1h",
    "active_minutes_last_3h",
    "active_minutes_last_24h",
    "oblast_active_raions_now",
    "oblast_active_share_now",
]


def evaluate_predictions(y_true, proba, model_name, split_name):
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba).astype(float)

    y_pred_05 = (proba >= 0.5).astype(int)

    if len(np.unique(y_true)) < 2:
        roc_auc = np.nan
    else:
        roc_auc = roc_auc_score(y_true, proba)

    positive_rows = int(y_true.sum())
    total_rows = len(y_true)

    k = max(1, int(total_rows * 0.10))
    top_idx = np.argpartition(proba, -k)[-k:]
    top_true = y_true[top_idx]

    precision_top_10 = float(top_true.sum() / k)
    recall_top_10 = float(top_true.sum() / positive_rows) if positive_rows else 0.0

    return {
        "model": model_name,
        "split": split_name,
        "rows": total_rows,
        "positive_rows": positive_rows,
        "positive_rate": positive_rows / total_rows if total_rows else 0,
        "pr_auc": average_precision_score(y_true, proba),
        "roc_auc": roc_auc,
        "brier": brier_score_loss(y_true, proba),
        "accuracy_at_0_5": accuracy_score(y_true, y_pred_05),
        "precision_at_0_5": precision_score(y_true, y_pred_05, zero_division=0),
        "recall_at_0_5": recall_score(y_true, y_pred_05, zero_division=0),
        "f1_at_0_5": f1_score(y_true, y_pred_05, zero_division=0),
        "precision_top_10_percent": precision_top_10,
        "recall_top_10_percent": recall_top_10,
    }


def historical_frequency_predictions(train_df, eval_df):
    global_rate = train_df[TARGET].mean()
    alpha = 30

    key_cols = ["oblast", "raion", "hour", "day_of_week"]

    stats = (
        train_df
        .groupby(key_cols)[TARGET]
        .agg(["sum", "count"])
        .reset_index()
    )

    stats["historical_proba"] = (
        (stats["sum"] + alpha * global_rate) / (stats["count"] + alpha)
    )

    merged = eval_df[key_cols].merge(
        stats[key_cols + ["historical_proba"]],
        on=key_cols,
        how="left",
    )

    return merged["historical_proba"].fillna(global_rate).to_numpy()


def make_lr_train_sample(train_df):
    if len(train_df) <= MAX_LR_TRAIN_ROWS:
        return train_df

    pos = train_df[train_df[TARGET] == 1]
    neg = train_df[train_df[TARGET] == 0]

    positive_rate = train_df[TARGET].mean()

    n_pos = min(len(pos), int(MAX_LR_TRAIN_ROWS * positive_rate))
    n_neg = MAX_LR_TRAIN_ROWS - n_pos

    sampled = pd.concat(
        [
            pos.sample(n=n_pos, random_state=RANDOM_STATE),
            neg.sample(n=n_neg, random_state=RANDOM_STATE),
        ],
        ignore_index=True,
    )

    sampled = sampled.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    return sampled


print("Reading dataset...")
df = pd.read_csv(IN_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

print("Dataset shape:", df.shape)
print(df["split"].value_counts())
print()

train_df = df[df["split"] == "train"].copy()
validation_df = df[df["split"] == "validation"].copy()
test_df = df[df["split"] == "test"].copy()

global_train_rate = train_df[TARGET].mean()

print("Train target rate:", global_train_rate)
print("Validation target rate:", validation_df[TARGET].mean())
print("Test target rate:", test_df[TARGET].mean())
print()

print("Training logistic regression...")
lr_train = make_lr_train_sample(train_df)

print("LR train sample rows:", len(lr_train))
print("LR train sample target rate:", lr_train[TARGET].mean())

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", one_hot, categorical_features),
        ("num", StandardScaler(with_mean=False), numeric_features),
    ]
)

log_reg = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(
                max_iter=200,
                solver="saga",
                class_weight="balanced",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)

log_reg.fit(lr_train[feature_cols], lr_train[TARGET])

print("Logistic regression trained")
print()

metrics = []
prediction_samples = []

for split_name, part in [
    ("validation", validation_df),
    ("test", test_df),
]:
    print(f"Evaluating {split_name}...")

    y_true = part[TARGET].to_numpy()

    pred_always_zero = np.zeros(len(part), dtype=float)

    pred_prevalence = np.full(
        len(part),
        global_train_rate,
        dtype=float,
    )

    pred_historical = historical_frequency_predictions(train_df, part)

    pred_oblast_activity = (
        part["oblast_active_share_now"]
        .clip(0, 1)
        .to_numpy(dtype=float)
    )

    pred_logistic = log_reg.predict_proba(part[feature_cols])[:, 1]

    predictions = {
        "always_no_alert": pred_always_zero,
        "train_prevalence": pred_prevalence,
        "historical_raion_hour_dow": pred_historical,
        "oblast_activity_share": pred_oblast_activity,
        "logistic_regression": pred_logistic,
    }

    for model_name, proba in predictions.items():
        metrics.append(
            evaluate_predictions(
                y_true=y_true,
                proba=proba,
                model_name=model_name,
                split_name=split_name,
            )
        )

    sample_size = min(50_000, len(part))
    sample_idx = part.sample(n=sample_size, random_state=RANDOM_STATE).index

    sample = part.loc[
        sample_idx,
        ["timestamp", "oblast", "raion", "split", TARGET],
    ].copy()

    for model_name, proba in predictions.items():
        proba_series = pd.Series(proba, index=part.index)
        sample[f"pred_{model_name}"] = proba_series.loc[sample_idx].values

    prediction_samples.append(sample)

metrics_df = pd.DataFrame(metrics)
metrics_df = metrics_df.sort_values(["split", "pr_auc"], ascending=[True, False])
metrics_df.to_csv(METRICS_PATH, index=False)

pred_sample_df = pd.concat(prediction_samples, ignore_index=True)
pred_sample_df.to_csv(PRED_SAMPLE_PATH, index=False)

print("Saved:")
print(METRICS_PATH)
print(PRED_SAMPLE_PATH)
print()

print("METRICS")
print(metrics_df.to_string(index=False))