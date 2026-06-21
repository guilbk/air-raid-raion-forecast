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

try:
    from catboost import CatBoostClassifier
except ImportError:
    raise ImportError(
        "CatBoost is not installed. Run: pip install catboost"
    )


IN_PATH = Path("data/processed/model_features_15min_purged.csv")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS_PATH = OUT_DIR / "catboost_metrics.csv"
PRED_SAMPLE_PATH = OUT_DIR / "catboost_predictions_sample.csv"
MODEL_PATH = OUT_DIR / "catboost_target_start_60m.cbm"

TARGET = "target_start_60m"
RANDOM_STATE = 42
MAX_TRAIN_ROWS = 800_000

feature_cols = [
    "oblast",
    "raion",
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

cat_features = [
    "oblast",
    "raion",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_night",
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


def make_train_sample(train_df):
    if len(train_df) <= MAX_TRAIN_ROWS:
        return train_df

    pos = train_df[train_df[TARGET] == 1]
    neg = train_df[train_df[TARGET] == 0]

    positive_rate = train_df[TARGET].mean()

    n_pos = min(len(pos), int(MAX_TRAIN_ROWS * positive_rate))
    n_neg = MAX_TRAIN_ROWS - n_pos

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

train_sample = make_train_sample(train_df)

print("Train full rows:", len(train_df))
print("Train sample rows:", len(train_sample))
print("Train sample target rate:", train_sample[TARGET].mean())
print("Validation rows:", len(validation_df))
print("Test rows:", len(test_df))
print()

cat_feature_indices = [feature_cols.index(col) for col in cat_features]

model = CatBoostClassifier(
    loss_function="Logloss",
    eval_metric="PRAUC",
    iterations=500,
    learning_rate=0.06,
    depth=6,
    l2_leaf_reg=6,
    random_seed=RANDOM_STATE,
    auto_class_weights="Balanced",
    verbose=50,
    allow_writing_files=False,
)

print("Training CatBoost...")
model.fit(
    train_sample[feature_cols],
    train_sample[TARGET],
    cat_features=cat_feature_indices,
    eval_set=(validation_df[feature_cols], validation_df[TARGET]),
    use_best_model=True,
)

model.save_model(MODEL_PATH)

metrics = []
prediction_samples = []

for split_name, part in [
    ("validation", validation_df),
    ("test", test_df),
]:
    print(f"Evaluating {split_name}...")

    y_true = part[TARGET].to_numpy()
    proba = model.predict_proba(part[feature_cols])[:, 1]

    metrics.append(
        evaluate_predictions(
            y_true=y_true,
            proba=proba,
            model_name="catboost",
            split_name=split_name,
        )
    )

    sample_size = min(50_000, len(part))
    sample = part.sample(n=sample_size, random_state=RANDOM_STATE).copy()
    proba_series = pd.Series(proba, index=part.index)

    sample_out = sample[
        ["timestamp", "oblast", "raion", "split", TARGET]
    ].copy()
    sample_out["pred_catboost"] = proba_series.loc[sample.index].values
    prediction_samples.append(sample_out)

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(METRICS_PATH, index=False)

pred_sample_df = pd.concat(prediction_samples, ignore_index=True)
pred_sample_df.to_csv(PRED_SAMPLE_PATH, index=False)

feature_importance = pd.DataFrame(
    {
        "feature": feature_cols,
        "importance": model.get_feature_importance(),
    }
).sort_values("importance", ascending=False)

feature_importance.to_csv(OUT_DIR / "catboost_feature_importance.csv", index=False)

print("Saved:")
print(METRICS_PATH)
print(PRED_SAMPLE_PATH)
print(MODEL_PATH)
print(OUT_DIR / "catboost_feature_importance.csv")
print()

print("METRICS")
print(metrics_df.to_string(index=False))
print()

print("FEATURE IMPORTANCE")
print(feature_importance.to_string(index=False))