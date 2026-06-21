import numpy as np
import pandas as pd
from pathlib import Path
import joblib

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
    one_hot_kwargs = {"handle_unknown": "ignore", "sparse_output": True}
except TypeError:
    from sklearn.preprocessing import OneHotEncoder
    one_hot_kwargs = {"handle_unknown": "ignore", "sparse": True}


IN_PATH = Path("data/processed/model_features_v2_15min_purged.csv")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS_PATH = OUT_DIR / "logistic_v3_search_metrics.csv"
FINAL_METRICS_PATH = OUT_DIR / "logistic_v3_final_metrics.csv"
FINAL_MODEL_PATH = OUT_DIR / "logistic_v3_final_model.joblib"
PRED_SAMPLE_PATH = OUT_DIR / "logistic_v3_predictions_sample.csv"

TARGET = "target_start_60m"
RANDOM_STATE = 42

MAX_SEARCH_TRAIN_ROWS = 900_000


SAFE_V1_FEATURES = [
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

SAFE_V1_CYCLIC_FEATURES = SAFE_V1_FEATURES + [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
]

HISTORY_V2_NO_REGION = SAFE_V1_CYCLIC_FEATURES + [
    "ends_last_1h",
    "ends_last_3h",
    "ends_last_24h",
    "active_prev_15m",
    "active_prev_30m",
    "active_prev_60m",
    "minutes_since_last_start",
    "minutes_since_last_end",
    "oblast_starts_last_1h",
    "oblast_starts_last_3h",
    "oblast_starts_last_24h",
    "country_active_raions_now",
    "country_active_share_now",
    "country_starts_last_1h",
    "country_starts_last_3h",
    "country_starts_last_24h",
]

NO_MONTH_FEATURES = [
    col for col in HISTORY_V2_NO_REGION
    if col not in ["month", "month_sin", "month_cos"]
]

NO_COUNTRY_FEATURES = [
    col for col in HISTORY_V2_NO_REGION
    if not col.startswith("country_")
]

TIME_ONLY_FEATURES = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_night",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
]

FEATURE_SETS = {
    "safe_v1": SAFE_V1_FEATURES,
    "safe_v1_cyclic": SAFE_V1_CYCLIC_FEATURES,
    "history_v2_no_region": HISTORY_V2_NO_REGION,
    "history_v2_no_month": NO_MONTH_FEATURES,
    "history_v2_no_country": NO_COUNTRY_FEATURES,
    "time_only": TIME_ONLY_FEATURES,
}

CATEGORICAL_CANDIDATES = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_night",
]

C_VALUES = [0.2, 0.5, 1.0, 2.0]


def evaluate_predictions(y_true, proba, model_name, split_name):
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba).astype(float)
    proba = np.clip(proba, 0, 1)

    y_pred_05 = (proba >= 0.5).astype(int)

    roc_auc = np.nan
    if len(np.unique(y_true)) >= 2:
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


def make_train_sample(train_df, max_rows):
    if len(train_df) <= max_rows:
        return train_df

    pos = train_df[train_df[TARGET] == 1]
    neg = train_df[train_df[TARGET] == 0]

    positive_rate = train_df[TARGET].mean()

    n_pos = min(len(pos), int(max_rows * positive_rate))
    n_neg = max_rows - n_pos

    sampled = pd.concat(
        [
            pos.sample(n=n_pos, random_state=RANDOM_STATE),
            neg.sample(n=n_neg, random_state=RANDOM_STATE),
        ],
        ignore_index=True,
    )

    sampled = sampled.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    return sampled


def build_pipeline(feature_cols, c_value):
    categorical_features = [
        col for col in CATEGORICAL_CANDIDATES
        if col in feature_cols
    ]

    numeric_features = [
        col for col in feature_cols
        if col not in categorical_features
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(**one_hot_kwargs), categorical_features),
            ("num", StandardScaler(with_mean=False), numeric_features),
        ]
    )

    model = LogisticRegression(
        max_iter=350,
        solver="saga",
        class_weight="balanced",
        C=c_value,
        random_state=RANDOM_STATE,
        tol=1e-3,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


print("Reading dataset...")
df = pd.read_csv(IN_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

print("Dataset shape:", df.shape)
print(df["split"].value_counts())
print()

train_df = df[df["split"] == "train"].copy()
validation_df = df[df["split"] == "validation"].copy()
test_df = df[df["split"] == "test"].copy()

print("Target rates:")
print("train:", train_df[TARGET].mean())
print("validation:", validation_df[TARGET].mean())
print("test:", test_df[TARGET].mean())
print()

search_train = make_train_sample(train_df, MAX_SEARCH_TRAIN_ROWS)

print("Search train rows:", len(search_train))
print("Search train target rate:", search_train[TARGET].mean())
print()

metrics = []
trained_configs = []

for feature_set_name, feature_cols in FEATURE_SETS.items():
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        print("Skipping", feature_set_name, "missing:", missing)
        continue

    for c_value in C_VALUES:
        model_name = f"logreg_{feature_set_name}_C{c_value}"

        print("Training", model_name)

        pipe = build_pipeline(feature_cols, c_value)
        pipe.fit(search_train[feature_cols], search_train[TARGET])

        for split_name, part in [
            ("validation", validation_df),
            ("test", test_df),
        ]:
            proba = pipe.predict_proba(part[feature_cols])[:, 1]

            metrics.append(
                evaluate_predictions(
                    y_true=part[TARGET],
                    proba=proba,
                    model_name=model_name,
                    split_name=split_name,
                )
            )

        trained_configs.append(
            {
                "model_name": model_name,
                "feature_set_name": feature_set_name,
                "c_value": c_value,
                "feature_cols": feature_cols,
            }
        )

metrics_df = pd.DataFrame(metrics)
metrics_df = metrics_df.sort_values(["split", "pr_auc"], ascending=[True, False])
metrics_df.to_csv(METRICS_PATH, index=False)

print()
print("SEARCH METRICS")
print(metrics_df.to_string(index=False))
print()

validation_metrics = metrics_df[metrics_df["split"] == "validation"].copy()
best_validation_model_name = validation_metrics.sort_values(
    "pr_auc",
    ascending=False,
).iloc[0]["model"]

best_config = None
for config in trained_configs:
    if config["model_name"] == best_validation_model_name:
        best_config = config
        break

if best_config is None:
    raise RuntimeError("Best config not found")

print("Best by validation:")
print(best_config["model_name"])
print("Feature set:", best_config["feature_set_name"])
print("C:", best_config["c_value"])
print()

print("Training final model on train + validation...")

final_train_df = df[df["split"].isin(["train", "validation"])].copy()
final_test_df = test_df.copy()

final_feature_cols = best_config["feature_cols"]
final_pipe = build_pipeline(final_feature_cols, best_config["c_value"])
final_pipe.fit(final_train_df[final_feature_cols], final_train_df[TARGET])

test_proba = final_pipe.predict_proba(final_test_df[final_feature_cols])[:, 1]

final_metrics = pd.DataFrame(
    [
        evaluate_predictions(
            y_true=final_test_df[TARGET],
            proba=test_proba,
            model_name="logistic_v3_final_train_plus_validation",
            split_name="test",
        )
    ]
)

final_metrics.to_csv(FINAL_METRICS_PATH, index=False)

joblib.dump(
    {
        "model": final_pipe,
        "feature_cols": final_feature_cols,
        "target": TARGET,
        "best_search_model": best_config["model_name"],
        "feature_set_name": best_config["feature_set_name"],
        "c_value": best_config["c_value"],
    },
    FINAL_MODEL_PATH,
)

sample_size = min(100_000, len(final_test_df))
sample = final_test_df.sample(n=sample_size, random_state=RANDOM_STATE).copy()
sample_proba = pd.Series(test_proba, index=final_test_df.index)

pred_sample = sample[
    ["timestamp", "oblast", "raion", "split", TARGET]
].copy()
pred_sample["pred_logistic_v3_final"] = sample_proba.loc[sample.index].values
pred_sample.to_csv(PRED_SAMPLE_PATH, index=False)

print("Saved:")
print(METRICS_PATH)
print(FINAL_METRICS_PATH)
print(FINAL_MODEL_PATH)
print(PRED_SAMPLE_PATH)
print()

print("FINAL METRICS")
print(final_metrics.to_string(index=False))