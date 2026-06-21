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

try:
    from catboost import CatBoostClassifier
except ImportError:
    raise ImportError("CatBoost is not installed. Run: pip install catboost")


IN_PATH = Path("data/processed/model_features_v2_15min_purged.csv")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS_PATH = OUT_DIR / "model_v2_metrics.csv"
PRED_SAMPLE_PATH = OUT_DIR / "model_v2_predictions_sample.csv"
CATBOOST_MODEL_PATH = OUT_DIR / "catboost_v2_target_start_60m.cbm"
FEATURE_IMPORTANCE_PATH = OUT_DIR / "catboost_v2_feature_importance.csv"
BLEND_PATH = OUT_DIR / "model_v2_blend_weights.csv"

TARGET = "target_start_60m"
RANDOM_STATE = 42
MAX_LR_TRAIN_ROWS = 900_000

feature_cols = [
    "oblast",
    "raion",
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
    "starts_last_1h",
    "starts_last_3h",
    "starts_last_24h",
    "ends_last_1h",
    "ends_last_3h",
    "ends_last_24h",
    "active_minutes_last_1h",
    "active_minutes_last_3h",
    "active_minutes_last_24h",
    "active_prev_15m",
    "active_prev_30m",
    "active_prev_60m",
    "minutes_since_last_start",
    "minutes_since_last_end",
    "oblast_active_raions_now",
    "oblast_active_share_now",
    "oblast_starts_last_1h",
    "oblast_starts_last_3h",
    "oblast_starts_last_24h",
    "country_active_raions_now",
    "country_active_share_now",
    "country_starts_last_1h",
    "country_starts_last_3h",
    "country_starts_last_24h",
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

numeric_features = [col for col in feature_cols if col not in cat_features]


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


def historical_frequency_predictions(train_df, eval_df):
    global_rate = train_df[TARGET].mean()
    alpha = 40

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


def find_best_blend(y_true, pred_cat, pred_lr, pred_hist):
    best = {
        "w_catboost": 0.0,
        "w_logistic": 0.0,
        "w_historical": 1.0,
        "validation_pr_auc": -1.0,
    }

    grid = np.arange(0, 1.0001, 0.05)

    for w_cat in grid:
        for w_lr in grid:
            if w_cat + w_lr > 1:
                continue

            w_hist = 1 - w_cat - w_lr
            blend = w_cat * pred_cat + w_lr * pred_lr + w_hist * pred_hist
            score = average_precision_score(y_true, blend)

            if score > best["validation_pr_auc"]:
                best = {
                    "w_catboost": float(w_cat),
                    "w_logistic": float(w_lr),
                    "w_historical": float(w_hist),
                    "validation_pr_auc": float(score),
                }

    return best


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

for col in cat_features:
    train_df[col] = train_df[col].astype(str)
    validation_df[col] = validation_df[col].astype(str)
    test_df[col] = test_df[col].astype(str)

print("Training logistic regression v2...")

lr_train = make_lr_train_sample(train_df)

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", one_hot, cat_features),
        ("num", StandardScaler(with_mean=False), numeric_features),
    ]
)

log_reg = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(
                max_iter=250,
                solver="saga",
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)

log_reg.fit(lr_train[feature_cols], lr_train[TARGET])

print("Logistic regression trained")
print()

cat_feature_indices = [feature_cols.index(col) for col in cat_features]

model = CatBoostClassifier(
    loss_function="Logloss",
    eval_metric="PRAUC",
    iterations=1500,
    learning_rate=0.035,
    depth=7,
    l2_leaf_reg=8,
    random_seed=RANDOM_STATE,
    verbose=100,
    allow_writing_files=False,
    od_type="Iter",
    od_wait=120,
    thread_count=-1,
)

print("Training CatBoost v2 on full train...")
model.fit(
    train_df[feature_cols],
    train_df[TARGET],
    cat_features=cat_feature_indices,
    eval_set=(validation_df[feature_cols], validation_df[TARGET]),
    use_best_model=True,
)

model.save_model(CATBOOST_MODEL_PATH)

print("Predicting validation and test...")

preds = {}

for split_name, part in [
    ("validation", validation_df),
    ("test", test_df),
]:
    preds[(split_name, "historical_raion_hour_dow")] = historical_frequency_predictions(train_df, part)
    preds[(split_name, "logistic_regression_v2")] = log_reg.predict_proba(part[feature_cols])[:, 1]
    preds[(split_name, "catboost_v2")] = model.predict_proba(part[feature_cols])[:, 1]

print("Finding best blend on validation...")

y_val = validation_df[TARGET].to_numpy()

best_blend = find_best_blend(
    y_true=y_val,
    pred_cat=preds[("validation", "catboost_v2")],
    pred_lr=preds[("validation", "logistic_regression_v2")],
    pred_hist=preds[("validation", "historical_raion_hour_dow")],
)

blend_df = pd.DataFrame([best_blend])
blend_df.to_csv(BLEND_PATH, index=False)

print("Best validation blend:")
print(blend_df.to_string(index=False))
print()

metrics = []

for split_name, part in [
    ("validation", validation_df),
    ("test", test_df),
]:
    y_true = part[TARGET].to_numpy()

    pred_hist = preds[(split_name, "historical_raion_hour_dow")]
    pred_lr = preds[(split_name, "logistic_regression_v2")]
    pred_cat = preds[(split_name, "catboost_v2")]

    pred_blend = (
        best_blend["w_catboost"] * pred_cat
        + best_blend["w_logistic"] * pred_lr
        + best_blend["w_historical"] * pred_hist
    )

    split_predictions = {
        "historical_raion_hour_dow": pred_hist,
        "logistic_regression_v2": pred_lr,
        "catboost_v2": pred_cat,
        "blend_v2": pred_blend,
    }

    for model_name, proba in split_predictions.items():
        metrics.append(
            evaluate_predictions(
                y_true=y_true,
                proba=proba,
                model_name=model_name,
                split_name=split_name,
            )
        )

metrics_df = pd.DataFrame(metrics)
metrics_df = metrics_df.sort_values(["split", "pr_auc"], ascending=[True, False])
metrics_df.to_csv(METRICS_PATH, index=False)

feature_importance = pd.DataFrame(
    {
        "feature": feature_cols,
        "importance": model.get_feature_importance(),
    }
).sort_values("importance", ascending=False)

feature_importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

prediction_samples = []

for split_name, part in [
    ("validation", validation_df),
    ("test", test_df),
]:
    sample_size = min(50_000, len(part))
    sample = part.sample(n=sample_size, random_state=RANDOM_STATE).copy()

    sample_out = sample[
        ["timestamp", "oblast", "raion", "split", TARGET]
    ].copy()

    for model_name in [
        "historical_raion_hour_dow",
        "logistic_regression_v2",
        "catboost_v2",
    ]:
        proba = pd.Series(preds[(split_name, model_name)], index=part.index)
        sample_out[f"pred_{model_name}"] = proba.loc[sample.index].values

    pred_hist = pd.Series(preds[(split_name, "historical_raion_hour_dow")], index=part.index)
    pred_lr = pd.Series(preds[(split_name, "logistic_regression_v2")], index=part.index)
    pred_cat = pd.Series(preds[(split_name, "catboost_v2")], index=part.index)

    sample_out["pred_blend_v2"] = (
        best_blend["w_catboost"] * pred_cat.loc[sample.index].values
        + best_blend["w_logistic"] * pred_lr.loc[sample.index].values
        + best_blend["w_historical"] * pred_hist.loc[sample.index].values
    )

    prediction_samples.append(sample_out)

pred_sample_df = pd.concat(prediction_samples, ignore_index=True)
pred_sample_df.to_csv(PRED_SAMPLE_PATH, index=False)

print("Saved:")
print(METRICS_PATH)
print(PRED_SAMPLE_PATH)
print(CATBOOST_MODEL_PATH)
print(FEATURE_IMPORTANCE_PATH)
print(BLEND_PATH)
print()

print("METRICS")
print(metrics_df.to_string(index=False))
print()

print("FEATURE IMPORTANCE")
print(feature_importance.to_string(index=False))