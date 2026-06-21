import numpy as np
import pandas as pd
from pathlib import Path

IN_PATH = Path("data/processed/model_targets_15min.csv")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = OUT_DIR / "model_features_15min.csv"
SUMMARY_PATH = OUT_DIR / "feature_summary.csv"

FREQ_MINUTES = 15
WINDOWS = {
    "1h": 4,
    "3h": 12,
    "24h": 96,
}

print("Reading targets...")
df = pd.read_csv(IN_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
df = df.sort_values(["oblast", "raion", "timestamp"]).reset_index(drop=True)

df["hour"] = df["timestamp"].dt.hour.astype("int8")
df["day_of_week"] = df["timestamp"].dt.dayofweek.astype("int8")
df["month"] = df["timestamp"].dt.month.astype("int8")
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype("int8")
df["is_night"] = df["hour"].isin([0, 1, 2, 3, 4, 5]).astype("int8")

parts = []
groups = df.groupby(["oblast", "raion"], sort=False)

for i, ((oblast, raion), g) in enumerate(groups, start=1):
    g = g.copy().reset_index(drop=True)

    starts = g["alert_started_now"].astype("int8")
    active = g["is_alert_active"].astype("int8")

    for name, steps in WINDOWS.items():
        g[f"starts_last_{name}"] = (
            starts.shift(1).rolling(steps, min_periods=1).sum().fillna(0).astype("int16")
        )
        g[f"active_minutes_last_{name}"] = (
            active.shift(1).rolling(steps, min_periods=1).sum().fillna(0).astype("int16") * FREQ_MINUTES
        )

    g["row_number_in_raion"] = np.arange(len(g), dtype=np.int32)
    g["has_full_history_24h"] = (g["row_number_in_raion"] >= WINDOWS["24h"]).astype("int8")

    parts.append(g)

    if i % 20 == 0:
        print(f"Processed {i}/{len(groups)} raions")

df = pd.concat(parts, ignore_index=True)

oblast_raion_counts = (
    df[["oblast", "raion"]]
    .drop_duplicates()
    .groupby("oblast")["raion"]
    .size()
    .to_dict()
)

df["oblast_raions_total"] = df["oblast"].map(oblast_raion_counts).astype("int16")
df["oblast_active_raions_now"] = (
    df.groupby(["timestamp", "oblast"])["is_alert_active"]
    .transform("sum")
    .astype("int16")
)
df["oblast_active_share_now"] = (
    df["oblast_active_raions_now"] / df["oblast_raions_total"]
).astype("float32")

df["split"] = "test"
df.loc[df["timestamp"] < pd.Timestamp("2025-11-01", tz="UTC"), "split"] = "train"
df.loc[
    (df["timestamp"] >= pd.Timestamp("2025-11-01", tz="UTC"))
    & (df["timestamp"] < pd.Timestamp("2026-01-01", tz="UTC")),
    "split",
] = "validation"

df["model_eligible"] = (
    (df["train_eligible"] == 1)
    & (df["has_full_history_24h"] == 1)
).astype("int8")

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

keep_cols = [
    "timestamp",
    "oblast",
    "raion",
    "split",
    "model_eligible",
    "is_alert_active",
    "target_start_15m",
    "target_start_30m",
    "target_start_60m",
    "target_start_120m",
] + feature_cols

out = df.loc[df["model_eligible"] == 1, keep_cols].copy()
out.to_csv(OUT_PATH, index=False)

summary_rows = [
    {"metric": "input_rows", "value": len(df)},
    {"metric": "output_model_rows", "value": len(out)},
    {"metric": "unique_oblasts", "value": out["oblast"].nunique()},
    {"metric": "unique_raions", "value": out["raion"].nunique()},
    {"metric": "min_timestamp", "value": out["timestamp"].min()},
    {"metric": "max_timestamp", "value": out["timestamp"].max()},
]

for split_name, part in out.groupby("split"):
    summary_rows.append({"metric": f"{split_name}_rows", "value": len(part)})
    summary_rows.append({
        "metric": f"{split_name}_target_start_60m_rate",
        "value": part["target_start_60m"].mean()
    })

summary = pd.DataFrame(summary_rows)
summary.to_csv(SUMMARY_PATH, index=False)

print("Saved:")
print(OUT_PATH)
print(SUMMARY_PATH)
print()
print(summary.to_string(index=False))