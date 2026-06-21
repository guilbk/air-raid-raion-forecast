import numpy as np
import pandas as pd
from pathlib import Path

IN_PATH = Path("data/processed/model_targets_15min.csv")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = OUT_DIR / "model_features_v2_15min_purged.csv"
SUMMARY_PATH = OUT_DIR / "feature_v2_summary.csv"

FREQ_MINUTES = 15
CAP_MINUTES = 7 * 24 * 60
PURGE_MINUTES = 120

WINDOWS = {
    "1h": 4,
    "3h": 12,
    "24h": 96,
}

SPLIT_TRAIN_END = pd.Timestamp("2025-11-01 00:00:00", tz="UTC")
SPLIT_VALIDATION_END = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")

PURGE_BOUNDARIES = [
    SPLIT_TRAIN_END,
    SPLIT_VALIDATION_END,
]


def minutes_since_previous_event(event_values):
    event_values = np.asarray(event_values, dtype=np.int8)
    n = len(event_values)
    idx = np.arange(n, dtype=np.int32)

    last_event_idx = np.maximum.accumulate(
        np.where(event_values == 1, idx, -1)
    )

    previous_event_idx = np.concatenate(
        [np.array([-1], dtype=np.int32), last_event_idx[:-1]]
    )

    minutes = np.where(
        previous_event_idx >= 0,
        (idx - previous_event_idx) * FREQ_MINUTES,
        CAP_MINUTES,
    )

    minutes = np.minimum(minutes, CAP_MINUTES).astype(np.int16)
    return minutes


print("Reading targets...")
df = pd.read_csv(IN_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
df = df.sort_values(["oblast", "raion", "timestamp"]).reset_index(drop=True)

int_cols = [
    "is_alert_active",
    "alert_started_now",
    "alert_ended_now",
    "target_start_15m",
    "target_start_30m",
    "target_start_60m",
    "target_start_120m",
    "train_eligible",
    "has_full_future_window",
]

for col in int_cols:
    if col in df.columns:
        df[col] = df[col].astype("int8")

print("Building time features...")

df["hour"] = df["timestamp"].dt.hour.astype("int8")
df["day_of_week"] = df["timestamp"].dt.dayofweek.astype("int8")
df["month"] = df["timestamp"].dt.month.astype("int8")
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype("int8")
df["is_night"] = df["hour"].isin([0, 1, 2, 3, 4, 5]).astype("int8")

df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24).astype("float32")
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24).astype("float32")

df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7).astype("float32")
df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7).astype("float32")

df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12).astype("float32")
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12).astype("float32")

print("Building raion history features...")

parts = []
groups = df.groupby(["oblast", "raion"], sort=False)

for i, ((oblast, raion), g) in enumerate(groups, start=1):
    g = g.copy().reset_index(drop=True)

    starts = g["alert_started_now"].astype("int8")
    ended = g["alert_ended_now"].astype("int8")
    active = g["is_alert_active"].astype("int8")

    for name, steps in WINDOWS.items():
        g[f"starts_last_{name}"] = (
            starts.shift(1)
            .rolling(steps, min_periods=1)
            .sum()
            .fillna(0)
            .astype("int16")
        )

        g[f"ends_last_{name}"] = (
            ended.shift(1)
            .rolling(steps, min_periods=1)
            .sum()
            .fillna(0)
            .astype("int16")
        )

        g[f"active_minutes_last_{name}"] = (
            active.shift(1)
            .rolling(steps, min_periods=1)
            .sum()
            .fillna(0)
            .astype("int16")
            * FREQ_MINUTES
        )

    g["active_prev_15m"] = active.shift(1).fillna(0).astype("int8")
    g["active_prev_30m"] = active.shift(1).rolling(2, min_periods=1).max().fillna(0).astype("int8")
    g["active_prev_60m"] = active.shift(1).rolling(4, min_periods=1).max().fillna(0).astype("int8")

    g["minutes_since_last_start"] = minutes_since_previous_event(starts.to_numpy())
    g["minutes_since_last_end"] = minutes_since_previous_event(ended.to_numpy())

    g["row_number_in_raion"] = np.arange(len(g), dtype=np.int32)
    g["has_full_history_24h"] = (g["row_number_in_raion"] >= WINDOWS["24h"]).astype("int8")

    parts.append(g)

    if i % 20 == 0:
        print(f"Processed {i}/{len(groups)} raions")

df = pd.concat(parts, ignore_index=True)

print("Building oblast context features...")

oblast_raion_counts = (
    df[["oblast", "raion"]]
    .drop_duplicates()
    .groupby("oblast")["raion"]
    .size()
    .to_dict()
)

oblast_ts = (
    df.groupby(["oblast", "timestamp"], as_index=False)
    .agg(
        oblast_started_raions_now=("alert_started_now", "sum"),
        oblast_active_raions_now=("is_alert_active", "sum"),
    )
    .sort_values(["oblast", "timestamp"])
    .reset_index(drop=True)
)

oblast_parts = []

for oblast, g in oblast_ts.groupby("oblast", sort=False):
    g = g.copy().reset_index(drop=True)

    starts = g["oblast_started_raions_now"].astype("int16")

    for name, steps in WINDOWS.items():
        g[f"oblast_starts_last_{name}"] = (
            starts.shift(1)
            .rolling(steps, min_periods=1)
            .sum()
            .fillna(0)
            .astype("int16")
        )

    g["oblast_raions_total"] = int(oblast_raion_counts[oblast])
    g["oblast_active_share_now"] = (
        g["oblast_active_raions_now"] / g["oblast_raions_total"]
    ).astype("float32")

    oblast_parts.append(g)

oblast_ts = pd.concat(oblast_parts, ignore_index=True)

df = df.merge(
    oblast_ts[
        [
            "oblast",
            "timestamp",
            "oblast_active_raions_now",
            "oblast_active_share_now",
            "oblast_starts_last_1h",
            "oblast_starts_last_3h",
            "oblast_starts_last_24h",
        ]
    ],
    on=["oblast", "timestamp"],
    how="left",
)

print("Building country context features...")

country_ts = (
    df.groupby("timestamp", as_index=False)
    .agg(
        country_started_raions_now=("alert_started_now", "sum"),
        country_active_raions_now=("is_alert_active", "sum"),
    )
    .sort_values("timestamp")
    .reset_index(drop=True)
)

country_starts = country_ts["country_started_raions_now"].astype("int16")

for name, steps in WINDOWS.items():
    country_ts[f"country_starts_last_{name}"] = (
        country_starts.shift(1)
        .rolling(steps, min_periods=1)
        .sum()
        .fillna(0)
        .astype("int16")
    )

country_raions_total = df[["oblast", "raion"]].drop_duplicates().shape[0]
country_ts["country_active_share_now"] = (
    country_ts["country_active_raions_now"] / country_raions_total
).astype("float32")

df = df.merge(
    country_ts[
        [
            "timestamp",
            "country_active_raions_now",
            "country_active_share_now",
            "country_starts_last_1h",
            "country_starts_last_3h",
            "country_starts_last_24h",
        ]
    ],
    on="timestamp",
    how="left",
)

print("Applying split and purge gap...")

df["split"] = "test"
df.loc[df["timestamp"] < SPLIT_TRAIN_END, "split"] = "train"
df.loc[
    (df["timestamp"] >= SPLIT_TRAIN_END)
    & (df["timestamp"] < SPLIT_VALIDATION_END),
    "split",
] = "validation"

df["purged"] = 0

for boundary in PURGE_BOUNDARIES:
    purge_start = boundary - pd.Timedelta(minutes=PURGE_MINUTES)
    mask = (df["timestamp"] >= purge_start) & (df["timestamp"] < boundary)
    df.loc[mask, "purged"] = 1

df["model_eligible"] = (
    (df["train_eligible"] == 1)
    & (df["has_full_history_24h"] == 1)
    & (df["purged"] == 0)
).astype("int8")

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

keep_cols = [
    "timestamp",
    "split",
    "model_eligible",
    "target_start_15m",
    "target_start_30m",
    "target_start_60m",
    "target_start_120m",
] + feature_cols

out = df.loc[df["model_eligible"] == 1, keep_cols].copy()

for col in feature_cols:
    if out[col].isna().any():
        print(f"WARNING nulls in {col}: {out[col].isna().sum()}")

out.to_csv(OUT_PATH, index=False)

summary_rows = [
    {"metric": "input_rows", "value": len(df)},
    {"metric": "output_model_rows", "value": len(out)},
    {"metric": "purged_rows", "value": int(df["purged"].sum())},
    {"metric": "unique_oblasts", "value": out["oblast"].nunique()},
    {"metric": "unique_raions", "value": out["raion"].nunique()},
    {"metric": "min_timestamp", "value": out["timestamp"].min()},
    {"metric": "max_timestamp", "value": out["timestamp"].max()},
    {"metric": "feature_count", "value": len(feature_cols)},
]

for split_name, part in out.groupby("split"):
    summary_rows.append({"metric": f"{split_name}_rows", "value": len(part)})
    summary_rows.append({"metric": f"{split_name}_min_timestamp", "value": part["timestamp"].min()})
    summary_rows.append({"metric": f"{split_name}_max_timestamp", "value": part["timestamp"].max()})
    summary_rows.append({
        "metric": f"{split_name}_target_start_60m_rate",
        "value": part["target_start_60m"].mean(),
    })

summary = pd.DataFrame(summary_rows)
summary.to_csv(SUMMARY_PATH, index=False)

print("Saved:")
print(OUT_PATH)
print(SUMMARY_PATH)
print()
print(summary.to_string(index=False))