import numpy as np
import pandas as pd
from pathlib import Path

IN_PATH = Path("data/interim/raion_time_grid_15min.csv")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = OUT_DIR / "model_targets_15min.csv"
SUMMARY_PATH = OUT_DIR / "target_summary.csv"

FREQ_MINUTES = 15
HORIZONS = [15, 30, 60, 120]
MAX_HORIZON = max(HORIZONS)
MAX_STEPS = MAX_HORIZON // FREQ_MINUTES

print("Reading time grid...")
df = pd.read_csv(IN_PATH)

df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

df = df.sort_values(["oblast", "raion", "timestamp"]).reset_index(drop=True)

for col in ["is_alert_active", "alert_started_now", "alert_ended_now"]:
    df[col] = df[col].astype("int8")

target_cols = [f"target_start_{h}m" for h in HORIZONS]

for col in target_cols:
    df[col] = 0

df["has_full_future_window"] = 0

parts = []

groups = df.groupby(["oblast", "raion"], sort=False)

for i, ((oblast, raion), g) in enumerate(groups, start=1):
    g = g.copy()
    starts = g["alert_started_now"].to_numpy(dtype=np.int8)
    n = len(g)

    cumsum = np.concatenate([[0], np.cumsum(starts)])
    idx = np.arange(n)

    for horizon in HORIZONS:
        steps = horizon // FREQ_MINUTES
        end_idx = np.minimum(idx + steps + 1, n)
        future_starts = cumsum[end_idx] - cumsum[idx + 1]
        g[f"target_start_{horizon}m"] = (future_starts > 0).astype("int8")

    g["has_full_future_window"] = ((idx + MAX_STEPS) < n).astype("int8")

    parts.append(g)

    if i % 20 == 0:
        print(f"Processed {i}/{len(groups)} raions")

df_out = pd.concat(parts, ignore_index=True)

df_out["train_eligible"] = (
    (df_out["is_alert_active"] == 0)
    & (df_out["alert_started_now"] == 0)
    & (df_out["has_full_future_window"] == 1)
).astype("int8")

df_out.to_csv(OUT_PATH, index=False)

rows = len(df_out)
train_rows = int(df_out["train_eligible"].sum())

summary_rows = [
    {"metric": "rows", "value": rows},
    {"metric": "train_eligible_rows", "value": train_rows},
    {"metric": "not_train_eligible_rows", "value": rows - train_rows},
    {"metric": "min_timestamp", "value": df_out["timestamp"].min()},
    {"metric": "max_timestamp", "value": df_out["timestamp"].max()},
    {"metric": "unique_oblasts", "value": df_out["oblast"].nunique()},
    {"metric": "unique_raions", "value": df_out["raion"].nunique()},
]

for horizon in HORIZONS:
    col = f"target_start_{horizon}m"

    total_pos = int(df_out[col].sum())
    train_pos = int(df_out.loc[df_out["train_eligible"] == 1, col].sum())

    total_rate = total_pos / rows if rows else 0
    train_rate = train_pos / train_rows if train_rows else 0

    summary_rows.extend(
        [
            {"metric": f"{col}_positive_rows_all", "value": total_pos},
            {"metric": f"{col}_positive_rate_all", "value": total_rate},
            {"metric": f"{col}_positive_rows_train", "value": train_pos},
            {"metric": f"{col}_positive_rate_train", "value": train_rate},
        ]
    )

summary = pd.DataFrame(summary_rows)
summary.to_csv(SUMMARY_PATH, index=False)

print("Saved:")
print(OUT_PATH)
print(SUMMARY_PATH)
print()
print(summary.to_string(index=False))