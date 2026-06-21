import numpy as np
import pandas as pd
from pathlib import Path

IN_PATH = Path("data/interim/raion_alert_intervals.csv")
OUT_DIR = Path("data/interim")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = OUT_DIR / "raion_time_grid_15min.csv"
SUMMARY_PATH = OUT_DIR / "time_grid_summary.csv"

START_DATE = "2025-01-01"
FREQ = "15min"

df = pd.read_csv(IN_PATH)

df["started_at"] = pd.to_datetime(df["started_at"], utc=True, errors="coerce")
df["finished_at"] = pd.to_datetime(df["finished_at"], utc=True, errors="coerce")

start_date = pd.Timestamp(START_DATE, tz="UTC")

df = df[df["finished_at"] >= start_date].copy()

min_time = max(df["started_at"].min().floor(FREQ), start_date)
max_time = df["finished_at"].max().ceil(FREQ)

timeline = pd.date_range(min_time, max_time, freq=FREQ, tz="UTC")

raions = (
    df[["oblast", "raion"]]
    .drop_duplicates()
    .sort_values(["oblast", "raion"])
    .reset_index(drop=True)
)

all_parts = []

for i, row in raions.iterrows():
    oblast = row["oblast"]
    raion = row["raion"]

    g = df[(df["oblast"] == oblast) & (df["raion"] == raion)].copy()

    active_diff = np.zeros(len(timeline) + 1, dtype=np.int16)
    direct_diff = np.zeros(len(timeline) + 1, dtype=np.int16)
    inherited_diff = np.zeros(len(timeline) + 1, dtype=np.int16)

    start_counts = np.zeros(len(timeline), dtype=np.int16)
    end_counts = np.zeros(len(timeline), dtype=np.int16)

    for _, alert in g.iterrows():
        s = alert["started_at"]
        e = alert["finished_at"]

        s_idx = np.searchsorted(timeline.values, s.to_datetime64(), side="left")
        e_idx = np.searchsorted(timeline.values, e.to_datetime64(), side="left")

        if s_idx < len(timeline) and e_idx > 0:
            s_idx = max(s_idx, 0)
            e_idx = min(e_idx, len(timeline))

            if s_idx < e_idx:
                active_diff[s_idx] += 1
                active_diff[e_idx] -= 1

                if alert["source_level"] in ["direct_raion", "hromada_to_raion_proxy"]:
                    direct_diff[s_idx] += 1
                    direct_diff[e_idx] -= 1

                if alert["source_level"] == "oblast_inherited":
                    inherited_diff[s_idx] += 1
                    inherited_diff[e_idx] -= 1

        start_bucket = s.floor(FREQ)
        end_bucket = e.floor(FREQ)

        start_pos = np.searchsorted(timeline.values, start_bucket.to_datetime64(), side="left")
        end_pos = np.searchsorted(timeline.values, end_bucket.to_datetime64(), side="left")

        if 0 <= start_pos < len(timeline):
            start_counts[start_pos] += 1

        if 0 <= end_pos < len(timeline):
            end_counts[end_pos] += 1

    active_count = np.cumsum(active_diff[:-1])
    direct_count = np.cumsum(direct_diff[:-1])
    inherited_count = np.cumsum(inherited_diff[:-1])

    part = pd.DataFrame(
        {
            "timestamp": timeline,
            "oblast": oblast,
            "raion": raion,
            "is_alert_active": (active_count > 0).astype(np.int8),
            "alert_started_now": (start_counts > 0).astype(np.int8),
            "alert_ended_now": (end_counts > 0).astype(np.int8),
            "direct_active": (direct_count > 0).astype(np.int8),
            "inherited_active": (inherited_count > 0).astype(np.int8),
        }
    )

    all_parts.append(part)

    if (i + 1) % 20 == 0:
        print(f"Processed {i + 1}/{len(raions)} raions")

grid = pd.concat(all_parts, ignore_index=True)

grid["source_level_now"] = "none"

grid.loc[
    (grid["direct_active"] == 1) & (grid["inherited_active"] == 0),
    "source_level_now"
] = "direct_or_hromada"

grid.loc[
    (grid["direct_active"] == 0) & (grid["inherited_active"] == 1),
    "source_level_now"
] = "oblast_inherited"

grid.loc[
    (grid["direct_active"] == 1) & (grid["inherited_active"] == 1),
    "source_level_now"
] = "mixed"

grid.to_csv(OUT_PATH, index=False)

summary = pd.DataFrame(
    {
        "metric": [
            "start_date",
            "min_timestamp",
            "max_timestamp",
            "freq",
            "raions",
            "rows",
            "active_rows",
            "started_rows",
            "ended_rows",
        ],
        "value": [
            START_DATE,
            grid["timestamp"].min(),
            grid["timestamp"].max(),
            FREQ,
            grid[["oblast", "raion"]].drop_duplicates().shape[0],
            len(grid),
            int(grid["is_alert_active"].sum()),
            int(grid["alert_started_now"].sum()),
            int(grid["alert_ended_now"].sum()),
        ],
    }
)

summary.to_csv(SUMMARY_PATH, index=False)

print("Saved:")
print(OUT_PATH)
print(SUMMARY_PATH)
print()
print(summary)
print()
print(grid.head())