import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/official_data_uk.csv")
OUT_DIR = Path("data/interim")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df_raw = pd.read_csv(RAW_PATH)

df = df_raw.copy()

df["started_at"] = pd.to_datetime(df["started_at"], utc=True, errors="coerce")
df["finished_at"] = pd.to_datetime(df["finished_at"], utc=True, errors="coerce")

df["oblast"] = df["oblast"].astype(str).str.strip()
df["raion"] = df["raion"].astype("string").str.strip()
df["hromada"] = df["hromada"].astype("string").str.strip()
df["level"] = df["level"].astype(str).str.strip()

df = df.dropna(subset=["oblast", "started_at", "finished_at"])
df = df[df["finished_at"] > df["started_at"]]

df["duration_min"] = (df["finished_at"] - df["started_at"]).dt.total_seconds() / 60
df = df[(df["duration_min"] >= 1) & (df["duration_min"] <= 24 * 60)]

raions = (
    df[df["raion"].notna()][["oblast", "raion"]]
    .drop_duplicates()
    .sort_values(["oblast", "raion"])
    .reset_index(drop=True)
)

raions.to_csv(OUT_DIR / "raions_from_alerts.csv", index=False)

direct_raion = df[df["raion"].notna()].copy()

direct_raion["source_level"] = direct_raion["level"].apply(
    lambda x: "direct_raion" if x == "raion" else "hromada_to_raion_proxy"
)

oblast_level = df[df["raion"].isna()].copy()

oblast_level = oblast_level.drop(columns=["raion"])

oblast_level = oblast_level.merge(
    raions,
    on="oblast",
    how="left"
)

oblast_level = oblast_level[oblast_level["raion"].notna()].copy()
oblast_level["source_level"] = "oblast_inherited"

keep_cols = [
    "oblast",
    "raion",
    "hromada",
    "level",
    "started_at",
    "finished_at",
    "duration_min",
    "source",
    "source_level",
]

raion_intervals = pd.concat(
    [
        direct_raion[keep_cols],
        oblast_level[keep_cols],
    ],
    ignore_index=True,
)

raion_intervals = raion_intervals.sort_values(
    ["started_at", "oblast", "raion"]
).reset_index(drop=True)

raion_intervals.to_csv(OUT_DIR / "raion_alert_intervals.csv", index=False)

summary = pd.DataFrame(
    {
        "metric": [
            "raw_rows",
            "clean_rows",
            "unique_oblasts",
            "unique_raions_from_alerts",
            "final_raion_interval_rows",
            "min_started_at",
            "max_started_at",
            "direct_raion_rows",
            "hromada_to_raion_proxy_rows",
            "oblast_inherited_rows",
        ],
        "value": [
            len(df_raw),
            len(df),
            df["oblast"].nunique(),
            raions["raion"].nunique(),
            len(raion_intervals),
            df["started_at"].min(),
            df["started_at"].max(),
            (raion_intervals["source_level"] == "direct_raion").sum(),
            (raion_intervals["source_level"] == "hromada_to_raion_proxy").sum(),
            (raion_intervals["source_level"] == "oblast_inherited").sum(),
        ],
    }
)

summary.to_csv(OUT_DIR / "data_quality_summary.csv", index=False)

print("Saved:")
print(OUT_DIR / "raions_from_alerts.csv")
print(OUT_DIR / "raion_alert_intervals.csv")
print(OUT_DIR / "data_quality_summary.csv")
print()
print(summary)