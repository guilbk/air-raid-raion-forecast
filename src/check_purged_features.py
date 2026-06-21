import pandas as pd

PATH = "data/processed/model_features_15min_purged.csv"
SUMMARY_PATH = "data/processed/purge_summary.csv"

df = pd.read_csv(PATH)
summary = pd.read_csv(SUMMARY_PATH)

df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

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

print("PURGE SUMMARY")
print(summary.to_string(index=False))
print()

print("SHAPE")
print(df.shape)
print()

print("SPLIT COUNTS")
print(df["split"].value_counts())
print()

print("SPLIT DATE RANGES")
print(df.groupby("split")["timestamp"].agg(["min", "max"]))
print()

print("TARGET RATE BY SPLIT")
print(df.groupby("split")["target_start_60m"].mean())
print()

print("NULLS IN FEATURES")
print(df[feature_cols].isna().sum())
print()

print("ROWS IN PURGE WINDOWS SHOULD BE ZERO")

checks = [
    ("before_validation", "2025-10-31 22:00:00+00:00", "2025-11-01 00:00:00+00:00"),
    ("before_test", "2025-12-31 22:00:00+00:00", "2026-01-01 00:00:00+00:00"),
]

for name, start, end in checks:
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    rows = df[(df["timestamp"] >= start) & (df["timestamp"] < end)]
    print(name, len(rows))

print()

bad_split_order = (
    df[df["split"] == "train"]["timestamp"].max()
    >= df[df["split"] == "validation"]["timestamp"].min()
)

bad_validation_order = (
    df[df["split"] == "validation"]["timestamp"].max()
    >= df[df["split"] == "test"]["timestamp"].min()
)

print("train overlaps validation:", bool(bad_split_order))
print("validation overlaps test:", bool(bad_validation_order))