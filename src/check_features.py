import pandas as pd

PATH = "data/processed/model_features_15min.csv"
SUMMARY_PATH = "data/processed/feature_summary.csv"

df = pd.read_csv(PATH)
summary = pd.read_csv(SUMMARY_PATH)

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

print("FEATURE SUMMARY")
print(summary.to_string(index=False))
print()

print("SHAPE")
print(df.shape)
print()

print("SPLIT COUNTS")
print(df["split"].value_counts())
print()

print("TARGET RATE BY SPLIT")
print(df.groupby("split")["target_start_60m"].mean())
print()

print("NULLS IN FEATURES")
print(df[feature_cols].isna().sum())
print()

print("BASIC SAMPLE")
print(df.head(10).to_string(index=False))