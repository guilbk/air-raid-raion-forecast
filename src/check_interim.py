import pandas as pd

df = pd.read_csv("data/interim/raion_alert_intervals.csv")
summary = pd.read_csv("data/interim/data_quality_summary.csv")

print(summary)
print()
print(df.head())
print()
print(df["source_level"].value_counts())
print()
print("unique oblast-raion pairs:", df[["oblast", "raion"]].drop_duplicates().shape[0])
print("rows:", len(df))