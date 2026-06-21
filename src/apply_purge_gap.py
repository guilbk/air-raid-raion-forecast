import pandas as pd
from pathlib import Path

IN_PATH = Path("data/processed/model_features_15min.csv")
OUT_PATH = Path("data/processed/model_features_15min_purged.csv")
SUMMARY_PATH = Path("data/processed/purge_summary.csv")

PURGE_MINUTES = 120

boundaries = [
    pd.Timestamp("2025-11-01 00:00:00", tz="UTC"),
    pd.Timestamp("2026-01-01 00:00:00", tz="UTC"),
]

df = pd.read_csv(IN_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

before_rows = len(df)

df["purged"] = 0

for boundary in boundaries:
    start = boundary - pd.Timedelta(minutes=PURGE_MINUTES)
    mask = (df["timestamp"] >= start) & (df["timestamp"] < boundary)
    df.loc[mask, "purged"] = 1

purged_rows = int(df["purged"].sum())

out = df[df["purged"] == 0].drop(columns=["purged"]).copy()
out.to_csv(OUT_PATH, index=False)

summary_rows = [
    {"metric": "input_rows", "value": before_rows},
    {"metric": "purged_rows", "value": purged_rows},
    {"metric": "output_rows", "value": len(out)},
    {"metric": "purge_minutes", "value": PURGE_MINUTES},
    {"metric": "min_timestamp", "value": out["timestamp"].min()},
    {"metric": "max_timestamp", "value": out["timestamp"].max()},
]

for split, part in out.groupby("split"):
    summary_rows.append({"metric": f"{split}_rows", "value": len(part)})
    summary_rows.append({"metric": f"{split}_min_timestamp", "value": part["timestamp"].min()})
    summary_rows.append({"metric": f"{split}_max_timestamp", "value": part["timestamp"].max()})
    summary_rows.append({"metric": f"{split}_target_start_60m_rate", "value": part["target_start_60m"].mean()})

summary = pd.DataFrame(summary_rows)
summary.to_csv(SUMMARY_PATH, index=False)

print("Saved:")
print(OUT_PATH)
print(SUMMARY_PATH)
print()
print(summary.to_string(index=False))