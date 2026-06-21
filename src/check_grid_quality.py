from pathlib import Path
import pandas as pd

input_path = Path("data/interim/raion_time_grid_15min.csv")
output_path = Path("data/interim/grid_quality_summary.csv")

required_columns = [
    "timestamp",
    "oblast",
    "raion",
    "is_alert_active",
    "alert_started_now",
    "alert_ended_now",
    "direct_active",
    "inherited_active",
    "source_level_now",
]

warnings = []

if not input_path.exists():
    raise FileNotFoundError(f"File not found: {input_path}")

df = pd.read_csv(input_path, usecols=required_columns)

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

for col in ["is_alert_active", "direct_active", "inherited_active"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

rows = len(df)
duplicate_mask = df.duplicated(subset=["timestamp", "oblast", "raion"], keep=False)
duplicate_rows = int(duplicate_mask.sum())
duplicate_groups = int(df.loc[duplicate_mask, ["timestamp", "oblast", "raion"]].drop_duplicates().shape[0])

if duplicate_rows > 0:
    warnings.append(f"Found duplicate rows by timestamp + oblast + raion: {duplicate_rows} rows, {duplicate_groups} groups")

expected_active = ((df["direct_active"] == 1) | (df["inherited_active"] == 1)).astype(int)
active_mismatch_mask = df["is_alert_active"] != expected_active
active_mismatch_rows = int(active_mismatch_mask.sum())

if active_mismatch_rows > 0:
    warnings.append(f"is_alert_active mismatch with direct_active OR inherited_active: {active_mismatch_rows} rows")

mixed_active_rows = int(((df["direct_active"] == 1) & (df["inherited_active"] == 1)).sum())

unique_oblasts = int(df["oblast"].nunique())
unique_raions = int(df["raion"].nunique())
unique_oblast_raion_pairs = int(df[["oblast", "raion"]].drop_duplicates().shape[0])

min_timestamp = df["timestamp"].min()
max_timestamp = df["timestamp"].max()
null_timestamps = int(df["timestamp"].isna().sum())

if null_timestamps > 0:
    warnings.append(f"Found invalid timestamps: {null_timestamps}")

raion_rows = df.groupby(["oblast", "raion"]).size()
raion_unique_timestamps = df.groupby(["oblast", "raion"])["timestamp"].nunique()

min_rows_per_raion = int(raion_rows.min())
max_rows_per_raion = int(raion_rows.max())
same_rows_per_raion = bool(min_rows_per_raion == max_rows_per_raion)

min_unique_timestamps_per_raion = int(raion_unique_timestamps.min())
max_unique_timestamps_per_raion = int(raion_unique_timestamps.max())
same_unique_timestamps_per_raion = bool(min_unique_timestamps_per_raion == max_unique_timestamps_per_raion)

if not same_rows_per_raion:
    warnings.append("Not every raion has the same number of rows")

if not same_unique_timestamps_per_raion:
    warnings.append("Not every raion has the same number of unique timestamps")

source_counts = df["source_level_now"].fillna("NaN").value_counts(dropna=False)

summary_rows = [
    ("rows", rows),
    ("duplicate_rows", duplicate_rows),
    ("duplicate_groups", duplicate_groups),
    ("active_mismatch_rows", active_mismatch_rows),
    ("mixed_direct_and_inherited_active_rows", mixed_active_rows),
    ("unique_oblasts", unique_oblasts),
    ("unique_raions", unique_raions),
    ("unique_oblast_raion_pairs", unique_oblast_raion_pairs),
    ("min_timestamp", min_timestamp),
    ("max_timestamp", max_timestamp),
    ("null_timestamps", null_timestamps),
    ("min_rows_per_raion", min_rows_per_raion),
    ("max_rows_per_raion", max_rows_per_raion),
    ("same_rows_per_raion", same_rows_per_raion),
    ("min_unique_timestamps_per_raion", min_unique_timestamps_per_raion),
    ("max_unique_timestamps_per_raion", max_unique_timestamps_per_raion),
    ("same_unique_timestamps_per_raion", same_unique_timestamps_per_raion),
]

for source_level, count in source_counts.items():
    summary_rows.append((f"source_level_now__{source_level}", int(count)))

summary = pd.DataFrame(summary_rows, columns=["metric", "value"])

output_path.parent.mkdir(parents=True, exist_ok=True)
summary.to_csv(output_path, index=False)

print("\nGRID QUALITY SUMMARY")
print(summary.to_string(index=False))

print("\nSOURCE LEVEL COUNTS")
print(source_counts.to_string())

if warnings:
    print("\nWARNINGS")
    for warning in warnings:
        print(f"- {warning}")
else:
    print("\nWARNINGS")
    print("- No major problems found")

print(f"\nSaved: {output_path}")