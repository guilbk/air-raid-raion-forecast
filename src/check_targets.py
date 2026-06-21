import pandas as pd

PATH = "data/processed/model_targets_15min.csv"
SUMMARY_PATH = "data/processed/target_summary.csv"

df = pd.read_csv(PATH)
summary = pd.read_csv(SUMMARY_PATH)

df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

print("TARGET SUMMARY")
print(summary.to_string(index=False))
print()

print("TARGET COUNTS ON TRAIN ELIGIBLE ROWS")
train = df[df["train_eligible"] == 1].copy()

for col in [
    "target_start_15m",
    "target_start_30m",
    "target_start_60m",
    "target_start_120m",
]:
    pos = int(train[col].sum())
    total = len(train)
    rate = pos / total if total else 0
    print(f"{col}: {pos} / {total} = {rate:.6f}")

print()
print("TRAIN ELIGIBLE VALUE COUNTS")
print(df["train_eligible"].value_counts())
print()

print("SANITY CHECK")
bad_active_train = ((df["train_eligible"] == 1) & (df["is_alert_active"] == 1)).sum()
print("train rows with active alert:", int(bad_active_train))
bad_start_now_train = ((df["train_eligible"] == 1) & (df["alert_started_now"] == 1)).sum()
print("train rows with alert_started_now:", int(bad_start_now_train))

bad_future = ((df["train_eligible"] == 1) & (df["has_full_future_window"] == 0)).sum()
print("train rows without full future window:", int(bad_future))
print()

print("EXAMPLE AROUND ONE POSITIVE TARGET")
example_rows = df[(df["train_eligible"] == 1) & (df["target_start_60m"] == 1)]

if len(example_rows) == 0:
    print("No positive examples found for target_start_60m")
else:
    ex = example_rows.iloc[0]
    oblast = ex["oblast"]
    raion = ex["raion"]
    ts = ex["timestamp"]

    one = df[(df["oblast"] == oblast) & (df["raion"] == raion)].copy()
    one = one.sort_values("timestamp").reset_index(drop=True)

    pos = one.index[one["timestamp"] == ts][0]
    start = max(pos - 4, 0)
    end = min(pos + 10, len(one))

    cols = [
        "timestamp",
        "oblast",
        "raion",
        "is_alert_active",
        "alert_started_now",
        "target_start_15m",
        "target_start_30m",
        "target_start_60m",
        "target_start_120m",
        "train_eligible",
    ]

    print(one.loc[start:end, cols].to_string(index=False))