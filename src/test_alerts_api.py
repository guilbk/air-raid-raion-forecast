import json
import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("ALERTS_IN_UA_TOKEN")
URL = "https://api.alerts.in.ua/v1/alerts/active.json"

OUT_DIR = Path("data/live")
OUT_DIR.mkdir(parents=True, exist_ok=True)

if not TOKEN:
    raise RuntimeError("ALERTS_IN_UA_TOKEN not found in .env")

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/json",
    "User-Agent": "air-raid-raion-forecast-test/1.0",
}

print("Testing alerts.in.ua API...")
print("URL:", URL)
print("Token loaded:", bool(TOKEN))
print()

response = requests.get(URL, headers=headers, timeout=20)

print("Status code:", response.status_code)
print("Content type:", response.headers.get("content-type"))
print()

if response.status_code == 401:
    raise RuntimeError("401 Unauthorized. Token is wrong or not active.")

if response.status_code == 403:
    raise RuntimeError("403 Forbidden. API access denied for this token.")

if response.status_code == 429:
    raise RuntimeError("429 Too Many Requests. Wait and try again later.")

response.raise_for_status()

data = response.json()

json_path = OUT_DIR / "live_active_alerts_raw_test.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

if isinstance(data, dict) and "alerts" in data:
    alerts = data["alerts"]
elif isinstance(data, list):
    alerts = data
else:
    alerts = []

print("Raw JSON saved to:", json_path)
print("Alerts count:", len(alerts))
print()

if not alerts:
    print("No active alerts returned.")
    print("This can be normal if there are no active alerts now.")
    raise SystemExit

df = pd.json_normalize(alerts)

csv_path = OUT_DIR / "live_active_alerts_test.csv"
df.to_csv(csv_path, index=False)

print("CSV saved to:", csv_path)
print()
print("Columns:")
print(list(df.columns))
print()

if "alert_type" in df.columns:
    print("Alert types:")
    print(df["alert_type"].value_counts(dropna=False))
    print()

if "location_type" in df.columns:
    print("Location types:")
    print(df["location_type"].value_counts(dropna=False))
    print()

print("First rows:")
print(df.head(20).to_string(index=False))