import os
import urllib.request

URL = "https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset/main/datasets/official_data_uk.csv"
OUT_PATH = "data/raw/official_data_uk.csv"

os.makedirs("data/raw", exist_ok=True)

print("Downloading dataset...")
urllib.request.urlretrieve(URL, OUT_PATH)
print(f"Saved to {OUT_PATH}")
