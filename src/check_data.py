import pandas as pd

path = "data/raw/official_data_uk.csv"

df = pd.read_csv(path)

print(df.head())
print(df.shape)
print(df.columns)