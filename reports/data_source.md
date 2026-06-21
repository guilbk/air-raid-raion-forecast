# Data source

The project uses the public Ukrainian air raid sirens dataset:

https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset

The main file is:

https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset/main/datasets/official_data_uk.csv

The raw CSV file is not stored in this repository because it is larger than the GitHub web upload limit. Instead, it can be downloaded by running:

```bash
python src/download_data.py
