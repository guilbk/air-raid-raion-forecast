# air-raid-raion-forecast

Time series analysis of air raid alerts in Ukraine.

This project builds a raion-level proxy time series dataset from historical Ukrainian air raid alert records and trains baseline forecasting models for a 60-minute alert-start risk task.

The final artifact is a Streamlit dashboard that shows:

- regional and raion-level risk scores
- historical alert analytics
- model evaluation results
- raion profile pages
- a live-alert architecture prepared for alerts.in.ua API

This is a research and demonstration project. It is not an official alert system and must not be used for safety decisions.

---

## Task

The main modeling task is:

> Predict whether a new air raid alert will start in a raion within the next 60 minutes.

The target column is:

```text
target_start_60m
