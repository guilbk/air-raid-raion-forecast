from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Air Raid Alert Forecast", layout="wide")

DATA_DIR = Path("data/processed")

FILES = {
    "metrics": DATA_DIR / "final_model_metrics.csv",
    "comparison": DATA_DIR / "final_model_comparison.csv",
    "latest": DATA_DIR / "latest_risk_snapshot.csv",
    "oblast": DATA_DIR / "risk_by_oblast.csv",
    "hour": DATA_DIR / "risk_by_hour.csv",
}

@st.cache_data
def load_csv(path):
    if not path.exists():
        return None
    return pd.read_csv(path)

def show_missing(name, path):
    st.warning(f"Missing file: `{path}`. Run the output preparation script first.")

def pick_col(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None

def show_table(df, cols=None):
    if df is None:
        return
    if cols:
        cols = [c for c in cols if c in df.columns]
        st.dataframe(df[cols], use_container_width=True)
    else:
        st.dataframe(df, use_container_width=True)

metrics = load_csv(FILES["metrics"])
comparison = load_csv(FILES["comparison"])
latest = load_csv(FILES["latest"])
oblast = load_csv(FILES["oblast"])
hour = load_csv(FILES["hour"])

st.title("Ukraine Air Raid Alert Forecast Dashboard")
st.caption("Educational ML dashboard. Not an official alert system.")

tabs = st.tabs([
    "Overview",
    "Latest risks",
    "Oblast analysis",
    "Hourly pattern",
    "Model comparison",
    "About",
])

with tabs[0]:
    st.subheader("Overview")
    st.warning("This dashboard shows model risk scores, not official warnings and not exact calibrated probabilities.")

    if metrics is None:
        show_missing("metrics", FILES["metrics"])
    else:
        row = metrics.iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Model", str(row.get("model", "unknown")))
        c2.metric("Test PR AUC", f"{float(row.get('pr_auc', 0)):.4f}")
        c3.metric("Test ROC AUC", f"{float(row.get('roc_auc', 0)):.4f}")
        c4.metric("Brier score", f"{float(row.get('brier', 0)):.4f}")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Positive rate", f"{float(row.get('positive_rate', 0)):.4f}")
        c6.metric("Precision @ 0.5", f"{float(row.get('precision_at_0_5', 0)):.4f}")
        c7.metric("Recall @ 0.5", f"{float(row.get('recall_at_0_5', 0)):.4f}")
        c8.metric("F1 @ 0.5", f"{float(row.get('f1_at_0_5', 0)):.4f}")

        st.subheader("Final metrics")
        show_table(metrics)

with tabs[1]:
    st.subheader("Latest risks")

    if latest is None:
        show_missing("latest risks", FILES["latest"])
    else:
        risk_col = pick_col(latest, ["risk_score", "pred_logistic_v3", "predicted_risk", "prediction", "pred"])
        oblast_col = "oblast" if "oblast" in latest.columns else None

        if oblast_col:
            oblasts = ["All"] + sorted(latest[oblast_col].dropna().unique().tolist())
            selected_oblast = st.selectbox("Filter by oblast", oblasts)
            view = latest.copy()
            if selected_oblast != "All":
                view = view[view[oblast_col] == selected_oblast]
        else:
            view = latest.copy()

        if risk_col:
            view = view.sort_values(risk_col, ascending=False)
            top_n = st.slider("Top N raions", 5, 50, 15)
            show_table(view.head(top_n))
        else:
            st.warning("No risk score column found.")
            show_table(view)

        if risk_col and "raion" in view.columns:
            chart_df = view.head(20).copy()
            chart_df["label"] = chart_df.get("oblast", "").astype(str) + " / " + chart_df["raion"].astype(str)
            fig = px.bar(
                chart_df.sort_values(risk_col),
                x=risk_col,
                y="label",
                orientation="h",
                title="Top raions by latest risk score",
                labels={risk_col: "Risk score", "label": "Raion"},
            )
            st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    st.subheader("Oblast analysis")

    if oblast is None:
        show_missing("oblast analysis", FILES["oblast"])
    else:
        show_table(oblast)

        needed = {"oblast", "avg_predicted_risk", "actual_positive_rate"}
        if needed.issubset(oblast.columns):
            chart_df = oblast.sort_values("avg_predicted_risk", ascending=False)
            fig = px.bar(
                chart_df,
                x="oblast",
                y=["avg_predicted_risk", "actual_positive_rate"],
                barmode="group",
                title="Average predicted risk vs actual positive rate by oblast",
                labels={
                    "oblast": "Oblast",
                    "value": "Rate",
                    "variable": "Metric",
                },
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Expected columns not found: oblast, avg_predicted_risk, actual_positive_rate.")

with tabs[3]:
    st.subheader("Hourly pattern")

    if hour is None:
        show_missing("hourly pattern", FILES["hour"])
    else:
        show_table(hour)

        hour_col = "hour" if "hour" in hour.columns else None
        if hour_col and {"avg_predicted_risk", "actual_positive_rate"}.issubset(hour.columns):
            chart_df = hour.sort_values(hour_col)
            fig = px.line(
                chart_df,
                x=hour_col,
                y=["avg_predicted_risk", "actual_positive_rate"],
                markers=True,
                title="Average predicted risk vs actual positive rate by hour",
                labels={
                    hour_col: "Hour",
                    "value": "Rate",
                    "variable": "Metric",
                },
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Expected columns not found: hour, avg_predicted_risk, actual_positive_rate.")

with tabs[4]:
    st.subheader("Model comparison")

    if comparison is None:
        show_missing("model comparison", FILES["comparison"])
    else:
        view = comparison.copy()
        if "split" in view.columns:
            splits = ["All"] + sorted(view["split"].dropna().unique().tolist())
            selected_split = st.selectbox("Split", splits, index=splits.index("test") if "test" in splits else 0)
            if selected_split != "All":
                view = view[view["split"] == selected_split]

        cols = [
            "model",
            "split",
            "rows",
            "positive_rate",
            "pr_auc",
            "roc_auc",
            "brier",
            "precision_at_0_5",
            "recall_at_0_5",
            "f1_at_0_5",
            "precision_top_10_percent",
            "recall_top_10_percent",
        ]
        show_table(view, cols)

        if {"model", "pr_auc"}.issubset(view.columns):
            chart_df = view.sort_values("pr_auc", ascending=False)
            fig = px.bar(
                chart_df,
                x="model",
                y="pr_auc",
                title="Model comparison by PR AUC",
                labels={"model": "Model", "pr_auc": "PR AUC"},
            )
            st.plotly_chart(fig, use_container_width=True)

with tabs[5]:
    st.subheader("About")
    st.markdown(
        """
This project builds a raion-level proxy time-series dataset of air raid alerts in Ukraine.

The target is `target_start_60m`: whether a new alert will start in the same raion within the next 60 minutes.

The model output is a relative risk score. It should not be interpreted as an official warning or as an exact calibrated probability.

Main limitations:

- This is not an official alert system.
- This dashboard must not be used for safety decisions.
- Historical raion-level data is incomplete.
- Some historical alerts are inherited from oblast-level records.
- The dataset is a raion-level proxy dataset, not a fully direct raion-level historical archive.
- The split is chronological, not random.
- A 120-minute purge gap is used before split boundaries to reduce temporal leakage.
- The model predicts alert patterns, not attacks.
        """
    )