import os
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    import requests
except ImportError:
    requests = None


APP_TITLE = "Air Raid Raion Forecast"
TARGET_NAME = "target_start_60m"

DATA_PATHS = {
    "metrics": Path("data/processed/final_model_metrics.csv"),
    "comparison": Path("data/processed/final_model_comparison.csv"),
    "latest_risk": Path("data/processed/latest_risk_snapshot.csv"),
    "risk_by_oblast": Path("data/processed/risk_by_oblast.csv"),
    "risk_by_hour": Path("data/processed/risk_by_hour.csv"),
    "prediction_sample": Path("data/processed/logistic_v3_test_predictions_sample.csv"),
    "intervals": Path("data/interim/raion_alert_intervals.csv"),
}


OBLAST_COORDS = {
    "Вінницька область": (49.2328, 28.4810),
    "Волинська область": (50.7472, 25.3254),
    "Дніпропетровська область": (48.4647, 35.0462),
    "Донецька область": (48.0159, 37.8029),
    "Житомирська область": (50.2547, 28.6587),
    "Закарпатська область": (48.6208, 22.2879),
    "Запорізька область": (47.8388, 35.1396),
    "Івано-Франківська область": (48.9226, 24.7111),
    "Київська область": (50.4501, 30.5234),
    "Кіровоградська область": (48.5079, 32.2623),
    "Луганська область": (48.5740, 39.3078),
    "Львівська область": (49.8397, 24.0297),
    "Миколаївська область": (46.9750, 31.9946),
    "Одеська область": (46.4825, 30.7233),
    "Полтавська область": (49.5883, 34.5514),
    "Рівненська область": (50.6199, 26.2516),
    "Сумська область": (50.9077, 34.7981),
    "Тернопільська область": (49.5535, 25.5948),
    "Харківська область": (49.9935, 36.2304),
    "Херсонська область": (46.6354, 32.6169),
    "Хмельницька область": (49.4229, 26.9871),
    "Черкаська область": (49.4444, 32.0598),
    "Чернівецька область": (48.2915, 25.9403),
    "Чернігівська область": (51.4982, 31.2893),
    "м. Київ": (50.4501, 30.5234),
}


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>
    .stApp {
        background: radial-gradient(circle at top, #172033 0%, #0b0f17 45%, #06080d 100%);
        color: #e8edf7;
    }

    [data-testid="stSidebar"] {
        background: #080d14;
        border-right: 1px solid #1f2b3d;
    }

    .main-card {
        background: rgba(13, 20, 32, 0.92);
        border: 1px solid rgba(92, 124, 173, 0.25);
        border-radius: 18px;
        padding: 18px;
        margin-bottom: 14px;
        box-shadow: 0 0 22px rgba(0, 0, 0, 0.24);
    }

    .small-note {
        color: #9fb0c9;
        font-size: 0.92rem;
        line-height: 1.45;
    }

    .warning-box {
        background: rgba(245, 158, 11, 0.10);
        border: 1px solid rgba(245, 158, 11, 0.35);
        border-radius: 14px;
        padding: 14px;
        color: #ffe6b3;
    }

    .risk-low {
        color: #22c55e;
        font-weight: 700;
    }

    .risk-medium {
        color: #eab308;
        font-weight: 700;
    }

    .risk-high {
        color: #f97316;
        font-weight: 700;
    }

    .risk-very-high {
        color: #ef4444;
        font-weight: 700;
    }

    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.92);
        border: 1px solid rgba(148, 163, 184, 0.18);
        padding: 14px;
        border-radius: 16px;
    }
</style>
""",
    unsafe_allow_html=True,
)


def safe_read_csv(path: Path, name: str) -> Tuple[pd.DataFrame, Optional[str]]:
    if not path.exists():
        return pd.DataFrame(), f"Файл не знайдено: `{path}`"
    try:
        return pd.read_csv(path), None
    except Exception as exc:
        return pd.DataFrame(), f"Не вдалося прочитати `{path}`: {exc}"


@st.cache_data(show_spinner=False)
def load_all_data():
    data = {}
    errors = []
    for key, path in DATA_PATHS.items():
        df, error = safe_read_csv(path, key)
        data[key] = df
        if error:
            errors.append(error)
    return data, errors


def find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def get_risk_col(df: pd.DataFrame) -> Optional[str]:
    return find_col(
        df,
        [
            "risk_score",
            "pred_logistic_v3",
            "pred_logistic",
            "predicted_risk",
            "avg_predicted_risk",
            "prediction",
            "score",
        ],
    )


def risk_level_from_score(score: float) -> str:
    if pd.isna(score):
        return "unknown"
    if score < 0.25:
        return "low"
    if score < 0.50:
        return "medium"
    if score < 0.75:
        return "high"
    return "very_high"


def risk_level_ua(level: str) -> str:
    mapping = {
        "low": "низький",
        "medium": "середній",
        "high": "високий",
        "very_high": "дуже високий",
        "unknown": "невідомо",
    }
    return mapping.get(str(level), str(level))


def risk_html(level: str, text: Optional[str] = None) -> str:
    level = str(level)
    label = text or risk_level_ua(level)
    css = {
        "low": "risk-low",
        "medium": "risk-medium",
        "high": "risk-high",
        "very_high": "risk-very-high",
    }.get(level, "")
    return f'<span class="{css}">{label}</span>'


def prepare_latest_risk(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    risk_col = get_risk_col(df)

    if risk_col and risk_col != "risk_score":
        df["risk_score"] = pd.to_numeric(df[risk_col], errors="coerce")
    elif "risk_score" in df.columns:
        df["risk_score"] = pd.to_numeric(df["risk_score"], errors="coerce")
    else:
        df["risk_score"] = pd.NA

    if "risk_level" not in df.columns:
        df["risk_level"] = df["risk_score"].apply(risk_level_from_score)

    if "risk_rank" not in df.columns:
        df["risk_rank"] = df["risk_score"].rank(ascending=False, method="first").astype("Int64")

    if "risk_percentile" not in df.columns:
        df["risk_percentile"] = df["risk_score"].rank(pct=True).fillna(0)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

    return df


def prepare_risk_by_oblast(df: pd.DataFrame, latest_df: pd.DataFrame) -> pd.DataFrame:
    if not df.empty:
        out = df.copy()
    elif not latest_df.empty:
        out = (
            latest_df.groupby("oblast", as_index=False)
            .agg(
                rows=("raion", "count"),
                avg_predicted_risk=("risk_score", "mean"),
                max_predicted_risk=("risk_score", "max"),
            )
        )
        if TARGET_NAME in latest_df.columns:
            out["actual_positive_rate"] = latest_df.groupby("oblast")[TARGET_NAME].mean().values
    else:
        return pd.DataFrame()

    if "avg_predicted_risk" not in out.columns:
        risk_col = get_risk_col(out)
        if risk_col:
            out["avg_predicted_risk"] = pd.to_numeric(out[risk_col], errors="coerce")

    if "actual_positive_rate" not in out.columns:
        out["actual_positive_rate"] = pd.NA

    out["lat"] = out["oblast"].map(lambda x: OBLAST_COORDS.get(x, (None, None))[0])
    out["lon"] = out["oblast"].map(lambda x: OBLAST_COORDS.get(x, (None, None))[1])
    return out


def prepare_risk_by_hour(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()

    if "hour" not in out.columns:
        possible_hour_col = find_col(out, ["timestamp_hour", "time_hour"])
        if possible_hour_col:
            out["hour"] = out[possible_hour_col]

    if "avg_predicted_risk" not in out.columns:
        risk_col = get_risk_col(out)
        if risk_col:
            out["avg_predicted_risk"] = pd.to_numeric(out[risk_col], errors="coerce")

    if "actual_positive_rate" not in out.columns:
        actual_col = find_col(out, ["positive_rate", "target_rate", "actual_rate"])
        if actual_col:
            out["actual_positive_rate"] = pd.to_numeric(out[actual_col], errors="coerce")

    return out


def load_live_alerts() -> Tuple[pd.DataFrame, str]:
    if load_dotenv is not None:
        load_dotenv()

    token = os.getenv("ALERTS_IN_UA_TOKEN")

    if not token:
        return pd.DataFrame(), "live mode unavailable: ALERTS_IN_UA_TOKEN not found"

    if requests is None:
        return pd.DataFrame(), "live mode unavailable: package `requests` is not installed"

    url = "https://api.alerts.in.ua/v1/alerts/active.json"

    try:
        response = requests.get(url, params={"token": token}, timeout=10)
        response.raise_for_status()
        payload = response.json()
        alerts = payload.get("alerts", payload if isinstance(payload, list) else [])
        return pd.DataFrame(alerts), "live mode available: active alerts loaded"
    except Exception as exc:
        return pd.DataFrame(), f"live mode unavailable: API request failed: {exc}"


def format_percent(value) -> str:
    try:
        if pd.isna(value):
            return "—"
        return f"{float(value) * 100:.1f}%"
    except Exception:
        return "—"


def format_score(value) -> str:
    try:
        if pd.isna(value):
            return "—"
        return f"{float(value):.3f}"
    except Exception:
        return "—"


def metric_value(metrics_df: pd.DataFrame, name: str):
    if metrics_df.empty or name not in metrics_df.columns:
        return None
    return metrics_df[name].iloc[0]


def build_oblast_map(oblast_df: pd.DataFrame) -> go.Figure:
    if oblast_df.empty or "avg_predicted_risk" not in oblast_df.columns:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            height=520,
            title="Немає даних для панелі ризику областей",
        )
        return fig

    plot_df = oblast_df.dropna(subset=["avg_predicted_risk"]).copy()

    if "lat" in plot_df.columns and "lon" in plot_df.columns and plot_df["lat"].notna().any():
        fig = px.scatter_geo(
            plot_df,
            lat="lat",
            lon="lon",
            hover_name="oblast",
            size="avg_predicted_risk",
            color="avg_predicted_risk",
            color_continuous_scale=["#22c55e", "#eab308", "#f97316", "#ef4444"],
            size_max=38,
            projection="natural earth",
            custom_data=[
                "avg_predicted_risk",
                "actual_positive_rate",
                "max_predicted_risk" if "max_predicted_risk" in plot_df.columns else "avg_predicted_risk",
            ],
        )

        fig.update_traces(
            hovertemplate=(
                "<b>%{hovertext}</b><br>"
                "Середній risk score: %{customdata[0]:.3f}<br>"
                "Фактична частка позитивів: %{customdata[1]:.3f}<br>"
                "Максимальний risk score: %{customdata[2]:.3f}<extra></extra>"
            )
        )

        fig.update_geos(
            visible=True,
            showcountries=True,
            countrycolor="#334155",
            showland=True,
            landcolor="#0f172a",
            showocean=True,
            oceancolor="#020617",
            lataxis_range=[44, 53],
            lonaxis_range=[21, 41],
        )

        fig.update_layout(
            template="plotly_dark",
            height=560,
            margin=dict(l=0, r=0, t=35, b=0),
            title="Панель ризику по областях",
            coloraxis_colorbar=dict(title="Risk score"),
        )
        return fig

    top = plot_df.sort_values("avg_predicted_risk", ascending=True).tail(15)
    fig = px.bar(
        top,
        x="avg_predicted_risk",
        y="oblast",
        orientation="h",
        color="avg_predicted_risk",
        color_continuous_scale=["#22c55e", "#eab308", "#f97316", "#ef4444"],
        labels={"avg_predicted_risk": "Risk score", "oblast": "Область"},
        title="Топ областей за середньою оцінкою ризику",
    )
    fig.update_layout(template="plotly_dark", height=560, margin=dict(l=0, r=0, t=45, b=0))
    return fig


def build_top_raions_chart(latest_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    if latest_df.empty or "risk_score" not in latest_df.columns:
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", height=420, title="Немає даних про райони")
        return fig

    top = latest_df.sort_values("risk_score", ascending=True).tail(top_n).copy()
    top["label"] = top["oblast"].astype(str) + " · " + top["raion"].astype(str)

    fig = px.bar(
        top,
        x="risk_score",
        y="label",
        orientation="h",
        color="risk_score",
        color_continuous_scale=["#22c55e", "#eab308", "#f97316", "#ef4444"],
        labels={"risk_score": "Risk score", "label": "Район"},
        title=f"Топ-{top_n} районів за оцінкою ризику",
    )
    fig.update_layout(template="plotly_dark", height=520, margin=dict(l=0, r=0, t=45, b=0))
    return fig


def build_hour_chart(hour_df: pd.DataFrame) -> go.Figure:
    if hour_df.empty or "hour" not in hour_df.columns:
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", height=420, title="Немає даних по годинах")
        return fig

    plot_df = hour_df.copy()
    plot_df["hour"] = pd.to_numeric(plot_df["hour"], errors="coerce")
    plot_df = plot_df.dropna(subset=["hour"]).sort_values("hour")

    fig = go.Figure()

    if "avg_predicted_risk" in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=plot_df["hour"],
                y=plot_df["avg_predicted_risk"],
                mode="lines+markers",
                name="Середній risk score",
            )
        )

    if "actual_positive_rate" in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=plot_df["hour"],
                y=plot_df["actual_positive_rate"],
                mode="lines+markers",
                name="Фактична частка позитивів",
            )
        )

    fig.update_layout(
        template="plotly_dark",
        height=420,
        title="Ризик і фактичні позитивні події по годинах доби",
        xaxis_title="Година доби",
        yaxis_title="Значення",
        margin=dict(l=0, r=0, t=45, b=0),
    )
    return fig


def build_raion_hour_starts(intervals_df: pd.DataFrame, oblast: str, raion: str) -> go.Figure:
    if intervals_df.empty:
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", height=420, title="Немає даних про тривоги району")
        return fig

    df = intervals_df.copy()
    if "started_at" not in df.columns:
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", height=420, title="У файлі немає started_at")
        return fig

    df = df[(df["oblast"] == oblast) & (df["raion"] == raion)].copy()

    if df.empty:
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", height=420, title="Для вибраного району немає інтервалів")
        return fig

    df["started_at"] = pd.to_datetime(df["started_at"], errors="coerce", utc=True)
    df = df.dropna(subset=["started_at"])
    df["hour"] = df["started_at"].dt.hour

    counts = df.groupby("hour", as_index=False).size()
    all_hours = pd.DataFrame({"hour": list(range(24))})
    counts = all_hours.merge(counts, on="hour", how="left").fillna({"size": 0})

    fig = px.bar(
        counts,
        x="hour",
        y="size",
        labels={"hour": "Година доби", "size": "Кількість стартів"},
        title=f"Старт тривог по годинах: {raion}",
    )
    fig.update_layout(template="plotly_dark", height=420, margin=dict(l=0, r=0, t=45, b=0))
    return fig


def show_missing_files(errors: list[str]):
    if errors:
        with st.expander("Файли, які не вдалося завантажити", expanded=False):
            for error in errors:
                st.warning(error)


data, load_errors = load_all_data()

metrics_df = data["metrics"]
comparison_df = data["comparison"]
latest_df = prepare_latest_risk(data["latest_risk"])
risk_by_oblast_df = prepare_risk_by_oblast(data["risk_by_oblast"], latest_df)
risk_by_hour_df = prepare_risk_by_hour(data["risk_by_hour"])
prediction_sample_df = data["prediction_sample"]
intervals_df = data["intervals"]


st.sidebar.title("📡 Навігація")
st.sidebar.caption("Raion-level proxy alert forecast")

mode = st.sidebar.radio(
    "Режим",
    ["Історичний знімок", "Live API"],
    index=0,
)

live_df, live_status = load_live_alerts()

if mode == "Live API":
    st.sidebar.info(live_status)
    if live_df.empty:
        st.sidebar.warning("Додаток працює на підготовлених CSV. Live API ще не активний.")
else:
    st.sidebar.info("Режим історичного знімка. API реальних тривог ще не використовується.")

show_missing_files(load_errors)

available_oblasts = sorted(latest_df["oblast"].dropna().unique()) if "oblast" in latest_df.columns and not latest_df.empty else []
selected_oblast = st.sidebar.selectbox("Область", available_oblasts) if available_oblasts else None

if selected_oblast and not latest_df.empty:
    available_raions = sorted(latest_df.loc[latest_df["oblast"] == selected_oblast, "raion"].dropna().unique())
else:
    available_raions = []

selected_raion = st.sidebar.selectbox("Район", available_raions) if available_raions else None


st.title("📡 Ukraine Air Raid Raion Forecast")
st.markdown(
    """
<div class="small-note">
Дослідницький додаток для аналізу повітряних тривог і оцінки ризику старту нової тривоги в районі протягом наступних 60 хвилин.
Модель показує <b>оцінку ризику</b>, а не точну ймовірність і не офіційне попередження.
</div>
""",
    unsafe_allow_html=True,
)

latest_ts = None
if not latest_df.empty and "timestamp" in latest_df.columns:
    latest_ts = latest_df["timestamp"].max()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Модель", "logistic_v3_final")
col2.metric("Target", "start 60m")
col3.metric("PR AUC test", "0.330")
col4.metric("ROC AUC test", "0.812")
col5.metric("Останній знімок", str(latest_ts) if latest_ts is not None else "—")

tabs = st.tabs(
    [
        "Головний екран",
        "Профіль району",
        "Аналітика",
        "Оцінка моделі",
        "Дані і обмеження",
    ]
)


with tabs[0]:
    st.subheader("Головний екран")

    st.markdown(
        """
<div class="warning-box">
<b>Режим історичного знімка.</b><br>
API реальних тривог ще не підключено. Після додавання токена додаток зможе отримувати live статуси тривог.
Зараз карта і таблиці працюють на підготовлених CSV файлах.
</div>
""",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.45, 1.0])

    with left:
        st.plotly_chart(build_oblast_map(risk_by_oblast_df), use_container_width=True)

    with right:
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        st.subheader("Топ ризикових районів")

        if latest_df.empty:
            st.warning("Немає `latest_risk_snapshot.csv` або файл порожній.")
        else:
            show_cols = [c for c in ["timestamp", "oblast", "raion", "risk_score", "risk_rank", "risk_percentile", "risk_level", TARGET_NAME] if c in latest_df.columns]
            top_risks = latest_df.sort_values("risk_score", ascending=False).head(15)[show_cols].copy()
            if "risk_score" in top_risks.columns:
                top_risks["risk_score"] = top_risks["risk_score"].map(lambda x: round(float(x), 4) if pd.notna(x) else x)
            st.dataframe(top_risks, use_container_width=True, hide_index=True)

        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    st.subheader("Вибраний район")

    if not selected_oblast or not selected_raion or latest_df.empty:
        st.info("Вибери область і район у боковому меню.")
    else:
        selected_row = latest_df[
            (latest_df["oblast"] == selected_oblast) & (latest_df["raion"] == selected_raion)
        ].sort_values("timestamp").tail(1)

        if selected_row.empty:
            st.warning("Для вибраного району немає risk snapshot.")
        else:
            row = selected_row.iloc[0]
            score = row.get("risk_score", pd.NA)
            level = row.get("risk_level", risk_level_from_score(score))
            rank = row.get("risk_rank", "—")

            a, b, c, d = st.columns(4)
            a.metric("Область", selected_oblast)
            b.metric("Район", selected_raion)
            c.metric("Risk score", format_score(score))
            d.metric("Ранг ризику", str(rank))

            st.markdown(
                f"Рівень ризику: {risk_html(level)}",
                unsafe_allow_html=True,
            )

            if st.button("Спрогнозувати ризик на 60 хв", type="primary"):
                st.success(
                    f"Оцінка ризику для {selected_raion}: {format_score(score)}. "
                    f"Інтерпретація: {risk_level_ua(level)} ризик старту нової тривоги протягом наступних 60 хвилин "
                    f"відносно інших районів у цьому історичному знімку."
                )
                st.caption(
                    "Це не точна ймовірність і не офіційне попередження. Модель використовується для ранжування ризику."
                )

    st.plotly_chart(build_top_raions_chart(latest_df), use_container_width=True)


with tabs[1]:
    st.subheader("Профіль району")

    if not selected_oblast or not selected_raion:
        st.info("Вибери область і район у боковому меню.")
    else:
        st.markdown(
            f"""
<div class="main-card">
<b>Область:</b> {selected_oblast}<br>
<b>Район:</b> {selected_raion}
</div>
""",
            unsafe_allow_html=True,
        )

        if intervals_df.empty:
            st.warning("Немає `data/interim/raion_alert_intervals.csv`.")
        else:
            df = intervals_df.copy()
            df = df[(df["oblast"] == selected_oblast) & (df["raion"] == selected_raion)].copy()

            if df.empty:
                st.warning("Для вибраного району немає інтервалів тривог.")
            else:
                if "started_at" in df.columns:
                    df["started_at"] = pd.to_datetime(df["started_at"], errors="coerce", utc=True)
                if "finished_at" in df.columns:
                    df["finished_at"] = pd.to_datetime(df["finished_at"], errors="coerce", utc=True)

                if "duration_min" not in df.columns and {"started_at", "finished_at"}.issubset(df.columns):
                    df["duration_min"] = (df["finished_at"] - df["started_at"]).dt.total_seconds() / 60

                df = df.sort_values("started_at", ascending=False)

                cols = [c for c in ["started_at", "finished_at", "duration_min", "source_level"] if c in df.columns]
                st.subheader("Останні 10 тривог")
                st.dataframe(df[cols].head(10), use_container_width=True, hide_index=True)

                st.plotly_chart(
                    build_raion_hour_starts(intervals_df, selected_oblast, selected_raion),
                    use_container_width=True,
                )


with tabs[2]:
    st.subheader("Аналітика")

    col_a, col_b = st.columns(2)

    with col_a:
        if risk_by_oblast_df.empty:
            st.warning("Немає `risk_by_oblast.csv`.")
        else:
            plot_df = risk_by_oblast_df.sort_values("avg_predicted_risk", ascending=True).tail(15)
            fig = px.bar(
                plot_df,
                x="avg_predicted_risk",
                y="oblast",
                orientation="h",
                color="avg_predicted_risk",
                color_continuous_scale=["#22c55e", "#eab308", "#f97316", "#ef4444"],
                title="Середній risk score по областях",
                labels={"avg_predicted_risk": "Risk score", "oblast": "Область"},
            )
            fig.update_layout(template="plotly_dark", height=520, margin=dict(l=0, r=0, t=45, b=0))
            st.plotly_chart(fig, use_container_width=True)

    with col_b:
        if risk_by_oblast_df.empty or "actual_positive_rate" not in risk_by_oblast_df.columns:
            st.warning("Немає фактичної частки позитивних подій по областях.")
        else:
            plot_df = risk_by_oblast_df.dropna(subset=["actual_positive_rate"]).sort_values("actual_positive_rate", ascending=True).tail(15)
            fig = px.bar(
                plot_df,
                x="actual_positive_rate",
                y="oblast",
                orientation="h",
                color="actual_positive_rate",
                color_continuous_scale=["#334155", "#eab308", "#ef4444"],
                title="Фактична частка позитивних подій по областях",
                labels={"actual_positive_rate": "Positive rate", "oblast": "Область"},
            )
            fig.update_layout(template="plotly_dark", height=520, margin=dict(l=0, r=0, t=45, b=0))
            st.plotly_chart(fig, use_container_width=True)

    st.plotly_chart(build_hour_chart(risk_by_hour_df), use_container_width=True)

    if not prediction_sample_df.empty:
        with st.expander("Приклад тестових прогнозів"):
            st.dataframe(prediction_sample_df.head(200), use_container_width=True)


with tabs[3]:
    st.subheader("Оцінка моделі")

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("PR AUC", "0.330244")
    m2.metric("ROC AUC", "0.812334")
    m3.metric("Brier", "0.168278")
    m4.metric("F1@0.5", "0.351705")
    m5.metric("Precision top 10%", "0.362957")
    m6.metric("Recall top 10%", "0.403474")

    st.markdown(
        """
<div class="main-card">
<b>Фінальна модель:</b> logistic_v3_final<br>
<b>Target:</b> target_start_60m — чи почнеться нова повітряна тривога в районі протягом наступних 60 хвилин.<br><br>
CatBoost тестувався, але не став фінальною моделлю, бо гірше переносився на майбутній тестовий період.
Фінально обрана логістична регресія, бо вона дала кращу якість на forward test split.
</div>
""",
        unsafe_allow_html=True,
    )

    st.subheader("Фінальні метрики")
    if metrics_df.empty:
        st.warning("Немає `final_model_metrics.csv`.")
    else:
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    st.subheader("Порівняння моделей")
    if comparison_df.empty:
        st.warning("Немає `final_model_comparison.csv`.")
    else:
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        if {"model", "pr_auc"}.issubset(comparison_df.columns):
            plot_df = comparison_df.copy()
            if "split" in plot_df.columns:
                plot_df = plot_df[plot_df["split"].astype(str).str.lower() == "test"].copy()
                if plot_df.empty:
                    plot_df = comparison_df.copy()

            fig = px.bar(
                plot_df.sort_values("pr_auc", ascending=True),
                x="pr_auc",
                y="model",
                orientation="h",
                color="pr_auc",
                color_continuous_scale=["#334155", "#eab308", "#22c55e"],
                title="PR AUC моделей",
                labels={"pr_auc": "PR AUC", "model": "Модель"},
            )
            fig.update_layout(template="plotly_dark", height=480, margin=dict(l=0, r=0, t=45, b=0))
            st.plotly_chart(fig, use_container_width=True)


with tabs[4]:
    st.subheader("Дані і обмеження")

    st.markdown(
        """
<div class="main-card">
<h4>Що робить додаток</h4>
Додаток показує дослідницьку оцінку ризику старту нової повітряної тривоги в районі протягом наступних 60 хвилин.
Він працює на підготовлених CSV файлах і не потребує API токена для запуску.
</div>

<div class="main-card">
<h4>Критичні обмеження</h4>
<ul>
<li>Це не офіційна система безпеки.</li>
<li>Це не заміна офіційних повідомлень про повітряну тривогу.</li>
<li>Модель дає <b>risk score</b>, а не точну ймовірність.</li>
<li>Датасет є <b>raion-level proxy dataset</b>.</li>
<li>Частина історичних районних рядків успадкована з рівня області.</li>
<li>Для реального часу потрібен токен alerts.in.ua API.</li>
<li>Поки live API не підключений, додаток працює в режимі історичного знімка.</li>
</ul>
</div>

<div class="main-card">
<h4>Як буде працювати live mode</h4>
Після додавання токена в <code>.env</code> функція <code>load_live_alerts()</code> зможе отримувати активні тривоги.
Потім ці дані можна буде перетворити у той самий формат <code>timestamp + oblast + raion</code>, який використовує dashboard.
</div>
""",
        unsafe_allow_html=True,
    )

    st.subheader("Статус live API")
    st.code(live_status)

    if not live_df.empty:
        st.dataframe(live_df.head(100), use_container_width=True)

    st.subheader("Очікувані файли")
    expected = pd.DataFrame(
        [{"file": str(path), "exists": path.exists()} for path in DATA_PATHS.values()]
    )
    st.dataframe(expected, use_container_width=True, hide_index=True)