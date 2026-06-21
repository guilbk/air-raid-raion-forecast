import os
import json
import time
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    import requests
except ImportError:
    requests = None

try:
    import joblib
except ImportError:
    joblib = None

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None


APP_TITLE = "Air Raid Raion Forecast"
TARGET_NAME = "target_start_60m"
LIVE_REFRESH_MINUTES = 15

DATA = {
    "metrics": Path("data/processed/final_model_metrics.csv"),
    "comparison": Path("data/processed/final_model_comparison.csv"),
    "latest_risk": Path("data/processed/latest_risk_snapshot.csv"),
    "risk_by_oblast": Path("data/processed/risk_by_oblast.csv"),
    "risk_by_hour": Path("data/processed/risk_by_hour.csv"),
    "prediction_sample": Path("data/processed/logistic_v3_test_predictions_sample.csv"),
    "intervals": Path("data/interim/raion_alert_intervals.csv"),
}

MODEL_PATHS = [
    Path("data/processed/logistic_v3_final_model.joblib"),
    Path("data/processed/logistic_v3_final.joblib"),
    Path("data/processed/logistic_v3_final.pkl"),
    Path("data/processed/logistic_v3_model.joblib"),
    Path("models/logistic_v3_final.joblib"),
    Path("models/logistic_v3_final.pkl"),
]

FEATURE_PATHS = [
    Path("data/processed/logistic_v3_feature_columns.json"),
    Path("data/processed/logistic_v3_feature_columns.txt"),
    Path("data/processed/final_feature_columns.json"),
    Path("data/processed/final_feature_columns.txt"),
]

LIVE_DIR = Path("data/live")
LIVE_DIR.mkdir(parents=True, exist_ok=True)

LIVE_SNAPSHOT_PATH = LIVE_DIR / "live_raion_snapshots.csv"
LIVE_RISK_PATH = LIVE_DIR / "live_risk_snapshot.csv"
LIVE_RAW_LAST_PATH = LIVE_DIR / "live_api_last_response.json"

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

FALLBACK_NUMERIC_FEATURES = [
    "hour",
    "day_of_week",
    "is_weekend",
    "is_night",
    "starts_last_1h",
    "starts_last_3h",
    "starts_last_24h",
    "active_minutes_last_1h",
    "active_minutes_last_3h",
    "active_minutes_last_24h",
    "oblast_active_raions_now",
    "oblast_active_share_now",
]


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
    background: radial-gradient(circle at top, #172033 0%, #0b0f17 48%, #06080d 100%);
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
.warning-box {
    background: rgba(245, 158, 11, 0.10);
    border: 1px solid rgba(245, 158, 11, 0.35);
    border-radius: 14px;
    padding: 14px;
    color: #ffe6b3;
}
.danger-box {
    background: rgba(239, 68, 68, 0.10);
    border: 1px solid rgba(239, 68, 68, 0.35);
    border-radius: 14px;
    padding: 14px;
    color: #fecaca;
}
.small-note {
    color: #9fb0c9;
    font-size: 0.92rem;
    line-height: 1.45;
}
.risk-low {color: #22c55e; font-weight: 800;}
.risk-medium {color: #eab308; font-weight: 800;}
.risk-high {color: #f97316; font-weight: 800;}
.risk-very-high {color: #ef4444; font-weight: 800;}
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


def read_csv_safe(path: Path, usecols=None) -> tuple[pd.DataFrame, str | None]:
    if not path.exists():
        return pd.DataFrame(), f"Файл не знайдено: {path}"
    try:
        return pd.read_csv(path, usecols=usecols), None
    except Exception as exc:
        return pd.DataFrame(), f"Не вдалося прочитати {path}: {exc}"


@st.cache_data(show_spinner=False)
def load_static_data():
    data = {}
    errors = []

    for key, path in DATA.items():
        df, error = read_csv_safe(path)
        data[key] = df
        if error:
            errors.append(error)

    return data, errors


def normalize_text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def get_risk_col(df: pd.DataFrame) -> str | None:
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


def risk_level(score) -> str:
    try:
        score = float(score)
    except Exception:
        return "unknown"
    if score < 0.25:
        return "low"
    if score < 0.50:
        return "medium"
    if score < 0.75:
        return "high"
    return "very_high"


def risk_ua(level: str) -> str:
    return {
        "low": "низький",
        "medium": "середній",
        "high": "високий",
        "very_high": "дуже високий",
        "already_active": "тривога активна",
        "unknown": "невідомо",
    }.get(str(level), str(level))


def risk_html(level: str, text: str | None = None) -> str:
    css = {
        "low": "risk-low",
        "medium": "risk-medium",
        "high": "risk-high",
        "very_high": "risk-very-high",
        "already_active": "risk-very-high",
    }.get(str(level), "")
    return f'<span class="{css}">{text or risk_ua(level)}</span>'


def fmt_score(x) -> str:
    try:
        if pd.isna(x):
            return "—"
        return f"{float(x):.3f}"
    except Exception:
        return "—"


def fmt_pct(x) -> str:
    try:
        if pd.isna(x):
            return "—"
        return f"{float(x) * 100:.1f}%"
    except Exception:
        return "—"


def floor_to_15min_utc(dt: datetime | None = None) -> pd.Timestamp:
    if dt is None:
        dt = datetime.now(timezone.utc)
    ts = pd.Timestamp(dt).tz_convert("UTC") if pd.Timestamp(dt).tzinfo else pd.Timestamp(dt, tz="UTC")
    minute = (ts.minute // 15) * 15
    return ts.replace(minute=minute, second=0, microsecond=0)


def prepare_latest_risk(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    risk_col = get_risk_col(out)

    if risk_col is not None and risk_col != "risk_score":
        out["risk_score"] = pd.to_numeric(out[risk_col], errors="coerce")
    elif "risk_score" in out.columns:
        out["risk_score"] = pd.to_numeric(out["risk_score"], errors="coerce")
    else:
        out["risk_score"] = np.nan

    if "risk_level" not in out.columns:
        out["risk_level"] = out["risk_score"].apply(risk_level)

    if "risk_rank" not in out.columns:
        out["risk_rank"] = out["risk_score"].rank(ascending=False, method="first").astype("Int64")

    if "risk_percentile" not in out.columns:
        out["risk_percentile"] = out["risk_score"].rank(pct=True)

    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)

    return out


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
            actual = latest_df.groupby("oblast")[TARGET_NAME].mean().reset_index(name="actual_positive_rate")
            out = out.merge(actual, on="oblast", how="left")
    else:
        return pd.DataFrame()

    if "avg_predicted_risk" not in out.columns:
        risk_col = get_risk_col(out)
        if risk_col:
            out["avg_predicted_risk"] = pd.to_numeric(out[risk_col], errors="coerce")

    if "actual_positive_rate" not in out.columns:
        out["actual_positive_rate"] = np.nan

    out["lat"] = out["oblast"].map(lambda x: OBLAST_COORDS.get(x, (None, None))[0])
    out["lon"] = out["oblast"].map(lambda x: OBLAST_COORDS.get(x, (None, None))[1])
    return out


def prepare_risk_by_hour(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()

    if "hour" not in out.columns:
        possible = find_col(out, ["timestamp_hour", "time_hour"])
        if possible:
            out["hour"] = out[possible]

    if "avg_predicted_risk" not in out.columns:
        risk_col = get_risk_col(out)
        if risk_col:
            out["avg_predicted_risk"] = pd.to_numeric(out[risk_col], errors="coerce")

    if "actual_positive_rate" not in out.columns:
        actual_col = find_col(out, ["positive_rate", "target_rate", "actual_rate"])
        if actual_col:
            out["actual_positive_rate"] = pd.to_numeric(out[actual_col], errors="coerce")

    return out


def build_raion_reference(latest_df: pd.DataFrame, intervals_df: pd.DataFrame) -> pd.DataFrame:
    parts = []

    if not latest_df.empty and {"oblast", "raion"}.issubset(latest_df.columns):
        parts.append(latest_df[["oblast", "raion"]].dropna().drop_duplicates())

    if not intervals_df.empty and {"oblast", "raion"}.issubset(intervals_df.columns):
        parts.append(intervals_df[["oblast", "raion"]].dropna().drop_duplicates())

    if not parts:
        return pd.DataFrame(columns=["oblast", "raion"])

    ref = pd.concat(parts, ignore_index=True).drop_duplicates()
    ref["oblast"] = ref["oblast"].map(normalize_text)
    ref["raion"] = ref["raion"].map(normalize_text)
    ref = ref[(ref["oblast"] != "") & (ref["raion"] != "")]
    return ref.sort_values(["oblast", "raion"]).reset_index(drop=True)


def load_token() -> str | None:
    if load_dotenv is not None:
        load_dotenv()
    token = os.getenv("ALERTS_IN_UA_TOKEN")
    if token:
        return token.strip()
    return None


def fetch_active_alerts() -> tuple[pd.DataFrame, str]:
    token = load_token()

    if not token:
        return pd.DataFrame(), "live unavailable: ALERTS_IN_UA_TOKEN не знайдено в .env"

    if requests is None:
        return pd.DataFrame(), "live unavailable: пакет requests не встановлено"

    url = "https://api.alerts.in.ua/v1/alerts/active.json"

    try:
        response = requests.get(url, params={"token": token}, timeout=12)
        response.raise_for_status()
        payload = response.json()

        LIVE_RAW_LAST_PATH.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if isinstance(payload, dict):
            alerts = payload.get("alerts", [])
        elif isinstance(payload, list):
            alerts = payload
        else:
            alerts = []

        df = pd.DataFrame(alerts)
        return df, f"live ok: отримано активних alert records: {len(df)}"
    except Exception as exc:
        return pd.DataFrame(), f"live error: {exc}"


def is_air_raid_alert(row: pd.Series) -> bool:
    alert_type = normalize_text(row.get("alert_type", "")).lower()
    alert_title = normalize_text(row.get("alert_type_title", "")).lower()
    title = normalize_text(row.get("title", "")).lower()

    if alert_type == "air_raid":
        return True
    if "повітря" in alert_title or "air" in alert_title:
        return True
    if "повітря" in title:
        return True
    return False


def get_alert_location(row: pd.Series) -> tuple[str, str, str]:
    location_type = normalize_text(row.get("location_type", "")).lower()

    oblast = normalize_text(
        row.get("location_oblast")
        or row.get("oblast")
        or row.get("region")
        or ""
    )

    raion = normalize_text(
        row.get("location_raion")
        or row.get("raion")
        or ""
    )

    title = normalize_text(
        row.get("location_title")
        or row.get("location")
        or row.get("title")
        or ""
    )

    if not oblast and "область" in title:
        oblast = title

    return location_type, oblast, raion


def build_live_raion_snapshot(api_df: pd.DataFrame, raion_ref: pd.DataFrame, ts: pd.Timestamp) -> pd.DataFrame:
    base = raion_ref.copy()
    base["timestamp"] = ts
    base["fetched_at"] = pd.Timestamp.now(tz="UTC")
    base["direct_active"] = 0
    base["inherited_active"] = 0
    base["api_records_matched"] = 0

    if api_df.empty or base.empty:
        base["is_alert_active"] = 0
        base["source_level_now"] = "none"
        return base[
            [
                "timestamp",
                "fetched_at",
                "oblast",
                "raion",
                "is_alert_active",
                "direct_active",
                "inherited_active",
                "source_level_now",
                "api_records_matched",
            ]
        ]

    air = api_df[api_df.apply(is_air_raid_alert, axis=1)].copy()

    for _, alert in air.iterrows():
        location_type, oblast, raion = get_alert_location(alert)

        if location_type == "oblast" and oblast:
            mask = base["oblast"] == oblast
            base.loc[mask, "inherited_active"] = 1
            base.loc[mask, "api_records_matched"] += 1
            continue

        if location_type in ["raion", "hromada", "city", "community"] and raion:
            if oblast:
                mask = (base["oblast"] == oblast) & (base["raion"] == raion)
            else:
                mask = base["raion"] == raion

            if mask.any():
                base.loc[mask, "direct_active"] = 1
                base.loc[mask, "api_records_matched"] += 1
            elif oblast:
                mask_oblast = base["oblast"] == oblast
                base.loc[mask_oblast, "inherited_active"] = 1
                base.loc[mask_oblast, "api_records_matched"] += 1

    base["is_alert_active"] = ((base["direct_active"] == 1) | (base["inherited_active"] == 1)).astype(int)

    conditions = [
        (base["direct_active"] == 0) & (base["inherited_active"] == 0),
        (base["direct_active"] == 1) & (base["inherited_active"] == 0),
        (base["direct_active"] == 0) & (base["inherited_active"] == 1),
        (base["direct_active"] == 1) & (base["inherited_active"] == 1),
    ]

    choices = ["none", "direct_or_hromada", "oblast_inherited", "mixed"]
    base["source_level_now"] = np.select(conditions, choices, default="none")

    return base[
        [
            "timestamp",
            "fetched_at",
            "oblast",
            "raion",
            "is_alert_active",
            "direct_active",
            "inherited_active",
            "source_level_now",
            "api_records_matched",
        ]
    ]


def read_live_log() -> pd.DataFrame:
    if not LIVE_SNAPSHOT_PATH.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(LIVE_SNAPSHOT_PATH)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        if "fetched_at" in df.columns:
            df["fetched_at"] = pd.to_datetime(df["fetched_at"], errors="coerce", utc=True)
        return df
    except Exception:
        return pd.DataFrame()


def append_live_snapshot(snapshot: pd.DataFrame):
    if snapshot.empty:
        return

    existing = read_live_log()

    combined = pd.concat([existing, snapshot], ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], errors="coerce", utc=True)
    combined = combined.dropna(subset=["timestamp", "oblast", "raion"])

    combined = combined.drop_duplicates(
        subset=["timestamp", "oblast", "raion"],
        keep="last",
    )

    combined = combined.sort_values(["timestamp", "oblast", "raion"])
    combined.to_csv(LIVE_SNAPSHOT_PATH, index=False)


def maybe_collect_live_snapshot(raion_ref: pd.DataFrame, force: bool = False) -> tuple[pd.DataFrame, str]:
    ts = floor_to_15min_utc()
    log = read_live_log()

    if not force and not log.empty and "timestamp" in log.columns:
        last_ts = log["timestamp"].max()
        if pd.notna(last_ts) and last_ts >= ts:
            latest = log[log["timestamp"] == last_ts].copy()
            return latest, f"live cached: останній знімок уже є для {last_ts}"

    api_df, status = fetch_active_alerts()

    if "error" in status or "unavailable" in status:
        if not log.empty and "timestamp" in log.columns:
            last_ts = log["timestamp"].max()
            latest = log[log["timestamp"] == last_ts].copy()
            return latest, status + " | використано останній live log"
        return pd.DataFrame(), status

    snapshot = build_live_raion_snapshot(api_df, raion_ref, ts)
    append_live_snapshot(snapshot)
    return snapshot, status + f" | saved snapshot: {ts}"


def compute_live_started_flags(log: pd.DataFrame) -> pd.DataFrame:
    if log.empty:
        return log

    df = log.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values(["oblast", "raion", "timestamp"])

    df["prev_active"] = df.groupby(["oblast", "raion"])["is_alert_active"].shift(1).fillna(0)
    df["alert_started_now"] = ((df["is_alert_active"] == 1) & (df["prev_active"] == 0)).astype(int)
    return df.drop(columns=["prev_active"])


def build_live_features(current_snapshot: pd.DataFrame, full_log: pd.DataFrame) -> pd.DataFrame:
    if current_snapshot.empty:
        return pd.DataFrame()

    current = current_snapshot.copy()
    current["timestamp"] = pd.to_datetime(current["timestamp"], errors="coerce", utc=True)
    current_ts = current["timestamp"].max()

    log = compute_live_started_flags(full_log)
    if log.empty:
        log = current.copy()
        log["alert_started_now"] = 0

    log["timestamp"] = pd.to_datetime(log["timestamp"], errors="coerce", utc=True)
    log = log.dropna(subset=["timestamp"])

    current["hour"] = current_ts.hour
    current["day_of_week"] = current_ts.dayofweek
    current["is_weekend"] = int(current_ts.dayofweek >= 5)
    current["is_night"] = int(current_ts.hour < 6 or current_ts.hour >= 22)

    windows = {
        "1h": 60,
        "3h": 180,
        "24h": 1440,
    }

    keys = ["oblast", "raion"]

    for label, minutes in windows.items():
        start_time = current_ts - pd.Timedelta(minutes=minutes)
        recent = log[(log["timestamp"] > start_time) & (log["timestamp"] <= current_ts)].copy()

        if recent.empty:
            starts = pd.DataFrame(columns=keys + [f"starts_last_{label}"])
            active = pd.DataFrame(columns=keys + [f"active_minutes_last_{label}"])
        else:
            starts = (
                recent.groupby(keys, as_index=False)["alert_started_now"]
                .sum()
                .rename(columns={"alert_started_now": f"starts_last_{label}"})
            )
            active = (
                recent.groupby(keys, as_index=False)["is_alert_active"]
                .sum()
                .rename(columns={"is_alert_active": f"active_minutes_last_{label}"})
            )
            active[f"active_minutes_last_{label}"] = active[f"active_minutes_last_{label}"] * 15

        current = current.merge(starts, on=keys, how="left")
        current = current.merge(active, on=keys, how="left")

    for col in [
        "starts_last_1h",
        "starts_last_3h",
        "starts_last_24h",
        "active_minutes_last_1h",
        "active_minutes_last_3h",
        "active_minutes_last_24h",
    ]:
        if col not in current.columns:
            current[col] = 0
        current[col] = pd.to_numeric(current[col], errors="coerce").fillna(0)

    oblast_stats = (
        current.groupby("oblast", as_index=False)
        .agg(
            oblast_active_raions_now=("is_alert_active", "sum"),
            oblast_total_raions=("raion", "count"),
        )
    )
    oblast_stats["oblast_active_share_now"] = (
        oblast_stats["oblast_active_raions_now"] / oblast_stats["oblast_total_raions"].replace(0, np.nan)
    ).fillna(0)

    current = current.merge(
        oblast_stats[["oblast", "oblast_active_raions_now", "oblast_active_share_now"]],
        on="oblast",
        how="left",
    )

    current["live_history_hours_available"] = 0.0
    if not log.empty:
        min_ts = log["timestamp"].min()
        current["live_history_hours_available"] = max((current_ts - min_ts).total_seconds() / 3600, 0)

    return current


def load_model():
    if joblib is None:
        return None, "joblib not installed"

    for path in MODEL_PATHS:
        if path.exists():
            try:
                return joblib.load(path), f"model loaded: {path}"
            except Exception as exc:
                return None, f"model load failed: {path}: {exc}"

    return None, "model file not found, fallback risk scoring will be used"


def load_feature_columns() -> list[str] | None:
    for path in FEATURE_PATHS:
        if not path.exists():
            continue

        try:
            if path.suffix.lower() == ".json":
                return json.loads(path.read_text(encoding="utf-8"))
            if path.suffix.lower() == ".txt":
                return [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
        except Exception:
            continue

    return None


def get_model_feature_columns(model, live_df: pd.DataFrame) -> list[str]:
    explicit = load_feature_columns()
    if explicit:
        return explicit

    if model is not None and hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    cols = []
    for col in ["oblast", "raion"] + FALLBACK_NUMERIC_FEATURES:
        if col in live_df.columns:
            cols.append(col)
    return cols


def fallback_risk_scores(live_df: pd.DataFrame, latest_df: pd.DataFrame, risk_by_oblast_df: pd.DataFrame) -> pd.Series:
    df = live_df.copy()

    base = pd.Series(0.10, index=df.index, dtype=float)

    if not latest_df.empty and {"oblast", "raion", "risk_score"}.issubset(latest_df.columns):
        prior = latest_df[["oblast", "raion", "risk_score"]].copy()
        prior = prior.rename(columns={"risk_score": "historical_risk_prior"})
        df = df.merge(prior, on=["oblast", "raion"], how="left")
        base = df["historical_risk_prior"].fillna(base).astype(float)

    elif not risk_by_oblast_df.empty and {"oblast", "avg_predicted_risk"}.issubset(risk_by_oblast_df.columns):
        prior = risk_by_oblast_df[["oblast", "avg_predicted_risk"]].copy()
        prior = prior.rename(columns={"avg_predicted_risk": "oblast_risk_prior"})
        df = df.merge(prior, on="oblast", how="left")
        base = df["oblast_risk_prior"].fillna(base).astype(float)

    active_share = pd.to_numeric(live_df.get("oblast_active_share_now", 0), errors="coerce").fillna(0)
    starts_3h = pd.to_numeric(live_df.get("starts_last_3h", 0), errors="coerce").fillna(0)
    active_3h = pd.to_numeric(live_df.get("active_minutes_last_3h", 0), errors="coerce").fillna(0)
    is_night = pd.to_numeric(live_df.get("is_night", 0), errors="coerce").fillna(0)

    risk = (
        0.55 * base
        + 0.25 * active_share
        + 0.04 * np.clip(starts_3h, 0, 3)
        + 0.001 * np.clip(active_3h, 0, 180)
        + 0.04 * is_night
    )

    return pd.Series(np.clip(risk, 0, 0.99), index=live_df.index)


def predict_live_risk(
    live_features: pd.DataFrame,
    model,
    model_status: str,
    latest_df: pd.DataFrame,
    risk_by_oblast_df: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    if live_features.empty:
        return pd.DataFrame(), "no live features"

    out = live_features.copy()
    out["risk_score"] = np.nan
    out["prediction_mode"] = "fallback_momentary"

    inactive_mask = out["is_alert_active"] == 0

    if model is not None and inactive_mask.any():
        feature_cols = get_model_feature_columns(model, out)

        X = out.loc[inactive_mask].copy()

        for col in feature_cols:
            if col not in X.columns:
                X[col] = 0

        X_model = X[feature_cols].copy()

        try:
            if hasattr(model, "predict_proba"):
                pred = model.predict_proba(X_model)[:, 1]
            else:
                pred = model.predict(X_model)
                pred = np.asarray(pred).astype(float)

            out.loc[inactive_mask, "risk_score"] = pred
            out.loc[inactive_mask, "prediction_mode"] = "logistic_v3_live_features"
            status = model_status + " | live prediction ok"
        except Exception as exc:
            pred = fallback_risk_scores(out.loc[inactive_mask], latest_df, risk_by_oblast_df)
            out.loc[inactive_mask, "risk_score"] = pred.values
            out.loc[inactive_mask, "prediction_mode"] = "fallback_after_model_error"
            status = model_status + f" | model prediction failed: {exc} | fallback used"
    else:
        if inactive_mask.any():
            pred = fallback_risk_scores(out.loc[inactive_mask], latest_df, risk_by_oblast_df)
            out.loc[inactive_mask, "risk_score"] = pred.values
        status = model_status + " | fallback used"

    out.loc[out["is_alert_active"] == 1, "risk_score"] = np.nan
    out.loc[out["is_alert_active"] == 1, "prediction_mode"] = "already_active_no_start_prediction"

    out["risk_level"] = out["risk_score"].apply(risk_level)
    out.loc[out["is_alert_active"] == 1, "risk_level"] = "already_active"

    out["risk_rank"] = out["risk_score"].rank(ascending=False, method="first").astype("Int64")
    out["risk_percentile"] = out["risk_score"].rank(pct=True)

    out.to_csv(LIVE_RISK_PATH, index=False)
    return out, status


def build_oblast_panel(oblast_df: pd.DataFrame, value_col: str, title: str) -> go.Figure:
    if oblast_df.empty or value_col not in oblast_df.columns:
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", height=520, title="Немає даних")
        return fig

    df = oblast_df.copy()
    df["lat"] = df["oblast"].map(lambda x: OBLAST_COORDS.get(x, (None, None))[0])
    df["lon"] = df["oblast"].map(lambda x: OBLAST_COORDS.get(x, (None, None))[1])

    geo_df = df.dropna(subset=["lat", "lon", value_col])

    if not geo_df.empty:
        fig = px.scatter_geo(
            geo_df,
            lat="lat",
            lon="lon",
            hover_name="oblast",
            size=value_col,
            color=value_col,
            color_continuous_scale=["#22c55e", "#eab308", "#f97316", "#ef4444"],
            size_max=42,
            projection="natural earth",
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
            height=570,
            title=title,
            margin=dict(l=0, r=0, t=45, b=0),
            coloraxis_colorbar=dict(title="score"),
        )
        return fig

    top = df.sort_values(value_col, ascending=True).tail(15)
    fig = px.bar(
        top,
        x=value_col,
        y="oblast",
        orientation="h",
        color=value_col,
        color_continuous_scale=["#22c55e", "#eab308", "#f97316", "#ef4444"],
        title=title,
    )
    fig.update_layout(template="plotly_dark", height=570, margin=dict(l=0, r=0, t=45, b=0))
    return fig


def build_live_oblast_summary(live_risk: pd.DataFrame) -> pd.DataFrame:
    if live_risk.empty:
        return pd.DataFrame()

    inactive = live_risk[live_risk["is_alert_active"] == 0].copy()

    risk_summary = (
        inactive.groupby("oblast", as_index=False)
        .agg(
            avg_live_risk=("risk_score", "mean"),
            max_live_risk=("risk_score", "max"),
            inactive_raions=("raion", "count"),
        )
    )

    active_summary = (
        live_risk.groupby("oblast", as_index=False)
        .agg(
            active_raions=("is_alert_active", "sum"),
            total_raions=("raion", "count"),
        )
    )

    out = active_summary.merge(risk_summary, on="oblast", how="left")
    out["active_share"] = out["active_raions"] / out["total_raions"].replace(0, np.nan)
    return out


def chart_top_live_risks(live_risk: pd.DataFrame, n: int = 20) -> go.Figure:
    if live_risk.empty:
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", height=520, title="Немає live predictions")
        return fig

    df = live_risk[live_risk["is_alert_active"] == 0].copy()
    df = df.dropna(subset=["risk_score"]).sort_values("risk_score", ascending=True).tail(n)
    df["label"] = df["oblast"] + " · " + df["raion"]

    fig = px.bar(
        df,
        x="risk_score",
        y="label",
        orientation="h",
        color="risk_score",
        color_continuous_scale=["#22c55e", "#eab308", "#f97316", "#ef4444"],
        title=f"Топ-{n} районів за live risk score",
        labels={"risk_score": "Risk score", "label": "Район"},
    )
    fig.update_layout(template="plotly_dark", height=620, margin=dict(l=0, r=0, t=45, b=0))
    return fig


def chart_live_active_history(log: pd.DataFrame) -> go.Figure:
    if log.empty:
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", height=420, title="Live log ще порожній")
        return fig

    df = log.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

    summary = (
        df.groupby("timestamp", as_index=False)
        .agg(
            active_raions=("is_alert_active", "sum"),
            direct_active=("direct_active", "sum"),
            inherited_active=("inherited_active", "sum"),
        )
        .sort_values("timestamp")
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=summary["timestamp"], y=summary["active_raions"], mode="lines+markers", name="active raions"))
    fig.add_trace(go.Scatter(x=summary["timestamp"], y=summary["direct_active"], mode="lines+markers", name="direct active"))
    fig.add_trace(go.Scatter(x=summary["timestamp"], y=summary["inherited_active"], mode="lines+markers", name="inherited active"))

    fig.update_layout(
        template="plotly_dark",
        height=420,
        title="Live log: активні райони у часі",
        xaxis_title="Timestamp",
        yaxis_title="Кількість районів",
        margin=dict(l=0, r=0, t=45, b=0),
    )
    return fig


def chart_risk_by_hour(hour_df: pd.DataFrame) -> go.Figure:
    if hour_df.empty or "hour" not in hour_df.columns:
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", height=420, title="Немає risk_by_hour.csv")
        return fig

    df = hour_df.copy()
    df["hour"] = pd.to_numeric(df["hour"], errors="coerce")
    df = df.dropna(subset=["hour"]).sort_values("hour")

    fig = go.Figure()

    if "avg_predicted_risk" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["hour"],
                y=df["avg_predicted_risk"],
                mode="lines+markers",
                name="Середній risk score",
            )
        )

    if "actual_positive_rate" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["hour"],
                y=df["actual_positive_rate"],
                mode="lines+markers",
                name="Фактична частка позитивів",
            )
        )

    fig.update_layout(
        template="plotly_dark",
        height=430,
        title="Ризик і фактичні позитиви по годинах",
        xaxis_title="Година",
        yaxis_title="Значення",
        margin=dict(l=0, r=0, t=45, b=0),
    )
    return fig


def chart_raion_hour_starts(intervals: pd.DataFrame, oblast: str, raion: str) -> go.Figure:
    if intervals.empty or "started_at" not in intervals.columns:
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", height=420, title="Немає історії району")
        return fig

    df = intervals[(intervals["oblast"] == oblast) & (intervals["raion"] == raion)].copy()
    if df.empty:
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", height=420, title="Для району немає інтервалів")
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
        title=f"Старт тривог по годинах: {raion}",
        labels={"hour": "Година доби", "size": "Кількість стартів"},
    )
    fig.update_layout(template="plotly_dark", height=420, margin=dict(l=0, r=0, t=45, b=0))
    return fig


static_data, static_errors = load_static_data()

metrics_df = static_data["metrics"]
comparison_df = static_data["comparison"]
latest_df = prepare_latest_risk(static_data["latest_risk"])
intervals_df = static_data["intervals"]
risk_by_oblast_df = prepare_risk_by_oblast(static_data["risk_by_oblast"], latest_df)
risk_by_hour_df = prepare_risk_by_hour(static_data["risk_by_hour"])
prediction_sample_df = static_data["prediction_sample"]

raion_ref = build_raion_reference(latest_df, intervals_df)
model, model_status = load_model()

st.sidebar.title("📡 Air Raid Forecast")
st.sidebar.caption("Live + historical dashboard")

auto_refresh = st.sidebar.checkbox("Автооновлення кожні 15 хв", value=True)
force_refresh = st.sidebar.button("Оновити live зараз", type="primary")

if auto_refresh:
    if st_autorefresh is not None:
        st_autorefresh(interval=LIVE_REFRESH_MINUTES * 60 * 1000, key="live_autorefresh")
    else:
        st.sidebar.warning("Для автооновлення встанови streamlit-autorefresh. Зараз працює ручне оновлення.")

if static_errors:
    with st.sidebar.expander("Проблеми з файлами"):
        for err in static_errors:
            st.warning(err)

live_snapshot, live_status = maybe_collect_live_snapshot(raion_ref, force=force_refresh)
live_log = read_live_log()

if not live_snapshot.empty:
    live_features = build_live_features(live_snapshot, live_log)
    live_risk, prediction_status = predict_live_risk(
        live_features,
        model,
        model_status,
        latest_df,
        risk_by_oblast_df,
    )
else:
    live_features = pd.DataFrame()
    live_risk = pd.DataFrame()
    prediction_status = "no live snapshot"

live_oblast_summary = build_live_oblast_summary(live_risk)

st.sidebar.info(live_status)
st.sidebar.caption(prediction_status)

oblast_options = sorted(raion_ref["oblast"].dropna().unique()) if not raion_ref.empty else []
selected_oblast = st.sidebar.selectbox("Область", oblast_options) if oblast_options else None

if selected_oblast:
    raion_options = sorted(raion_ref.loc[raion_ref["oblast"] == selected_oblast, "raion"].dropna().unique())
else:
    raion_options = []

selected_raion = st.sidebar.selectbox("Район", raion_options) if raion_options else None

st.title("📡 Ukraine Air Raid Raion Forecast")
st.markdown(
    """
<div class="small-note">
Додаток показує live статус активних тривог, збирає знімки кожні 15 хвилин і дає приблизний risk score старту нової тривоги на 60 хвилин.
Risk score — це оцінка ризику для ранжування, не точна ймовірність і не офіційне попередження.
</div>
""",
    unsafe_allow_html=True,
)

m1, m2, m3, m4, m5, m6 = st.columns(6)

current_ts = live_snapshot["timestamp"].max() if not live_snapshot.empty and "timestamp" in live_snapshot.columns else None
active_raions = int(live_snapshot["is_alert_active"].sum()) if not live_snapshot.empty else 0
active_oblasts = int(live_snapshot.loc[live_snapshot["is_alert_active"] == 1, "oblast"].nunique()) if not live_snapshot.empty else 0
live_log_points = int(live_log["timestamp"].nunique()) if not live_log.empty and "timestamp" in live_log.columns else 0
top_score = live_risk["risk_score"].max() if not live_risk.empty and "risk_score" in live_risk.columns else np.nan

m1.metric("Live timestamp", str(current_ts) if current_ts is not None else "—")
m2.metric("Активні райони", active_raions)
m3.metric("Активні області", active_oblasts)
m4.metric("Live знімків", live_log_points)
m5.metric("Max live risk", fmt_score(top_score))
m6.metric("Test PR AUC", "0.330")

tabs = st.tabs(
    [
        "Live моніторинг",
        "Прогноз району",
        "Інфографіка",
        "Історія району",
        "Оцінка моделі",
        "Дані і обмеження",
    ]
)

with tabs[0]:
    st.subheader("Live моніторинг")

    st.markdown(
        """
<div class="danger-box">
Це не офіційна карта тривог і не система безпеки. Для реальних рішень використовуй тільки офіційні джерела.
</div>
""",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.35, 1.0])

    with left:
        if not live_oblast_summary.empty:
            st.plotly_chart(
                build_oblast_panel(live_oblast_summary, "active_share", "Live: частка активних районів по областях"),
                use_container_width=True,
            )
        elif not risk_by_oblast_df.empty:
            st.plotly_chart(
                build_oblast_panel(risk_by_oblast_df, "avg_predicted_risk", "Fallback: історичний risk score по областях"),
                use_container_width=True,
            )
        else:
            st.warning("Немає даних для мапи/панелі.")

    with right:
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        st.subheader("Активні зараз")

        if live_risk.empty:
            st.warning("Live risk snapshot порожній.")
        else:
            active_now = live_risk[live_risk["is_alert_active"] == 1].copy()
            cols = [c for c in ["timestamp", "oblast", "raion", "source_level_now"] if c in active_now.columns]
            st.dataframe(active_now[cols].head(40), use_container_width=True, hide_index=True)

        st.markdown("</div>", unsafe_allow_html=True)

    st.plotly_chart(chart_top_live_risks(live_risk, 25), use_container_width=True)
    st.plotly_chart(chart_live_active_history(live_log), use_container_width=True)

with tabs[1]:
    st.subheader("Прогноз району на 60 хвилин")

    if not selected_oblast or not selected_raion:
        st.info("Вибери область і район у боковому меню.")
    elif live_risk.empty:
        st.warning("Live risk snapshot ще не створено.")
    else:
        row = live_risk[
            (live_risk["oblast"] == selected_oblast)
            & (live_risk["raion"] == selected_raion)
        ]

        if row.empty:
            st.warning("Для вибраного району немає live-рядка.")
        else:
            row = row.iloc[0]
            is_active = int(row.get("is_alert_active", 0))
            score = row.get("risk_score", np.nan)
            level = row.get("risk_level", "unknown")
            rank = row.get("risk_rank", "—")
            mode = row.get("prediction_mode", "—")

            a, b, c, d = st.columns(4)
            a.metric("Область", selected_oblast)
            b.metric("Район", selected_raion)
            c.metric("Статус", "активна тривога" if is_active else "немає активної")
            d.metric("Risk rank", str(rank))

            if is_active:
                st.markdown(
                    f"""
<div class="danger-box">
У вибраному районі зараз активна тривога. Модель тренувалась прогнозувати <b>старт нової тривоги</b>,
тому для активного району risk score старту не показується.
</div>
""",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
<div class="main-card">
<h3>Risk score на 60 хв: {fmt_score(score)}</h3>
Рівень ризику: {risk_html(level)}<br>
Режим прогнозу: <code>{mode}</code><br><br>
Це приблизна оцінка ризику, побудована на моментних live-даних і live-історії, яку додаток накопичує під час роботи.
</div>
""",
                    unsafe_allow_html=True,
                )

                if st.button("Спрогнозувати ризик на 60 хв", type="primary"):
                    st.success(
                        f"{selected_raion}: risk score = {fmt_score(score)}. "
                        f"Інтерпретація: {risk_ua(level)} ризик старту нової тривоги протягом наступних 60 хвилин."
                    )

            with st.expander("Показати feature row для live-прогнозу"):
                show_cols = [
                    "timestamp",
                    "oblast",
                    "raion",
                    "is_alert_active",
                    "source_level_now",
                    "hour",
                    "day_of_week",
                    "is_night",
                    "starts_last_1h",
                    "starts_last_3h",
                    "starts_last_24h",
                    "active_minutes_last_1h",
                    "active_minutes_last_3h",
                    "active_minutes_last_24h",
                    "oblast_active_raions_now",
                    "oblast_active_share_now",
                    "live_history_hours_available",
                    "risk_score",
                    "risk_level",
                    "prediction_mode",
                ]
                show_cols = [c for c in show_cols if c in live_risk.columns]
                st.dataframe(live_risk.loc[[row.name], show_cols], use_container_width=True, hide_index=True)

with tabs[2]:
    st.subheader("Інфографіка")

    col1, col2 = st.columns(2)

    with col1:
        if not live_oblast_summary.empty:
            plot = live_oblast_summary.sort_values("active_share", ascending=True).tail(15)
            fig = px.bar(
                plot,
                x="active_share",
                y="oblast",
                orientation="h",
                color="active_share",
                color_continuous_scale=["#22c55e", "#eab308", "#f97316", "#ef4444"],
                title="Live: частка активних районів по областях",
                labels={"active_share": "Active share", "oblast": "Область"},
            )
            fig.update_layout(template="plotly_dark", height=500, margin=dict(l=0, r=0, t=45, b=0))
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if not live_oblast_summary.empty:
            plot = live_oblast_summary.dropna(subset=["avg_live_risk"]).sort_values("avg_live_risk", ascending=True).tail(15)
            fig = px.bar(
                plot,
                x="avg_live_risk",
                y="oblast",
                orientation="h",
                color="avg_live_risk",
                color_continuous_scale=["#22c55e", "#eab308", "#f97316", "#ef4444"],
                title="Live: середній risk score неактивних районів",
                labels={"avg_live_risk": "Risk score", "oblast": "Область"},
            )
            fig.update_layout(template="plotly_dark", height=500, margin=dict(l=0, r=0, t=45, b=0))
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    col3, col4 = st.columns(2)

    with col3:
        if not risk_by_oblast_df.empty:
            plot = risk_by_oblast_df.sort_values("avg_predicted_risk", ascending=True).tail(15)
            fig = px.bar(
                plot,
                x="avg_predicted_risk",
                y="oblast",
                orientation="h",
                color="avg_predicted_risk",
                color_continuous_scale=["#22c55e", "#eab308", "#f97316", "#ef4444"],
                title="Історично: середній risk score по областях",
                labels={"avg_predicted_risk": "Risk score", "oblast": "Область"},
            )
            fig.update_layout(template="plotly_dark", height=500, margin=dict(l=0, r=0, t=45, b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Немає risk_by_oblast.csv")

    with col4:
        if not risk_by_oblast_df.empty and "actual_positive_rate" in risk_by_oblast_df.columns:
            plot = risk_by_oblast_df.dropna(subset=["actual_positive_rate"]).sort_values("actual_positive_rate", ascending=True).tail(15)
            fig = px.bar(
                plot,
                x="actual_positive_rate",
                y="oblast",
                orientation="h",
                color="actual_positive_rate",
                color_continuous_scale=["#334155", "#eab308", "#ef4444"],
                title="Історично: фактична частка positive target",
                labels={"actual_positive_rate": "Positive rate", "oblast": "Область"},
            )
            fig.update_layout(template="plotly_dark", height=500, margin=dict(l=0, r=0, t=45, b=0))
            st.plotly_chart(fig, use_container_width=True)

    st.plotly_chart(chart_risk_by_hour(risk_by_hour_df), use_container_width=True)

    if not live_risk.empty:
        risk_values = live_risk.loc[live_risk["is_alert_active"] == 0, "risk_score"].dropna()
        if not risk_values.empty:
            fig = px.histogram(
                risk_values,
                nbins=30,
                title="Live: розподіл risk score серед неактивних районів",
                labels={"value": "Risk score", "count": "Кількість"},
            )
            fig.update_layout(template="plotly_dark", height=420, margin=dict(l=0, r=0, t=45, b=0))
            st.plotly_chart(fig, use_container_width=True)

with tabs[3]:
    st.subheader("Історія вибраного району")

    if not selected_oblast or not selected_raion:
        st.info("Вибери область і район у боковому меню.")
    elif intervals_df.empty:
        st.warning("Немає data/interim/raion_alert_intervals.csv")
    else:
        df = intervals_df[
            (intervals_df["oblast"] == selected_oblast)
            & (intervals_df["raion"] == selected_raion)
        ].copy()

        if df.empty:
            st.warning("Для вибраного району немає історичних інтервалів.")
        else:
            df["started_at"] = pd.to_datetime(df["started_at"], errors="coerce", utc=True)
            df["finished_at"] = pd.to_datetime(df["finished_at"], errors="coerce", utc=True)

            if "duration_min" not in df.columns:
                df["duration_min"] = (df["finished_at"] - df["started_at"]).dt.total_seconds() / 60

            latest_alerts = df.sort_values("started_at", ascending=False)
            show_cols = [c for c in ["started_at", "finished_at", "duration_min", "source_level"] if c in latest_alerts.columns]

            a, b, c = st.columns(3)
            a.metric("Історичних інтервалів", len(df))
            b.metric("Середня тривалість", f"{df['duration_min'].mean():.1f} хв")
            c.metric("Макс. тривалість", f"{df['duration_min'].max():.1f} хв")

            st.subheader("Останні 10 тривог")
            st.dataframe(latest_alerts[show_cols].head(10), use_container_width=True, hide_index=True)

            st.plotly_chart(chart_raion_hour_starts(intervals_df, selected_oblast, selected_raion), use_container_width=True)

            if "source_level" in df.columns:
                counts = df["source_level"].fillna("unknown").value_counts().reset_index()
                counts.columns = ["source_level", "count"]
                fig = px.pie(
                    counts,
                    names="source_level",
                    values="count",
                    title="Типи джерела історичних інтервалів району",
                )
                fig.update_layout(template="plotly_dark", height=420)
                st.plotly_chart(fig, use_container_width=True)

with tabs[4]:
    st.subheader("Оцінка моделі")

    a, b, c, d, e, f = st.columns(6)
    a.metric("PR AUC", "0.330244")
    b.metric("ROC AUC", "0.812334")
    c.metric("Brier", "0.168278")
    d.metric("F1@0.5", "0.351705")
    e.metric("Precision top 10%", "0.362957")
    f.metric("Recall top 10%", "0.403474")

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
        st.warning("Немає final_model_metrics.csv")
    else:
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    st.subheader("Порівняння моделей")
    if comparison_df.empty:
        st.warning("Немає final_model_comparison.csv")
    else:
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        if {"model", "pr_auc"}.issubset(comparison_df.columns):
            plot = comparison_df.copy()
            if "split" in plot.columns:
                test_plot = plot[plot["split"].astype(str).str.lower() == "test"].copy()
                if not test_plot.empty:
                    plot = test_plot

            fig = px.bar(
                plot.sort_values("pr_auc", ascending=True),
                x="pr_auc",
                y="model",
                orientation="h",
                color="pr_auc",
                color_continuous_scale=["#334155", "#eab308", "#22c55e"],
                title="PR AUC моделей",
                labels={"pr_auc": "PR AUC", "model": "Модель"},
            )
            fig.update_layout(template="plotly_dark", height=520, margin=dict(l=0, r=0, t=45, b=0))
            st.plotly_chart(fig, use_container_width=True)

with tabs[5]:
    st.subheader("Дані і обмеження")

    st.markdown(
        """
<div class="main-card">
<h4>Що робить live mode</h4>
Додаток кожні 15 хвилин отримує активні тривоги з API, приводить їх до районної сітки,
зберігає знімок у <code>data/live/live_raion_snapshots.csv</code> і будує приблизний live risk score.
</div>

<div class="main-card">
<h4>Чому прогноз приблизний</h4>
Фінальна модель тренувалась на історичних features. У live mode перші години роботи доступні тільки моментні дані.
Rolling features за 1/3/24 години стають кращими тільки після накопичення live log.
</div>

<div class="main-card">
<h4>Критичні обмеження</h4>
<ul>
<li>Це не офіційна система безпеки.</li>
<li>Це не заміна офіційних повітряних тривог.</li>
<li>Модель дає <b>risk score</b>, а не точну ймовірність.</li>
<li>Для активних районів прогноз старту нової тривоги не показується, бо тривога вже активна.</li>
<li>Датасет є <b>raion-level proxy dataset</b>.</li>
<li>Частина історичних районних рядків успадкована з рівня області.</li>
<li>Якщо модельний файл не знайдено, додаток використовує fallback scoring.</li>
</ul>
</div>
""",
        unsafe_allow_html=True,
    )

    st.subheader("Стан файлів")
    files = pd.DataFrame(
        [
            {"file": str(path), "exists": path.exists()}
            for path in list(DATA.values()) + [LIVE_SNAPSHOT_PATH, LIVE_RISK_PATH, LIVE_RAW_LAST_PATH]
        ]
    )
    st.dataframe(files, use_container_width=True, hide_index=True)

    st.subheader("Live API status")
    st.code(live_status)
    st.code(prediction_status)

    if LIVE_RAW_LAST_PATH.exists():
        with st.expander("Остання raw API відповідь"):
            try:
                st.json(json.loads(LIVE_RAW_LAST_PATH.read_text(encoding="utf-8")))
            except Exception:
                st.text(LIVE_RAW_LAST_PATH.read_text(encoding="utf-8")[:5000])

    if not live_log.empty:
        with st.expander("Live log sample"):
            st.dataframe(live_log.tail(300), use_container_width=True, hide_index=True)

    if not live_risk.empty:
        with st.expander("Live risk snapshot"):
            st.dataframe(live_risk, use_container_width=True, hide_index=True)