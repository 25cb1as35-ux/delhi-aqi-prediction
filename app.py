"""
Delhi AQI Oracle - redesigned Streamlit dashboard
--------------------------------------------------
* Cinematic entry page with drifting clouds
* Sticky glass top-navigation bar (no sidebar)
* Blurred Delhi backdrop + slowly floating clouds on every page
* Animated AQI hero block (pollution / clean-air artwork + floating dust particles)
* Restyled map, forecast, bar, calendar & heat-map charts
* Health advisories whose animations react to the live AQI
"""
import os
import io
import base64
import random
import pickle
import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
import pydeck as pdk
import plotly.graph_objects as go
import streamlit as st

try:
    from pymongo import MongoClient
except ImportError:  # app still works (without logging) if pymongo is missing
    MongoClient = None

# =====================================================================
# 1. PAGE SETUP
# =====================================================================
st.set_page_config(
    page_title="Delhi AQI Oracle",
    layout="wide",
    page_icon="🌫️",
    initial_sidebar_state="collapsed",
)

ROOT = Path(__file__).parent if "__file__" in globals() else Path.cwd()
OUT = ROOT / "outputs"
ASSETS = ROOT / "assets"

PAGES = ["Live Dashboard", "Historical Trends", "Model Accuracy", "Health Advisories"]
PAGE_ICONS = {"Live Dashboard": "🏠", "Historical Trends": "📈",
              "Model Accuracy": "🎯", "Health Advisories": "🩺"}

for k, v in {"entered": False, "page": PAGES[0]}.items():
    st.session_state.setdefault(k, v)


# =====================================================================
# 2. SECRETS & DATABASE
# =====================================================================
def _secret(name, default=""):
    try:
        return st.secrets[name]
    except Exception:
        return os.environ.get(name, default)


# REPLACE THESE (or use .streamlit/secrets.toml) FOR LOCAL TESTING
WAQI_TOKEN = _secret("WAQI_TOKEN", "YOUR_WAQI_TOKEN_HERE")
MONGO_URL = _secret("MONGO_URL", "YOUR_MONGO_URL_HERE")


def _is_configured(value):
    return bool(value) and "YOUR_" not in str(value)


@st.cache_resource(show_spinner=False)
def get_database():
    """Returns the MongoDB collection, or None if Mongo is not reachable."""
    if MongoClient is None or not _is_configured(MONGO_URL):
        return None
    try:
        client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=4000)
        client.admin.command("ping")
        return client.aqi_database.predictions
    except Exception:
        return None


predictions_collection = get_database()

LOCATIONS_DATA = {
    "Anand Vihar": {"coords": (28.6469, 77.3160), "pincode": "110092", "file": "anand_vihar.xlsx"},
    "Dwarka": {"coords": (28.5921, 77.0460), "pincode": "110075", "file": "dwarka.xlsx"},
    "IGI Airport": {"coords": (28.5562, 77.1000), "pincode": "110037", "file": "igi_airport.xlsx"},
    "Okhla Phase III": {"coords": (28.5432, 77.2725), "pincode": "110020", "file": "okhla.xlsx"},
    "Rohini": {"coords": (28.7320, 77.1123), "pincode": "110085", "file": "rohini.xlsx"},
}
ALL_LOCATIONS = list(LOCATIONS_DATA.keys())
st.session_state.setdefault("station", ALL_LOCATIONS[0])


# =====================================================================
# 3. MODELS & MATH  (your original logic)
# =====================================================================
@st.cache_resource(show_spinner=False)
def load_models():
    try:
        with open(OUT / "best_model_1h_final.pkl", "rb") as f: b1 = pickle.load(f)
        with open(OUT / "best_model_6h_final.pkl", "rb") as f: b6 = pickle.load(f)
        with open(OUT / "best_model_24h_final.pkl", "rb") as f: b24 = pickle.load(f)
        return b1["model"], b1["columns"], b6["model"], b6["columns"], b24["model"], b24["columns"]
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def load_clean():
    return pd.read_pickle(OUT / "clean.pkl")


def calculate_cpcb_aqi(pm25, pm10):
    if pm25 <= 30: aqi_25 = pm25 * 50 / 30
    elif pm25 <= 60: aqi_25 = 50 + (pm25 - 30) * 50 / 30
    elif pm25 <= 90: aqi_25 = 100 + (pm25 - 60) * 100 / 30
    elif pm25 <= 120: aqi_25 = 200 + (pm25 - 90) * 100 / 30
    elif pm25 <= 250: aqi_25 = 300 + (pm25 - 120) * 100 / 130
    else: aqi_25 = 400 + (pm25 - 250) * 100 / 130
    if pm10 <= 50: aqi_10 = pm10 * 50 / 50
    elif pm10 <= 100: aqi_10 = 50 + (pm10 - 50) * 50 / 50
    elif pm10 <= 250: aqi_10 = 100 + (pm10 - 100) * 100 / 150
    elif pm10 <= 350: aqi_10 = 200 + (pm10 - 250) * 100 / 100
    elif pm10 <= 430: aqi_10 = 300 + (pm10 - 350) * 100 / 80
    else: aqi_10 = 400 + (pm10 - 430) * 100 / 80
    return max(aqi_25, aqi_10)


def get_theme(aqi):
    if aqi <= 50: return "#00e400", "Good"
    if aqi <= 100: return "#ffff00", "Moderate"
    if aqi <= 200: return "#ff7e00", "Poor"
    if aqi <= 300: return "#ff0000", "Very Poor"
    return "#8f3f97", "Severe"


def ink(aqi):
    """Readable text colour on top of the AQI colour."""
    return "#111827" if aqi <= 200 else "#ffffff"


def hex_rgb(h):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)]


HEADLINES = {
    "Good": "The air is clean and fresh. A perfect day to be outdoors!",
    "Moderate": "Air is acceptable. Very sensitive people should take it easy.",
    "Poor": "Breathing discomfort is likely for sensitive groups on long exposure.",
    "Very Poor": "Breathing this air for long can cause respiratory illness.",
    "Severe": "Hazardous air. It affects healthy people too. Stay indoors.",
}


# =====================================================================
# 4. FETCH & PREDICT
# =====================================================================
def _fetch_station(name, info, wx_now, wx_6h, wx_24h):
    lat, lon = info["coords"]
    # --- Primary source: WAQI (your original logic) ---
    if _is_configured(WAQI_TOKEN):
        try:
            url = f"https://api.waqi.info/feed/geo:{lat};{lon}/?token={WAQI_TOKEN}"
            res = requests.get(url, timeout=4).json()
            if res["status"] == "ok":
                d = res["data"]
                live_temp = float(d["iaqi"].get("t", {}).get("v", 25.0))
                live_wind = float(d["iaqi"].get("w", {}).get("v", 5.0))
                pm25 = float(d["iaqi"].get("pm25", {}).get("v", 0))
                pm10 = float(d["iaqi"].get("pm10", {}).get("v", 0))
                return name, {
                    "aqi": calculate_cpcb_aqi(pm25, pm10), "temp": live_temp, "wind": live_wind,
                    "pm25": pm25, "pm10": pm10, "wx_6h": wx_6h, "wx_24h": wx_24h, "source": "WAQI",
                }
        except Exception:
            pass
    # --- Fallback source: Open-Meteo air quality (free, no key needed) ---
    try:
        aq = requests.get(
            "https://air-quality-api.open-meteo.com/v1/air-quality",
            params={"latitude": lat, "longitude": lon, "current": "pm10,pm2_5"}, timeout=6,
        ).json()["current"]
        pm25, pm10 = float(aq["pm2_5"]), float(aq["pm10"])
        return name, {
            "aqi": calculate_cpcb_aqi(pm25, pm10), "temp": wx_now["temp"], "wind": wx_now["wind"],
            "pm25": pm25, "pm10": pm10, "wx_6h": wx_6h, "wx_24h": wx_24h, "source": "Open-Meteo",
        }
    except Exception:
        return name, None


@st.cache_data(ttl=900, show_spinner=False)
def get_live_data():
    try:
        wx_url = ("https://api.open-meteo.com/v1/forecast?latitude=28.6139&longitude=77.2090"
                  "&hourly=temperature_2m,wind_speed_10m&current=temperature_2m,wind_speed_10m&forecast_days=2")
        wx_json = requests.get(wx_url, timeout=6).json()
        wx_res = wx_json["hourly"]
        wx_6h = {"temp": float(wx_res["temperature_2m"][6]), "wind": float(wx_res["wind_speed_10m"][6])}
        wx_24h = {"temp": float(wx_res["temperature_2m"][24]), "wind": float(wx_res["wind_speed_10m"][24])}
        cur = wx_json.get("current", {})
        wx_now = {"temp": float(cur.get("temperature_2m", 25.0)), "wind": float(cur.get("wind_speed_10m", 5.0))}
    except Exception:
        wx_now = wx_6h = wx_24h = {"temp": 25.0, "wind": 5.0}

    with ThreadPoolExecutor(max_workers=5) as pool:  # fetch all stations in parallel
        results = list(pool.map(lambda kv: _fetch_station(kv[0], kv[1], wx_now, wx_6h, wx_24h), LOCATIONS_DATA.items()))
    data = {name: d for name, d in results if d}
    return data, datetime.datetime.now().strftime("%d %b %Y, %I:%M %p")


def _parse_dates(col):
    """CPCB exports use dd-mm-yyyy. Use day-first parsing only when it reads more rows
    than the default parser (so files that already worked behave exactly as before)."""
    if pd.api.types.is_datetime64_any_dtype(col):
        return col
    default = pd.to_datetime(col, errors="coerce")
    try:
        dayfirst = pd.to_datetime(col.astype(str), errors="coerce", dayfirst=True, format="mixed")
    except Exception:
        return default
    return dayfirst if dayfirst.notna().sum() > default.notna().sum() else default


@st.cache_data(ttl=3600, show_spinner=False)
def load_station_excel(file_name):
    file_path = ROOT / file_name
    if not file_path.exists(): return None

    df_raw = pd.read_excel(file_path, header=None)
    header_row = 0
    for i in range(min(15, len(df_raw))):
        row_str = str(df_raw.iloc[i].values).lower()
        if 'date' in row_str and ('pm2.5' in row_str or 'pm25' in row_str):
            header_row = i
            break

    df = pd.read_excel(file_path, header=header_row)
    df.columns = [str(c).lower().strip() for c in df.columns]
    time_col = [c for c in df.columns if 'date' in c or 'time' in c][0]
    pm25_col = [c for c in df.columns if 'pm2.5' in c or 'pm25' in c][0]

    df = df[[time_col, pm25_col]].rename(columns={time_col: "dt", pm25_col: "pm2_5"})
    df["dt"] = _parse_dates(df["dt"])
    df["pm2_5"] = pd.to_numeric(df["pm2_5"], errors='coerce')
    df = df.dropna()
    df = df[df["pm2_5"] > 0]

    df = df.set_index("dt").resample("1h").mean().reset_index()
    df["aqi_index"] = df["pm2_5"].apply(lambda x: calculate_cpcb_aqi(x, 0))
    return df


@st.cache_data(ttl=900, show_spinner=False)
def predict(station_name, live_weather, _m1, _c1, _m6, _c6, _m24, _c24):
    """Your original prediction logic.  Cached for 15 min per live reading so that
    clicking through the nav bar doesn't log duplicate rows into MongoDB."""
    m1, c1, m6, c6, m24, c24 = _m1, _c1, _m6, _c6, _m24, _c24
    clean = load_clean()
    df = clean.loc[station_name].copy()

    for col in ['aqi_index', 'temp_c', 'humidity', 'windspeed_kph']:
        if col in df.columns: df[col] = df[col].astype(float)

    for i in range(1, 25):
        if len(df) >= i:
            df.iloc[-i, df.columns.get_loc('aqi_index')] = live_weather['aqi']
            df.iloc[-i, df.columns.get_loc('temp_c')] = live_weather['temp']
            df.iloc[-i, df.columns.get_loc('windspeed_kph')] = live_weather['wind']

    g = df.copy()
    for L in (1, 3, 6, 12, 24): g[f"aqi_lag{L}"] = g.aqi_index.shift(L)
    t = g.index
    g["hour"] = t.hour; g["month"] = t.month; g["dow"] = t.dayofweek
    g["hour_sin"] = np.sin(2 * np.pi * t.hour / 24); g["hour_cos"] = np.cos(2 * np.pi * t.hour / 24)

    last_row = g.iloc[[-1]].copy()
    for loc in ALL_LOCATIONS:
        last_row[f"location_{loc}"] = 1 if loc == station_name else 0
    last_row = last_row.drop(columns=["location"], errors="ignore")

    def run_pred(model, cols, horizon, f_wind, f_temp):
        last_row[f"future_wind_{horizon}h"] = f_wind
        last_row[f"future_temp_{horizon}h"] = f_temp
        X = last_row.reindex(columns=cols, fill_value=0).select_dtypes(include=[np.number])
        return float(model.predict(X)[0])

    p1 = run_pred(m1, c1, 1, live_weather['wind'], live_weather['temp'])
    p6 = run_pred(m6, c6, 6, live_weather['wx_6h']['wind'], live_weather['wx_6h']['temp'])
    p24 = run_pred(m24, c24, 24, live_weather['wx_24h']['wind'], live_weather['wx_24h']['temp'])

    # SAVE TO MONGODB
    db_msg = None
    if predictions_collection is not None:
        try:
            predictions_collection.insert_one({
                "timestamp": datetime.datetime.now(),
                "location": station_name,
                "actual_aqi": live_weather['aqi'],
                "pred_1h": p1, "pred_6h": p6, "pred_24h": p24,
            })
        except Exception as e:
            db_msg = f"Failed to save to database: {e}"
    return p1, p6, p24, db_msg


# =====================================================================
# 5. VISUAL HELPERS
# =====================================================================
def html(markup):
    """Render raw HTML. Strips indentation & blank lines so Markdown never
    turns our HTML into code blocks."""
    clean = "\n".join(line.strip() for line in markup.splitlines() if line.strip())
    st.markdown(clean, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def img_uri(name, blur=0, max_w=1600):
    path = ASSETS / name
    if not path.exists():
        return None
    try:
        from PIL import Image, ImageFilter
        im = Image.open(path).convert("RGB")
        im.thumbnail((max_w, max_w))
        if blur:
            im = im.filter(ImageFilter.GaussianBlur(blur))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=80)
        data = buf.getvalue()
    except Exception:
        data = path.read_bytes()
    return "data:image/jpeg;base64," + base64.b64encode(data).decode()


CLOUD_SVG = (
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 400 200'><defs>"
    "<filter id='b' x='-20%' y='-20%' width='140%' height='140%'><feGaussianBlur stdDeviation='7'/></filter>"
    "<radialGradient id='g' cx='50%' cy='35%' r='65%'><stop offset='0' stop-color='#ffffff'/>"
    "<stop offset='1' stop-color='#d6e0ef'/></radialGradient></defs>"
    "<g filter='url(#b)' fill='url(#g)'>"
    "<ellipse cx='120' cy='130' rx='85' ry='45'/><ellipse cx='195' cy='100' rx='85' ry='68'/>"
    "<ellipse cx='275' cy='128' rx='88' ry='48'/><ellipse cx='175' cy='142' rx='130' ry='38'/>"
    "<ellipse cx='245' cy='82' rx='55' ry='48'/><ellipse cx='140' cy='98' rx='48' ry='40'/></g></svg>"
)
CLOUD_URI = "data:image/svg+xml;base64," + base64.b64encode(CLOUD_SVG.encode()).decode()


def sky_layer(n=8, seed=3, strength=0.55, top_range=(2, 70), size=(220, 520)):
    rng = random.Random(seed)
    clouds = []
    for _ in range(n):
        dur = rng.uniform(80, 170)
        clouds.append(
            f'<div class="cloud" style="top:{rng.uniform(*top_range):.1f}%;'
            f'width:{rng.uniform(*size):.0f}px;opacity:{rng.uniform(0.45, 1.0) * strength:.2f};'
            f'animation-duration:{dur:.0f}s;animation-delay:-{rng.uniform(0, dur):.0f}s"></div>'
        )
    return '<div class="sky">' + "".join(clouds) + "</div>"


def particles(n, seed=7, rgb="255,214,170", max_size=6.0, rise=140):
    rng = random.Random(seed)
    dots = []
    for _ in range(int(n)):
        s = rng.uniform(1.5, max_size)
        o = rng.uniform(0.35, 0.95)
        d = rng.uniform(7, 20)
        dots.append(
            f'<span class="pt" style="left:{rng.uniform(0, 100):.1f}%;top:{rng.uniform(5, 100):.1f}%;'
            f'width:{s:.1f}px;height:{s:.1f}px;--dx:{rng.uniform(-70, 70):.0f}px;--rise:-{rise}px;'
            f'--o:{o:.2f};background:rgba({rgb},{o:.2f});box-shadow:0 0 {s * 2:.0f}px rgba({rgb},.8);'
            f'animation-duration:{d:.1f}s;animation-delay:-{rng.uniform(0, d):.1f}s"></span>'
        )
    return '<div class="particles">' + "".join(dots) + "</div>"


def section(title, subtitle="", icon=""):
    html(f"""
    <div class="section-title fade-up">
        <span class="section-icon">{icon}</span>
        <div><h2>{title}</h2><p>{subtitle}</p></div>
    </div>""")


def stat_card(label, value, sub="", color="#38bdf8", delay=0):
    return (f'<div class="stat fade-up" style="--accent:{color};animation-delay:{delay}ms">'
            f'<div class="stat-label">{label}</div><div class="stat-value">{value}</div>'
            f'<div class="stat-sub">{sub}</div></div>')


def style_fig(fig, height=360):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1", family="Poppins, Inter, sans-serif", size=12),
        hoverlabel=dict(bgcolor="rgba(15,23,42,.95)", bordercolor="rgba(255,255,255,.15)",
                        font=dict(color="white", family="Poppins, sans-serif")),
        legend=dict(orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="rgba(255,255,255,.05)", zeroline=False, linecolor="rgba(255,255,255,.1)"),
        yaxis=dict(gridcolor="rgba(255,255,255,.07)", zeroline=False),
    )
    return fig


def show_chart(fig, key=None):
    st.plotly_chart(fig.to_dict(), config={"displayModeBar": False}, key=key)


def wide_button(label, **kw):
    try:
        return st.button(label, width="stretch", **kw)
    except TypeError:  # older Streamlit
        return st.button(label, use_container_width=True, **kw)


AQI_SCALE = [[0.0, "#00e400"], [0.1, "#a3e635"], [0.2, "#ffff00"], [0.4, "#ff7e00"],
             [0.6, "#ff0000"], [0.8, "#8f3f97"], [1.0, "#7e0023"]]
BANDS = [(0, 50, "#00e400"), (50, 100, "#ffff00"), (100, 200, "#ff7e00"),
         (200, 300, "#ff0000"), (300, 500, "#8f3f97")]


# =====================================================================
# 6. GLOBAL CSS
# =====================================================================
GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp, .stMarkdown, button, input { font-family: 'Poppins', 'Inter', 'Segoe UI', sans-serif; }
[data-testid="stHeader"], [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
[data-testid="stToolbar"], [data-testid="stDecoration"], footer, #MainMenu { display: none !important; }
.stApp { background: #050914; color: #e5e7eb; }
[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, rgba(4,8,20,__OV1__) 0%, rgba(4,8,20,__OV2__) 55%, rgba(4,8,20,.93) 100%), url('__BG__') center / cover fixed;
}
[data-testid="stMainBlockContainer"], .block-container { position: relative; z-index: 1; padding-top: 1rem !important; padding-bottom: 3rem; max-width: 1320px; }
[data-testid="stElementContainer"]:has(.global-layer) { position: absolute; }
/* ---------- clouds ---------- */
.sky { position: fixed; inset: 0; z-index: -1; pointer-events: none; overflow: hidden; }
.cloud { position: absolute; left: 0; aspect-ratio: 2 / 1; background: url('__CLOUD__') center / contain no-repeat;
         animation-name: drift; animation-timing-function: linear; animation-iteration-count: infinite; will-change: transform; }
@keyframes drift { from { transform: translateX(-45vw); } to { transform: translateX(115vw); } }
/* ---------- particles ---------- */
.particles { position: absolute; inset: 0; overflow: hidden; pointer-events: none; }
.pt { position: absolute; border-radius: 50%; animation-name: ptFloat; animation-timing-function: linear; animation-iteration-count: infinite; }
@keyframes ptFloat { 0% { transform: translate(0,0) scale(1); opacity: 0; } 15% { opacity: var(--o); }
                     85% { opacity: var(--o); } 100% { transform: translate(var(--dx), var(--rise)) scale(.5); opacity: 0; } }
/* ---------- generic animations ---------- */
.fade-up { animation: fadeUp .8s cubic-bezier(.2,.7,.2,1) both; }
@keyframes fadeUp { from { opacity: 0; transform: translateY(24px); } to { opacity: 1; transform: none; } }
@keyframes pulseDot { 0% { box-shadow: 0 0 0 0 rgba(74,222,128,.7); } 70% { box-shadow: 0 0 0 10px rgba(74,222,128,0); } 100% { box-shadow: 0 0 0 0 rgba(74,222,128,0); } }
@keyframes kenburns { from { transform: scale(1.02); } to { transform: scale(1.14) translateX(-2%); } }
@keyframes ringFill { from { stroke-dashoffset: var(--C); } to { stroke-dashoffset: var(--off); } }
@keyframes growX { from { width: 0; } }
@keyframes slideMarker { from { left: 0; opacity: 0; } }
@keyframes pop { from { opacity: 0; transform: scale(.6); } to { opacity: 1; transform: scale(1); } }
@keyframes shimmer { 0% { background-position: -200% 0; } 100% { background-position: 200% 0; } }
/* ---------- top nav ---------- */
div:has(> .st-key-topnav) { position: sticky; top: .6rem; z-index: 999; }
.st-key-topnav { position: sticky; top: .6rem; z-index: 999; background: rgba(10,15,30,.86); backdrop-filter: blur(18px) saturate(1.4);
    -webkit-backdrop-filter: blur(18px) saturate(1.4); border: 1px solid rgba(255,255,255,.10); border-radius: 20px;
    padding: .55rem 1rem; box-shadow: 0 12px 40px rgba(0,0,0,.35); animation: fadeUp .7s both; }
.brand { display: flex; align-items: center; gap: .65rem; }
.brand-logo { width: 42px; height: 42px; border-radius: 13px; display: grid; place-items: center; font-size: 22px;
    background: linear-gradient(135deg,#38bdf8,#6366f1 60%,#a855f7); box-shadow: 0 6px 18px rgba(99,102,241,.5); }
.brand-name { font-weight: 700; font-size: 1.12rem; color: #fff; line-height: 1.1; }
.brand-sub { font-size: .72rem; color: #94a3b8; letter-spacing: .08em; text-transform: uppercase; }
.st-key-topnav button { background: transparent !important; border: 1px solid transparent !important; color: #cbd5e1 !important;
    border-radius: 12px !important; font-weight: 500 !important; transition: all .25s ease !important; min-height: 2.6rem; }
.st-key-topnav button:hover { background: rgba(255,255,255,.08) !important; color: #fff !important; transform: translateY(-1px); }
.st-key-topnav button[kind="primary"], .st-key-topnav [data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg,#0ea5e9,#6366f1) !important; color: #fff !important;
    box-shadow: 0 6px 20px rgba(99,102,241,.45) !important; }
.st-key-topnav [data-baseweb="select"] > div { background: rgba(255,255,255,.07) !important; border: 1px solid rgba(255,255,255,.14) !important; border-radius: 12px !important; }
/* ---------- cards ---------- */
[class*="st-key-card"] { background: rgba(12,18,34,.58); backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(255,255,255,.08); border-radius: 22px; padding: 1.1rem 1.2rem 1rem; box-shadow: 0 18px 50px rgba(0,0,0,.28);
    animation: fadeUp .8s both; }
.glass { background: rgba(12,18,34,.58); backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(255,255,255,.08); border-radius: 22px; padding: 1.2rem 1.3rem; box-shadow: 0 18px 50px rgba(0,0,0,.28); }
.card-title { font-weight: 600; color: #f1f5f9; font-size: 1.02rem; margin-bottom: .1rem; }
.card-sub { color: #94a3b8; font-size: .82rem; margin-bottom: .4rem; }
.section-title { display: flex; align-items: center; gap: .9rem; margin: 1.8rem 0 .9rem; }
.section-title h2 { margin: 0 !important; padding: 0 !important; font-size: 1.45rem !important; font-weight: 700 !important; color: #fff !important; }
.section-title p { margin: 0; color: #94a3b8; font-size: .88rem; }
.section-icon { width: 46px; height: 46px; border-radius: 14px; display: grid; place-items: center; font-size: 22px;
    background: rgba(255,255,255,.07); border: 1px solid rgba(255,255,255,.1); }
.page-head { margin: 1.4rem 0 .4rem; }
.page-head h1 { font-size: 2.3rem !important; font-weight: 800 !important; margin: 0 !important; padding: 0 !important;
    background: linear-gradient(90deg,#fff,#c7d2fe 60%,#7dd3fc); -webkit-background-clip: text; background-clip: text; color: transparent !important; }
.page-head p { color: #94a3b8; margin: .2rem 0 0; }
.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; }
.stat { position: relative; overflow: hidden; background: rgba(12,18,34,.6); backdrop-filter: blur(14px); border: 1px solid rgba(255,255,255,.08);
    border-radius: 20px; padding: 1.1rem 1.2rem; }
.stat::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px; background: var(--accent); box-shadow: 0 0 18px var(--accent); }
.stat-label { color: #94a3b8; font-size: .8rem; text-transform: uppercase; letter-spacing: .08em; }
.stat-value { color: #fff; font-size: 1.9rem; font-weight: 700; margin-top: .15rem; }
.stat-sub { color: #cbd5e1; font-size: .82rem; }
/* ---------- hero (live AQI) ---------- */
.hero { position: relative; overflow: hidden; border-radius: 28px; min-height: 400px; margin-top: 1.2rem;
    border: 1px solid rgba(255,255,255,.12); box-shadow: 0 30px 80px rgba(0,0,0,.45), 0 0 0 1px rgba(255,255,255,.03) inset; animation: fadeUp .9s both; }
.hero-bg { position: absolute; inset: 0; background-size: cover; background-position: center; animation: kenburns 28s ease-in-out infinite alternate; }
.hero-tint { position: absolute; inset: 0; background: linear-gradient(90deg, rgba(5,8,18,.92) 0%, rgba(5,8,18,.7) 42%, rgba(5,8,18,.15) 75%, rgba(5,8,18,.35) 100%),
    radial-gradient(circle at 80% 50%, color-mix(in srgb, var(--c) 30%, transparent) 0%, transparent 55%); }
.hero-content { position: relative; z-index: 2; display: flex; justify-content: space-between; align-items: center; gap: 2rem; padding: 2.4rem 2.6rem; flex-wrap: wrap; min-height: 400px; }
.live-badge { display: inline-flex; align-items: center; gap: .5rem; padding: .32rem .8rem; border-radius: 999px; font-size: .75rem; font-weight: 600;
    letter-spacing: .1em; color: #bbf7d0; background: rgba(34,197,94,.14); border: 1px solid rgba(74,222,128,.35); }
.live-dot { width: 8px; height: 8px; border-radius: 50%; background: #4ade80; animation: pulseDot 1.8s infinite; }
.hero h1.station { font-size: 2.6rem !important; font-weight: 800 !important; color: #fff !important; margin: .8rem 0 .1rem !important; padding: 0 !important; line-height: 1.1; }
.hero .meta { color: #cbd5e1; font-size: .9rem; }
.cat-pill { display: inline-block; margin: 1rem 0 .6rem; padding: .45rem 1.1rem; border-radius: 999px; font-weight: 700; font-size: 1rem;
    background: var(--c); color: var(--ink); box-shadow: 0 0 30px color-mix(in srgb, var(--c) 60%, transparent); }
.headline { color: #e2e8f0; font-size: 1.05rem; max-width: 470px; }
.chips { display: flex; flex-wrap: wrap; gap: .6rem; margin-top: 1.3rem; }
.chip { background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.14); backdrop-filter: blur(8px); border-radius: 14px; padding: .55rem .85rem; min-width: 98px; }
.chip b { display: block; color: #fff; font-size: 1.05rem; }
.chip span { color: #94a3b8; font-size: .72rem; text-transform: uppercase; letter-spacing: .06em; }
.ring-wrap { position: relative; width: 250px; height: 250px; margin-right: 1rem; }
.ring { width: 250px; height: 250px; transform: rotate(-90deg); filter: drop-shadow(0 0 18px color-mix(in srgb, var(--c) 70%, transparent)); }
.ring-bg { fill: rgba(5,8,18,.55); stroke: rgba(255,255,255,.1); stroke-width: 12; }
.ring-fg { fill: none; stroke-width: 12; stroke-linecap: round; animation: ringFill 1.8s cubic-bezier(.2,.7,.2,1) both; }
.ring-center { position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; }
.ring-center .num { font-size: 4.6rem; font-weight: 800; color: #fff; line-height: 1; text-shadow: 0 0 30px color-mix(in srgb, var(--c) 80%, transparent); }
.ring-center .lbl { color: #cbd5e1; letter-spacing: .25em; font-size: .78rem; margin-top: .3rem; }
.scale { position: relative; margin: 1.1rem .4rem 0; }
.scale-bar { height: 12px; border-radius: 999px; background: linear-gradient(90deg,#00e400 0%,#ffff00 15%,#ff7e00 30%,#ff0000 50%,#8f3f97 70%,#7e0023 100%); box-shadow: 0 0 20px rgba(0,0,0,.4) inset; }
.scale-marker { position: absolute; top: -9px; transform: translateX(-50%); animation: slideMarker 1.6s cubic-bezier(.2,.7,.2,1) both; }
.scale-marker div { width: 30px; height: 30px; border-radius: 50%; background: #fff; border: 4px solid var(--c); box-shadow: 0 0 18px var(--c); }
.scale-labels { display: flex; justify-content: space-between; color: #94a3b8; font-size: .74rem; margin-top: .5rem; }
/* ---------- cigarettes ---------- */
.cig { display: flex; align-items: center; gap: 1.2rem; flex-wrap: wrap; }
.cig-icons span { display: inline-block; font-size: 1.6rem; animation: pop .5s both; }
.cig-num { font-size: 2.2rem; font-weight: 800; color: #fda4af; }
/* ---------- forecast ---------- */
.fc-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; }
.fc { position: relative; overflow: hidden; border-radius: 22px; padding: 1.2rem 1.3rem; background: rgba(12,18,34,.62); backdrop-filter: blur(14px);
    border: 1px solid rgba(255,255,255,.08); transition: transform .3s ease, box-shadow .3s ease; }
.fc:hover { transform: translateY(-5px); box-shadow: 0 18px 40px rgba(0,0,0,.4), 0 0 0 1px var(--c); }
.fc::after { content: ""; position: absolute; right: -40px; top: -40px; width: 130px; height: 130px; border-radius: 50%;
    background: radial-gradient(circle, color-mix(in srgb, var(--c) 45%, transparent), transparent 70%); }
.fc-h { color: #94a3b8; font-size: .8rem; text-transform: uppercase; letter-spacing: .1em; }
.fc-v { font-size: 2.7rem; font-weight: 800; color: #fff; line-height: 1.15; }
.fc-tag { display: inline-block; padding: .18rem .65rem; border-radius: 999px; font-size: .76rem; font-weight: 700; background: var(--c); color: var(--ink); }
.fc-delta { font-weight: 600; font-size: .9rem; margin-left: .5rem; }
.bar { height: 7px; border-radius: 999px; background: rgba(255,255,255,.08); margin-top: .9rem; overflow: hidden; }
.bar > div { height: 100%; border-radius: 999px; background: var(--c); box-shadow: 0 0 12px var(--c); animation: growX 1.4s cubic-bezier(.2,.7,.2,1) both; }
/* ---------- station ranking ---------- */
.rank-row { display: grid; grid-template-columns: 28px 1fr 56px; gap: .7rem; align-items: center; padding: .62rem .2rem; border-bottom: 1px dashed rgba(255,255,255,.07); }
.rank-row.active { background: rgba(255,255,255,.05); border-radius: 12px; padding-left: .5rem; }
.rank-no { width: 26px; height: 26px; border-radius: 8px; display: grid; place-items: center; font-size: .78rem; font-weight: 700; background: rgba(255,255,255,.08); color: #cbd5e1; }
.rank-name { color: #e2e8f0; font-weight: 500; font-size: .92rem; }
.rank-aqi { text-align: right; font-weight: 800; color: var(--c); font-size: 1.15rem; }
.legend { display: flex; flex-wrap: wrap; gap: .5rem .9rem; margin-top: .7rem; font-size: .78rem; color: #cbd5e1; }
.legend i { display: inline-block; width: 11px; height: 11px; border-radius: 4px; margin-right: .35rem; vertical-align: -1px; }
/* ---------- calendar ---------- */
.cal { display: grid; grid-template-columns: repeat(7, 1fr); gap: .5rem; }
.cal-h { text-align: center; color: #64748b; font-size: .74rem; text-transform: uppercase; letter-spacing: .1em; padding-bottom: .2rem; }
.cal-cell { position: relative; aspect-ratio: 1.35 / 1; border-radius: 14px; padding: .4rem .55rem; display: flex; flex-direction: column; justify-content: space-between;
    background: linear-gradient(145deg, color-mix(in srgb, var(--c) 92%, white), color-mix(in srgb, var(--c) 70%, black)); color: var(--ink);
    box-shadow: 0 6px 18px color-mix(in srgb, var(--c) 25%, transparent); animation: pop .45s both; transition: transform .2s ease, box-shadow .2s ease; cursor: default; }
.cal-cell:hover { transform: translateY(-4px) scale(1.06); box-shadow: 0 12px 26px color-mix(in srgb, var(--c) 55%, transparent); z-index: 2; }
.cal-cell.empty { background: transparent; box-shadow: none; }
.cal-cell.nodata { background: rgba(255,255,255,.04); color: #475569; box-shadow: none; border: 1px dashed rgba(255,255,255,.08); }
.cal-day { font-size: .72rem; font-weight: 600; opacity: .8; }
.cal-aqi { font-size: 1.15rem; font-weight: 800; text-align: right; }
.cal-month { position: absolute; top: .35rem; right: .5rem; font-size: .6rem; font-weight: 700; text-transform: uppercase; opacity: .75; }
/* ---------- health ---------- */
.adv { position: relative; overflow: hidden; border-radius: 28px; padding: 2.2rem 2.4rem; margin-top: 1rem; display: flex; gap: 2rem; align-items: center; flex-wrap: wrap;
    background: linear-gradient(120deg, color-mix(in srgb, var(--c) 30%, rgba(8,12,24,.85)) 0%, rgba(8,12,24,.82) 60%);
    border: 1px solid color-mix(in srgb, var(--c) 45%, transparent); box-shadow: 0 25px 70px rgba(0,0,0,.4), 0 0 60px color-mix(in srgb, var(--c) 18%, transparent); animation: fadeUp .9s both; }
.adv-icon { position: relative; z-index: 2; width: 130px; height: 130px; border-radius: 50%; display: grid; place-items: center; font-size: 4.2rem;
    background: color-mix(in srgb, var(--c) 22%, rgba(0,0,0,.3)); border: 2px solid color-mix(in srgb, var(--c) 70%, transparent); }
.adv-icon::before, .adv-icon::after { content: ""; position: absolute; inset: -2px; border-radius: 50%; border: 2px solid var(--c); animation: ripple var(--rip, 2.4s) ease-out infinite; }
.adv-icon::after { animation-delay: calc(var(--rip, 2.4s) / 2); }
@keyframes ripple { from { transform: scale(1); opacity: .8; } to { transform: scale(1.8); opacity: 0; } }
.adv-text { position: relative; z-index: 2; flex: 1; min-width: 260px; }
.adv-text h2 { color: #fff !important; font-size: 2rem !important; font-weight: 800 !important; margin: 0 0 .3rem !important; padding: 0 !important; }
.adv-text p { color: #e2e8f0; font-size: 1.02rem; margin: 0; }
.a-sway { display: inline-block; animation: sway 3s ease-in-out infinite; transform-origin: bottom center; }
.a-glow { display: inline-block; animation: glow 3s ease-in-out infinite; }
.a-bob { display: inline-block; animation: bob 2.4s ease-in-out infinite; }
.a-blink { display: inline-block; animation: blink 1.2s ease-in-out infinite; }
.a-siren { display: inline-block; animation: siren .6s ease-in-out infinite; }
@keyframes sway { 0%,100% { transform: rotate(-10deg); } 50% { transform: rotate(10deg); } }
@keyframes glow { 0%,100% { transform: scale(1); filter: drop-shadow(0 0 0 #fde047); } 50% { transform: scale(1.12); filter: drop-shadow(0 0 22px #fde047); } }
@keyframes bob { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-12px); } }
@keyframes blink { 0%,100% { transform: scale(1) rotate(0); opacity: 1; } 25% { transform: scale(1.1) rotate(-6deg); } 50% { opacity: .55; } 75% { transform: scale(1.1) rotate(6deg); } }
@keyframes siren { 0%,100% { transform: translateX(0) scale(1); filter: drop-shadow(0 0 6px red); } 25% { transform: translateX(-5px) scale(1.08); } 50% { filter: drop-shadow(0 0 28px red); } 75% { transform: translateX(5px) scale(1.08); } }
.act-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: .9rem; }
.act { background: rgba(12,18,34,.6); backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,.08); border-radius: 20px; padding: 1.1rem; text-align: center;
    animation: fadeUp .7s both; transition: transform .25s ease; }
.act:hover { transform: translateY(-5px); }
.act .emo { font-size: 2.3rem; display: inline-block; animation: bob 3s ease-in-out infinite; }
.act .name { color: #e2e8f0; font-weight: 600; margin: .35rem 0 .5rem; font-size: .93rem; }
.badge { display: inline-block; padding: .22rem .7rem; border-radius: 999px; font-size: .74rem; font-weight: 700; }
.b-ok { background: rgba(34,197,94,.18); color: #86efac; border: 1px solid rgba(34,197,94,.4); }
.b-warn { background: rgba(245,158,11,.18); color: #fcd34d; border: 1px solid rgba(245,158,11,.4); }
.b-bad { background: rgba(239,68,68,.18); color: #fca5a5; border: 1px solid rgba(239,68,68,.45); }
.cond-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; }
.cond { background: rgba(12,18,34,.62); backdrop-filter: blur(14px); border: 1px solid rgba(255,255,255,.08); border-radius: 22px; padding: 1.3rem 1.4rem; animation: fadeUp .8s both; }
.cond-head { display: flex; align-items: center; gap: .9rem; }
.cond-emo { font-size: 2.4rem; display: inline-block; }
.cond-name { color: #fff; font-weight: 700; font-size: 1.08rem; }
.cond-risk { font-size: .8rem; font-weight: 600; }
.beat { animation: beat var(--spd, 1s) ease-in-out infinite; }
.breathe { animation: breathe var(--spd, 4s) ease-in-out infinite; }
.sneeze { animation: sneeze var(--spd, 3s) ease-in-out infinite; }
@keyframes beat { 0%,100% { transform: scale(1); } 15% { transform: scale(1.25); } 30% { transform: scale(1); } 45% { transform: scale(1.18); } 60% { transform: scale(1); } }
@keyframes breathe { 0%,100% { transform: scale(.9); } 50% { transform: scale(1.18); } }
@keyframes sneeze { 0%,70%,100% { transform: rotate(0) translateX(0); } 76% { transform: rotate(-14deg) translateX(-4px); } 84% { transform: rotate(10deg) translateX(6px) scale(1.12); } 92% { transform: rotate(-4deg); } }
.meter { height: 9px; border-radius: 999px; background: rgba(255,255,255,.08); margin: .9rem 0 .8rem; overflow: hidden; }
.meter > div { height: 100%; border-radius: 999px; background: linear-gradient(90deg,#22c55e,#eab308,#f97316,#ef4444,#a855f7); background-size: 400px 100%;
    animation: growX 1.6s cubic-bezier(.2,.7,.2,1) both; }
.dd { font-size: .88rem; color: #cbd5e1; margin: .35rem 0; line-height: 1.5; }
.dd b { color: #fff; }
/* ---------- widgets ---------- */
div[role="radiogroup"] { gap: .4rem; flex-wrap: wrap; }
div[role="radiogroup"] label { background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.1); padding: .35rem .9rem !important; border-radius: 999px; transition: all .2s; }
div[role="radiogroup"] label:has(input:checked) { background: linear-gradient(135deg,#0ea5e9,#6366f1); border-color: transparent; }
div[role="radiogroup"] label > div:first-child, [data-testid="stRadioOption"] > div > div:first-child:not([data-testid]) { display: none; }
[data-testid="stAlert"] { border-radius: 16px; backdrop-filter: blur(10px); }
.footer { text-align: center; color: #64748b; font-size: .78rem; margin-top: 3rem; }
/* ---------- landing ---------- */
.landing { position: relative; min-height: 70vh; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; padding: 2rem 1rem 0; }
.landing .eyebrow { display: inline-flex; gap: .5rem; align-items: center; padding: .4rem 1rem; border-radius: 999px; background: rgba(255,255,255,.1);
    border: 1px solid rgba(255,255,255,.2); color: #e0f2fe; font-size: .8rem; letter-spacing: .18em; text-transform: uppercase; backdrop-filter: blur(8px); animation: fadeUp 1s both; }
.landing h1 { font-size: clamp(2.4rem, 6vw, 4.8rem) !important; font-weight: 800 !important; line-height: 1.08 !important; color: #fff !important; margin: 1.3rem 0 .6rem !important;
    padding: 0 !important; text-shadow: 0 8px 40px rgba(0,0,0,.45); max-width: 1000px; }
.landing h1 .w { display: inline-block; opacity: 0; animation: fadeUp .9s cubic-bezier(.2,.7,.2,1) forwards; margin-right: .22em; }
.landing h1 .grad { background: linear-gradient(90deg,#7dd3fc,#c4b5fd,#f9a8d4,#7dd3fc); background-size: 200% 100%; -webkit-background-clip: text; background-clip: text;
    color: transparent; animation: fadeUp .9s cubic-bezier(.2,.7,.2,1) forwards, shimmer 6s linear infinite; }
.landing .lead { color: #e2e8f0; font-size: 1.15rem; max-width: 680px; opacity: 0; animation: fadeUp 1s 1.4s forwards; text-shadow: 0 2px 20px rgba(0,0,0,.6); }
.teaser { margin-top: 0; display: inline-flex; align-items: center; gap: 1rem; padding: .8rem 1.4rem; border-radius: 20px; background: rgba(8,12,24,.55);
    border: 1px solid rgba(255,255,255,.15); backdrop-filter: blur(12px); opacity: 0; animation: fadeUp 1s 1.8s forwards; }
.teaser .t-num { font-size: 2rem; font-weight: 800; color: var(--c); text-shadow: 0 0 20px var(--c); }
.teaser .t-txt { text-align: left; color: #cbd5e1; font-size: .88rem; line-height: 1.35; }
.features { display: flex; gap: .8rem; justify-content: center; flex-wrap: wrap; margin-top: 1.4rem; opacity: 0; animation: fadeUp 1s 2.1s forwards; }
.features span { color: #cbd5e1; font-size: .82rem; padding: .35rem .8rem; border-radius: 999px; background: rgba(255,255,255,.07); border: 1px solid rgba(255,255,255,.12); }
.st-key-enter_btn { opacity: 0; animation: fadeUp 1s 2.3s forwards; }
.st-key-enter_btn button { height: 3.6rem; border-radius: 999px !important; font-size: 1.12rem !important; font-weight: 700 !important; border: none !important; color: #fff !important;
    background: linear-gradient(135deg,#0ea5e9,#6366f1,#a855f7) !important; background-size: 200% 100% !important; box-shadow: 0 10px 40px rgba(99,102,241,.6) !important;
    animation: shimmer 5s linear infinite; transition: transform .25s ease !important; }
.st-key-enter_btn button:hover { transform: translateY(-3px) scale(1.03); }
.fog { position: fixed; left: 0; right: 0; bottom: 0; height: 32vh; pointer-events: none; z-index: -1;
    background: linear-gradient(0deg, rgba(226,232,240,.18), transparent); }
@media (max-width: 760px) { .hero-content { padding: 1.6rem; } .ring-wrap, .ring { width: 200px; height: 200px; } .ring-center .num { font-size: 3.6rem; } }
</style>
"""


def inject_global(landing=False):
    bg = img_uri("delhi_backdrop.jpg", blur=0 if landing else 7, max_w=1600 if landing else 1280) or ""
    css = (GLOBAL_CSS.replace("__BG__", bg).replace("__CLOUD__", CLOUD_URI)
           .replace("__OV1__", ".35" if landing else ".55").replace("__OV2__", ".55" if landing else ".82"))
    if landing:
        layer = (sky_layer(n=9, seed=11, strength=0.6, top_range=(0, 45), size=(260, 560))
                 + sky_layer(n=5, seed=5, strength=0.45, top_range=(72, 92), size=(600, 950))
                 + '<div class="fog"></div>')
    else:
        layer = sky_layer(n=8, seed=3, strength=0.42)
    html(css + f'<div class="global-layer">{layer}</div>')


# =====================================================================
# 7. ENTRY (LANDING) PAGE
# =====================================================================
def enter_site():
    st.session_state.entered = True


if not st.session_state.entered:
    inject_global(landing=True)
    words = "You are about to discover".split()
    title = "".join(f'<span class="w" style="animation-delay:{i * 0.12:.2f}s">{w}</span>' for i, w in enumerate(words))
    title += ('<br><span class="w" style="animation-delay:.7s">the</span><span class="w grad" style="animation-delay:.8s">real&nbsp;story</span>'
              '<span class="w" style="animation-delay:1s">of the air</span>'
              '<span class="w" style="animation-delay:1.15s">you breathe.</span>')
    html(f"""
    <div class="landing">
        {particles(45, seed=21, rgb="255,236,200", max_size=4, rise=220)}
        <div class="eyebrow">☁️ Delhi AQI Oracle</div>
        <h1>{title}</h1>
        <p class="lead">Every breath tells a story. See live air quality across Delhi, AI forecasts for the next 24 hours,
        and what it really means for your health.</p>
        <div class="features"><span>📡 5 live CPCB stations</span><span>🔮 1h · 6h · 24h AI forecasts</span>
        <span>📅 30-day history</span><span>🩺 Health guidance</span></div>
    </div>""")
    teaser_slot = st.empty()
    _, mid, _ = st.columns([1.4, 1, 1.4])
    with mid:
        with st.container(key="enter_btn"):
            wide_button("Enter & reveal the truth  →", key="enter", on_click=enter_site)
    live_data, _updated = get_live_data()          # fetched AFTER the page is drawn
    if live_data:
        avg = float(np.mean([v["aqi"] for v in live_data.values()]))
        c, tag = get_theme(avg)
        worst = max(live_data.items(), key=lambda kv: kv[1]["aqi"])
        with teaser_slot:
            html(f'''<div style="text-align:center"><div class="teaser" style="--c:{c}"><div class="t-num">{avg:.0f}</div>
                 <div class="t-txt">Delhi's average AQI <b>right now</b><br>
                 <b style="color:{c}">{tag}</b> · worst at {worst[0]} ({worst[1]["aqi"]:.0f})</div></div></div>''')
    st.stop()


# =====================================================================
# 8. TOP NAVIGATION BAR
# =====================================================================
inject_global(landing=False)


def go_to(p):
    st.session_state.page = p


with st.container(key="topnav"):
    cols = st.columns([2.3, 1.25, 1.35, 1.25, 1.35, 1.6], vertical_alignment="center")
    with cols[0]:
        html("""<div class="brand"><div class="brand-logo">☁️</div>
                <div><div class="brand-name">Delhi AQI Oracle</div><div class="brand-sub">Air · Forecast · Health</div></div></div>""")
    for i, p in enumerate(PAGES):
        with cols[i + 1]:
            wide_button(f"{PAGE_ICONS[p]}  {p}", key=f"nav_{i}", on_click=go_to, args=(p,),
                        type="primary" if st.session_state.page == p else "secondary")
    with cols[5]:
        st.selectbox("Station", ALL_LOCATIONS, key="station", label_visibility="collapsed")

page = st.session_state.page
station = st.session_state.station
models = load_models()
live_data, updated_at = get_live_data()


def page_head(title, sub):
    html(f'<div class="page-head fade-up"><h1>{title}</h1><p>{sub}</p></div>')


def need_live(station_name):
    if station_name not in live_data:
        st.warning(f"📡 Live data for **{station_name}** is unavailable right now. Please try again in a few minutes "
                   f"or pick another station from the top bar.")
        st.stop()
    return live_data[station_name]


def footer():
    srcs = sorted({v.get("source", "WAQI") for v in live_data.values()}) or ["—"]
    html(f'<div class="footer">Live data: {", ".join(srcs)} · Weather: Open-Meteo · AQI computed on the CPCB (India) scale · '
         f'Updated {updated_at}<br>Made with 💙 for cleaner Delhi air</div>')


# =====================================================================
# PAGE 1: LIVE DASHBOARD
# =====================================================================
if page == "Live Dashboard":
    d = need_live(station)
    station_pincode = LOCATIONS_DATA[station]["pincode"]
    current_aqi = d["aqi"]
    pm25 = d["pm25"]
    color, tag = get_theme(current_aqi)

    # ---------- hero block with artwork + dust particles ----------
    art = img_uri("delhi_clean.jpg" if current_aqi <= 100 else "delhi_polluted.jpg") or ""
    n_particles = int(np.clip(current_aqi / 5, 10, 90))
    dust_rgb = "220,255,230" if current_aqi <= 100 else "255,205,150"
    C = 2 * np.pi * 108
    off = C * (1 - min(current_aqi, 500) / 500)
    marker_pos = min(current_aqi, 500) / 500 * 100

    html(f"""
    <div class="hero" style="--c:{color};--ink:{ink(current_aqi)}">
        <div class="hero-bg" style="background-image:url('{art}')"></div>
        <div class="hero-tint"></div>
        {particles(n_particles, seed=int(current_aqi) + 3, rgb=dust_rgb, max_size=6.5, rise=180)}
        <div class="hero-content">
            <div>
                <span class="live-badge"><span class="live-dot"></span>LIVE · {d.get('source', 'WAQI')}</span>
                <h1 class="station">{station}</h1>
                <div class="meta">📍 New Delhi {station_pincode} &nbsp;·&nbsp; 🕒 {updated_at}</div>
                <div class="cat-pill">{tag} Air Quality</div>
                <div class="headline">{HEADLINES[tag]}</div>
                <div class="chips">
                    <div class="chip"><span>🌡️ Temp</span><b>{d['temp']:.1f} °C</b></div>
                    <div class="chip"><span>💨 Wind</span><b>{d['wind']:.1f} km/h</b></div>
                    <div class="chip"><span>🌫️ PM2.5</span><b>{pm25:.0f}</b></div>
                    <div class="chip"><span>🌪️ PM10</span><b>{d['pm10']:.0f}</b></div>
                </div>
            </div>
            <div class="ring-wrap">
                <svg class="ring" viewBox="0 0 250 250">
                    <circle cx="125" cy="125" r="108" class="ring-bg"/>
                    <circle cx="125" cy="125" r="108" class="ring-fg" style="stroke:{color};stroke-dasharray:{C:.1f};stroke-dashoffset:{off:.1f};--C:{C:.1f};--off:{off:.1f}"/>
                </svg>
                <div class="ring-center"><div class="num">{current_aqi:.0f}</div><div class="lbl">AQI · CPCB</div></div>
            </div>
        </div>
    </div>
    <div class="scale" style="--c:{color}">
        <div class="scale-bar"></div>
        <div class="scale-marker" style="left:{marker_pos:.1f}%"><div></div></div>
        <div class="scale-labels"><span>0 Good</span><span>100 Moderate</span><span>200 Poor</span><span>300 Very Poor</span><span>400+ Severe</span><span>500</span></div>
    </div>""")

    # ---------- cigarette equivalence ----------
    cigs = pm25 / 22
    shown = int(min(np.ceil(cigs), 12))
    icons = "".join(f'<span style="animation-delay:{0.15 * i:.2f}s">🚬</span>' for i in range(shown))
    html(f"""
    <div class="glass fade-up" style="margin-top:1.6rem">
        <div class="cig">
            <div class="cig-num">{cigs:.1f}</div>
            <div style="flex:1;min-width:220px"><div class="card-title">cigarettes per day</div>
            <div class="card-sub" style="margin:0">Breathing the air at {station} today is like smoking this many cigarettes (based on PM2.5 ÷ 22).</div></div>
            <div class="cig-icons">{icons}{'<span>➕</span>' if cigs > 12 else ''}</div>
        </div>
    </div>""")

    # ---------- AI forecast ----------
    section("AI Forecast", "XGBoost predictions for the next 1, 6 and 24 hours", "🔮")
    if models is None:
        st.info("🧠 Model files were not found in the `outputs/` folder, so forecasts are hidden. "
                "Add `best_model_1h_final.pkl`, `best_model_6h_final.pkl`, `best_model_24h_final.pkl` and `clean.pkl`.")
    else:
        try:
            pred_1h, pred_6h, pred_24h, db_msg = predict(station, d, *models)
            if db_msg:
                st.error(db_msg)
            cards = []
            for i, (lbl, val) in enumerate([("Next 1 Hour", pred_1h), ("Next 6 Hours", pred_6h), ("Next 24 Hours", pred_24h)]):
                c, t = get_theme(val)
                delta = val - current_aqi
                arrow, dcol = ("▲", "#f87171") if delta > 0 else ("▼", "#4ade80")
                cards.append(f"""
                <div class="fc fade-up" style="--c:{c};--ink:{ink(val)};animation-delay:{i * 120}ms">
                    <div class="fc-h">⏱ {lbl}</div>
                    <div class="fc-v">{val:.0f}<span class="fc-delta" style="color:{dcol}">{arrow} {abs(delta):.0f}</span></div>
                    <span class="fc-tag">{t}</span>
                    <div class="bar"><div style="width:{min(val, 500) / 5:.1f}%"></div></div>
                </div>""")
            html('<div class="fc-grid">' + "".join(cards) + "</div>")

            now = datetime.datetime.now()
            xs = [f"Now<br>{now:%H:%M}", f"+1h<br>{now + datetime.timedelta(hours=1):%H:%M}",
                  f"+6h<br>{now + datetime.timedelta(hours=6):%H:%M}", f"+24h<br>{now + datetime.timedelta(hours=24):%a %H:%M}"]
            ys = [current_aqi, pred_1h, pred_6h, pred_24h]
            fig = go.Figure()
            ymax = max(ys) * 1.3 + 20
            for lo, hi, bc in BANDS:
                if lo < ymax:
                    fig.add_hrect(y0=lo, y1=min(hi, ymax), fillcolor=bc, opacity=0.06, line_width=0)
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines+markers+text", line=dict(color="#a5b4fc", width=4, shape="spline"),
                fill="tozeroy", fillcolor="rgba(129,140,248,.12)",
                marker=dict(size=18, color=[get_theme(v)[0] for v in ys], line=dict(color="white", width=3)),
                text=[f"{v:.0f}" for v in ys], textposition="top center", textfont=dict(color="white", size=14),
                hovertemplate="%{x}<br>AQI <b>%{y:.0f}</b><extra></extra>"))
            style_fig(fig, 300)
            fig.update_yaxes(range=[0, ymax], title="AQI")
            fig.update_xaxes(range=[-0.3, 3.3])
            fig.update_traces(cliponaxis=False)
            with st.container(key="card_forecast"):
                html('<div class="card-title">Forecast trajectory</div><div class="card-sub">Shaded bands show CPCB categories</div>')
                show_chart(fig, key="fc_chart")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    # ---------- map + ranking ----------
    section("Live Station Map", "Circle size and colour scale with the current AQI", "📍")
    left, right = st.columns([2.1, 1], gap="medium")
    map_data = []
    for k, v in live_data.items():
        col_hex, t = get_theme(v["aqi"])
        rgb = hex_rgb(col_hex)
        map_data.append({
            "location": k, "lat": LOCATIONS_DATA[k]["coords"][0], "lon": LOCATIONS_DATA[k]["coords"][1],
            "aqi": round(v["aqi"]), "aqi_txt": f"{v['aqi']:.0f}", "tag": t, "pm25": round(v["pm25"]), "temp": round(v["temp"], 1),
            "color": rgb + [235], "halo": rgb + [55], "radius": 700 + v["aqi"] * 9, "elev": v["aqi"] * 25,
            "label": k + (" ★" if k == station else ""),
        })
    map_df = pd.DataFrame(map_data)
    with left:
        with st.container(key="card_map"):
            view3d = st.toggle("🏙️ 3D column view", value=False)
            layers = []
            if view3d:
                layers.append(pdk.Layer("ColumnLayer", map_df, get_position=["lon", "lat"], get_elevation="elev",
                                        elevation_scale=1, radius=650, get_fill_color="color", pickable=True, extruded=True, auto_highlight=True))
            else:
                layers += [
                    pdk.Layer("ScatterplotLayer", map_df, get_position=["lon", "lat"], get_fill_color="halo",
                              get_radius="radius", radius_min_pixels=18, stroked=False),
                    pdk.Layer("ScatterplotLayer", map_df, get_position=["lon", "lat"], get_fill_color="color", get_radius=520,
                              radius_min_pixels=16, stroked=True, get_line_color=[255, 255, 255, 230], line_width_min_pixels=2.5,
                              pickable=True, auto_highlight=True),
                    pdk.Layer("TextLayer", map_df, get_position=["lon", "lat"], get_text="aqi_txt", get_color=[15, 23, 42],
                              get_size=13, font_weight=800, get_alignment_baseline="'center'", font_family="'Poppins', sans-serif"),
                ]
            layers.append(pdk.Layer("TextLayer", map_df, get_position=["lon", "lat"], get_text="label", get_color=[241, 245, 249],
                                    get_size=13, get_pixel_offset=[0, 34], font_weight=600, font_family="'Poppins', sans-serif",
                                    outline_width=3, outline_color=[2, 6, 23, 255], font_settings={"sdf": True}))
            st.pydeck_chart(pdk.Deck(
                layers=layers,
                initial_view_state=pdk.ViewState(latitude=28.63, longitude=77.19, zoom=10.1, pitch=50 if view3d else 25, bearing=-10 if view3d else 0),
                map_provider="carto", map_style=pdk.map_styles.CARTO_DARK,
                tooltip={"html": "<div style='font-family:Poppins,sans-serif'><b style='font-size:15px'>{location}</b><br/>"
                                 "AQI <b>{aqi}</b> · {tag}<br/>PM2.5 {pm25} · 🌡️ {temp}°C</div>",
                         "style": {"background": "rgba(15,23,42,.95)", "color": "white", "border-radius": "12px",
                                   "padding": "10px 12px", "border": "1px solid rgba(255,255,255,.15)"}},
            ), height=470)
    with right:
        rows = ""
        for i, r in enumerate(sorted(map_data, key=lambda x: -x["aqi"])):
            c = get_theme(r["aqi"])[0]
            rows += f"""
            <div class="rank-row {'active' if r['location'] == station else ''}" style="--c:{c}">
                <div class="rank-no">{i + 1}</div>
                <div><div class="rank-name">{r['location']}</div>
                <div class="bar" style="--c:{c};margin-top:.4rem"><div style="width:{min(r['aqi'], 500) / 5:.0f}%;animation-delay:{i * 120}ms"></div></div></div>
                <div class="rank-aqi">{r['aqi']}</div>
            </div>"""
        legend = "".join(f'<span><i style="background:{c}"></i>{n}</span>' for n, c in
                         [("Good", "#00e400"), ("Moderate", "#ffff00"), ("Poor", "#ff7e00"), ("Very Poor", "#ff0000"), ("Severe", "#8f3f97")])
        html(f"""<div class="glass fade-up" style="min-height:560px">
                 <div class="card-title">Station ranking</div><div class="card-sub">Most polluted first · ★ = selected</div>
                 {rows}<div class="legend">{legend}</div></div>""")
    footer()

# =====================================================================
# PAGE 2: HISTORICAL TRENDS
# =====================================================================
elif page == "Historical Trends":
    page_head("Historical Trends", f"Official CPCB data for <b style='color:#e2e8f0'>{station}</b>")
    file_name = LOCATIONS_DATA[station]["file"]
    df = load_station_excel(file_name)
    if df is None:
        st.error(f"Data file `{file_name}` not found next to app.py.")
        st.stop()

    last30 = df.tail(30 * 24).copy()
    valid = last30.dropna(subset=["aqi_index"])
    if not valid.empty:
        peak = valid.loc[valid["aqi_index"].idxmax()]
        low = valid.loc[valid["aqi_index"].idxmin()]
        avg30 = valid["aqi_index"].mean()
        unhealthy = (valid["aqi_index"] > 200).mean() * 100
        html('<div class="stats">'
             + stat_card("30-day average", f"{avg30:.0f}", get_theme(avg30)[1], get_theme(avg30)[0], 0)
             + stat_card("Peak AQI", f"{peak['aqi_index']:.0f}", f"{peak['dt']:%d %b, %I %p}", get_theme(peak['aqi_index'])[0], 100)
             + stat_card("Cleanest hour", f"{low['aqi_index']:.0f}", f"{low['dt']:%d %b, %I %p}", get_theme(low['aqi_index'])[0], 200)
             + stat_card("Hours above 200", f"{unhealthy:.0f}%", "Very Poor or worse", "#f87171", 300)
             + "</div>")

    section("AQI Chart", "Average AQI over the selected timeframe", "📊")
    with st.container(key="card_bar"):
        timeframe = st.radio("Select Timeframe", ["Last 24 Hours", "Last 7 Days", "Last 30 Days"],
                             horizontal=True, label_visibility="collapsed", index=1)
        if timeframe == "Last 24 Hours":
            df_chart = df.tail(24).copy()
        elif timeframe == "Last 7 Days":
            df_chart = df.tail(7 * 24).copy()
        else:
            df_chart = df.tail(30 * 24).copy()

        if timeframe == "Last 24 Hours":
            chart_data = df_chart.assign(date_str=df_chart['dt'].dt.strftime('%d %b %H:%M'))
        else:
            chart_data = df_chart.groupby(df_chart['dt'].dt.date)['aqi_index'].mean().reset_index()
            chart_data['date_str'] = pd.to_datetime(chart_data['dt']).dt.strftime('%a %d %b')
        chart_data = chart_data.dropna(subset=["aqi_index"])

        fig_bar = go.Figure(go.Bar(
            x=chart_data['date_str'], y=chart_data['aqi_index'],
            marker=dict(color=[get_theme(a)[0] for a in chart_data['aqi_index']], opacity=0.9,
                        line=dict(color="rgba(255,255,255,.35)", width=1), cornerradius=8),
            text=chart_data['aqi_index'].round(0).astype(int), textposition='outside',
            textfont=dict(color='#e2e8f0', size=11),
            hovertemplate="<b>%{x}</b><br>AQI %{y:.0f}<extra></extra>",
        ))
        for lvl, lab in [(100, "Moderate"), (200, "Poor"), (300, "Very Poor")]:
            if chart_data['aqi_index'].max() > lvl * 0.7:
                fig_bar.add_hline(y=lvl, line=dict(color="rgba(255,255,255,.25)", dash="dot", width=1),
                                  annotation_text=lab, annotation_font=dict(color="#94a3b8", size=10), annotation_position="top left")
        style_fig(fig_bar, 380)
        fig_bar.update_layout(bargap=0.28, yaxis_title="Average AQI")
        fig_bar.update_yaxes(range=[0, chart_data['aqi_index'].max() * 1.18 if len(chart_data) else 100])
        show_chart(fig_bar, key="bar")

    with st.container(key="card_trend"):
        html('<div class="card-title">Hourly trend</div><div class="card-sub">Every hour in the selected window</div>')
        tr = df_chart.dropna(subset=["aqi_index"])
        fig_tr = go.Figure(go.Scatter(
            x=tr['dt'], y=tr['aqi_index'], mode="lines", line=dict(width=2.5, color="#a5b4fc", shape="spline", smoothing=0.6),
            fill="tozeroy", fillgradient=dict(type="vertical", colorscale=[[0, "rgba(99,102,241,0)"], [1, "rgba(244,114,182,.45)"]]),
            hovertemplate="%{x|%d %b %H:%M}<br>AQI <b>%{y:.0f}</b><extra></extra>"))
        style_fig(fig_tr, 280)
        show_chart(fig_tr, key="trend")

    # ---------- calendar ----------
    section("AQI Calendar", "Daily average for the last 30 days — hover a day for details", "📅")
    df_recent = df.sort_values("dt", ascending=False).head(30 * 24).copy()
    df_recent['date'] = df_recent['dt'].dt.date
    daily = df_recent.groupby('date')['aqi_index'].mean().sort_index()
    if len(daily):
        start, end = daily.index.min(), daily.index.max()
        cells = "".join(f'<div class="cal-h">{d}</div>' for d in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        cells += '<div class="cal-cell empty"></div>' * start.weekday()
        day, i = start, 0
        while day <= end:
            v = daily.get(day, np.nan)
            month_tag = f'<span class="cal-month">{day:%b}</span>' if (day.day == 1 or day == start) else ""
            if pd.isna(v):
                cells += f'<div class="cal-cell nodata" title="{day:%a %d %b %Y} · no data"><span class="cal-day">{day.day}</span>{month_tag}<span class="cal-aqi">–</span></div>'
            else:
                c, t = get_theme(v)
                cells += (f'<div class="cal-cell" style="--c:{c};--ink:{ink(v)};animation-delay:{i * 22}ms" '
                          f'title="{day:%A, %d %b %Y}&#10;AQI {v:.0f} · {t}"><span class="cal-day">{day.day}</span>'
                          f'{month_tag}<span class="cal-aqi">{v:.0f}</span></div>')
            day += datetime.timedelta(days=1)
            i += 1
        legend = "".join(f'<span><i style="background:{c}"></i>{n}</span>' for n, c in
                         [("Good ≤50", "#00e400"), ("Moderate ≤100", "#ffff00"), ("Poor ≤200", "#ff7e00"),
                          ("Very Poor ≤300", "#ff0000"), ("Severe 300+", "#8f3f97")])
        html(f"""<div class="glass fade-up"><div class="card-title">{start:%d %b} – {end:%d %b %Y}</div>
                 <div class="card-sub">{station}</div><div class="cal">{cells}</div><div class="legend">{legend}</div></div>""")

    # ---------- hour x weekday pattern ----------
    section("Daily Pattern", "When is the air worst? Average AQI by hour and weekday (last 30 days)", "🕒")
    pat = valid.assign(h=valid['dt'].dt.hour, wd=valid['dt'].dt.dayofweek).pivot_table(index="wd", columns="h", values="aqi_index", aggfunc="mean")
    if not pat.empty:
        pat = pat.reindex(index=range(7), columns=range(24))
        fig_hm = go.Figure(go.Heatmap(
            z=pat.values, x=[f"{h:02d}:00" for h in range(24)], y=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            colorscale=AQI_SCALE, zmin=0, zmax=500, xgap=3, ygap=3,
            colorbar=dict(title="AQI", thickness=12, outlinewidth=0, tickfont=dict(color="#94a3b8")),
            hovertemplate="%{y} %{x}<br>AQI <b>%{z:.0f}</b><extra></extra>"))
        style_fig(fig_hm, 320)
        fig_hm.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
        fig_hm.update_xaxes(gridcolor="rgba(0,0,0,0)")
        with st.container(key="card_heat"):
            show_chart(fig_hm, key="heat")
    footer()

# =====================================================================
# PAGE 3: MODEL ACCURACY
# =====================================================================
elif page == "Model Accuracy":
    page_head("Model Accuracy", "Real-world performance of the AI — every Live Dashboard visit logs a prediction to MongoDB")
    if predictions_collection is None:
        st.info("🗄️ MongoDB is not connected. Add `MONGO_URL` to `.streamlit/secrets.toml` (or env vars) to start logging predictions.")
        st.stop()
    try:
        cursor = predictions_collection.find().sort("timestamp", 1)
        df_log = pd.DataFrame(list(cursor))
    except Exception as e:
        st.error(f"Could not load data from MongoDB: {e}")
        df_log = pd.DataFrame()

    if df_log.empty:
        st.info("⏳ Collecting data... Visit the Live Dashboard to build history.")
    else:
        df_log['timestamp'] = pd.to_datetime(df_log['timestamp'])
        df_station = df_log[df_log['location'] == station].sort_values('timestamp').copy()
        if len(df_station) < 2:
            st.warning(f"Need more data for **{station}**. Open the Live Dashboard for this station a few more times.")
        else:
            df_station['error'] = (df_station['actual_aqi'] - df_station['pred_1h']).abs()
            mae = df_station['error'].mean()
            rmse = float(np.sqrt((df_station['error'] ** 2).mean()))
            within = (df_station['error'] <= 25).mean() * 100
            html('<div class="stats">'
                 + stat_card("Mean Absolute Error", f"{mae:.1f}", "AQI points", "#38bdf8", 0)
                 + stat_card("RMSE", f"{rmse:.1f}", "AQI points", "#a78bfa", 100)
                 + stat_card("Within ±25 AQI", f"{within:.0f}%", "of 1h predictions", "#4ade80", 200)
                 + stat_card("Predictions logged", f"{len(df_station)}", station, "#f472b6", 300)
                 + "</div>")

            section("Real vs Predicted", f"Actual live AQI vs 1-hour forecast for {station}", "📉")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_station['timestamp'], y=df_station['pred_1h'], mode='lines+markers', name='Predicted 1H AQI',
                                     line=dict(color='#fb923c', dash='dash', width=2.5, shape="spline"), marker=dict(size=7)))
            fig.add_trace(go.Scatter(x=df_station['timestamp'], y=df_station['actual_aqi'], mode='lines+markers', name='Actual AQI (Live)',
                                     line=dict(color='#4ade80', width=3.5, shape="spline"), marker=dict(size=8, line=dict(color="white", width=1.5)),
                                     fill="tonexty", fillcolor="rgba(148,163,184,.12)"))
            style_fig(fig, 420)
            fig.update_layout(hovermode="x unified", yaxis_title="AQI")
            with st.container(key="card_acc"):
                show_chart(fig, key="acc")

            section("Error Distribution", "How far off the 1-hour forecast usually is", "🎯")
            fig_h = go.Figure(go.Histogram(x=df_station['actual_aqi'] - df_station['pred_1h'], nbinsx=25,
                                           marker=dict(color="#818cf8", line=dict(color="rgba(255,255,255,.4)", width=1), cornerradius=6),
                                           hovertemplate="Error %{x}<br>%{y} predictions<extra></extra>"))
            fig_h.add_vline(x=0, line=dict(color="#4ade80", dash="dot"))
            style_fig(fig_h, 300)
            fig_h.update_layout(xaxis_title="Actual − Predicted (AQI)", yaxis_title="Count", bargap=0.08)
            with st.container(key="card_hist"):
                show_chart(fig_h, key="hist")
    footer()

# =====================================================================
# PAGE 4: HEALTH ADVISORIES
# =====================================================================
elif page == "Health Advisories":
    d = need_live(station)
    current_aqi = d["aqi"]
    color, tag = get_theme(current_aqi)
    page_head("Health Advisories", f"Personalised guidance for <b style='color:#e2e8f0'>{station}</b> · animations react to the live AQI")

    if current_aqi > 300:
        icon, anim, title, msg = "🚨", "a-siren", "Hazardous!", "Avoid all outdoor physical activity. Keep windows closed and run air purifiers."
    elif current_aqi > 200:
        icon, anim, title, msg = "⚠️", "a-blink", "Very Poor!", "Sensitive groups (asthma, heart) should stay indoors. Masks are mandatory outside."
    elif current_aqi > 100:
        icon, anim, title, msg = "😷", "a-bob", "Poor air", "Safe for most for short periods, but sensitive groups should limit prolonged outdoor exertion."
    elif current_aqi > 50:
        icon, anim, title, msg = "🌤️", "a-glow", "Moderate", "Air is acceptable. Unusually sensitive people should reduce heavy outdoor exertion."
    else:
        icon, anim, title, msg = "🌿", "a-sway", "Good — breathe easy!", "Air quality is great. Enjoy outdoor activities!"
    ripple = max(0.7, 3.0 - current_aqi / 170)   # ripples get faster as pollution rises
    html(f"""
    <div class="adv" style="--c:{color};--rip:{ripple:.2f}s">
        {particles(int(np.clip(current_aqi / 8, 6, 60)), seed=9, rgb="255,255,255" if current_aqi <= 100 else "255,200,150", max_size=5, rise=160)}
        <div class="adv-icon"><span class="{anim}">{icon}</span></div>
        <div class="adv-text">
            <div style="color:#cbd5e1;font-size:.85rem;letter-spacing:.12em;text-transform:uppercase">Current AQI in {station}</div>
            <h2>{current_aqi:.0f} · {title}</h2>
            <p>{msg}</p>
        </div>
    </div>""")

    # ---------- activity guide ----------
    section("What can I do today?", "Quick activity guide based on the current AQI", "🧭")

    def status(ok_max, warn_max, ok, warn, bad):
        if current_aqi <= ok_max: return "b-ok", ok
        if current_aqi <= warn_max: return "b-warn", warn
        return "b-bad", bad

    acts = [
        ("🏃", "Outdoor exercise", status(100, 200, "Go for it", "Keep it light", "Avoid")),
        ("🪟", "Open windows", status(100, 200, "Ventilate", "Briefly only", "Keep closed")),
        ("😷", "N95 mask", status(100, 200, "Optional", "Recommended", "Essential")),
        ("🧒", "Kids playing outside", status(100, 150, "Safe", "Limit time", "Indoors")),
        ("🌬️", "Air purifier", status(50, 200, "Not needed", "Helpful", "Run all day")),
        ("🚴", "Walk / cycle commute", status(100, 200, "Enjoy", "Short trips", "Use transport")),
    ]
    tiles = "".join(f'<div class="act" style="animation-delay:{i * 90}ms"><span class="emo" style="animation-delay:{i * 0.3:.1f}s">{e}</span>'
                    f'<div class="name">{n}</div><span class="badge {b}">{s}</span></div>' for i, (e, n, (b, s)) in enumerate(acts))
    html(f'<div class="act-grid">{tiles}</div>')

    # ---------- specific conditions ----------
    section("Specific Conditions", "Risk meters & animations scale with the live AQI", "🩺")

    def risk(mult):
        pct = float(np.clip(current_aqi / 400 * 100 * mult, 4, 100))
        lab = "Low" if pct < 25 else "Moderate" if pct < 50 else "High" if pct < 75 else "Very High"
        col = "#4ade80" if pct < 25 else "#facc15" if pct < 50 else "#fb923c" if pct < 75 else "#f87171"
        return pct, lab, col

    spd_heart = max(0.45, 1.5 - current_aqi / 350)
    spd_lung = max(1.2, 4.5 - current_aqi / 120)
    spd_sneeze = max(1.2, 4.0 - current_aqi / 140)
    conditions = [
        ("🫁", "breathe", spd_lung, "Asthma & Respiratory Issues", 1.25, "High when AQI is Severe.",
         "Keep inhalers handy, use air purifiers indoors, wear an N95 mask if going out.",
         "Do not exercise outdoors, avoid smoking."),
        ("❤️", "beat", spd_heart, "Heart Conditions", 1.15, "High. Air pollution can increase blood pressure and heart rate.",
         "Stay indoors during early morning (worst AQI), monitor BP.",
         "Avoid strenuous outdoor activities."),
        ("🤧", "sneeze", spd_sneeze, "Allergies & Sinus", 0.95, "Moderate to High.",
         "Keep windows closed, take prescribed antihistamines.",
         "Do not hang clothes outside to dry."),
        ("👵", "a-bob", 2.6, "Elderly, Children & Pregnancy", 1.1, "Higher sensitivity to fine particles.",
         "Plan outings for afternoons, keep hydrated, consult a doctor if breathless.",
         "Avoid busy roads and construction areas."),
    ]
    cards = ""
    for i, (emo, cls, spd, name, mult, note, dos, donts) in enumerate(conditions):
        pct, lab, col = risk(mult)
        cards += f"""
        <div class="cond" style="animation-delay:{i * 120}ms">
            <div class="cond-head">
                <span class="cond-emo {cls}" style="--spd:{spd:.2f}s">{emo}</span>
                <div><div class="cond-name">{name}</div><div class="cond-risk" style="color:{col}">Risk now: {lab}</div></div>
            </div>
            <div class="meter"><div style="width:{pct:.0f}%;animation-delay:{0.2 + i * 0.15:.2f}s"></div></div>
            <div class="dd"><b>Risk level:</b> {note}</div>
            <div class="dd">✅ <b>Do's:</b> {dos}</div>
            <div class="dd">❌ <b>Don'ts:</b> {donts}</div>
        </div>"""
    html(f'<div class="cond-grid">{cards}</div>')
    footer()
