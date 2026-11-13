import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import pydeck as pdk
from datetime import datetime
from pathlib import Path

# --- 1. SETUP ---
st.set_page_config(page_title="Delhi AQI Oracle", layout="wide", page_icon="🌫️")
ROOT = Path(__file__).parent if "__file__" in globals() else Path.cwd()
OUT = ROOT / "outputs"

# PASTE YOUR WAQI TOKEN HERE
try:
    WAQI_TOKEN = st.secrets["WAQI_TOKEN"] 
except KeyError:
    st.error("WAQI_TOKEN not found in Streamlit secrets.")
    st.stop()

LOCATIONS_DATA = {
    "Anand Vihar": {"coords": (28.6469, 77.3160), "pincode": "110092"},
    "Connaught Place": {"coords": (28.6304, 77.2177), "pincode": "110001"},
    "Dwarka": {"coords": (28.5921, 77.0460), "pincode": "110075"},
    "IGI Airport": {"coords": (28.5562, 77.1000), "pincode": "110037"},
    "Okhla Phase III": {"coords": (28.5432, 77.2725), "pincode": "110020"},
    "Rohini": {"coords": (28.7320, 77.1123), "pincode": "110085"}
}
ALL_LOCATIONS = list(LOCATIONS_DATA.keys())

# --- 2. LOAD MODELS ---
@st.cache_resource
def load_models():
    with open(OUT / "best_model_1h_final.pkl", "rb") as f: b1 = pickle.load(f)
    with open(OUT / "best_model_6h_final.pkl", "rb") as f: b6 = pickle.load(f)
    with open(OUT / "best_model_24h_final.pkl", "rb") as f: b24 = pickle.load(f)
    
    m1, c1 = b1["model"], b1["columns"]
    m6, c6 = b6["model"], b6["columns"]
    m24, c24 = b24["model"], b24["columns"]
    return m1, c1, m6, c6, m24, c24

# --- 3. OFFICIAL CPCB MATH ---
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

# --- 4. FAST FETCH (WAQI + FUTURE WEATHER) ---
@st.cache_data(ttl=900)
def get_live_data():
    data = {}
    
    # 1. Fetch Future Weather from Open-Meteo (for +1h, +6h, +24h)
    try:
        wx_url = "https://api.open-meteo.com/v1/forecast?latitude=28.6139&longitude=77.2090&hourly=temperature_2m,wind_speed_10m&forecast_days=2"
        wx_res = requests.get(wx_url, timeout=5).json()["hourly"]
        wx_now = {"temp": float(wx_res["temperature_2m"][0]), "wind": float(wx_res["wind_speed_10m"][0])}
        wx_1h = {"temp": float(wx_res["temperature_2m"][1]), "wind": float(wx_res["wind_speed_10m"][1])}
        wx_6h = {"temp": float(wx_res["temperature_2m"][6]), "wind": float(wx_res["wind_speed_10m"][6])}
        wx_24h = {"temp": float(wx_res["temperature_2m"][24]), "wind": float(wx_res["wind_speed_10m"][24])}
    except:
        wx_now = wx_1h = wx_6h = wx_24h = {"temp": 25.0, "wind": 5.0}

    # 2. Fetch Live AQI from WAQI
    for name, info in LOCATIONS_DATA.items():
        lat, lon = info["coords"]
        try:
            url = f"https://api.waqi.info/feed/geo:{lat};{lon}/?token={WAQI_TOKEN}"
            res = requests.get(url, timeout=3).json()
            if res["status"] == "ok":
                d = res["data"]
                pm25 = float(d["iaqi"].get("pm25", {}).get("v", 0))
                pm10 = float(d["iaqi"].get("pm10", {}).get("v", 0))
                official_aqi = calculate_cpcb_aqi(pm25, pm10)
                data[name] = {
                    "aqi": official_aqi, 
                    "temp": wx_now["temp"], 
                    "wind": wx_now["wind"], 
                    "pm25": pm25, "pm10": pm10,
                    "wx_1h": wx_1h, "wx_6h": wx_6h, "wx_24h": wx_24h
                }
        except: pass
    return data

# --- 5. PREDICTION LOGIC ---
def predict(station_name, live_weather, m1, c1, m6, c6, m24, c24):
    clean = pd.read_pickle(OUT / "clean.pkl")
    df = clean.loc[station_name].copy()
    
    for col in ['aqi_index', 'temp_c', 'humidity', 'windspeed_kph']:
        if col in df.columns: df[col] = df[col].astype(float)
    
    # Anchor the last 24 hours so the model knows this is a sustained event
    for i in range(1, 25):
        if len(df) >= i:
            df.iloc[-i, df.columns.get_loc('aqi_index')] = live_weather['aqi']
            df.iloc[-i, df.columns.get_loc('temp_c')] = live_weather['temp']
            df.iloc[-i, df.columns.get_loc('windspeed_kph')] = live_weather['wind']

    g = df.copy()
    for L in (1, 3, 6, 12, 24): g[f"aqi_lag{L}"] = g.aqi_index.shift(L)
    
    t = g.index
    g["hour"]=t.hour; g["month"]=t.month; g["dow"]=t.dayofweek
    g["is_weekend"]=(t.dayofweek>=5).astype(int)
    g["hour_sin"]=np.sin(2*np.pi*t.hour/24); g["hour_cos"]=np.cos(2*np.pi*t.hour/24)
    g["month_sin"]=np.sin(2*np.pi*t.month/12); g["month_cos"]=np.cos(2*np.pi*t.month/12)
    
    last_row = g.iloc[[-1]].copy()
    
    for loc in ALL_LOCATIONS:
        last_row[f"location_{loc}"] = 1 if loc == station_name else 0
    last_row = last_row.drop(columns=["location"], errors="ignore")

    def run_pred(model, cols, horizon, f_wind, f_temp):
        # Inject the future weather features the model was trained on
        last_row[f"future_wind_{horizon}h"] = f_wind
        last_row[f"future_temp_{horizon}h"] = f_temp
        
        X = last_row.reindex(columns=cols, fill_value=0).select_dtypes(include=[np.number])
        return model.predict(X)[0]
        
    pred_1h = run_pred(m1, c1, 1, live_weather['wx_1h']['wind'], live_weather['wx_1h']['temp'])
    pred_6h = run_pred(m6, c6, 6, live_weather['wx_6h']['wind'], live_weather['wx_6h']['temp'])
    pred_24h = run_pred(m24, c24, 24, live_weather['wx_24h']['wind'], live_weather['wx_24h']['temp'])
        
    return pred_1h, pred_6h, pred_24h

# --- 6. UI RENDER ---
m1, c1, m6, c6, m24, c24 = load_models()
live_data = get_live_data()

st.title("🌫️ Delhi AQI Oracle")
station = st.sidebar.selectbox("Select Station", ALL_LOCATIONS)
station_pincode = LOCATIONS_DATA[station]["pincode"]

if station not in live_data:
    st.error("Live data unavailable. API limits reached. Please refresh in a few minutes.")
    st.stop()

current_aqi = live_data[station]["aqi"]

def get_theme(aqi):
    if aqi <= 50: return "#00e400", "Good"
    if aqi <= 100: return "#ffff00", "Moderate"
    if aqi <= 200: return "#ff7e00", "Poor"
    if aqi <= 300: return "#ff0000", "Very Poor"
    return "#8f3f97", "Severe"

color, tag = get_theme(current_aqi)

st.markdown(f"""
    <div style="background-color: {color}; padding: 20px; border-radius: 15px; text-align: center; color: white; margin-bottom: 20px;">
        <h2 style="margin: 0;">{station} - {station_pincode}</h2>
        <h4 style="margin: 0; margin-bottom: 10px;">{tag} Air Quality</h4>
        <h1 style="font-size: 90px; margin: 0; font-weight: bold;">{current_aqi:.0f}</h1>
        <p style="font-size: 18px; margin: 0;">Temp: {live_data[station]['temp']:.1f} °C | PM2.5: {live_data[station]['pm25']} | PM10: {live_data[station]['pm10']}</p>
    </div>
""", unsafe_allow_html=True)

pred_1h, pred_6h, pred_24h = predict(station, live_data[station], m1, c1, m6, c6, m24, c24)

st.subheader("🔮 AI Forecast (XGBoost)")
c1, c2, c3 = st.columns(3)
c1.metric("Next 1 Hour", f"{pred_1h:.0f}", delta=f"{pred_1h - current_aqi:+.0f}")
c2.metric("Next 6 Hours", f"{pred_6h:.0f}", delta=f"{pred_6h - current_aqi:+.0f}")
c3.metric("Next 24 Hours", f"{pred_24h:.0f}", delta=f"{pred_24h - current_aqi:+.0f}")

st.divider()

st.subheader("📍 Live Station Map")
map_data = []
for k, v in live_data.items():
    map_data.append({
        "location": k, "pincode": LOCATIONS_DATA[k]["pincode"],
        "lat": LOCATIONS_DATA[k]["coords"][0], "lon": LOCATIONS_DATA[k]["coords"][1], "aqi": v["aqi"]
    })
map_df = pd.DataFrame(map_data)
map_df['color'] = map_df['aqi'].apply(lambda x: [int(get_theme(x)[0].lstrip('#')[i:i+2], 16) for i in (0,2,4)])

layer = pdk.Layer("ScatterplotLayer", map_df, get_position=["lon", "lat"], get_color="color", get_radius="aqi * 10", pickable=True)
st.pydeck_chart(pdk.Deck(
    layers=[layer], 
    initial_view_state=pdk.ViewState(latitude=28.6139, longitude=77.2090, zoom=9), 
    map_style="dark",
    tooltip={"text": "{location} ({pincode})\nAQI: {aqi}"}
))