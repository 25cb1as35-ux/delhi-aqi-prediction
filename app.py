import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import pydeck as pdk
import datetime
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# --- MONGODB IMPORTS ---
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

# --- 1. SETUP & STYLING ---
st.set_page_config(page_title="Delhi AQI Oracle", layout="wide", page_icon="🌫️", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    h1, h2, h3 { color: #FAFAFA !important; font-family: 'Helvetica Neue', sans-serif; }
    .stMetric { background-color: #1E2129; padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
    .stMetric label { color: #A0AAB5 !important; }
    .stMetric div { color: #FAFAFA !important; }
    [data-testid="stSidebar"] { background-color: #161A23; }
</style>
""", unsafe_allow_html=True)

ROOT = Path(__file__).parent if "__file__" in globals() else Path.cwd()
OUT = ROOT / "outputs"

# --- 2. SECRETS & DATABASE ---
try:
    WAQI_TOKEN = st.secrets["WAQI_TOKEN"]
    MONGO_URL = st.secrets["MONGO_URL"]
except Exception:
    # REPLACE THESE WITH YOUR ACTUAL KEYS FOR LOCAL TESTING
    WAQI_TOKEN = "YOUR_WAQI_TOKEN_HERE"
    MONGO_URL = "YOUR_MONGO_URL_HERE" 

@st.cache_resource
def get_database():
    try:
        client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=5000)
        db = client.aqi_database
        return db.predictions
    except ConnectionFailure:
        st.error("Failed to connect to MongoDB. Check your URL and Password.")
        st.stop()

predictions_collection = get_database()

LOCATIONS_DATA = {
    "Anand Vihar": {"coords": (28.6469, 77.3160), "pincode": "110092", "file": "anand_vihar.xlsx"},
    "Dwarka": {"coords": (28.5921, 77.0460), "pincode": "110075", "file": "dwarka.xlsx"},
    "IGI Airport": {"coords": (28.5562, 77.1000), "pincode": "110037", "file": "igi_airport.xlsx"},
    "Okhla Phase III": {"coords": (28.5432, 77.2725), "pincode": "110020", "file": "okhla.xlsx"},
    "Rohini": {"coords": (28.7320, 77.1123), "pincode": "110085", "file": "rohini.xlsx"}
}
ALL_LOCATIONS = list(LOCATIONS_DATA.keys())

# --- 3. LOAD MODELS & MATH ---
@st.cache_resource
def load_models():
    with open(OUT / "best_model_1h_final.pkl", "rb") as f: b1 = pickle.load(f)
    with open(OUT / "best_model_6h_final.pkl", "rb") as f: b6 = pickle.load(f)
    with open(OUT / "best_model_24h_final.pkl", "rb") as f: b24 = pickle.load(f)
    return b1["model"], b1["columns"], b6["model"], b6["columns"], b24["model"], b24["columns"]

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

# --- 4. FETCH & PREDICT ---
@st.cache_data(ttl=900)
def get_live_data():
    data = {}
    try:
        wx_url = "https://api.open-meteo.com/v1/forecast?latitude=28.6139&longitude=77.2090&hourly=temperature_2m,wind_speed_10m&forecast_days=2"
        wx_res = requests.get(wx_url, timeout=5).json()["hourly"]
        wx_6h = {"temp": float(wx_res["temperature_2m"][6]), "wind": float(wx_res["wind_speed_10m"][6])}
        wx_24h = {"temp": float(wx_res["temperature_2m"][24]), "wind": float(wx_res["wind_speed_10m"][24])}
    except:
        wx_6h = wx_24h = {"temp": 25.0, "wind": 5.0}

    for name, info in LOCATIONS_DATA.items():
        lat, lon = info["coords"]
        try:
            url = f"https://api.waqi.info/feed/geo:{lat};{lon}/?token={WAQI_TOKEN}"
            res = requests.get(url, timeout=3).json()
            if res["status"] == "ok":
                d = res["data"]
                live_temp = float(d["iaqi"].get("t", {}).get("v", 25.0))
                live_wind = float(d["iaqi"].get("w", {}).get("v", 5.0))
                pm25 = float(d["iaqi"].get("pm25", {}).get("v", 0))
                pm10 = float(d["iaqi"].get("pm10", {}).get("v", 0))
                official_aqi = calculate_cpcb_aqi(pm25, pm10)
                data[name] = {
                    "aqi": official_aqi, "temp": live_temp, "wind": live_wind, 
                    "pm25": pm25, "pm10": pm10, "wx_6h": wx_6h, "wx_24h": wx_24h
                }
        except: pass
    return data

@st.cache_data(ttl=3600)
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
    df["dt"] = pd.to_datetime(df["dt"], errors='coerce')
    df["pm2_5"] = pd.to_numeric(df["pm2_5"], errors='coerce')
    df = df.dropna()
    df = df[df["pm2_5"] > 0] 
    
    df = df.set_index("dt").resample("1h").mean().reset_index()
    df["aqi_index"] = df["pm2_5"].apply(lambda x: calculate_cpcb_aqi(x, 0))
    return df

def predict(station_name, live_weather, m1, c1, m6, c6, m24, c24):
    clean = pd.read_pickle(OUT / "clean.pkl")
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
    g["hour"]=t.hour; g["month"]=t.month; g["dow"]=t.dayofweek
    g["hour_sin"]=np.sin(2*np.pi*t.hour/24); g["hour_cos"]=np.cos(2*np.pi*t.hour/24)
    
    last_row = g.iloc[[-1]].copy()
    for loc in ALL_LOCATIONS:
        last_row[f"location_{loc}"] = 1 if loc == station_name else 0
    last_row = last_row.drop(columns=["location"], errors="ignore")

    def run_pred(model, cols, horizon, f_wind, f_temp):
        last_row[f"future_wind_{horizon}h"] = f_wind
        last_row[f"future_temp_{horizon}h"] = f_temp
        X = last_row.reindex(columns=cols, fill_value=0).select_dtypes(include=[np.number])
        return model.predict(X)[0]

    p1 = run_pred(m1, c1, 1, live_weather['wind'], live_weather['temp'])
    p6 = run_pred(m6, c6, 6, live_weather['wx_6h']['wind'], live_weather['wx_6h']['temp'])
    p24 = run_pred(m24, c24, 24, live_weather['wx_24h']['wind'], live_weather['wx_24h']['temp'])
    
    # SAVE TO MONGODB
    try:
        predictions_collection.insert_one({
            "timestamp": datetime.datetime.now(),
            "location": station_name,
            "actual_aqi": live_weather['aqi'],
            "pred_1h": p1,
            "pred_6h": p6,
            "pred_24h": p24
        })
    except Exception as e:
        st.error(f"Failed to save to database: {e}")
    
    return p1, p6, p24

def get_theme(aqi):
    if aqi <= 50: return "#00e400", "Good"
    if aqi <= 100: return "#ffff00", "Moderate"
    if aqi <= 200: return "#ff7e00", "Poor"
    if aqi <= 300: return "#ff0000", "Very Poor"
    return "#8f3f97", "Severe"

# --- 5. UI NAVIGATION ---
st.sidebar.title("🌫️ AQI Oracle")
page = st.sidebar.radio("Navigation", ["Live Dashboard", "Historical Trends", "Model Accuracy", "Health Advisories"])

m1, c1, m6, c6, m24, c24 = load_models()
live_data = get_live_data()

# ==========================================
# PAGE 1: LIVE DASHBOARD
# ==========================================
if page == "Live Dashboard":
    station = st.sidebar.selectbox("Select Station", ALL_LOCATIONS)
    station_pincode = LOCATIONS_DATA[station]["pincode"]
    current_aqi = live_data[station]["aqi"]
    pm25 = live_data[station]["pm25"]
    color, tag = get_theme(current_aqi)

    st.markdown(f"""
        <div style="background: linear-gradient(135deg, {color} 0%, #1E2129 150%); padding: 30px; border-radius: 20px; text-align: center; color: white; margin-bottom: 20px; box-shadow: 0 10px 20px rgba(0,0,0,0.5);">
            <h2 style="margin: 0; font-weight: 300; letter-spacing: 2px;">{station} ({station_pincode})</h2>
            <h1 style="font-size: 100px; margin: 10px 0; font-weight: bold; text-shadow: 2px 2px 8px rgba(0,0,0,0.5);">{current_aqi:.0f}</h1>
            <h3 style="margin: 0; margin-bottom: 15px;">{tag} Air Quality</h3>
            <div style="display: flex; justify-content: center; gap: 30px; font-size: 16px; opacity: 0.9;">
                <span>🌡️ {live_data[station]['temp']:.1f} °C</span>
                <span>💨 PM2.5: {pm25}</span>
                <span>🌪️ PM10: {live_data[station]['pm10']}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    cigs = pm25 / 22
    st.info(f"🚬 **Health Impact:** Breathing the air here is equivalent to smoking **{cigs:.1f} cigarettes** per day.")

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
            "location": k, "lat": LOCATIONS_DATA[k]["coords"][0], 
            "lon": LOCATIONS_DATA[k]["coords"][1], "aqi": v["aqi"],
            "color": [int(get_theme(v["aqi"])[0].lstrip('#')[i:i+2], 16) for i in (0,2,4)]
        })
    map_df = pd.DataFrame(map_data)

    layer_dots = pdk.Layer("ScatterplotLayer", map_df, get_position=["lon", "lat"], get_color="color", get_radius="aqi * 15", pickable=True, opacity=0.8)
    layer_text = pdk.Layer("TextLayer", map_df, get_position=["lon", "lat"], get_text="aqi", get_color=[255, 255, 255], get_size=14, get_alignment_baseline="'center'")

    st.pydeck_chart(pdk.Deck(
        layers=[layer_dots, layer_text], 
        initial_view_state=pdk.ViewState(latitude=28.6139, longitude=77.2090, zoom=10, pitch=30), 
        map_style="dark",
        tooltip={"text": "{location}\nAQI: {aqi}"}
    ))

# ==========================================
# PAGE 2: HISTORICAL TRENDS
# ==========================================
elif page == "Historical Trends":
    st.title("📈 Historical Trends (Official CPCB Data)")
    station = st.sidebar.selectbox("Select Station", ALL_LOCATIONS)
    file_name = LOCATIONS_DATA[station]["file"]
    
    df = load_station_excel(file_name)
    
    if df is None:
        st.error(f"Data file {file_name} not found.")
        st.stop()

    st.subheader("Daily AQI Chart")
    timeframe = st.radio("Select Timeframe", ["Last 24 Hours", "Last 7 Days", "Last 30 Days"], horizontal=True)
    
    if timeframe == "Last 24 Hours":
        df_chart = df.tail(24)
    elif timeframe == "Last 7 Days":
        df_chart = df.tail(7 * 24)
    else:
        df_chart = df.tail(30 * 24)

    df_chart['date_str'] = df_chart['dt'].dt.strftime('%d %b %H:%M')
    
    if timeframe == "Last 24 Hours":
        chart_data = df_chart
    else:
        chart_data = df_chart.groupby(df_chart['dt'].dt.date)['aqi_index'].mean().reset_index()
        chart_data['date_str'] = chart_data['dt'].astype(str)

    fig_bar = go.Figure(data=[go.Bar(
        x=chart_data['date_str'], y=chart_data['aqi_index'],
        marker_color=[get_theme(a)[0] for a in chart_data['aqi_index']],
        text=chart_data['aqi_index'].round(0),
        textposition='outside', textfont=dict(color='white')
    )])
    fig_bar.update_layout(plot_bgcolor='#0E1117', paper_bgcolor='#0E1117', font=dict(color='white'), xaxis_title="Time", yaxis_title="Average AQI", height=350, margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig_bar.to_dict(), use_container_width=True, config={'displayModeBar': False})

    # FIXED ERROR: Calendar
    st.subheader("📅 AQI Calendar")
    df_recent = df.sort_values("dt", ascending=False).head(30 * 24)
    df_recent['date'] = df_recent['dt'].dt.date
    daily = df_recent.groupby('date')['aqi_index'].mean().reset_index()
    
    # THE FIX: Fill NaNs with 0 before converting to string
    daily['aqi_index'] = daily['aqi_index'].fillna(0) 
    daily['date_str'] = daily['date'].astype(str) 
    
    fig_cal = go.Figure()
    for i, row in daily.iterrows():
        day_idx = row['date'].weekday()
        week_idx = (row['date'].day - 1) // 7
        fig_cal.add_trace(go.Scatter(
            x=[day_idx], y=[week_idx], mode="markers+text",
            marker=dict(symbol="square", size=50, color=get_theme(row['aqi_index'])[0]),
            text=str(int(row['aqi_index'])), textfont=dict(color="black", size=10),
            hoverinfo="text", hovertext=f"{row['date_str']}<br>AQI: {int(row['aqi_index'])}", showlegend=False
        ))
        
    fig_cal.update_layout(
        title="Last 30 Days Calendar",
        plot_bgcolor='#0E1117', paper_bgcolor='#0E1117', font=dict(color='white'),
        xaxis=dict(showgrid=False, showticklabels=False, range=[-0.5, 6.5]),
        yaxis=dict(showgrid=False, showticklabels=False, autorange="reversed"),
        height=300, margin=dict(l=0, r=0, t=40, b=0)
    )
    st.plotly_chart(fig_cal.to_dict(), use_container_width=True, config={'displayModeBar': False})

# ==========================================
# PAGE 3: MODEL ACCURACY (NEW!)
# ==========================================
elif page == "Model Accuracy":
    st.title("📊 Model Accuracy Dashboard")
    st.write("This page tracks the real-world performance of the AI. Every time you load the Live Dashboard, it saves a prediction to MongoDB.")
    
    try:
        cursor = predictions_collection.find().sort("timestamp", 1)
        df_log = pd.DataFrame(list(cursor))
    except Exception as e:
        st.error(f"Could not load data from MongoDB: {e}")
        df_log = pd.DataFrame()

    if df_log.empty:
        st.info("Collecting data... Please refresh the Live Dashboard to build history.")
    else:
        df_log['timestamp'] = pd.to_datetime(df_log['timestamp'])
        station = st.selectbox("Select Station to Analyze", ALL_LOCATIONS)
        df_station = df_log[df_log['location'] == station].sort_values('timestamp')
        
        if len(df_station) < 2:
            st.warning("Need more data for this specific station. Refresh the Live Dashboard a few more times.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_station['timestamp'], y=df_station['actual_aqi'], mode='lines+markers', name='Actual AQI (Live)', line=dict(color='#00e400', width=3)))
            fig.add_trace(go.Scatter(x=df_station['timestamp'], y=df_station['pred_1h'], mode='lines+markers', name='Predicted 1H AQI', line=dict(color='#ff7e00', dash='dash')))
            
            fig.update_layout(title=f"Real vs Predicted AQI for {station}", xaxis_title="Time", yaxis_title="AQI", template="plotly_dark", hovermode="x unified")
            st.plotly_chart(fig.to_dict(), use_container_width=True, config={'displayModeBar': False})

            df_station['error'] = abs(df_station['actual_aqi'] - df_station['pred_1h'])
            mae = df_station['error'].mean()
            
            c1, c2 = st.columns(2)
            c1.metric("Mean Absolute Error (MAE)", f"{mae:.2f} AQI points")
            c2.metric("Total Predictions Logged", f"{len(df_station)}")

# ==========================================
# PAGE 4: HEALTH ADVISORIES
# ==========================================
elif page == "Health Advisories":
    st.title("🏥 Health Advisories")
    station = st.sidebar.selectbox("Select Station", ALL_LOCATIONS)
    current_aqi = live_data[station]["aqi"]
    
    st.write(f"Current AQI in **{station}** is **{current_aqi:.0f}**.")
    
    if current_aqi > 300:
        st.error("🚨 **Hazardous!** Avoid all outdoor physical activity. Keep windows closed.")
    elif current_aqi > 200:
        st.warning("⚠️ **Very Poor!** Sensitive groups (Asthma, Heart) should stay indoors. Masks mandatory.")
    else:
        st.success("✅ **Moderate/Poor.** Safe for most, but sensitive groups should limit prolonged outdoor exertion.")

    st.divider()
    st.subheader("Specific Conditions")

    with st.expander("🫁 Asthma & Respiratory Issues"):
        st.write("**Risk Level:** High when AQI is Severe.")
        st.write("✅ **Do's:** Keep inhalers handy, use air purifiers indoors, wear an N95 mask if going out.")
        st.write("❌ **Don'ts:** Do not exercise outdoors, avoid smoking.")

    with st.expander("❤️ Heart Conditions"):
        st.write("**Risk Level:** High. Air pollution can increase blood pressure and heart rate.")
        st.write("✅ **Do's:** Stay indoors during early morning (worst AQI), monitor BP.")
        st.write("❌ **Don'ts:** Avoid strenuous outdoor activities.")

    with st.expander("🤧 Allergies & Sinus"):
        st.write("**Risk Level:** Moderate to High.")
        st.write("✅ **Do's:** Keep windows closed, take prescribed antihistamines.")
        st.write("❌ **Don'ts:** Do not hang clothes outside to dry.")
