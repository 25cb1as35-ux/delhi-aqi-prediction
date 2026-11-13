import requests
import pandas as pd
import time
from datetime import datetime
from pathlib import Path

WAQI_TOKEN = "98843231627b634af609f4cc503fd5b84c01d788"
DB_FILE = Path("outputs/aqi_weather_history.csv")

stations = {
    "Anand Vihar": (28.6469, 77.3160), "Connaught Place": (28.6304, 77.2177),
    "Dwarka": (28.5921, 77.0460), "IGI Airport": (28.5562, 77.1000),
    "Okhla Phase III": (28.5432, 77.2725), "Rohini": (28.7320, 77.1123)
}

def fetch_and_save():
    now = datetime.now().strftime("%Y-%m-%d %H:00:00")
    rows = []
    
    # Get live weather (Delhi center)
    wx_url = "https://api.open-meteo.com/v1/forecast?latitude=28.6139&longitude=77.2090&current=temperature_2m,relative_humidity_2m,wind_speed_10m"
    try:
        wx = requests.get(wx_url).json().get("current", {})
    except:
        wx = {}

    # Get live AQI
    for name, (lat, lon) in stations.items():
        try:
            res = requests.get(f"https://api.waqi.info/feed/geo:{lat};{lon}/?token={WAQI_TOKEN}").json()
            if res["status"] == "ok":
                d = res["data"]
                rows.append({
                    "dt": now, "location": name,
                    "aqi_index": float(d["aqi"]),
                    "temp_c": float(wx.get("temperature_2m", 20)),
                    "humidity": float(wx.get("relative_humidity_2m", 60)),
                    "windspeed_kph": float(wx.get("wind_speed_10m", 5))
                })
        except: pass
        
    if rows:
        df = pd.DataFrame(rows)
        # If file exists, append, else create
        df.to_csv(DB_FILE, mode='a', header=not DB_FILE.exists(), index=False)
        print(f"✅ Added {len(rows)} rows at {now}")

if __name__ == "__main__":
    while True:
        fetch_and_save()
        print("Sleeping for 1 hour...")
        time.sleep(3600) # Wait 1 hour