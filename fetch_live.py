import requests
import pandas as pd
from datetime import datetime

API_KEY = "e50e446d36b0fab3e2faa1f354927a57" # Your key
LAT, LON = 28.6139, 77.2090 # Central Delhi

def get_live_data():
    # 1. Fetch Weather (Fixed URL on one line)
    wx_url = f"https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"
    wx_response = requests.get(wx_url)
    wx = wx_response.json()
    
    # Safety check: Did the API fail?
    if wx.get("cod") != 200:
        print(f"⚠️ Weather API Error: {wx.get('message')}")
        print("If it says 'Invalid API key', wait 1 hour and try again.")
        return None

    # 2. Fetch Air Pollution
    aqi_url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={API_KEY}"
    aqi_response = requests.get(aqi_url)
    aqi = aqi_response.json()
    
    # Extract data safely
    try:
        data = {
            "dt": [datetime.now()],
            "temp_c": [wx["main"]["temp"]],
            "humidity": [wx["main"]["humidity"]],
            "pressure_mb": [wx["main"]["pressure"]],
            "windspeed_kph": [wx["wind"]["speed"] * 3.6], # m/s to km/h
            "aqi_index": [aqi["list"][0]["components"]["pm2_5"] * 2], # Rough PM2.5 to AQI
            "pm2_5": [aqi["list"][0]["components"]["pm2_5"]],
            "pm10": [aqi["list"][0]["components"]["pm10"]]
        }
        
        df = pd.DataFrame(data)
        df.set_index("dt", inplace=True)
        return df
        
    except KeyError as e:
        print("⚠️ Could not find the data in the API response.")
        print("Here is what the API actually sent:")
        print(wx) # Print the raw response so we can see what's wrong
        return None

if __name__ == "__main__":
    print("Fetching live data...")
    live_df = get_live_data()
    if live_df is not None:
        print("\n✅ Success! Live data:")
        print(live_df)