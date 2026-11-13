# Delhi AQI Oracle 🌫️

An end-to-end, real-time Machine Learning pipeline and dashboard for forecasting Air Quality Index (AQI) across 6 major monitoring stations in Delhi.

## 🚀 Features
- **Real-Time Data Ingestion:** Fetches live ground-truth data using WAQI and calculates official CPCB AQI.
- **Live Weather Integration:** Pulls live meteorological data from Open-Meteo to feed future weather features to the model.
- **Advanced Forecasting:** Uses XGBoost models trained on historical 2025 data + 2026 government data using a Quantile Loss function (90th percentile) to predict extreme winter pollution events.
- **Predictions:** Forecasts AQI for +1 Hour, +6 Hours, and +24 Hours.
- **Interactive UI:** Streamlit dashboard with a live map, health advisories, and pollutant breakdown.

## 🛠️ Tech Stack
- **Machine Learning:** XGBoost, Scikit-Learn, Pandas, Numpy
- **Data Sources:** WAQI API, Open-Meteo API, CPCB (Government of India)
- **Frontend:** Streamlit, PyDeck (Maps)

## 🏃‍♂️ How to Run Locally
1. Clone the repository.
2. Install requirements: `pip install -r requirements.txt`
3. Add your WAQI API token to `app.py`.
4. Run the app: `streamlit run app.py`
