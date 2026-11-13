# 🌫️ Delhi AQI Oracle: Real-Time Machine Learning Forecasting

**Live Demo:**
**Research Paper:** See `paper_draft.md` in the repository.

## 📖 Overview
Delhi AQI Oracle is a production-grade, real-time Air Quality Index (AQI) forecasting platform. Unlike standard AQI websites that only show current data, this application uses a custom-trained Machine Learning pipeline to predict AQI for the **next 1 hour, 6 hours, and 24 hours** across 6 major Delhi monitoring stations.

## 🚀 The Research (Scenario A vs. Scenario B)
This project is the culmination of a deep research study comparing two data scenarios:
*   **Scenario A:** A clean, tabular Kaggle Churn dataset (Classification).
*   **Scenario B (This Project):** A highly messy, real-world Delhi AQI dataset (Time-Series Regression).

### Key Findings:
1.  **Model Ranking Inversion:** The model that won on the clean churn data (Tuned XGBoost) completely failed on the time-series AQI data, proving that model superiority is *data-regime conditional*.
2.  **Data Leakage Audit:** A critical feature leakage was discovered and resolved (R² dropping from 0.99 to -1.27 when all history was removed), proving the pipeline's temporal integrity.
3.  **The 24-Hour Time-Travel Problem:** Standard ML models predict mean-reversion (causing a fake drop in AQI). This project solves it using **Quantile Loss (90th percentile)** and integrating future meteorological forecasts (Open-Meteo).

## 🛠️ Tech Stack
*   **Machine Learning:** XGBoost, Scikit-Learn (Quantile Regression, Time-Series Splits).
*   **Data Engineering:** Pandas, NumPy, WAQI API (Live Ground Data), Open-Meteo API (Weather).
*   **Frontend/UI:** Streamlit, PyDeck (Interactive Mapping).
*   **Data Sources:** Historical 2025 Kaggle CSV, Official CPCB 2026 Government `.xlsx` files.

## ⚙️ How it Works
1.  Fetches live PM2.5 and PM10 from CPCB sensors via WAQI.
2.  Applies the official Government of India CPCB mathematical formula to calculate real-time AQI.
3.  Pulls live weather forecasts (wind speed, temperature).
4.  Feeds this data into 3 specialized XGBoost models (1h, 6h, 24h) trained on 2025 data and 2026 government data.
5.  Outputs a smooth, accurate forecast and visualizes it on an interactive dark-mode map.

## 📊 Model Performance (1-Hour Forecast)
*   **Best Model:** XGBoost (Quantile 0.9)
*   **RMSE:** 3.41
*   **R²:** 0.9883
*   **Improvement over Persistence Baseline:** 12.6%

## 💻 Run Locally
```bash
git clone https://github.com/yourusername/delhi-aqi-prediction.git
cd delhi-aqi-prediction
pip install -r requirements.txt
streamlit run app.py
