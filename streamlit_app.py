# streamlit_app.py
import os
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import requests
from datetime import datetime
import joblib

from hybrid_predictor import predict_hybrid
from styles import apply_custom_styles, render_result_card

# -------------------------------------------------------------------
# CONFIG & STYLES
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Epidemic AI | Disease Outbreak Predictor",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_styles()

from dotenv import load_dotenv
load_dotenv()

# -------------------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------------------
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
FEATURE_COLUMNS = [
    "temp_max", "temp_min", "temp_avg", "rain_mm", "wind_kmh", 
    "humidity", "pressure", "cloud_cover", "dew_point", 
    "temp_range", "heat_index", "rain_3day_avg", "rain_7day_sum", 
    "month", "day_of_year"
]

# -------------------------------------------------------------------
# DATA LOADERS
# -------------------------------------------------------------------
@st.cache_resource
def get_resources():
    encoder = joblib.load("label_encoder.pkl")
    return encoder

encoder = get_resources()
class_names = list(encoder.classes_)

def idx_to_label(idx: int) -> str:
    return encoder.inverse_transform([idx])[0]

# -------------------------------------------------------------------
# WEATHER API
# -------------------------------------------------------------------
def get_live_weather(city: str):
    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    )

    r = requests.get(url)
    data = r.json()

    if r.status_code != 200:
        raise ValueError(data.get("message", "Weather API error"))

    temp_max = float(data["main"]["temp_max"])
    temp_min = float(data["main"]["temp_min"])
    temp_avg = (temp_max + temp_min) / 2

    humidity = float(data["main"]["humidity"])
    pressure = float(data["main"]["pressure"])
    cloud_cover = float(data.get("clouds", {}).get("all", 0))
    wind_kmh = float(data.get("wind", {}).get("speed", 0)) * 3.6
    rain_mm = float(data.get("rain", {}).get("1h", 0.0))

    dew_point = temp_min
    temp_range = temp_max - temp_min
    heat_index = temp_avg + humidity * 0.1

    today = datetime.now()

    features = [
        temp_max, temp_min, temp_avg, rain_mm, wind_kmh,
        humidity, pressure, cloud_cover, dew_point,
        temp_range, heat_index, 
        rain_mm,          # 3-day avg (proxy)
        rain_mm * 3,      # 7-day sum (proxy)
        today.month,
        today.timetuple().tm_yday
    ]

    return features, data

# -------------------------------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------------------------------
with st.sidebar:
    st.markdown("<h1 style='text-align: center; color: #00d2ff;'>🦠 Epidemic AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; opacity: 0.7;'>Advanced Disease Outbreak Prediction System</p>", unsafe_allow_html=True)
    st.divider()
    
    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "🔧 Manual Analysis", "📦 Batch Processing", "🧪 Model Insights"],
        index=0
    )
    
    st.divider()
    st.info("Ensuring global health security through predictive analytics.")

# -------------------------------------------------------------------
# PAGE: DASHBOARD (LIVE WEATHER)
# -------------------------------------------------------------------
if page == "🏠 Dashboard":
    st.header("🌍 Real-time Outbreak Dashboard")
    st.write("Predict disease risk based on current meteorological conditions in any city.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📍 Location")
        city_input = st.text_input("Enter City Name", "Chennai", placeholder="e.g. London, Tokyo, Mumbai")
        predict_btn = st.button("🚀 Analyze Risk", use_container_width=True)
        
    if predict_btn:
        with st.spinner(f"Fetching weather for {city_input}..."):
            try:
                features, weather_raw = get_live_weather(city_input)
                result = predict_hybrid(features)
                risk_score = result["risk_score"]
                
                with col2:
                    render_result_card(result["final_prediction"], risk_score)
                    
                    # Mini weather report
                    st.subheader("🌡 Current Conditions")
                    w_cols = st.columns(3)
                    w_cols[0].metric("Temp", f"{weather_raw['main']['temp']}°C")
                    w_cols[1].metric("Humidity", f"{weather_raw['main']['humidity']}%")
                    w_cols[2].metric("Wind", f"{weather_raw['wind']['speed']} m/s")
                
                st.divider()
                
                # Probability Distribution
                st.subheader("📊 Probability Distribution")
                df_probs = pd.DataFrame({
                    "Disease": class_names,
                    "Probability (%)": [p * 100 for p in result["ann_probabilities"]]
                }).sort_values(by="Probability (%)", ascending=False)
                
                fig = px.bar(
                    df_probs, 
                    x="Probability (%)", 
                    y="Disease", 
                    orientation='h',
                    color="Probability (%)",
                    color_continuous_scale="Viridis",
                    template="plotly_dark",
                    height=400
                )
                fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error fetching data: {str(e)}")

# -------------------------------------------------------------------
# PAGE: MANUAL ANALYSIS
# -------------------------------------------------------------------
elif page == "🔧 Manual Analysis":
    st.header("🔧 Manual Parameter Synthesis")
    st.write("Input specific weather parameters to simulate outbreak scenarios.")
    
    with st.form("manual_form"):
        c1, c2, c3 = st.columns(3)
        temp_max = c1.number_input("Max Temp (°C)", 0.0, 50.0, 30.0)
        temp_min = c2.number_input("Min Temp (°C)", 0.0, 40.0, 24.0)
        humidity = c3.number_input("Humidity (%)", 0.0, 100.0, 70.0)
        
        c4, c5, c6 = st.columns(3)
        rain_mm = c4.number_input("Rainfall (mm)", 0.0, 500.0, 10.0)
        pressure = c5.number_input("Pressure (hPa)", 900.0, 1100.0, 1013.0)
        wind_kmh = c6.number_input("Wind Speed (km/h)", 0.0, 150.0, 8.0)
        
        with st.expander("Advanced Parameters"):
            a1, a2 = st.columns(2)
            cloud_cover = a1.slider("Cloud Cover (%)", 0, 100, 50)
            dew_point = a2.slider("Dew Point (°C)", 0, 40, 22)
            
            a3, a4 = st.columns(2)
            rain_3day = a3.number_input("Past 3-Day Rain Avg (mm)", 0.0, 500.0, 10.0)
            rain_7day = a4.number_input("Past 7-Day Rain Sum (mm)", 0.0, 1000.0, 30.0)
            
            a5, a6 = st.columns(2)
            m_now = datetime.now().month
            d_now = datetime.now().timetuple().tm_yday
            month = a5.selectbox("Month", range(1, 13), index=m_now-1)
            day_of_year = a6.number_input("Day of Year (1-366)", 1, 366, d_now)
            
        submit = st.form_submit_button("🔮 Generate Prediction")
        
    if submit:
        # Derived features
        temp_avg = (temp_max + temp_min) / 2
        temp_range = temp_max - temp_min
        heat_index = temp_avg + (humidity * 0.1)
        
        features = [
            temp_max, temp_min, temp_avg, rain_mm, wind_kmh,
            humidity, pressure, cloud_cover, dew_point,
            temp_range, heat_index, rain_3day, rain_7day,
            month, day_of_year
        ]
        
        result = predict_hybrid(features)
        
        st.divider()
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            render_result_card(result["final_prediction"], result["risk_score"])
        
        with res_col2:
            st.subheader("🗳️ Ensemble Voting")
            v_cols = st.columns(3)
            v_cols[0].metric("ANN", idx_to_label(result["votes"]["ann"]))
            v_cols[1].metric("Random Forest", idx_to_label(result["votes"]["rf"]))
            v_cols[2].metric("XGBoost", idx_to_label(result["votes"]["xgb"]))

# -------------------------------------------------------------------
# PAGE: BATCH PROCESSING
# -------------------------------------------------------------------
elif page == "📦 Batch Processing":
    st.header("📦 Batch Prediction Pipeline")
    st.write("Upload a CSV file containing multiple weather records for bulk processing.")
    
    st.warning("⚠️ CSV must contain columns: " + ", ".join(FEATURE_COLUMNS))
    
    uploaded = st.file_uploader("Upload CSV", type="csv")
    
    if uploaded:
        df = pd.read_csv(uploaded)
        
        if not all(col in df.columns for col in FEATURE_COLUMNS):
            st.error("Missing required columns. Please check your CSV format.")
        else:
            if st.button("⚡ Process All Records"):
                progress_bar = st.progress(0)
                preds = []
                risks = []
                
                for i, row in df.iterrows():
                    features = [row[c] for c in FEATURE_COLUMNS]
                    res = predict_hybrid(features)
                    preds.append(res["final_prediction"])
                    risks.append(res["risk_score"])
                    progress_bar.progress((i + 1) / len(df))
                
                df["Predicted Disease"] = preds
                df["Risk Score (%)"] = risks
                
                st.success(f"Successfully processed {len(df)} records!")
                st.dataframe(df.style.background_gradient(subset=["Risk Score (%)"], cmap="YlOrRd"))
                
                # Download link
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Results", csv, "predictions.csv", "text/csv")

# -------------------------------------------------------------------
# PAGE: MODEL INSIGHTS
# -------------------------------------------------------------------
elif page == "🧪 Model Insights":
    st.header("🧪 Architectural Overview")
    
    st.info("""
    **Hybrid Ensemble Engine**
    - **Primary Model:** PyTorch Artificial Neural Network (ANN)
    - **Robustness Layer:** Random Forest & XGBoost Classifiers
    - **Optimization:** PCA-based Dimensionality Reduction
    """)
    
    st.markdown("""
    ### Statistical Foundations
    The system analyzes 15 meteorological variables to identify signatures associated with high-risk epidemiological outbreaks. 
    By combining deep learning with ensemble methods, we achieve higher accuracy than individual models.
    
    - **Sensitivity:** Optimized for early detection of potential outbreaks.
    - **Integration:** Hooks into OpenWeather API for real-time validation.
    """)
    
    # Simple architecture diagram or logo
    st.image("https://img.icons8.com/nolan/256/artificial-intelligence.png", width=128)

# -------------------------------------------------------------------
# FOOTER
# -------------------------------------------------------------------
st.divider()
st.caption("© 2026 Epidemic AI. Prepared for predictive clinical and environmental monitoring.")
