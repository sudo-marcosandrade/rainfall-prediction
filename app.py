import joblib
import pandas as pd
import streamlit as st

from src.config import MODELS_DIR


st.set_page_config(page_title="Rainfall Prediction", layout="wide")

st.title("Rainfall Prediction")
st.caption("Predict the probability of rain tomorrow from today's weather conditions.")

model_path = MODELS_DIR / "rainfall_model.joblib"

if not model_path.exists():
    st.warning("Train the model first with: python -m src.train")
    st.stop()

model = joblib.load(model_path)

left, right = st.columns(2)

with left:
    location = st.selectbox("Location", ["Sydney", "Melbourne", "Brisbane", "Perth", "Adelaide", "Albury"])
    min_temp = st.number_input("Minimum temperature", value=13.0)
    max_temp = st.number_input("Maximum temperature", value=24.0)
    rainfall = st.number_input("Rainfall today", min_value=0.0, value=0.0)
    humidity_3pm = st.slider("Humidity at 3pm", 0, 100, 55)

with right:
    pressure_9am = st.number_input("Pressure at 9am", value=1013.0)
    pressure_3pm = st.number_input("Pressure at 3pm", value=1010.0)
    wind_gust_speed = st.number_input("Wind gust speed", min_value=0.0, value=35.0)
    rain_today = st.selectbox("Rain today?", ["No", "Yes"])
    month = st.slider("Month", 1, 12, 7)

sample = pd.DataFrame(
    [
        {
            "Location": location,
            "MinTemp": min_temp,
            "MaxTemp": max_temp,
            "Rainfall": rainfall,
            "Evaporation": None,
            "Sunshine": None,
            "WindGustDir": "W",
            "WindGustSpeed": wind_gust_speed,
            "WindDir9am": "W",
            "WindDir3pm": "W",
            "WindSpeed9am": None,
            "WindSpeed3pm": None,
            "Humidity9am": None,
            "Humidity3pm": humidity_3pm,
            "Pressure9am": pressure_9am,
            "Pressure3pm": pressure_3pm,
            "Cloud9am": None,
            "Cloud3pm": None,
            "Temp9am": None,
            "Temp3pm": None,
            "RainToday": rain_today,
            "Year": 2017,
            "Month": month,
            "Day": 15,
        }
    ]
)

probability = model.predict_proba(sample)[0, 1]

st.metric("Probability of rain tomorrow", f"{probability:.1%}")
st.progress(float(probability))

if probability >= 0.5:
    st.success("Prediction: rain tomorrow")
else:
    st.info("Prediction: no rain tomorrow")
