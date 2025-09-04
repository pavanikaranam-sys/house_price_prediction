import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st
import pickle
import joblib
import sklearn
import numpy as np
import base64
import os
import json, re

st.set_page_config(
    page_title="Bangalore House price Prediction",
    page_icon="🏡️",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("🏡️ House Price Prediction")
st.write("Welcome to house price prediction app")


with open("house_price_prediction.pkl", "rb") as f:
    model = pickle.load(f)

with open("columns.json", "r") as f:
    data_columns = json.load(f)['data_columns']

location_cols = data_columns[3:]
location_map = {loc.replace("location_", ""): loc for loc in location_cols}

with st.sidebar:
    st.subheader("Please fill below house details:")
    location_selected = st.selectbox("Location", list(location_map.keys()))
    total_sqft = st.slider("Total sqft", 300, 5000)
    bhk = st.slider("No. of Bedrooms", 1,16,1)
    bath = st.slider("No. of Bathrooms", 1,16,1)

if total_sqft/bhk<300:
    st.warning("these type of houses does not exist")
if bath>bhk+1:
    st.warning("these type of houses does not exist")


st.markdown("⚠️ **Disclaimer**")

st.markdown("""
🔹️ Data in this app is limited to houses in Bangalore.  
🔹️ This app is for educational use only.  
🔹️ Do not rely on these predictions for financial or legal decisions.
""")
        
with st.expander("📘 How to Use This App"):
    st.markdown("""
    1. Enter information of house you are searching for.  
    2. Click **Predict** to check related house details.
    """)


if st.button("Predict"):

    try:
        x = np.zeros(len(data_columns))
        x[0] = total_sqft
        x[1] = bath
        x[2] = bhk

        loc_clean = location_selected
        
        if loc_clean in location_map:
            loc_column = location_map[loc_clean]          # e
            loc_index = data_columns.index(loc_column)
            x[loc_index] = 1

        x_df = pd.DataFrame([x], columns=data_columns)

        prediction = model.predict(x_df)[0]
        st.success(f"💰 Predicted Price: {prediction:.2f} Lakhs")

    except Exception as e:
        st.error(f"Error: {e}")

st.markdown("---")
st.markdown(
    "<center><small>Made with ❤️ by Pavani Karanam | 2025</small></center>",
    unsafe_allow_html=True
)
