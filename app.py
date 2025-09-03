import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st
import pickle
import joblib
import sklearn
import numpy as np
from fpdf import FPDF
import base64
import os
import json, re

st.set_page_config(
    page_title="House price Prediction",
    page_icon="🏡️",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("🏡️ House Price Prediction")
st.write("Welcome to house price prediction app")

# Use relative paths for deployment
#df = pd.read_csv("/home/user/Desktop/bhk/model/home_prices.csv")

model = joblib.load("house_price.pkl")

# Load feature columns
with open("columns.json", "r") as f:
    data_columns = json.load(f)['data_columns']

# Extract location columns (all cols from index 3 onwards are one-hot encoded locations)
location_cols = data_columns[3:]
# Map clean names to actual one-hot columns
location_map = {loc.replace("location_", ""): loc for loc in location_cols}

with st.sidebar:
    st.subheader("Please fill below house details:")
    location_selected = st.selectbox("Location", list(location_map.keys()))
    total_sqft = st.slider("Total sqft", 300, 5000)
    bhk = st.slider("No. of Bedrooms", 1,16,1)
    bath = st.slider("No. of Bathrooms", 1,16,1)

if total_sqft/bhk<300:
    st.warning("these type of houses does not exist")
if bath>bhk:
    st.warning("these type of houses does not exist")


with st.expander("📘 How to Use This App"):
    st.markdown("""
    1. Enter information of house you are searching for.  
    2. Click **Predict** to check related house details.
    """)

# Prediction button
if st.button("Predict"):

    try:
        # Prepare feature vector
        x = np.zeros(len(data_columns))
        x[0] = total_sqft
        x[1] = bath
        x[2] = bhk

        loc_clean = location_selected  # selected from dropdown
        if loc_clean in location_map:
            loc_column = location_map[loc_clean]          # exact column name
            loc_index = data_columns.index(loc_column)
            x[loc_index] = 1

        # Convert numpy array into dataframe with column names
        x_df = pd.DataFrame([x], columns=data_columns)

        # Make prediction
        prediction = model.predict(x_df)[0]
        new_prediction=round(prediction,2)
        st.success(f"💰 Predicted Price: {new_prediction} Lakhs")

    except Exception as e:
        st.error(f"Error: {e}")
