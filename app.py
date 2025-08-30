import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st
import pickle
import numpy as np
from fpdf import FPDF
import base64
import os
import json, re

st.title("🏡️ House Price Prediction")
st.markdown("Kindly provide the required details in the form below.")

df = pd.read_csv("/home/user/Desktop/bhk/model/home_prices.csv")

# Load model
with open("/home/user/Desktop/bhk/server/house_price.pkl", "rb") as f:
    model = pickle.load(f)

# Load feature columns
with open("/home/user/Desktop/bhk/server/columns.json", "r") as f:
    data_columns = json.load(f)['data_columns']



# Extract locations (all cols from index 3 onwards are one-hot encoded locations)
locations = data_columns[3:]
location_map = {loc.replace("location_", "").lower(): loc for loc in locations}

with st.sidebar:
    st.subheader("Please fill below house details:")
    with st.form("house_form"):
        location = st.selectbox("Location", list(location_map.keys()))
        total_sqft = st.slider("Total sqft", 300, 30000)
        bhk = st.slider("No. of Bedrooms", 1, 20, 2)
        bath = st.slider("No. of Bathrooms", 1, 20, 2)
        submitted = st.form_submit_button("Submit")
        if submitted:
            st.balloons()
            st.write("✅ Location:", location)
            st.write("✅ Total sqft:", total_sqft)
            st.write("✅ Bedrooms:", bhk)
            st.write("✅ Bathrooms:", bath)

st.write("Available features:", data_columns)

def normalize(text):
    text = text.lower()                     # lowercase
    text = text.strip()                     # remove extra spaces
    text = re.sub(r'[^a-z0-9 ]', ' ', text) # replace punctuation with space
    text = re.sub(r'\s+', ' ', text)        # collapse multiple spaces
    return text

df.columns=[normalize(c) for c in df.columns]
data_columns_lower=[normalize(col) for col in data_columns]
locations=[normalize(c) for c in locations]

data_columns=[c.strip().lower() for c in data_columns]
locations=[c.replace("location_", "").strip().lower() for c in locations]

st.write("Sample from data_columns:", data_columns[:20])
st.write("Sample from locations:", locations[:20])

    
# Prediction button
if st.button("Predict"):
    
    try:
        # Prepare feature vector
        x = np.zeros(len(data_columns))
        x[0] = total_sqft
        x[1] = bath
        x[2] = bhk

        # Directly match location (since your JSON already has raw names like 'Bellandur')
        if locations in data_columns:
            loc_index = data_columns.index(location)
            x[loc_index] = 1
        else:
            st.error(f"Location '{location}' not found in features")
            st.stop()

        # Convert numpy array into dataframe with column names
        x_df = pd.DataFrame([x], columns=data_columns)

        # Make prediction
        prediction = model.predict(x_df)[0]
        st.success(f"💰 Predicted Price: {round(prediction, 2)} Lakhs")

    except Exception as e:
        st.error(f"Error: {e}")

