import json

import requests
import streamlit as st

from data_processing import DataProcessor

# API URL
url = "http://api:8000/predict"

# Initialize data generator and load data
data_generator = DataProcessor(username="postgres", password="postgres")
ufc_fighters = data_generator.load_fighters()

# Fetch the names of all fighters
fighter_names = ufc_fighters["fighter_name"]

# Main title
st.title("UFC Fight Predictor")

# User Inputs
st.markdown("### Select Fighters")
col1, col2 = st.columns(2)

with col1:
    selected_fighter1 = st.selectbox("Fighter 1", fighter_names)

with col2:
    selected_fighter2 = st.selectbox("Fighter 2", fighter_names)

# Making a GET request to the FastAPI server running on localhost
if st.button("Predict"):
    response = requests.get(
        url, params={"fighter1": selected_fighter1, "fighter2": selected_fighter2}
    )
    if response.status_code == 200:
        result = response.json()
        result = json.loads(result)
        st.write(f"Win Probability: {result['win_prob']}")
        st.write(f"Loss Probability: {result['loss_prob']}")
    else:
        st.write("Could not get a valid response from the API.")
