import json

import requests
import streamlit as st

from data_processing import DataProcessor

# Define the base URL of your API
api_base_url = "http://api:8000"

# Initialize data generator and load data
data_generator = DataProcessor(username="postgres", password="postgres")
ufc_fighters = data_generator.load_fighters()

# Fetch the names of all fighters
fighter_names = ufc_fighters["fighter_name"]

# Main title and styling
st.title("🥊 UFC Fight Predictor 🥊")
st.markdown(
    """
<style>
.big-font {
    font-size:18px !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# Model selection
st.markdown("### Select Prediction Model")
model_choice = st.radio("Choose a model for prediction:", ("XGBoost", "Deep Learning"))

# User Inputs for fighters
st.markdown("### Select Fighters")
col1, col2 = st.columns(2)

with col1:
    selected_fighter1 = st.selectbox("Fighter 1", fighter_names)

with col2:
    selected_fighter2 = st.selectbox("Fighter 2", fighter_names)

# Prediction button
if st.button("Predict Fight Outcome"):
    # Define the specific API endpoint based on the selected model
    api_endpoint = "/predict_xgboost" if model_choice == "XGBoost" else "/predict_dl"

    # Define the API URL
    api_url = f"{api_base_url}{api_endpoint}"

    # Create a payload with fighter names
    payload = {"fighter1": selected_fighter1, "fighter2": selected_fighter2}

    # Make the API POST request
    response = requests.get(api_url, params=payload)

    # Display the result
    if response.status_code == 200:
        result = response.json()
        win_prob = result.get("win_prob", "N/A")
        loss_prob = result.get("loss_prob", "N/A")
        st.markdown(
            f"<p class='big-font'>Win Probability: {win_prob}</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<p class='big-font'>Loss Probability: {loss_prob}</p>",
            unsafe_allow_html=True,
        )
    else:
        st.error("Could not get a valid response from the API.")

# Additional styling
st.markdown(
    """
<style>
.streamlit-button {
    margin-top: 1rem;
}
</style>
""",
    unsafe_allow_html=True,
)
