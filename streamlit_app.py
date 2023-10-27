import streamlit as st
from ufc_fight_predictor import DataProcessor

# Initialize data generator and load data
data_generator = DataProcessor(username="postgres", password="postgres")
ufc_fighters = data_generator.load_fighters()

# Fetch the names of all fighters
fighter_names = ufc_fighters['fighter_name']

# Main title
st.title("UFC Fight Predictor")

# User Inputs
st.markdown("### Select Fighters")
col1, col2 = st.columns(2)

with col1:
    selected_fighter1 = st.selectbox("Fighter 1", fighter_names)

with col2:
    selected_fighter2 = st.selectbox("Fighter 2", fighter_names)

# Add Calculate button
calculate_button = st.button("Calculate")

if calculate_button:
    # Display selected fighters
    st.markdown("## Selected Fighters")
    st.write(f"🥷 Fighter 1: **{selected_fighter1}**")
    st.write(f"🥷 Fighter 2: **{selected_fighter2}**")

    # Perform some operation based on user input (e.g., prediction)
    difference = data_generator.calculate_differences_on_inference(selected_fighter1, selected_fighter2)
    result = f"The difference in UFC wins between {selected_fighter1} and {selected_fighter2} is {int(difference.ufc_wins_diff.values[0])}"

    # Display result
    st.markdown("## Prediction")
    st.write(result)

    # Further visualization (e.g., charts, graphs) can go here
