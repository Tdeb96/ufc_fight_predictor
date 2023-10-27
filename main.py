from ufc_fight_predictor import DataProcessor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)  # Set the desired logging level

# Create an instance of DataProcessor
data_processor = DataProcessor(username="postgres", password="postgres")

# Load data from the database
bouts, fighters = data_processor.load_data()

# Preprocess the loaded data
data_processor.preprocess_training_data(bouts, fighters)
