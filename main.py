import logging

from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models
from xgboost import XGBClassifier

from data_processing import DataProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)  # Set the desired logging level

# Create an instance of DataProcessor
data_processor = DataProcessor(
    username="postgres", password="postgres", server="localhost"
)

# Load data from the database
logging.info("Loading raw data from the database...")
bouts, fighters = data_processor.load_data()

# Preprocess the loaded data
logging.info("Preprocessing data...")
model_input = data_processor.preprocess_training_data(bouts, fighters)

# Split the data into X and y
X = model_input.drop(columns=["fighter1", "fighter2", "win"])
y = model_input["win"]

# Train XGBoost classifier with best parameters from the xgboost notebook
params = {
    "learning_rate": 0.1,
    "max_depth": 3,
    "n_estimators": 150,
    "reg_alpha": 0.0,
    "reg_lambda": 0.1,
}
logging.info("Training XGBClassifier...")
model = XGBClassifier(**params)
model.fit(X, y)
logging.info(
    "XGBClassifier trained successfully! Training accuracy: {}".format(
        model.score(X, y)
    )
)

# Save the model to the api/models directory
model.save_model("api/models/model.json")

# Train simple DL model with parameters obtained from neural_net notebook
params = {"num_units": 32, "dropout": 0.1, "optimizer": "adam"}

# Scale input parameters
scaler = StandardScaler().fit(X)
X = scaler.transform(X)


model = models.Sequential(
    [
        layers.Dense(params["num_units"], activation="relu"),
        layers.Dropout(params["dropout"]),
        layers.Dense(1, activation="sigmoid"),
    ]
)
model.compile(
    optimizer=params["optimizer"],
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

# Train model and capture history
logging.info("Training simple DL model...")
history = model.fit(X, y, epochs=20)
logging.info(
    "Simple DL model trained successfully! Training accuracy: {}".format(
        model.evaluate(X, y)[1]
    )
)

# Save model to api/models directory
model.save("api/models/dl.keras")

logging.info("Training of both models completed successfully!")
