import logging

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from xgboost import XGBClassifier

from data_processing import DataProcessor

logging.basicConfig(level=logging.INFO)

app = FastAPI()
data_processor = DataProcessor(username="postgres", password="postgres")

# Load models
xgboost_clf = XGBClassifier()
xgboost_clf.load_model("api/models/model.json")

dl_model = load_model("api/models/dl.keras")


def format_prediction(prob_class_0: float, prob_class_1: float) -> dict:
    return {
        "win_prob": prob_class_1,
        "loss_prob": prob_class_0,
    }


@app.get("/", response_model=dict)
def read_root():
    return {"message": "Welcome to the API"}


@app.get("/predict_xgboost")
def predict_xgboost(fighter1: str, fighter2: str):
    difference_df = data_processor.calculate_diff_on_inference(fighter1, fighter2)
    difference_df.drop("event_date", axis=1, inplace=True)
    difference_df.drop("fighter1", axis=1, inplace=True)
    difference_df.drop("fighter2", axis=1, inplace=True)
    logging.info(difference_df.columns)
    prediction = xgboost_clf.predict_proba(difference_df)
    return format_prediction(float(prediction[0][0]), float(prediction[0][1]))


@app.get("/predict_dl", response_model=dict)
def predict_dl(fighter1: str, fighter2: str):
    difference_df = data_processor.calculate_differences_on_inference(
        fighter1, fighter2
    )
    difference_np = np.array(difference_df).reshape(1, -1)
    prediction = dl_model.predict(difference_np)
    return format_prediction(float(1 - prediction[0][0]), float(prediction[0][0]))
