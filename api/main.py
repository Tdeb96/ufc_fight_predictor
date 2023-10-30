import json

from fastapi import FastAPI
from xgboost import XGBClassifier

from data_processing import DataProcessor

app = FastAPI()
data_processor = DataProcessor(username="postgres", password="postgres")

clf = XGBClassifier()
clf.load_model("api/models/model.xgb")


@app.get("/")
def read_root():
    return {"message": "Welcome to the API"}


@app.get("/predict")
def predict(fighter1: str, fighter2: str):
    difference_df = data_processor.calculate_differences_on_inference(
        fighter1, fighter2
    )
    prediction = clf.predict_proba(difference_df)

    output_dict = {
        "win_prob": float(prediction[0][1]),
        "loss_prob": float(prediction[0][0]),
    }

    # Convert to json
    output_json = json.dumps(output_dict)
    return output_json
