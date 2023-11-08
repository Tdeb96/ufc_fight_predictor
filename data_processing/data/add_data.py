import logging

import pandas as pd
from utils import get_db_engine

## Of course just hardcoded because of local development
POSTGRES_USERNAME = "postgres"
POSTGRES_PASSWORD = "postgres"

engine = get_db_engine(POSTGRES_USERNAME, POSTGRES_PASSWORD)
bouts = pd.read_csv("data/bouts.csv")
fighters = pd.read_csv("data/fighters.csv")
fighters_cleaned = pd.read_csv("data/fighters_cleaned.csv")
model_input = pd.read_csv("data/model_input.csv")
time_based_inference_df = pd.read_csv("data/time_based_inference_df.csv")

logging.info("Uploading bouts and fighters csv files to postgres server")
with engine.connect() as conn:
    conn.execute("CREATE SCHEMA IF NOT EXISTS ufc")
    bouts.to_sql(
        "bouts",
        conn,
        "ufc",
        if_exists="replace",
    )
    fighters.to_sql("fighters", conn, "ufc", if_exists="replace", index=False)
    fighters_cleaned.to_sql(
        "fighters_cleaned", conn, "ufc", if_exists="replace", index=False
    )
    model_input.to_sql("model_input", conn, "ufc", if_exists="replace", index=False)
    time_based_inference_df.to_sql(
        "time_based_inference_df", conn, "ufc", if_exists="replace", index=False
    )
