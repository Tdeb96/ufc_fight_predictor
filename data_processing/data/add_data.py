import logging

import pandas as pd
from utils import get_db_engine

## Of course just hardcoded because of local development
POSTGRES_USERNAME = "postgres"
POSTGRES_PASSWORD = "postgres"

with open("table_generation.sql", "r") as file:
    sql_commands = file.read()

engine = get_db_engine(POSTGRES_USERNAME, POSTGRES_PASSWORD)
bouts = pd.read_csv("data/bouts.csv")
fighters = pd.read_csv("data/fighters.csv")
model_input = pd.read_csv("data/model_input.csv")
time_based_inference_df = pd.read_csv("data/time_based_inference_df.csv")
df_bouts_double = pd.read_csv("data/bouts_double.csv")
df_fighter_stats_no_age = pd.read_csv("data/fighter_stats_no_age.csv")
odds = pd.read_csv("data/odds.csv")

logging.info("Uploading bouts and fighters csv files to postgres server")
with engine.connect() as conn:
    conn.execute(sql_commands)
    bouts.to_sql("bouts", conn, "ufc", if_exists="replace", index=False)
    fighters.to_sql("fighters", conn, "ufc", if_exists="replace", index=False)
    model_input.to_sql("model_input", conn, "ufc", if_exists="replace", index=False)
    time_based_inference_df.to_sql(
        "time_based_inference_df", conn, "ufc", if_exists="replace", index=False
    )
    df_bouts_double.to_sql(
        "bouts_double", conn, "ufc", if_exists="replace", index=False
    )
    df_fighter_stats_no_age.to_sql(
        "fighter_stats_no_age", conn, "ufc", if_exists="replace", index=False
    )
    odds.to_sql("odds", conn, "ufc", if_exists="replace", index=False)
