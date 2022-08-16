from sqlalchemy import create_engine
from sqlalchemy.engine.base import Engine
import pandas as pd 
import os
import logging
import csv

def get_db_engine(
    username: str,
    password: str,
    protocol: str = "postgresql",
    server: str = "timescale",
    port: int = 5432,
    dbname: str = "ufc",
) -> Engine:
    engine = create_engine(
        f"{protocol}://" f"{username}:" f"{password}@" f"{server}:" f"{port}/" f"{dbname}"
    )
    return engine

engine = get_db_engine(os.environ.get("POSTGRES_USERNAME"), os.environ.get("POSTGRES_PASSWORD"))
bouts = pd.read_csv('data/bouts.csv')
fighters = pd.read_csv('data/fighters.csv', quoting=csv.QUOTE_NONE)

logging.info('Uploading bouts and fighters csv files to postgres server')
with engine.connect() as conn:
    conn.execute("CREATE SCHEMA IF NOT EXISTS ufc")
    bouts.to_sql('bouts', conn, 'ufc', if_exists='replace')
    fighters.to_sql('bouts', conn, 'ufc', if_exists='replace')