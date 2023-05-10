import pandas as pd 
import logging
from utils import get_db_engine

## Of course just hardcoded because of local development
POSTGRES_USERNAME = 'postgres'
POSTGRES_PASSWORD = 'postgres'

engine = get_db_engine(POSTGRES_USERNAME, POSTGRES_PASSWORD)
bouts = pd.read_csv('data/bouts.csv')
fighters = pd.read_csv('data/fighters.csv')

logging.info('Uploading bouts and fighters csv files to postgres server')
with engine.connect() as conn:
    conn.execute("CREATE SCHEMA IF NOT EXISTS ufc")
    bouts.to_sql('bouts', conn, 'ufc', if_exists='replace')
    fighters.to_sql('fighters', conn, 'ufc', if_exists='replace')