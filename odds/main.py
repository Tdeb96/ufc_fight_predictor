import logging
import os

import pandas as pd
import requests
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine.base import Engine

load_dotenv()

# Fixed variables
SPORT = "mma_mixed_martial_arts"
ODDS_FORMAT = "decimal"
DATE_FORMAT = "iso"
REGIONS = "eu"

# import api_key from .env
API_KEY = os.getenv("API_KEY")


class OddsLoader:
    def __init__(
        self,
        username: str,
        password: str,
        protocol: str = "postgresql",
        server: str = "timescale",
        port: int = 5432,
        dbname: str = "ufc",
    ):
        self.db_engine = self.get_db_engine(
            username, password, protocol, server, port, dbname
        )
        self.logger = logging.getLogger(__name__)

    def get_db_engine(
        self,
        username: str,
        password: str,
        protocol: str = "postgresql",
        server: str = "timescale",
        port: int = 5432,
        dbname: str = "ufc",
    ) -> Engine:
        engine = create_engine(
            f"{protocol}://"
            f"{username}:"
            f"{password}@"
            f"{server}:"
            f"{port}/"
            f"{dbname}",
            isolation_level="AUTOCOMMIT",
        )
        return engine

    def get_historic_odds(self, date: str) -> pd.DataFrame:
        # Turn date from 'YYYY-MM-DD' to iso format
        date = pd.to_datetime(date).isoformat()

        historic_odds_response = requests.get(
            f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds-history",
            params={
                "api_key": API_KEY,
                "oddsFormat": ODDS_FORMAT,
                "dateFormat": DATE_FORMAT,
                "regions": REGIONS,
                "date": date + "Z",
            },
        )

        df_historic_odds = pd.json_normalize(
            historic_odds_response.json()["data"],
            record_path=["bookmakers", "markets", "outcomes"],
            meta=[
                "id",
                "sport_key",
                "sport_title",
                "commence_time",
                "home_team",
                "away_team",
                ["bookmakers", "key"],
                ["bookmakers", "title"],
                ["bookmakers", "last_update"],
            ],
            errors="ignore",
        )
        # Convert commence_time and bookmakers.last_update from iso to datetime
        df_historic_odds["commence_time"] = pd.to_datetime(
            df_historic_odds["commence_time"]
        )
        df_historic_odds["bookmakers.last_update"] = pd.to_datetime(
            df_historic_odds["bookmakers.last_update"]
        )
        return df_historic_odds

    def flatten_odds_response(self, odds_response) -> pd.DataFrame:
        flattened_data = []

        for event in odds_response.json():
            for bookmaker in event["bookmakers"]:
                for market in bookmaker["markets"]:
                    for outcome in market["outcomes"]:
                        flattened_data.append(
                            {
                                "id": event["id"],
                                "sport_key": event["sport_key"],
                                "sport_title": event["sport_title"],
                                "commence_time": pd.to_datetime(event["commence_time"]),
                                "home_team": event["home_team"],
                                "away_team": event["away_team"],
                                "bookmaker_key": bookmaker["key"],
                                "bookmaker_title": bookmaker["title"],
                                "bookmaker_last_update": pd.to_datetime(
                                    bookmaker["last_update"]
                                ),
                                "market_key": market["key"],
                                "market_last_update": pd.to_datetime(
                                    market["last_update"]
                                ),
                                "outcome_name": outcome["name"],
                                "outcome_price": outcome["price"],
                            }
                        )
        return pd.DataFrame(flattened_data)

    def get_future_odds(self) -> pd.DataFrame:
        odds_response = requests.get(
            f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds",
            params={
                "api_key": API_KEY,
                "oddsFormat": ODDS_FORMAT,
                "dateFormat": DATE_FORMAT,
                "regions": REGIONS,
            },
        )

        df_future_odds = self.flatten_odds_response(odds_response)
        df_future_odds["commence_time"] = pd.to_datetime(
            df_future_odds["commence_time"]
        )
        df_future_odds["market_last_update"] = pd.to_datetime(
            df_future_odds["market_last_update"]
        )

        return df_future_odds

    def get_bouts_double(self) -> pd.DataFrame:
        bouts_double = pd.read_sql_query(
            "SELECT * FROM ufc.bouts_double where event_date not in (select distinct event_date from ufc.odds)",
            con=self.db_engine,
            parse_dates=["date"],
        )
        return bouts_double

    def get_odds_after_2022(self, bouts_double) -> pd.DataFrame:
        def process_group(group):
            self.logger.info(
                f"Processing group with event_date {group['event_date'].iloc[0]}"
            )
            event_date = group["event_date"].iloc[
                0
            ]  # Get the event date from the group
            df_odds = self.get_historic_odds(event_date)

            # Merge the group and df_odds
            df_merged = pd.merge(
                group,
                df_odds,
                how="left",
                left_on=["fighter", "opponent"],
                right_on=["home_team", "away_team"],
            )
            df_merged = df_merged.drop(columns=["home_team", "away_team"])
            df_merged.dropna(inplace=True)

            return df_merged

        # Filter bouts_double to the fights after 2022
        bouts_double_after_2022 = bouts_double[
            bouts_double["event_date"] > "2022-01-01"
        ]

        # Group by event_date and apply the process_group function
        output_df = bouts_double_after_2022.groupby(
            "event_date", group_keys=False
        ).apply(process_group)

        # Reset the index of the resulting DataFrame
        output_df.reset_index(drop=True, inplace=True)

        return output_df

    def run_historic_odds(self):
        df_bouts = self.get_bouts_double()
        odds = self.get_odds_after_2022(df_bouts)
        self.logger.info(f"Inserting {len(odds)} rows into ufc.odds")
        odds.to_sql(
            "odds", con=self.db_engine, schema="ufc", if_exists="append", index=False
        )
        return odds

    def run_future_odds(self):
        df_future_odds = self.get_future_odds()
        self.logger.info(f"Inserting {len(df_future_odds)} rows into ufc.odds")
        df_future_odds.to_sql(
            "odds", con=self.db_engine, schema="ufc", if_exists="append", index=False
        )
        return df_future_odds


odds = OddsLoader(username="postgres", password="postgres", server="localhost")
odds.run_historic_odds()
