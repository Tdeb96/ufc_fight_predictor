import logging

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine.base import Engine


class DataProcessor:
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

    def load_data(self):
        with self.db_engine.connect() as conn:
            self.logger.info("Loading data from the database...")
            bouts = pd.read_sql("SELECT * FROM ufc.bouts", con=conn)
            bouts.drop(["event_url"], axis=1, inplace=True)
            fighters = pd.read_sql("SELECT * FROM ufc.fighters", con=conn)
            fighters.drop(["fighter_url"], axis=1, inplace=True)
            self.logger.info("Data loaded successfully.")
        return bouts, fighters

    def load_fighters(self):
        with self.db_engine.connect() as conn:
            self.logger.info("Loading data from the database...")
            fighters = pd.read_sql(
                "SELECT * FROM ufc.fighters_cleaned order by ufc_wins desc", con=conn
            )
            self.logger.info("Data loaded successfully.")
        return fighters

    def preprocess_training_data(self, bouts: pd.DataFrame, fighters: pd.DataFrame):
        self.logger.info("Starting data preprocessing...")

        # First convert bouts date to datetime
        bouts["event_date"] = pd.to_datetime(bouts["event_date"], format="%B %d %Y")
        self.logger.info("Converted event_date to datetime.")

        # Filter out the fighters who did not have any fights yet
        fighters_that_fought = set(bouts.fighter1).union(set(bouts.fighter2))
        fighters = fighters.loc[fighters.fighter_name.isin(fighters_that_fought)]
        self.logger.info("Filtered fighters who had fights.")

        # Extract total wins and losses
        fighters["total_wins"] = fighters.fighter_record.map(
            lambda x: x.split("Record: ")[1].split("-")[0]
        )
        fighters["total_losses"] = fighters.fighter_record.map(
            lambda x: x.split("Record: ")[1].split("-")[1]
        )
        fighters["total_draws"] = fighters.fighter_record.map(
            lambda x: x.split("Record: ")[1].split("-")[2].split("(")[0]
        )
        self.logger.info("Extracted total wins, losses, and draws.")

        # Drop fighter_record and index columns
        fighters = fighters.drop(["fighter_record"], axis=1)

        # Clean date column
        fighters.date_of_birth = fighters.date_of_birth.replace("--", None)
        fighters["date_of_birth"] = pd.to_datetime(
            fighters["date_of_birth"], format="%b %d %Y"
        )
        self.logger.info("Cleaned date_of_birth column.")

        # Turn height into centimeters
        fighters.height = fighters.height.replace("--", None)

        # Drop na values for height
        fighters = fighters.dropna(subset=["height"])

        fighters["height_feet"] = fighters.height.map(lambda x: int(x.split("' ")[0]))
        fighters["height_inch"] = fighters.height.map(
            lambda x: int(x.split("' ")[1].replace('"', ""))
        )
        fighters["height_cm"] = (
            30.48 * fighters["height_feet"] + 2.54 * fighters["height_inch"]
        )
        fighters = fighters.drop(["height", "height_feet", "height_inch"], axis=1)
        self.logger.info("Converted height to centimeters.")

        # check if there are fighters with the same name
        duplicated_fighters = fighters[
            fighters.duplicated(subset="fighter_name", keep=False)
        ]
        if not duplicated_fighters.empty:
            self.logger.warning("There are fighters with the same name.")

        # note that we have several fighters who have the same names
        # Fortunately, they belong to different weight classes
        fighters.loc[
            (fighters.fighter_name == "Michael McDonald") & (fighters.weight == 205),
            "fighter_name",
        ] = "Michael McDonald 205"
        fighters.loc[
            (fighters.fighter_name == "Joey Gomez") & (fighters.weight == 155),
            "fighter_name",
        ] = "Joey Gomez 155"
        fighters.loc[
            (fighters.fighter_name == "Mike Davis") & (fighters.weight == 145),
            "fighter_name",
        ] = "Mike Davis 145"
        fighters.loc[
            (fighters.fighter_name == "Bruno Silva") & (fighters.weight == 205),
            "fighter_name",
        ] = "Bruno Silva 125"

        # Some fighters do not have statistics available, and we will remove those fighters.
        fighters = fighters.loc[
            ~(
                (fighters["slpm"] == 0)
                & (fighters["strike_acc"] == 0)
                & (fighters["sapm"] == 0)
                & (fighters["strike_def"] == 0)
                & (fighters["td_avg"] == 0)
                & (fighters["td_acc"] == 0)
                & (fighters["td_def"] == 0)
                & (fighters["sub_avg"] == 0)
            )
        ].copy()
        self.logger.info("Removed fighters with no statistics.")

        fighters = fighters.loc[fighters["date_of_birth"] != "--", :].copy()
        fighters.date_of_birth = pd.to_datetime(fighters.date_of_birth)

        # Get the fighters record in the ufc
        def get_ufc_fights(fighter, bouts):
            """Extracts the total number of fights fought in the ufc"""
            bouts_test = bouts.loc[
                (bouts.fighter1 == fighter) | (bouts.fighter2 == fighter), :
            ].copy()
            wins = len(
                bouts_test.loc[
                    (bouts_test.winner == fighter) & (bouts_test.win == True), :
                ].copy()
            )
            losses = len(
                bouts_test.loc[
                    (bouts_test.winner != fighter) & (bouts_test.win == True), :
                ].copy()
            )
            nc = len(bouts_test) - wins - losses
            return wins, losses, nc

        fighters[["ufc_wins", "ufc_losses", "ufc_nc"]] = [
            get_ufc_fights(fighter, bouts) for fighter in fighters.fighter_name
        ]
        self.logger.info("Extracted UFC wins, losses, and no-contests.")

        # Remove the index, id, and stance columns if they exist
        if "index" in fighters.columns:
            fighters = fighters.drop(["index"], axis=1)

        fighters = fighters.drop(["id", "stance", "weight"], axis=1)

        # Write the fighters df to the db as fighters_cleaned
        fighters.to_sql(
            "fighters_cleaned",
            self.db_engine,
            schema="ufc",
            if_exists="replace",
            index=False,
        )

        # convert all dtypes except fighter name and date of birth to float
        columns_to_float = fighters.columns[1:]
        columns_to_float = columns_to_float.drop(["date_of_birth"])
        fighters[columns_to_float] = fighters[columns_to_float].astype(float)

        # First drop all the fights without a winner from the dataset
        bouts["win"] = bouts["win"].astype(bool)
        bouts = bouts.loc[bouts.win == True, :].copy()

        # Filter out relevant columns from the bouts df
        bouts = bouts[["fighter1", "fighter2", "winner"]].copy()

        # Create a loser column which will be equal to the fighter who lost the bout
        bouts["loser"] = bouts.apply(
            lambda x: x["fighter1"] if x["fighter2"] == x["winner"] else x["fighter2"],
            axis=1,
        )

        # drop fighter1 and fighter2 columns
        bouts = bouts.drop(["fighter1", "fighter2"], axis=1)

        # randomly distribute the winner and the loser columns over the fighter1 and fighter2 columns
        bouts["fighter1"] = bouts.apply(
            lambda x: x["winner"] if np.random.rand() > 0.5 else x["loser"], axis=1
        )
        bouts["fighter2"] = bouts.apply(
            lambda x: x["winner"] if x["fighter1"] == x["loser"] else x["loser"], axis=1
        )

        # set the win column to 1 if the fighter1 is the winner and 0 otherwise
        bouts["win"] = bouts.apply(
            lambda x: 1 if x["fighter1"] == x["winner"] else 0, axis=1
        )

        # reorder the columns to be fighter1, fighter2, win
        bouts = bouts[["fighter1", "fighter2", "win"]].copy()

        # Quick check to see if the positive case for our model occurs roughly 50% of the time
        positive_cases_ratio = sum(bouts.win) / len(bouts)
        self.logger.info(f"Positive cases ratio: {positive_cases_ratio}")

        bouts = bouts[["fighter1", "fighter2", "win"]].copy()

        # Merge the bouts dataframe with the difference in statistics between the two fighters
        bouts = bouts.merge(
            fighters, left_on="fighter1", right_on="fighter_name", how="left"
        )
        bouts = bouts.merge(
            fighters,
            left_on="fighter2",
            right_on="fighter_name",
            how="left",
            suffixes=("_fighter1", "_fighter2"),
        )

        # Calculate the difference in statistics between the two fighters
        bouts["reach_diff"] = bouts["reach_fighter1"] - bouts["reach_fighter2"]
        bouts["height_diff"] = bouts["height_cm_fighter1"] - bouts["height_cm_fighter2"]
        bouts["age_diff"] = (
            pd.to_datetime(bouts["date_of_birth_fighter1"])
            - pd.to_datetime(bouts["date_of_birth_fighter2"])
        ).dt.days / 365.25
        bouts["slpm_diff"] = bouts["slpm_fighter1"] - bouts["slpm_fighter2"]
        bouts["td_avg_diff"] = bouts["td_avg_fighter1"] - bouts["td_avg_fighter2"]
        bouts["strike_acc_diff"] = (
            bouts["strike_acc_fighter1"] - bouts["strike_acc_fighter2"]
        )
        bouts["td_acc_diff"] = bouts["td_acc_fighter1"] - bouts["td_acc_fighter2"]
        bouts["sapm_diff"] = bouts["sapm_fighter1"] - bouts["sapm_fighter2"]
        bouts["td_def_diff"] = bouts["td_def_fighter1"] - bouts["td_def_fighter2"]
        bouts["strike_def_diff"] = (
            bouts["strike_def_fighter1"] - bouts["strike_def_fighter2"]
        )
        bouts["sub_avg_diff"] = bouts["sub_avg_fighter1"] - bouts["sub_avg_fighter2"]
        bouts["total_wins_diff"] = (
            bouts["total_wins_fighter1"] - bouts["total_wins_fighter2"]
        )
        bouts["total_losses_diff"] = (
            bouts["total_losses_fighter1"] - bouts["total_losses_fighter2"]
        )
        bouts["total_draws_diff"] = (
            bouts["total_draws_fighter1"] - bouts["total_draws_fighter2"]
        )
        bouts["ufc_wins_diff"] = bouts["ufc_wins_fighter1"] - bouts["ufc_wins_fighter2"]
        bouts["ufc_losses_diff"] = (
            bouts["ufc_losses_fighter1"] - bouts["ufc_losses_fighter2"]
        )
        bouts["ufc_nc_diff"] = bouts["ufc_nc_fighter1"] - bouts["ufc_nc_fighter2"]

        # Drop the columns that we will not use for our model
        bouts.drop(
            [
                "fighter_name_fighter1",
                "reach_fighter1",
                "date_of_birth_fighter1",
                "slpm_fighter1",
                "td_avg_fighter1",
                "strike_acc_fighter1",
                "td_acc_fighter1",
                "sapm_fighter1",
                "td_def_fighter1",
                "strike_def_fighter1",
                "sub_avg_fighter1",
                "total_wins_fighter1",
                "total_losses_fighter1",
                "total_draws_fighter1",
                "height_cm_fighter1",
                "ufc_wins_fighter1",
                "ufc_losses_fighter1",
                "ufc_nc_fighter1",
                "fighter_name_fighter2",
                "reach_fighter2",
                "date_of_birth_fighter2",
                "slpm_fighter2",
                "td_avg_fighter2",
                "strike_acc_fighter2",
                "td_acc_fighter2",
                "sapm_fighter2",
                "td_def_fighter2",
                "strike_def_fighter2",
                "sub_avg_fighter2",
                "total_wins_fighter2",
                "total_losses_fighter2",
                "total_draws_fighter2",
                "height_cm_fighter2",
                "ufc_wins_fighter2",
                "ufc_losses_fighter2",
                "ufc_nc_fighter2",
            ],
            axis=1,
            inplace=True,
        )

        # Drop all nan values for now
        bouts = bouts.dropna()

        # Write the bouts df to the database
        bouts.to_sql(
            "model_input",
            self.db_engine,
            schema="ufc",
            if_exists="replace",
            index=False,
        )
        self.logger.info(
            "Data preprocessing complete. Model input saved to the database."
        )

    def calculate_differences_on_inference(
        self, fighter1: str, fighter2: str
    ) -> pd.DataFrame:
        # Extract fighter statistics for fighter1 and fighter2 from table fighters_cleaned
        fighter1_stats = pd.read_sql(
            f"SELECT * FROM ufc.fighters_cleaned WHERE fighter_name='{fighter1}'",
            con=self.db_engine,
        )
        fighter2_stats = pd.read_sql(
            f"SELECT * FROM ufc.fighters_cleaned WHERE fighter_name='{fighter2}'",
            con=self.db_engine,
        )

        # Create empty output dataframe
        output_df = pd.DataFrame()

        # Convert all columns except age_of_birth to float
        cols_to_convert = [
            col
            for col in fighter1_stats.columns
            if col not in ("date_of_birth", "fighter_name")
        ]
        fighter1_stats[cols_to_convert] = fighter1_stats[cols_to_convert].astype(float)
        fighter2_stats[cols_to_convert] = fighter2_stats[cols_to_convert].astype(float)

        # Calculate the difference in statistics between the two fighters
        output_df["reach_diff"] = fighter1_stats["reach"] - fighter2_stats["reach"]
        output_df["height_diff"] = (
            fighter1_stats["height_cm"] - fighter2_stats["height_cm"]
        )
        output_df["age_diff"] = (
            pd.to_datetime(fighter1_stats["date_of_birth"])
            - pd.to_datetime(fighter2_stats["date_of_birth"])
        ).dt.days / 365.25
        output_df["slpm_diff"] = fighter1_stats["slpm"] - fighter2_stats["slpm"]
        output_df["td_avg_diff"] = fighter1_stats["td_avg"] - fighter2_stats["td_avg"]
        output_df["strike_acc_diff"] = (
            fighter1_stats["strike_acc"] - fighter2_stats["strike_acc"]
        )
        output_df["td_acc_diff"] = fighter1_stats["td_acc"] - fighter2_stats["td_acc"]
        output_df["sapm_diff"] = fighter1_stats["sapm"] - fighter2_stats["sapm"]
        output_df["td_def_diff"] = fighter1_stats["td_def"] - fighter2_stats["td_def"]
        output_df["strike_def_diff"] = (
            fighter1_stats["strike_def"] - fighter2_stats["strike_def"]
        )
        output_df["sub_avg_diff"] = (
            fighter1_stats["sub_avg"] - fighter2_stats["sub_avg"]
        )
        output_df["total_wins_diff"] = (
            fighter1_stats["total_wins"] - fighter2_stats["total_wins"]
        )
        output_df["total_losses_diff"] = (
            fighter1_stats["total_losses"] - fighter2_stats["total_losses"]
        )
        output_df["total_draws_diff"] = (
            fighter1_stats["total_draws"] - fighter2_stats["total_draws"]
        )
        output_df["ufc_wins_diff"] = (
            fighter1_stats["ufc_wins"] - fighter2_stats["ufc_wins"]
        )
        output_df["ufc_losses_diff"] = (
            fighter1_stats["ufc_losses"] - fighter2_stats["ufc_losses"]
        )
        output_df["ufc_nc_diff"] = fighter1_stats["ufc_nc"] - fighter2_stats["ufc_nc"]

        return output_df
