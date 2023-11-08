import logging
from datetime import date

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sqlalchemy import create_engine
from sqlalchemy.engine.base import Engine
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)


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

    def load_inference_data_fighter(self, fighter_name: str, time):
        with self.db_engine.connect() as conn:
            self.logger.info(
                "Loading data for fighter {} from the database...".format(fighter_name)
            )
            query = "SELECT * FROM ufc.time_based_inference_df WHERE fighter = '{}' AND date <= '{}' order by date desc limit 1".format(
                fighter_name, time
            )
            fighter = pd.read_sql(query, con=conn)
            self.logger.info("Data loaded successfully.")
        return fighter

    def load_1_row_from_model_input(self):
        with self.db_engine.connect() as conn:
            self.logger.info("Loading 1 row from model_input table...")
            query = "SELECT * FROM ufc.model_input limit 1"
            df = pd.read_sql(query, con=conn)
            df.drop("index", axis=1, inplace=True)
            df.drop("win", axis=1, inplace=True)
        return df

    def filter_fighters(
        self, fighters: pd.DataFrame, bouts: pd.DataFrame
    ) -> pd.DataFrame:
        fighters_that_fought = set(bouts.fighter1).union(set(bouts.fighter2))
        filtered_fighters = fighters[
            fighters.fighter_name.isin(fighters_that_fought)
        ].copy()
        filtered_fighters.date_of_birth = filtered_fighters.date_of_birth.replace(
            "--", None
        )
        filtered_fighters.height = filtered_fighters.height.replace("--", None)
        filtered_fighters = filtered_fighters.dropna(subset=["height", "date_of_birth"])
        self.logger.info("Filtered fighters who have fought.")
        return filtered_fighters

    def parse_dates(self, fighters: pd.DataFrame) -> pd.DataFrame:
        fighters = fighters.copy()
        fighters["date_of_birth"] = pd.to_datetime(
            fighters["date_of_birth"].replace("--", None),
            errors="coerce",
            format="%b %d %Y",
        )
        self.logger.info("Parsed dates for fighters.")
        return fighters

    def handle_name_duplicates(self, fighters: pd.DataFrame) -> pd.DataFrame:
        name_changes = {
            ("Michael McDonald", 205): "Michael McDonald 205",
            ("Joey Gomez", 155): "Joey Gomez 155",
            ("Mike Davis", 155): "Mike Davis 155",
            ("Bruno Silva", 125): "Bruno Silva 125",
        }
        for (name, weight), new_name in name_changes.items():
            fighters.loc[
                (fighters.fighter_name == name) & (fighters.weight == weight),
                "fighter_name",
            ] = new_name
        self.logger.info("Handled duplicate names based on weight classes.")
        return fighters

    def transform_height_to_cm(self, fighters: pd.DataFrame) -> pd.DataFrame:
        fighters = fighters.copy()
        fighters.height = fighters.height.replace("--", None)
        fighters.dropna(subset=["height"], inplace=True)
        height_split = fighters["height"].str.extract(
            r'(?P<feet>\d+)\' (?P<inches>\d+)"'
        )
        fighters["height_cm"] = 30.48 * height_split["feet"].astype(
            float
        ) + 2.54 * height_split["inches"].astype(float)
        fighters.drop(["height"], axis=1, inplace=True)
        self.logger.info("Transformed fighter heights into centimeters.")
        return fighters

    def clean_fighter_data(self, fighters: pd.DataFrame) -> pd.DataFrame:
        fighters = fighters.copy()
        fighters = fighters[
            ["fighter_name", "height_cm", "reach", "date_of_birth", "fighter_record"]
        ]
        fighters.rename(columns={"fighter_name": "fighter"}, inplace=True)
        fighters[["total_wins", "total_losses"]] = fighters[
            "fighter_record"
        ].str.extract(r"Record: (\d+)-(\d+)")
        fighters.drop("fighter_record", axis=1, inplace=True)
        self.logger.info("Cleaned fighter data.")
        return fighters

    def predict_missing_reach(self, fighters: pd.DataFrame) -> pd.DataFrame:
        fighters = fighters.copy()
        fighters_with_reach = fighters.dropna(subset=["reach"]).copy()
        fighters_missing_reach = fighters[fighters.reach.isna()].copy()
        lr = LinearRegression()
        lr.fit(fighters_with_reach[["height_cm"]], fighters_with_reach["reach"])
        fighters_missing_reach["reach"] = lr.predict(
            fighters_missing_reach[["height_cm"]]
        )
        fighters = pd.concat([fighters_with_reach, fighters_missing_reach])
        self.logger.info("Predicted missing reach values using linear regression.")
        return fighters

    def fix_fighters_duplicate_names_bouts(self, bouts: pd.DataFrame) -> pd.DataFrame:
        bouts = bouts.copy()
        bouts.loc[
            (bouts.fighter1 == "Michael McDonald")
            & (bouts.weight_class == "Light heavyweight"),
            "fighter1",
        ] = "Michael McDonald 205"
        bouts.loc[
            (bouts.fighter1 == "Joey Gomez") & (bouts.weight_class == "Lightweight"),
            "fighter1",
        ] = "Joey Gomez 155"
        bouts.loc[
            (bouts.fighter1 == "Mike Davis") & (bouts.weight_class == "Lightweight "),
            "fighter1",
        ] = "Mike Davis 145"
        bouts.loc[
            (bouts.fighter1 == "Bruno Silva") & (bouts.weight_class == "Flyweight"),
            "fighter1",
        ] = "Bruno Silva 125"
        bouts.loc[
            (bouts.fighter2 == "Michael McDonald")
            & (bouts.weight_class == "Light heavyweight"),
            "fighter2",
        ] = "Michael McDonald 205"
        bouts.loc[
            (bouts.fighter2 == "Joey Gomez") & (bouts.weight_class == "Lightweight"),
            "fighter2",
        ] = "Joey Gomez 155"
        bouts.loc[
            (bouts.fighter2 == "Mike Davis") & (bouts.weight_class == "Lightweight "),
            "fighter2",
        ] = "Mike Davis 145"
        bouts.loc[
            (bouts.fighter2 == "Bruno Silva") & (bouts.weight_class == "Flyweight"),
            "fighter2",
        ] = "Bruno Silva 125"
        return bouts

    def clean_bouts_data(self, bouts: pd.DataFrame) -> pd.DataFrame:
        bouts = bouts.copy()
        bouts["event_date"] = pd.to_datetime(
            bouts["event_date"], errors="coerce", format="%B %d %Y"
        )
        bouts.drop(["index", "id"], axis=1, inplace=True)
        bouts = bouts.loc[bouts.win != 0]
        self.logger.info("Cleaned bouts data.")
        return bouts

    def calculate_fight_time(self, bouts: pd.DataFrame) -> pd.DataFrame:
        bouts = bouts.copy()
        bouts["fight_duration"] = round(
            (bouts["round_"] - 1) * 5
            + bouts["time_minutes"]
            + bouts["time_seconds"] / 60,
            2,
        )
        bouts.drop(["round_", "time_minutes", "time_seconds"], axis=1, inplace=True)
        self.logger.info("Calculated fight time in minutes.")
        return bouts

    def convert_bouts_wide_to_long(self, bouts: pd.DataFrame) -> pd.DataFrame:
        # Convert the control time columns by turning m:ss into seconds
        for col in ["control_time_1", "control_time_2"]:
            bouts[col] = bouts[col].replace("--", "0:00")
            bouts[col] = bouts[col].fillna("0:00")
            bouts[col] = (
                bouts[col].str.split(":").apply(lambda x: int(x[0]) * 60 + int(x[1]))
            )

        # Create separate dataframes for each fighter and then concatenate
        columns_1 = [col for col in bouts.columns if col.endswith("_1")]
        columns_2 = [col for col in bouts.columns if col.endswith("_2")]
        columns_shared_cleaned = [
            col.rstrip("_1") if col.endswith("_1") else col for col in columns_1
        ]
        common_columns = [
            col
            for col in bouts.columns
            if not col.endswith("_1") and not col.endswith("_2")
        ]
        df_fighter1 = bouts[common_columns + columns_1].copy()
        df_fighter2 = bouts[common_columns + columns_2].copy()
        df_fighter1.columns = [
            col.rstrip("_1") if col.endswith("_1") else col
            for col in df_fighter1.columns
        ]
        df_fighter2.columns = [
            col.rstrip("_2") if col.endswith("_2") else col
            for col in df_fighter2.columns
        ]

        # add columns_shared_cleaned of df_fighter2 to df_fighter1 with the postfix _received
        columns_shared_cleaned_received = [
            col + "_received" for col in columns_shared_cleaned
        ]
        df_fighter1[columns_shared_cleaned_received] = df_fighter2[
            columns_shared_cleaned
        ]

        # same for fighter1
        df_fighter2[columns_shared_cleaned_received] = df_fighter1[
            columns_shared_cleaned
        ]

        # Add the fighter names for each dataframe
        df_fighter1["fighter"] = bouts["fighter1"]
        df_fighter1["opponent"] = bouts["fighter2"]
        df_fighter2["fighter"] = bouts["fighter2"]
        df_fighter2["opponent"] = bouts["fighter1"]

        # Add a 'win' column to each dataframe based on the winner
        df_fighter1["win"] = bouts["winner"] == bouts["fighter1"]
        df_fighter2["win"] = bouts["winner"] == bouts["fighter2"]

        bouts_long = pd.concat([df_fighter1, df_fighter2], ignore_index=True)

        # Reorder columns to match the desired output
        desired_columns_order = ["event_name", "event_date", "win", "fighter"] + [
            col
            for col in bouts_long.columns
            if col not in ["event_name", "event_date", "win", "fighter"]
        ]
        bouts_long = bouts_long[desired_columns_order]

        # Drop some reduntant columns
        bouts_long.drop(["event_name", "winner"], axis=1, inplace=True)

        # After some analysis, it seems like for some old fights, we have no data for some of the statistics, so we will drop those rows
        bouts_long = bouts_long.dropna(subset=["sig_head_landed"])

        # Besides the old fights, we have 2 other columns with missing values, as these aren't important for our analysis, we will just delete the columns
        bouts_long = bouts_long.drop(
            [
                "sig_strike_perc",
                "takedown_perc",
                "sig_strike_perc_received",
                "takedown_perc_received",
                "fighter1",
                "fighter2",
            ],
            axis=1,
        )
        bouts_long = bouts_long.sort_values(by="event_date")

        self.logger.info("Converted bouts dataframe from wide to long format.")
        return bouts_long

    def compute_cumulative_stats(self, bouts_long: pd.DataFrame) -> pd.DataFrame:
        bouts_long = bouts_long.copy()
        # Add stats
        bouts_long["won_by_ko"] = bouts_long["win"] & (
            bouts_long["win_method_type"] == "KO/TKO"
        )
        bouts_long["lost_by_ko"] = ~bouts_long["win"] & (
            bouts_long["win_method_type"] == "KO/TKO"
        )
        bouts_long["won_by_sub"] = bouts_long["win"] & (
            bouts_long["win_method_type"] == "Submission"
        )
        bouts_long["lost_by_sub"] = ~bouts_long["win"] & (
            bouts_long["win_method_type"] == "Submission"
        )
        bouts_long["won_by_decision"] = bouts_long["win"] & (
            bouts_long["win_method_type"].str.contains("Decision")
        )
        bouts_long["lost_by_decision"] = ~bouts_long["win"] & (
            bouts_long["win_method_type"].str.contains("Decision")
        )
        self.logger.info("Computed cumulative stats for each fighter.")
        return bouts_long

    def extract_fighter_data(
        self, fighter_name: str, bouts_long: pd.DataFrame
    ) -> pd.DataFrame:
        """Extracts the fighter data from the bouts_long dataframe and returns it in a new dataframe"""
        fighter_data = bouts_long.loc[bouts_long.fighter == fighter_name].copy()
        fighter_data.sort_values(by=["event_date", "fighter", "opponent"], inplace=True)
        output = self.initialize_fighter_output(fighter_data)
        output = self.calculate_cumulative_stats(output, fighter_data)
        output = self.calculate_time_based_stats(output)
        output = self.calculate_per_minute_stats(output, fighter_data)
        output = self.shift_stats_for_fight_start(output)
        return output.iloc[1:, :]  # Drop the first row filled with NaNs after shifting

    def initialize_fighter_output(self, fighter_data: pd.DataFrame) -> pd.DataFrame:
        """Initializes the output DataFrame with basic fight information"""
        return fighter_data[["event_date", "fighter", "opponent"]].copy()

    def calculate_cumulative_stats(
        self, output: pd.DataFrame, fighter_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculates cumulative stats for the fighter"""
        output["wins"] = fighter_data.win.cumsum()
        output["losses"] = (~fighter_data.win).cumsum()
        output[
            "win_streak"
        ] = fighter_data.win.cumsum() - fighter_data.win.cumsum().where(
            ~fighter_data.win
        ).ffill().fillna(
            0
        )
        output["losing_streak"] = (~fighter_data.win).cumsum() - (
            ~fighter_data.win
        ).cumsum().where(fighter_data.win).ffill().fillna(0)
        output["total_time_in_octagon"] = fighter_data["fight_duration"].cumsum()
        output["title_fights"] = fighter_data["title_fight"].cumsum()
        output["performance_bonuses"] = fighter_data["performance_bonus"].cumsum()
        output["won_by_ko"] = fighter_data.won_by_ko.cumsum()
        output["won_by_sub"] = fighter_data.won_by_sub.cumsum()
        output["won_by_decision"] = fighter_data.won_by_decision.cumsum()
        output["lost_by_decision"] = fighter_data.lost_by_decision.cumsum()
        output["lost_by_sub"] = fighter_data.lost_by_sub.cumsum()
        output["lost_by_ko"] = fighter_data.lost_by_ko.cumsum()
        output["knocked_out_in_previous_fight"] = (
            output["lost_by_ko"].shift(1).fillna(0)
        )
        return output

    def calculate_time_based_stats(self, output: pd.DataFrame) -> pd.DataFrame:
        """Calculates stats that are based on time since last fight"""
        output["months_since_last_fight"] = abs(
            (output["event_date"] - output["event_date"].shift(-1)).dt.days / 30
        )
        return output

    def calculate_per_minute_stats(
        self, output: pd.DataFrame, fighter_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculates statistics per minute of fight time"""
        stats_to_cumsum = self.get_cumulative_stats_list()
        for stat in stats_to_cumsum:
            output[stat] = fighter_data[stat].cumsum() / output["total_time_in_octagon"]
        output = output.rename(
            columns={stat: stat + "_per_minute" for stat in stats_to_cumsum}
        )
        output["sig_strikes_to_head"] = (
            output["sig_head_landed_received_per_minute"]
            * output["total_time_in_octagon"]
        )
        return output

    def shift_stats_for_fight_start(self, output: pd.DataFrame) -> pd.DataFrame:
        """Shifts stats to represent the state before the start of the next fight"""
        output.iloc[:, 3:] = output.iloc[:, 3:].shift(1)
        return output

    def get_cumulative_stats_list(self) -> list:
        """Returns the list of stats to be cumulatively summed"""
        return [
            "knock_down",
            "knock_down_received",
            "sig_strikes",
            "sig_strikes_received",
            "total_strike",
            "total_strike_received",
            "takedowns",
            "takedowns_received",
            "submission_attempt",
            "submission_attempt_received",
            "reversals",
            "reversals_received",
            "control_time",
            "control_time_received",
            "sig_head_landed",
            "sig_body_landed",
            "sig_leg_landed",
            "sig_head_landed_received",
            "sig_body_landed_received",
            "sig_leg_landed_received",
            "sig_head_attempted",
            "sig_body_attempted",
            "sig_leg_attempted",
            "sig_head_attempted_received",
            "sig_body_attempted_received",
            "sig_leg_attempted_received",
            "sig_distance_landed",
            "sig_clinch_landed",
            "sig_ground_landed",
            "sig_distance_landed_received",
            "sig_clinch_landed_received",
            "sig_ground_landed_received",
            "sig_distance_attempted",
            "sig_clinch_attempted",
            "sig_ground_attempted",
            "sig_distance_attempted_received",
            "sig_clinch_attempted_received",
            "sig_ground_attempted_received",
        ]

    def aggregate_fighter_stats(self, bouts_long: pd.DataFrame) -> pd.DataFrame:
        fighter_list = bouts_long.fighter.unique()
        df_bouts_processed = pd.DataFrame()

        self.logger.info("Aggregating fighter stats...")
        # Loop over fighter_list with a progress bar using tqdm
        for fighter_name in tqdm(fighter_list):
            fighter_data = self.extract_fighter_data(fighter_name, bouts_long)
            df_bouts_processed = pd.concat(
                [df_bouts_processed, fighter_data], ignore_index=True
            )

        self.logger.info("Aggregated fighter stats.")
        return df_bouts_processed

    def merge_fighter_records(
        self, fighters: pd.DataFrame, df_bouts_processed: pd.DataFrame
    ) -> pd.DataFrame:
        # Merge the fighter records with their stats

        # Add the maximum wins and losses from the df_bouts_processed dataframe for each fighter in the fighters dataframe
        ufc_wins = df_bouts_processed.groupby("fighter")["wins"].max()
        ufc_losses = df_bouts_processed.groupby("fighter")["losses"].max()

        fighters = fighters.merge(ufc_wins, on="fighter", how="left").fillna(0)
        fighters = fighters.merge(ufc_losses, on="fighter", how="left").fillna(0)

        fighters["wins_outside_ufc"] = (
            fighters["total_wins"].astype(int) - fighters["wins"]
        )
        fighters["losses_outside_ufc"] = (
            fighters["total_losses"].astype(int) - fighters["losses"]
        )
        fighters = fighters.drop(
            ["total_wins", "total_losses", "wins", "losses"], axis=1
        )
        self.logger.info("Merged fighter records with their statistics.")
        return fighters

    def select_relevant_columns(self, bouts: pd.DataFrame) -> pd.DataFrame:
        bouts = bouts[["event_date", "fighter1", "fighter2", "winner"]].copy()
        self.logger.info("Selected relevant columns from bouts dataframe.")
        return bouts

    def determine_win_loss(self, bouts: pd.DataFrame) -> pd.DataFrame:
        bouts["loser"] = bouts.apply(
            lambda x: x["fighter2"] if x["fighter1"] == x["winner"] else x["fighter1"],
            axis=1,
        )
        self.logger.info("Determined winners and losers for each bout.")
        return bouts

    def randomize_fighter_positions(self, bouts: pd.DataFrame) -> pd.DataFrame:
        np.random.seed(42)
        bouts.sort_values(by=["event_date", "fighter1", "fighter2"], inplace=True)
        bouts["fighter1"] = bouts.apply(
            lambda x: x["winner"] if np.random.rand() > 0.5 else x["loser"], axis=1
        )
        bouts["fighter2"] = bouts.apply(
            lambda x: x["loser"] if x["fighter1"] == x["winner"] else x["winner"],
            axis=1,
        )
        bouts["win"] = bouts.apply(
            lambda x: 1 if x["fighter1"] == x["winner"] else 0, axis=1
        )
        bouts = bouts[["event_date", "fighter1", "fighter2", "win"]].copy()
        self.logger.info("Randomized fighter positions in the bouts dataframe.")
        return bouts

    def compute_inference_df(
        self, df_bouts_processed: pd.DataFrame, fighters: pd.DataFrame
    ) -> pd.DataFrame:
        inference_df = df_bouts_processed.merge(
            fighters, left_on=["fighter"], right_on=["fighter"], how="left"
        ).copy()
        inference_df.rename(columns={"event_date": "date"}, inplace=True)
        inference_df.drop("opponent", axis=1, inplace=True)
        self.logger.info("Computed inference dataframe")
        return inference_df

    def compute_differential_statistics(
        self,
        bouts: pd.DataFrame,
        df_bouts_processed: pd.DataFrame,
        fighters: pd.DataFrame,
    ) -> pd.DataFrame:
        bouts_diff = self.calculate_fighter_diff(bouts, df_bouts_processed)
        bouts_diff = self.merge_fighter_ages(bouts_diff, bouts, fighters)
        self.logger.info("Computed differential statistics for fighters.")
        return bouts_diff

    def calculate_fighter_diff(
        self, bouts: pd.DataFrame, df_bouts_processed: pd.DataFrame
    ) -> pd.DataFrame:
        # Assuming bouts has columns fighter1, fighter2, and win set correctly at this point
        bouts_fighter1 = bouts.merge(
            df_bouts_processed,
            left_on=["event_date", "fighter1", "fighter2"],
            right_on=["event_date", "fighter", "opponent"],
            how="left",
        )
        bouts_fighter2 = bouts.merge(
            df_bouts_processed,
            left_on=["event_date", "fighter1", "fighter2"],
            right_on=["event_date", "opponent", "fighter"],
            how="left",
        )
        columns_to_diff = [
            col
            for col in df_bouts_processed.columns
            if col not in ["event_date", "fighter", "opponent"]
        ]
        for col in columns_to_diff:
            bouts_fighter1[col + "_diff"] = bouts_fighter1[col] - bouts_fighter2[col]
        bouts_fighter1.drop(
            columns_to_diff + ["fighter", "opponent"], axis=1, inplace=True
        )
        return bouts_fighter1

    def merge_fighter_ages(
        self, bouts_diff: pd.DataFrame, bouts: pd.DataFrame, fighters: pd.DataFrame
    ) -> pd.DataFrame:
        fighter_1 = bouts.merge(
            fighters, left_on=["fighter1"], right_on=["fighter"], how="left"
        )
        fighter_1["age"] = (
            fighter_1["event_date"] - fighter_1["date_of_birth"]
        ).dt.days / 365.25

        fighter_2 = bouts.merge(
            fighters, left_on=["fighter2"], right_on=["fighter"], how="left"
        )
        fighter_2["age"] = (
            fighter_2["event_date"] - fighter_2["date_of_birth"]
        ).dt.days / 365.25

        columns_to_diff = [
            col for col in fighters.columns if col not in ["fighter", "date_of_birth"]
        ]
        columns_to_diff.append("age")
        for col in columns_to_diff:
            fighter_1[col + "_diff"] = fighter_1[col] - fighter_2[col]
        fighter_1.drop(
            columns_to_diff + ["date_of_birth", "fighter"], axis=1, inplace=True
        )

        bouts_diff = bouts_diff.merge(
            fighter_1,
            left_on=["event_date", "fighter1", "fighter2", "win"],
            right_on=["event_date", "fighter1", "fighter2", "win"],
            how="left",
        )
        return bouts_diff

    def drop_unusable_data(self, bouts_full: pd.DataFrame) -> pd.DataFrame:
        bouts_full.dropna(inplace=True)
        self.logger.info(
            "Dropped rows with missing data that can't be used in the model."
        )
        return bouts_full

    def preprocess_training_data(
        self, bouts: pd.DataFrame, fighters: pd.DataFrame
    ) -> pd.DataFrame:
        self.logger.info("Starting data preprocessing...")
        fighters = self.filter_fighters(fighters, bouts)
        fighters = self.parse_dates(fighters)
        fighters = self.handle_name_duplicates(fighters)
        fighters = self.transform_height_to_cm(fighters)
        fighters = self.clean_fighter_data(fighters)
        fighters = self.predict_missing_reach(fighters)
        bouts = self.fix_fighters_duplicate_names_bouts(bouts)
        bouts = self.clean_bouts_data(bouts)
        bouts = self.calculate_fight_time(bouts)
        bouts_long = self.convert_bouts_wide_to_long(bouts)
        bouts_long = self.compute_cumulative_stats(bouts_long)
        df_bouts_processed = self.aggregate_fighter_stats(bouts_long)
        fighters = self.merge_fighter_records(fighters, df_bouts_processed)
        bouts = self.select_relevant_columns(bouts)
        bouts = self.determine_win_loss(bouts)
        bouts = self.randomize_fighter_positions(bouts)
        bouts_diff = self.compute_differential_statistics(
            bouts, df_bouts_processed, fighters
        )
        bouts_full = self.drop_unusable_data(bouts_diff)
        self.write_to_database(bouts_full, "model_input")
        self.logger.info("Data preprocessing complete.")
        return bouts_full

    def write_to_database(self, bouts_full: pd.DataFrame, tablename: str):
        bouts_full.to_sql(
            tablename,
            self.db_engine,
            schema="ufc",
            if_exists="replace",
            index=False,
        )
        self.logger.info("Model input saved to the database.")

    def calculate_age(self, date_of_birth: str, event_date: str) -> float:
        date_of_birth = pd.to_datetime(date_of_birth)
        event_date = pd.to_datetime(event_date)
        age = (event_date - date_of_birth).days / 365.25
        return age

    def calculate_time_based_inference_df(
        self, bouts: pd.DataFrame, fighters: pd.DataFrame
    ) -> pd.DataFrame:
        """The function generated the dataframe that is equal to the df_bouts_processed merged with the fighters dataframe in the main function above
        it should basically contain all of the model parameters that we want to calculate the diff for between fighters"""
        self.logger.info("Starting data preprocessing...")
        fighters = self.filter_fighters(fighters, bouts)
        fighters = self.parse_dates(fighters)
        fighters = self.handle_name_duplicates(fighters)
        fighters = self.transform_height_to_cm(fighters)
        fighters = self.clean_fighter_data(fighters)
        fighters = self.predict_missing_reach(fighters)
        bouts = self.fix_fighters_duplicate_names_bouts(bouts)
        bouts = self.clean_bouts_data(bouts)
        bouts = self.calculate_fight_time(bouts)
        bouts_long = self.convert_bouts_wide_to_long(bouts)
        bouts_long = self.compute_cumulative_stats(bouts_long)
        df_bouts_processed = self.aggregate_fighter_stats(bouts_long)
        fighters = self.merge_fighter_records(fighters, df_bouts_processed)
        inference_df = self.compute_inference_df(df_bouts_processed, fighters)
        inference_df = self.drop_unusable_data(inference_df)
        self.write_to_database(inference_df, "time_based_inference_df")
        self.logger.info("Finished calculating time based inference dataframe.")

    def calculate_diff_on_inference(
        self,
        fighter_1: str,
        fighter_2: str,
        fighter_1_date=date.today().isoformat(),
        fighter_2_date=date.today().isoformat(),
    ) -> pd.DataFrame:
        """This function should calculate the diff between the two fighters for the time based inference dataframe"""

        # Retrieve the correct data
        fighter_1_data = self.load_inference_data_fighter(fighter_1, fighter_1_date)
        fighter_2_data = self.load_inference_data_fighter(fighter_2, fighter_2_date)
        df_for_columns = self.load_1_row_from_model_input()

        # Calculate the diff
        df_diff = pd.DataFrame()
        columns_to_diff = [
            col
            for col in fighter_1_data.columns
            if col not in ["date", "fighter", "date_of_birth"]
        ]
        for col in columns_to_diff:
            df_diff[col + "_diff"] = fighter_1_data[col] - fighter_2_data[col]

        # Calculate the diff for the time related columns
        age_fighter_1 = self.calculate_age(
            fighter_1_data["date_of_birth"].values[0], fighter_1_date
        )
        age_fighter_2 = self.calculate_age(
            fighter_2_data["date_of_birth"].values[0], fighter_2_date
        )
        age_diff = age_fighter_1 - age_fighter_2
        df_diff["age_diff"] = age_diff

        # Add fighter names and a fake event_date
        df_diff["fighter1"] = fighter_1
        df_diff["fighter2"] = fighter_2
        df_diff["event_date"] = fighter_1_date

        return df_diff[df_for_columns.columns]


# username = 'postgres'
# password = 'postgres'
# processor = DataProcessor(username, password, server='localhost')
# # bouts, fighters = processor.load_data()
# # # preprocessed_data = processor.preprocess_training_data(bouts, fighters)
# # # processor.calculate_time_based_inference_df(bouts, fighters)
# processor.calculate_diff_on_inference('Khabib Nurmagomedov',  'Justin Gaethje', '2020-10-24','2020-10-24')
