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


class DataProcessorDoublingUp:
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

    def load_training_data(self) -> pd.DataFrame:
        self.logger.info("Loading training data...")
        bouts_double = pd.read_sql_table("bouts_double", self.db_engine, schema="ufc")
        merged_fighter_stats = pd.read_sql_table(
            "fighter_stats_no_age", self.db_engine, schema="ufc"
        )
        merged_fighter_stats = self.calculate_age_helper(merged_fighter_stats)
        training_df = self.get_training_columns(bouts_double, merged_fighter_stats)
        training_df = self.drop_unusable_data(training_df)
        return training_df

    def load_male_fight_training_data(self) -> pd.DataFrame:
        query = """with uniques as (
            (select distinct fighter1 as fighter from ufc.bouts
            where weight_class not like '%%Women%%')
            union
            (select distinct fighter2 as fighter from ufc.bouts
            where weight_class not like '%%Women%%'))
            select distinct fighter from uniques"""
        male_fighter_names = pd.read_sql(query, self.db_engine)
        bouts_double = pd.read_sql_table("bouts_double", self.db_engine, schema="ufc")
        merged_fighter_stats = pd.read_sql_table(
            "fighter_stats_no_age", self.db_engine, schema="ufc"
        )
        bouts_double = bouts_double[
            bouts_double["fighter"].isin(male_fighter_names.fighter)
        ]
        merged_fighter_stats = merged_fighter_stats[
            merged_fighter_stats["fighter"].isin(male_fighter_names.fighter)
        ]
        merged_fighter_stats = self.calculate_age_helper(merged_fighter_stats)
        training_df = self.get_training_columns(bouts_double, merged_fighter_stats)
        training_df = self.drop_unusable_data(training_df)
        return training_df

    def load_male_data_after_year(self, year: int) -> pd.DataFrame:
        training_df = self.load_male_fight_training_data()
        training_df = training_df[training_df["event_date"] > f"{year}-01-01"]
        return training_df

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
            query = "SELECT * FROM ufc.fighter_stats_no_age WHERE fighter = '{}' AND date <= '{}' order by date desc limit 1".format(
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
        bouts["sig_strikes_attempted_1"] = (
            bouts["sig_head_attempted_1"]
            + bouts["sig_body_attempted_1"]
            + bouts["sig_leg_attempted_1"]
        )
        bouts["sig_strikes_attempted_2"] = (
            bouts["sig_head_attempted_2"]
            + bouts["sig_body_attempted_2"]
            + bouts["sig_leg_attempted_2"]
        )
        bouts["takedowns_attempted_1"] = round(
            bouts["takedowns_1"] / (bouts["takedown_perc_1"] / 100)
        ).fillna(0)
        bouts["takedowns_attempted_2"] = round(
            bouts["takedowns_2"] / (bouts["takedown_perc_2"] / 100)
        ).fillna(0)
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
        bouts_long["championship_rounds_experience"] = bouts_long["fight_duration"] > 15
        self.logger.info("Computed cumulative stats for each fighter.")
        return bouts_long

    def extract_fighter_data(
        self, fighter_name: str, bouts_long: pd.DataFrame
    ) -> pd.DataFrame:
        """Extracts the fighter data from the bouts_long dataframe and returns it in a new dataframe"""
        fighter_data = bouts_long.loc[bouts_long.fighter == fighter_name].copy()
        fighter_data.sort_values(by=["event_date", "fighter", "opponent"], inplace=True)
        output = self.initialize_fighter_output(fighter_data)
        output = self.calculate_result_based_stats(output, fighter_data)
        output = self.calculate_time_based_stats(output)
        output, per_minute_columns_last_fight = self.calculate_per_minute_stats(
            output, fighter_data
        )
        (
            output,
            accuracy_columns_last_fight,
        ) = self.calculate_accuracy_and_defense_scores(output, fighter_data)
        output, octagon_control_columns = self.calculate_octagon_control(
            output, fighter_data
        )
        combined_new_columns = (
            per_minute_columns_last_fight
            + accuracy_columns_last_fight
            + octagon_control_columns
            + ["knocked_out_in_previous_fight"]
        )
        output = self.calculate_last_3_fights_average(
            output, fighter_data, combined_new_columns
        )
        output = self.get_career_best_and_worst(
            output, fighter_data, combined_new_columns
        )
        output = self.shift_stats_for_fight_start(output)

        return output.iloc[1:, :]  # Drop the first row filled with NaNs after shifting

    def initialize_fighter_output(self, fighter_data: pd.DataFrame) -> pd.DataFrame:
        """Initializes the output DataFrame with basic fight information"""
        return fighter_data[["event_date", "fighter", "opponent"]].copy()

    def get_career_best_and_worst(
        self, output, fighter_data: pd.DataFrame, stat_columns: list
    ) -> pd.DataFrame:
        """Retrieves the highest stat so far for a list of columns."""

        fight_stat_list = self.get_fight_stats_lists()
        all_stats = sum(
            fight_stat_list, []
        )  # Flatten the list of lists into a single list of stats

        new_columns = pd.DataFrame()
        for col in all_stats:
            new_columns = new_columns.copy()
            new_columns[f"{col}_career_best"] = fighter_data[col].cummax()
            new_columns[f"{col}_career_worst"] = fighter_data[col].cummin()

        for col in stat_columns:
            new_columns = new_columns.copy()
            new_columns[f"{col}_career_best"] = output[col].cummax()
            new_columns[f"{col}_career_worst"] = output[col].cummin()

        output = pd.concat([output, new_columns], axis=1)

        return output

    def calculate_result_based_stats(
        self, output: pd.DataFrame, fighter_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculates cumulative stats and derived statistics for the fighter"""

        # Helper function to calculate win rates
        def calculate_win_rate(wins, fights):
            return wins / fights.clip(lower=1)

        # Cumulative counts
        cumulative_wins = fighter_data.win.cumsum()
        cumulative_losses = (~fighter_data.win).cumsum()

        output["wins"] = cumulative_wins
        output["losses"] = cumulative_losses
        output["fights"] = cumulative_wins + cumulative_losses
        output["total_time_in_octagon"] = fighter_data["fight_duration"].cumsum()
        output["championship_rounds_experience"] = fighter_data[
            "championship_rounds_experience"
        ].cumsum()

        # Streak calculations
        win_streak_calc = cumulative_wins - cumulative_wins.where(
            ~fighter_data.win
        ).ffill().fillna(0)
        loss_streak_calc = cumulative_losses - cumulative_losses.where(
            fighter_data.win
        ).ffill().fillna(0)

        output["win_streak"] = win_streak_calc
        output["losing_streak"] = loss_streak_calc

        # Binary and ratio-based features
        for method in ["ko", "sub", "decision"]:
            output[f"won_by_{method}"] = fighter_data[f"won_by_{method}"].cumsum()
            output[f"lost_by_{method}"] = fighter_data[f"lost_by_{method}"].cumsum()
            output[f"{method}_win_rate"] = calculate_win_rate(
                output[f"won_by_{method}"], output["fights"]
            )
            output[f"{method}_loss_rate"] = calculate_win_rate(
                output[f"lost_by_{method}"], output["fights"]
            )

        # fight specialness
        output["title_fights"] = fighter_data["title_fight"].cumsum()
        output["performance_bonuses"] = fighter_data["performance_bonus"].cumsum()
        output["knocked_out_in_previous_fight"] = output["lost_by_ko"]

        # More complex derived statistics
        output["win_rate"] = calculate_win_rate(cumulative_wins, output["fights"])
        output["finish_rate"] = calculate_win_rate(
            output["won_by_ko"] + output["won_by_sub"], output["fights"]
        )
        output["average_fight_duration"] = output["total_time_in_octagon"] / output[
            "fights"
        ].clip(lower=1)
        output["performance_bonus_rate"] = calculate_win_rate(
            output["performance_bonuses"], output["fights"]
        )
        output["win_by_ko_or_sub_rate"] = calculate_win_rate(
            output["won_by_ko"] + output["won_by_sub"], cumulative_wins
        )
        output["win_to_loss_ratio"] = cumulative_wins / cumulative_losses.replace(0, 1)

        # Fighting trends and performance under pressure
        output["recent_form"] = (
            fighter_data["win"].rolling(min_periods=1, window=3).mean()
        )
        output["comeback_wins"] = ((cumulative_losses > 0) & fighter_data.win).cumsum()
        output["comeback_win_rate"] = calculate_win_rate(
            output["comeback_wins"], output["fights"]
        )

        # Title fights and performance bonuses
        output["title_fights"] = fighter_data["title_fight"].cumsum()
        output["performance_bonuses"] = fighter_data["performance_bonus"].cumsum()

        # Additional performance metrics
        output["finish_to_win_streak_ratio"] = calculate_win_rate(
            (output["won_by_ko"] + output["won_by_sub"]).cummax(),
            win_streak_calc.cummax(),
        )
        output["finish_dominance_ratio"] = calculate_win_rate(
            output["won_by_ko"] + output["won_by_sub"],
            output["lost_by_ko"] + output["lost_by_sub"],
        )

        return output

    def calculate_last_3_fights_average(
        self, output: pd.DataFrame, fighter_data: pd.DataFrame, new_columns
    ) -> pd.DataFrame:
        """Calculates the average of each stat from fight_stat_list for the last 3 fights."""
        output = output.copy()
        fight_stat_list = self.get_fight_stats_lists()
        all_stats = sum(
            fight_stat_list, []
        )  # Flatten the list of lists into a single list of stats

        for stat in all_stats:
            # Calculate the rolling mean with a minimum of 1 period for the first fight
            rolling_avg = fighter_data[stat].rolling(window=3, min_periods=1).mean()
            # For the second fight, fill the NaN by averaging the first two fights
            if len(rolling_avg) > 1:  # Check to ensure there's at least 2 data points
                rolling_avg.iloc[1] = fighter_data[stat].iloc[:2].mean()
            output[f"{stat}_3_prev_avg"] = rolling_avg

        for stat in new_columns:
            # Calculate the rolling mean with a minimum of 1 period for the first fight
            rolling_avg = output[stat].rolling(window=3, min_periods=1).mean()
            # For the second fight, fill the NaN by averaging the first two fights
            if len(rolling_avg) > 1:  # Check to ensure there's at least 2 data points
                rolling_avg.iloc[1] = output[stat].iloc[:2].mean()
            output[f"{stat}_3_prev_avg"] = rolling_avg

        return output

    def calculate_time_based_stats(self, output: pd.DataFrame) -> pd.DataFrame:
        """Calculates stats that are based on time since last fight"""
        output["months_since_last_fight"] = abs(
            (output["event_date"] - output["event_date"].shift(-1)).dt.days / 30
        )
        return output

    def calculate_per_minute_stats(
        self, output: pd.DataFrame, fighter_data: pd.DataFrame
    ) -> (pd.DataFrame, list):
        """Calculates statistics per minute of fight time"""
        output = output.copy()

        new_colums = []  # List to store the names of new columns added

        # Get the lists of stats
        (
            offensive_landed_stats,
            offensive_attempted_stats,
            defensive_landed_stats,
            defensive_attempted_stats,
        ) = self.get_fight_stats_lists()

        # Flatten the list of stats to include in cumsum calculation
        all_stats = (
            offensive_landed_stats
            + offensive_attempted_stats
            + defensive_landed_stats
            + defensive_attempted_stats
        )

        # Calculate per minute stats for each stat in the list
        for stat in all_stats:
            per_minute_stat_career = f"{stat}_per_minute_career"
            per_minute_stat_last_fight = f"{stat}_per_minute_prev"

            # Only add the last_fight stat to the new_columns list
            new_colums.append(per_minute_stat_last_fight)

            output[per_minute_stat_career] = fighter_data[stat].cumsum() / (
                output["total_time_in_octagon"]
            )
            output[per_minute_stat_last_fight] = fighter_data[stat] / (
                fighter_data["fight_duration"]
            )

        return output, new_colums

    def calculate_accuracy_and_defense_scores(
        self, output: pd.DataFrame, fighter_data: pd.DataFrame
    ) -> (pd.DataFrame, list):
        """Calculates the average accuracy and defense scores for the fighter."""

        new_columns = []  # List to store the names of new columns added

        # Define pairs of offensive and defensive stats that have both landed and attempted
        offensive_stats_pairs = {
            "sig_strikes": "sig_strikes_attempted",
            "sig_head_landed": "sig_head_attempted",
            "sig_body_landed": "sig_body_attempted",
            "sig_leg_landed": "sig_leg_attempted",
            "sig_clinch_landed": "sig_clinch_attempted",
            "sig_ground_landed": "sig_ground_attempted",
        }

        # Create defensive stats pairs by appending '_received' to each offensive stat pair
        defensive_stats_pairs = {
            k + "_received": v + "_received" for k, v in offensive_stats_pairs.items()
        }

        output = output.copy()

        # Calculate average accuracy for offensive stats
        for landed, attempted in offensive_stats_pairs.items():
            accuracy_last_fight_col = f"{landed}_accuracy_prev"
            accuracy_avg_col = f"{landed}_accuracy_avg"

            landed_cumulative = fighter_data[landed].cumsum()
            attempted_cumulative = (
                fighter_data[attempted].cumsum().replace(0, 1)
            )  # Avoid division by zero
            output[accuracy_last_fight_col] = landed_cumulative / attempted_cumulative
            output[accuracy_avg_col] = (
                (landed_cumulative / attempted_cumulative).expanding().mean()
            )
            new_columns += [accuracy_last_fight_col]

        # Calculate average defense for defensive stats
        for landed, attempted in defensive_stats_pairs.items():
            defense_last_fight_col = f"{landed}_defense_prev"
            defense_avg_col = f"{landed}_defense_avg"
            landed_cumulative = fighter_data[landed].cumsum()
            attempted_cumulative = fighter_data[attempted].cumsum().replace(0, 1)
            output[defense_last_fight_col] = 1 - (
                landed_cumulative / attempted_cumulative
            )
            output[defense_avg_col] = (
                (1 - (landed_cumulative / attempted_cumulative)).expanding().mean()
            )
            new_columns += [defense_last_fight_col]

        return output, new_columns

    def calculate_octagon_control(
        self, output: pd.DataFrame, fighter_data: pd.DataFrame
    ) -> (pd.DataFrame, list):
        """Calculates metrics indicating how often a fighter controls the fight and the mean scores of these over time."""
        # Calculate control ratios
        output["control_ratio_prev"] = fighter_data["control_time"] / fighter_data[
            "fight_duration"
        ].replace(0, 1)
        output["takedown_success_ratio_prev"] = fighter_data[
            "takedowns"
        ] / fighter_data["takedowns_attempted"].replace(0, 1)

        # Calculate the mean scores over time for the control metrics
        output["control_ratio_mean"] = output["control_ratio_prev"].expanding().mean()
        output["takedown_success_ratio_mean"] = (
            output["takedown_success_ratio_prev"].expanding().mean()
        )

        return output, [
            "control_ratio_prev",
            "takedown_success_ratio_prev",
            "control_ratio_mean",
            "takedown_success_ratio_mean",
        ]

    def get_fight_stats_lists(self):
        offensive_landed_stats = [
            "knock_down",
            "sig_strikes",
            "total_strike",
            "takedowns",
            "submission_attempt",
            "reversals",
            "control_time",
            "sig_head_landed",
            "sig_body_landed",
            "sig_leg_landed",
            "sig_distance_landed",
            "sig_clinch_landed",
            "sig_ground_landed",
        ]

        offensive_attempted_stats = [
            "sig_head_attempted",
            "sig_body_attempted",
            "sig_leg_attempted",
            "sig_distance_attempted",
            "sig_clinch_attempted",
            "sig_ground_attempted",
        ]

        defensive_landed_stats = [
            "knock_down_received",
            "sig_strikes_received",
            "total_strike_received",
            "takedowns_received",
            "submission_attempt_received",
            "reversals_received",
            "control_time_received",
            "sig_head_landed_received",
            "sig_body_landed_received",
            "sig_leg_landed_received",
            "sig_distance_landed_received",
            "sig_clinch_landed_received",
            "sig_ground_landed_received",
        ]

        defensive_attempted_stats = [
            "sig_head_attempted_received",
            "sig_body_attempted_received",
            "sig_leg_attempted_received",
            "sig_distance_attempted_received",
            "sig_clinch_attempted_received",
            "sig_ground_attempted_received",
        ]

        return (
            offensive_landed_stats,
            offensive_attempted_stats,
            defensive_landed_stats,
            defensive_attempted_stats,
        )

    def shift_stats_for_fight_start(self, output: pd.DataFrame) -> pd.DataFrame:
        """Shifts stats to represent the state before the start of the next fight"""
        output.iloc[:, 3:] = output.iloc[:, 3:].shift(1)
        return output

    def aggregate_fighter_stats(self, bouts_long: pd.DataFrame) -> pd.DataFrame:
        fighter_list = bouts_long.fighter.unique()
        # fighter_list = ['Khabib Nurmagomedov', 'Conor McGregor']
        fighter_stats = pd.DataFrame()

        self.logger.info("Aggregating time based fighter stats...")
        for fighter_name in tqdm(fighter_list):
            fighter_data = self.extract_fighter_data(fighter_name, bouts_long)
            fighter_stats = pd.concat([fighter_stats, fighter_data], ignore_index=True)

        self.logger.info("Aggregated fighter stats.")
        return fighter_stats

    def merge_fighter_records(
        self, fighters: pd.DataFrame, fighter_stats: pd.DataFrame
    ) -> pd.DataFrame:
        # Merge the fighter records with their stats

        # Add the maximum wins and losses from the fighter_stats dataframe for each fighter in the fighters dataframe
        ufc_wins = fighter_stats.groupby("fighter")["wins"].max()
        ufc_losses = fighter_stats.groupby("fighter")["losses"].max()

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
        bouts = bouts[["event_date", "fighter", "opponent", "win"]].copy()
        self.logger.info("Selected relevant columns from bouts dataframe.")
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
        fighter_stats_fights = self.aggregate_fighter_stats(bouts_long)
        fighter_stats_non_changing = self.merge_fighter_records(
            fighters, fighter_stats_fights
        )
        merged_fighter_stats = self.merge_fighter_stats(
            fighter_stats_fights, fighter_stats_non_changing
        )
        bouts_double = self.select_relevant_columns(bouts_long)
        self.write_to_database(merged_fighter_stats, "fighter_stats_no_age_double")
        self.write_to_database(bouts_double, "bouts_double")
        self.logger.info("Data preprocessing complete.")

    def get_training_columns(
        self, bouts_long: pd.DataFrame, merged_fighter_stats: pd.DataFrame
    ) -> pd.DataFrame:
        bouts_input = bouts_long.copy()
        bouts_long = bouts_long.drop(["win"], axis=1)
        df_fighter = bouts_long.merge(
            merged_fighter_stats,
            left_on=["event_date", "fighter", "opponent"],
            right_on=["event_date", "fighter", "opponent"],
            how="left",
        )
        df_opponent = bouts_long.merge(
            merged_fighter_stats,
            left_on=["event_date", "opponent", "fighter"],
            right_on=["event_date", "fighter", "opponent"],
            how="left",
            suffixes=("", "_drop"),
        )

        df_opponent = df_opponent[
            df_opponent.columns.drop(list(df_opponent.filter(regex="_drop")))
        ]

        columns_to_diff = [
            col
            for col in merged_fighter_stats.columns
            if col not in ["event_date", "fighter", "opponent", "win"]
        ]
        df_diff = bouts_long.copy()

        # reset index of df_diff, df_fighter and df_opponent
        df_diff = df_diff.reset_index(drop=True)
        df_fighter = df_fighter.reset_index(drop=True)
        df_opponent = df_opponent.reset_index(drop=True)

        for col in columns_to_diff:
            df_diff = df_diff.copy()
            # Calculate diff df
            df_diff[col + "_diff"] = df_fighter[col] - df_opponent[col]

            # Change name from col to col + _'fighter' for the col in df_fighter
            df_fighter.rename(columns={col: col + "_fighter"}, inplace=True)

            # Change name from col to col + _'opponent' for the col in df_opponent
            df_opponent.rename(columns={col: col + "_opponent"}, inplace=True)

        output_df = bouts_input.merge(
            df_fighter,
            left_on=["event_date", "fighter", "opponent"],
            right_on=["event_date", "fighter", "opponent"],
            how="left",
        )
        output_df = output_df.merge(
            df_opponent,
            left_on=["event_date", "fighter", "opponent"],
            right_on=["event_date", "fighter", "opponent"],
            how="left",
        )
        output_df = output_df.merge(
            df_diff,
            left_on=["event_date", "fighter", "opponent"],
            right_on=["event_date", "fighter", "opponent"],
            how="left",
        )

        return output_df

    def get_training_columns_inference(
        self, bouts: pd.DataFrame, fighter_1: pd.DataFrame, fighter_2: pd.DataFrame
    ) -> pd.DataFrame:
        bouts_input = bouts.copy()

        columns_to_diff = [
            col
            for col in fighter_1.columns
            if col not in ["event_date", "fighter", "opponent", "win"]
        ]
        df_diff = bouts.copy()
        for col in columns_to_diff:
            df_diff = df_diff.copy()
            # Calculate diff df
            df_diff[col + "_diff"] = fighter_1[col] - fighter_2[col]

            # Change name from col to col + _'fighter' for the col
            fighter_1.rename(columns={col: col + "_fighter"}, inplace=True)

            # Change name from col to col + _'opponent' for the col
            fighter_2.rename(columns={col: col + "_opponent"}, inplace=True)

        output_df = bouts_input.merge(
            fighter_1, left_on=["fighter"], right_on=["fighter"], how="left"
        )
        output_df = output_df.merge(
            fighter_2, left_on=["opponent"], right_on=["fighter"], how="left"
        )
        output_df = output_df.merge(
            df_diff,
            left_on=["event_date", "fighter", "opponent"],
            right_on=["event_date", "fighter", "opponent"],
            how="left",
        )

        return output_df

    def calculate_age_helper(self, merged_fighter_stats: pd.DataFrame) -> pd.DataFrame:
        merged_fighter_stats = merged_fighter_stats.copy()
        merged_fighter_stats["age"] = (
            merged_fighter_stats["event_date"] - merged_fighter_stats["date_of_birth"]
        ).dt.days / 365.25
        merged_fighter_stats.drop("date_of_birth", axis=1, inplace=True)
        return merged_fighter_stats

    def merge_fighter_stats(
        self,
        fighter_stats_fights: pd.DataFrame,
        fighter_stats_non_changing: pd.DataFrame,
    ) -> pd.DataFrame:
        fighter_stats = fighter_stats_fights.merge(
            fighter_stats_non_changing,
            left_on=["fighter"],
            right_on=["fighter"],
            how="left",
        )
        return fighter_stats

    def write_to_database(self, df: pd.DataFrame, tablename: str):
        df.to_sql(
            tablename,
            self.db_engine,
            schema="ufc",
            if_exists="replace",
            index=False,
        )
        self.logger.info(f"table {tablename} saved to the database.")

    def calculate_diff_on_inference(
        self,
        fighter_1: str,
        fighter_2: str,
        fighter_1_date=date.today().isoformat(),
        fighter_2_date=date.today().isoformat(),
    ) -> pd.DataFrame:
        """This function should calculate the diff between the two fighters for the time based inference dataframe"""

        # Get 1 row of model input
        df_for_columns = self.load_1_row_from_model_input()

        # Retrieve the correct data
        fighter_1_data = self.load_inference_data_fighter(fighter_1, fighter_1_date)
        fighter_2_data = self.load_inference_data_fighter(fighter_2, fighter_2_date)

        fighter_1_data["event_date"] = pd.to_datetime(fighter_1_date)
        fighter_2_data["event_date"] = pd.to_datetime(fighter_2_date)

        fighter_1_data["age"] = (
            fighter_1_data["event_date"] - fighter_1_data["date_of_birth"]
        ).dt.days / 365.25
        fighter_2_data["age"] = (
            fighter_2_data["event_date"] - fighter_2_data["date_of_birth"]
        ).dt.days / 365.25

        # Combine all function inputs in the first row of a dataframe
        bouts = pd.DataFrame()
        bouts["event_date"] = [
            fighter_1_date
        ]  # This date will get dropped during model infererence anyway
        bouts["fighter"] = [fighter_1]
        bouts["opponent"] = [fighter_2]

        training_df = self.get_training_columns_inference(
            bouts, fighter_1_data, fighter_2_data
        )
        training_df = self.drop_unusable_data(training_df)

        return training_df[df_for_columns.columns]


username = "postgres"
password = "postgres"
processor = DataProcessorDoublingUp(username, password, server="localhost")
# processor.load_male_fight_training_data()
processor.load_male_data_after_year(2015)
# bouts, fighters = processor.load_data()
# preprocessed_data = processor.preprocess_training_data(bouts, fighters)
# processor.calculate_diff_on_inference('Khabib Nurmagomedov',  'Justin Gaethje', '2020-10-24','2020-10-24')
