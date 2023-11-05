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

    def extract_fighter_data(
        self, fighter_name: str, bouts_long: pd.DataFrame
    ) -> pd.DataFrame:
        """Extracts the fighter data from the bouts_long dataframe and returns it in a new dataframe"""
        fighter_data = bouts_long.loc[bouts_long.fighter == fighter_name].copy()
        output = fighter_data[["event_date", "fighter", "opponent"]].copy()
        # Creat a count for the number of wins over date
        output["wins"] = fighter_data.win.cumsum()

        # Create a count for the number of losses over date
        output["losses"] = (~fighter_data.win).cumsum()

        # Win streak
        output[
            "win_streak"
        ] = fighter_data.win.cumsum() - fighter_data.win.cumsum().where(
            ~fighter_data.win
        ).ffill().fillna(
            0
        )

        # Losing streak
        output["losing_streak"] = (~fighter_data.win).cumsum() - (
            ~fighter_data.win
        ).cumsum().where(fighter_data.win).ffill().fillna(0)

        # Count the total time spend in the octagon over date
        output["total_time_in_octagon"] = fighter_data["fight_duration"].cumsum()

        # title fights over time
        output["title_fights"] = fighter_data["title_fight"].cumsum()

        # performance bonuses over time
        output["performance_bonuses"] = fighter_data["performance_bonus"].cumsum()

        # win stats over date
        output["won_by_ko"] = fighter_data.won_by_ko.cumsum()
        output["won_by_sub"] = fighter_data.won_by_sub.cumsum()
        output["won_by_decision"] = fighter_data.won_by_decision.cumsum()
        output["lost_by_decision"] = fighter_data.lost_by_decision.cumsum()
        output["lost_by_sub"] = fighter_data.lost_by_sub.cumsum()
        output["lost_by_ko"] = fighter_data.lost_by_ko.cumsum()

        # Knocked out in the previous fight
        output["knocked_out_in_previous_fight"] = (
            output["lost_by_ko"].shift(1).fillna(0)
        )

        # Months since last fight
        output["months_since_last_fight"] = abs(
            (output["event_date"] - output["event_date"].shift(-1)).dt.days / 30
        )

        # Knockdown-related statistics
        knockdown_stats = ["knock_down", "knock_down_received"]

        # Significant strikes-related statistics
        sig_strikes_stats = ["sig_strikes", "sig_strikes_received"]

        # Total strikes-related statistics
        total_strikes_stats = ["total_strike", "total_strike_received"]

        # Takedowns-related statistics
        takedowns_stats = ["takedowns", "takedowns_received"]

        # Submission attempts-related statistics
        submission_stats = ["submission_attempt", "submission_attempt_received"]

        # Reversals-related statistics
        reversals_stats = ["reversals", "reversals_received"]

        # Control time-related statistics
        control_time_stats = ["control_time", "control_time_received"]

        # Significant strikes by target location (head, body, leg) - landed
        sig_strikes_by_location_landed_stats = [
            "sig_head_landed",
            "sig_body_landed",
            "sig_leg_landed",
            "sig_head_landed_received",
            "sig_body_landed_received",
            "sig_leg_landed_received",
        ]

        # Significant strikes by target location (head, body, leg) - attempted
        sig_strikes_by_location_attempted_stats = [
            "sig_head_attempted",
            "sig_body_attempted",
            "sig_leg_attempted",
            "sig_head_attempted_received",
            "sig_body_attempted_received",
            "sig_leg_attempted_received",
        ]

        # Significant strikes by fight position (distance, clinch, ground) - landed
        sig_strikes_by_position_landed_stats = [
            "sig_distance_landed",
            "sig_clinch_landed",
            "sig_ground_landed",
            "sig_distance_landed_received",
            "sig_clinch_landed_received",
            "sig_ground_landed_received",
        ]

        # Significant strikes by fight position (distance, clinch, ground) - attempted
        sig_strikes_by_position_attempted_stats = [
            "sig_distance_attempted",
            "sig_clinch_attempted",
            "sig_ground_attempted",
            "sig_distance_attempted_received",
            "sig_clinch_attempted_received",
            "sig_ground_attempted_received",
        ]

        # Compile all stats into a single list
        stats_to_cumsum = (
            knockdown_stats
            + sig_strikes_stats
            + total_strikes_stats
            + takedowns_stats
            + submission_stats
            + reversals_stats
            + control_time_stats
            + sig_strikes_by_location_landed_stats
            + sig_strikes_by_location_attempted_stats
            + sig_strikes_by_position_landed_stats
            + sig_strikes_by_position_attempted_stats
        )

        # Assuming 'temp' is your DataFrame and 'output' is a DataFrame to store the results
        for stat in stats_to_cumsum:
            output[stat] = fighter_data[stat].cumsum()

        # Assuming 'temp' is your DataFrame and 'output' is a DataFrame to store the results
        for stat in stats_to_cumsum:
            output[stat] = fighter_data[stat].cumsum()

        # Change all of the stats_to_cumsum now to be devided by the total time in the octagon
        for stat in stats_to_cumsum:
            output[stat] = output[stat] / output["total_time_in_octagon"]

        # Rename the columns in stats_to_cumsum to have a _per_minute suffix
        output = output.rename(
            columns={stat: stat + "_per_minute" for stat in stats_to_cumsum}
        )

        # Career damage taken
        output["sig_strikes_to_head"] = (
            output["sig_head_landed_received_per_minute"]
            * output["total_time_in_octagon"]
        )

        # Shift all rows except event_date and fighter by 1
        ## Now we have to think, which characteristics are known about the fighter when the fight starts? The answer is the characteristics after the last fight
        output.iloc[:, 3:] = output.iloc[:, 3:].shift(1)

        # Drop the first row as it will be filled with NaNs
        output = output.iloc[1:, :]

        return output

    def preprocess_training_data(
        self, bouts: pd.DataFrame, fighters: pd.DataFrame
    ) -> pd.DataFrame:
        # Data prep for the fighters table

        # Filter out the fighters who did not have any fights yet
        fighters_that_fought = set(bouts.fighter1).union(set(bouts.fighter2))
        fighters = fighters.loc[fighters.fighter_name.isin(fighters_that_fought)]

        fighters.date_of_birth = fighters.date_of_birth.replace("--", None)
        fighters["date_of_birth"] = pd.to_datetime(
            fighters["date_of_birth"], format="%b %d %Y"
        )

        fighters = fighters.copy()

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
            (fighters.fighter_name == "Mike Davis") & (fighters.weight == 155),
            "fighter_name",
        ] = "Mike Davis 155"
        fighters.loc[
            (fighters.fighter_name == "Bruno Silva") & (fighters.weight == 125),
            "fighter_name",
        ] = "Bruno Silva 125"

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

        # Only keep the height, reach and date of birth
        fighters = fighters[
            ["fighter_name", "height_cm", "reach", "date_of_birth", "fighter_record"]
        ]

        # Rename fighter_name to fighter
        fighters = fighters.rename(columns={"fighter_name": "fighter"})

        # Extract total wins and losses
        fighters["total_wins"] = fighters.fighter_record.map(
            lambda x: x.split("Record: ")[1].split("-")[0]
        )
        fighters["total_losses"] = fighters.fighter_record.map(
            lambda x: x.split("Record: ")[1].split("-")[1]
        )

        # drop the fighter_record column
        fighters = fighters.drop("fighter_record", axis=1)

        # We can just drop the fighters with no date of birth
        fighters = fighters.dropna(subset=["date_of_birth"])

        # We will compute the missing reach values with a quick linear regression based on height
        fighters_reach = fighters.copy()
        fighters_reach = fighters_reach.dropna(subset=["reach"])

        # We will use the height to predict the reach
        from sklearn.linear_model import LinearRegression

        lr = LinearRegression()
        lr.fit(fighters_reach[["height_cm"]], fighters_reach["reach"])

        # Predict the reach for the fighters with missing reach
        fighters_reach_missing = fighters.copy()
        fighters_reach_missing = fighters_reach_missing[
            fighters_reach_missing.reach.isna()
        ]
        fighters_reach_missing["reach"] = lr.predict(
            fighters_reach_missing[["height_cm"]]
        )

        # Merge the two datasets
        fighters = pd.concat([fighters_reach, fighters_reach_missing])

        # First convert bouts date to datetime
        bouts["event_date"] = pd.to_datetime(bouts["event_date"], format="%B %d %Y")

        # We need to make the same name changes in the bouts dataframe for the duplicate fighters
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

        # Drop index, id and event_url column
        bouts.drop(["index", "id", "event_url"], axis=1, inplace=True)

        # Drop all bouts without a win
        bouts = bouts.loc[bouts.win != 0]
        bouts.head(5)

        # Engineer a fight time feature in minutes
        bouts["fight_duration"] = round(
            (bouts["round_"] - 1) * 5
            + bouts["time_minutes"]
            + bouts["time_seconds"] / 60,
            2,
        )
        bouts = bouts.drop(["round_", "time_minutes", "time_seconds"], axis=1)

        # Fix the control time columns by turning m:ss into seconds
        bouts["control_time_1"] = bouts["control_time_1"].replace("--", "0:00")
        bouts["control_time_2"] = bouts["control_time_2"].replace("--", "0:00")
        bouts["control_time_1"] = bouts["control_time_1"].fillna("0:00")
        bouts["control_time_2"] = bouts["control_time_2"].fillna("0:00")
        bouts["control_time_1"] = bouts["control_time_1"].map(
            lambda x: int(x.split(":")[0]) * 60 + int(x.split(":")[1])
        )
        bouts["control_time_2"] = bouts["control_time_2"].map(
            lambda x: int(x.split(":")[0]) * 60 + int(x.split(":")[1])
        )

        # Turn the table from wide to long format

        # Extract the fighter-specific columns again with the correct dataframe
        columns_1 = [col for col in bouts.columns if col.endswith("_1")]
        columns_2 = [col for col in bouts.columns if col.endswith("_2")]
        columns_shared_cleaned = [
            col.rstrip("_1") if col.endswith("_1") else col for col in columns_1
        ]

        # Columns that are common for both fighters
        common_columns = [
            col
            for col in bouts.columns
            if not col.endswith("_1")
            and not col.endswith("_2")
            and col not in ["fighter1", "fighter2"]
        ]

        # Create two separate dataframes for each fighter
        df_fighter1 = bouts[common_columns + columns_1].copy()
        df_fighter2 = bouts[common_columns + columns_2].copy()

        # Rename the columns by stripping the '_1' or '_2' suffix to match the desired format
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

        # Concatenate the two dataframes
        bouts_long = pd.concat(
            [df_fighter1, df_fighter2], ignore_index=True, sort=False
        )

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
            ],
            axis=1,
        )

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

        fighter_list = bouts_long.fighter.unique()

        # Run the extract_fighter_data for all fighters, combine together in one dataframe
        df_bouts_processed = pd.concat(
            [
                self.extract_fighter_data(fighter_name, bouts_long)
                for fighter_name in fighter_list
            ],
            ignore_index=True,
            sort=False,
        )

        # Add the maximum wins and losses from the df_bouts_processed dataframe for each fighter in the fighters dataframe
        ufc_wins = df_bouts_processed.groupby("fighter")["wins"].max()
        ufc_losses = df_bouts_processed.groupby("fighter")["losses"].max()

        fighters = fighters.merge(ufc_wins, on="fighter", how="left").fillna(0)
        fighters = fighters.merge(ufc_losses, on="fighter", how="left").fillna(0)

        # subtracts wins from total_wins to get the number of wins outside of the UFC
        fighters["wins_outside_ufc"] = (
            fighters["total_wins"].astype(int) - fighters["wins"]
        )
        fighters["losses_outside_ufc"] = (
            fighters["total_losses"].astype(int) - fighters["losses"]
        )

        # drop all other columns
        fighters = fighters.drop(
            ["total_wins", "total_losses", "wins", "losses"], axis=1
        )

        # Filter out relevant columns from the bouts df
        bouts = bouts[["event_date", "fighter1", "fighter2", "winner"]].copy()

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
        bouts = bouts[["event_date", "fighter1", "fighter2", "win"]].copy()

        # get all the columns from the df_bouts_processed dataframe except event_date, fighter and opponent
        columns_to_diff_bouts = list(df_bouts_processed.columns)
        columns_to_diff_bouts.remove("event_date")
        columns_to_diff_bouts.remove("fighter")
        columns_to_diff_bouts.remove("opponent")

        # left join the bouts dataframe with the df_bouts_processed dataframe on event_date, fighter1 = fighter and fighter2 = opponent
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

        bouts_diff = bouts_fighter1.copy()
        bouts_diff[columns_to_diff_bouts] = (
            bouts_fighter1[columns_to_diff_bouts]
            - bouts_fighter2[columns_to_diff_bouts]
        )
        bouts_diff.drop(["fighter", "opponent"], axis=1, inplace=True)

        fighter_columns_to_diff = list(fighters.columns)
        fighter_columns_to_diff.remove("fighter")
        fighter_columns_to_diff.remove("date_of_birth")
        fighter_columns_to_diff.append("age")

        # Fighter diff
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

        fighter_diff = fighter_1.copy()
        fighter_diff[fighter_columns_to_diff] = (
            fighter_1[fighter_columns_to_diff] - fighter_2[fighter_columns_to_diff]
        )

        # drop date_of_birth column
        fighter_diff = fighter_diff.drop(["date_of_birth"], axis=1)
        fighter_diff = fighter_diff.drop(["fighter"], axis=1)

        # Append _diff to all columns in bouts_diff and fighter_diff corresponding to columns_to_diff_bouts and fighter_columns_to_diff
        bouts_diff.columns = [
            col + "_diff" if col in columns_to_diff_bouts else col
            for col in bouts_diff.columns
        ]
        fighter_diff.columns = [
            col + "_diff" if col in fighter_columns_to_diff else col
            for col in fighter_diff.columns
        ]

        # left join bouts with bouts_diff and fighter_diff, avoiding duplicate columns
        bouts_full = bouts.merge(
            bouts_diff,
            left_on=["event_date", "fighter1", "fighter2", "win"],
            right_on=["event_date", "fighter1", "fighter2", "win"],
            how="left",
        )
        bouts_full = bouts_full.merge(
            fighter_diff,
            left_on=["event_date", "fighter1", "fighter2", "win"],
            right_on=["event_date", "fighter1", "fighter2", "win"],
            how="left",
        )

        # If we have missing data, this means that it was the first fight for one of the fighters in the bout. We unfortunately can't use these rows in our model and have to drop them
        bouts_full = bouts_full.dropna()

        # Write the bouts df to the database
        bouts_full.to_sql(
            "model_input",
            self.db_engine,
            schema="ufc",
            if_exists="replace",
            index=False,
        )
        self.logger.info(
            "Data preprocessing complete. Model input saved to the database."
        )
        return bouts

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
