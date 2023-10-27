from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.engine.url import URL
from sqlalchemy.ext.declarative import declarative_base

from .settings import DATABASE

DeclarativeBase = declarative_base()


def db_connect():
    """
    Performs database connection using database settings from settings.py.
    Returns sqlalchemy engine instance
    """
    return create_engine(
        URL(**DATABASE), connect_args={"options": "-csearch_path={}".format("ufc")}
    )


def create_bouts_table(engine):
    """"""
    DeclarativeBase.metadata.create_all(engine)


class Bouts(DeclarativeBase):
    """Sqlalchemy bouts model"""

    __tablename__ = "bouts"

    id = Column(Integer, primary_key=True)
    event_url = Column("event_url", String)
    event_name = Column("event_name", String)
    event_date = Column("event_date", String)
    win = Column("win", String)
    winner = Column("winner", String)
    fighter1 = Column("fighter1", String)
    fighter2 = Column("fighter2", String)
    weight_class = Column("weight_class", String)
    title_fight = Column("title_fight", String)
    performance_bonus = Column("performance_bonus", String)
    win_method_type = Column("win_method_type", String)
    round_ = Column("round_", String)
    time_minutes = Column("time_minutes", Integer)
    time_seconds = Column("time_seconds", Integer)
    knock_down_1 = Column("knock_down_1", Integer)
    knock_down_2 = Column("knock_down_2", Integer)
    sig_strikes_1 = Column("sig_strikes_1", Integer)
    sig_strikes_2 = Column("sig_strikes_2", Integer)
    sig_strike_perc_1 = Column("sig_strike_perc_1", Integer)
    sig_strike_perc_2 = Column("sig_strike_perc_2", Integer)
    total_strike_1 = Column("total_strike_1", Integer)
    total_strike_2 = Column("total_strike_2", Integer)
    takedowns_1 = Column("takedowns_1", Integer)
    takedowns_2 = Column("takedowns_2", Integer)
    takedown_perc_1 = Column("takedown_perc_1", Integer)
    takedown_perc_2 = Column("takedown_perc_2", Integer)
    submission_attempt_1 = Column("submission_attempt_1", Integer)
    submission_attempt_2 = Column("submission_attempt_2", Integer)
    reversals_1 = Column("reversals_1", Integer)
    reversals_2 = Column("reversals_2", Integer)
    control_time_1 = Column("control_time_1", String)
    control_time_2 = Column("control_time_2", String)
    sig_head_landed_1 = Column("sig_head_landed_1", Integer)
    sig_head_landed_2 = Column("sig_head_landed_2", Integer)
    sig_head_attempted_1 = Column("sig_head_attempted_1", Integer)
    sig_head_attempted_2 = Column("sig_head_attempted_2", Integer)
    sig_body_landed_1 = Column("sig_body_landed_1", Integer)
    sig_body_landed_2 = Column("sig_body_landed_2", Integer)
    sig_body_attempted_1 = Column("sig_body_attempted_1", Integer)
    sig_body_attempted_2 = Column("sig_body_attempted_2", Integer)
    sig_leg_landed_1 = Column("sig_leg_landed_1", Integer)
    sig_leg_landed_2 = Column("sig_leg_landed_2", Integer)
    sig_leg_attempted_1 = Column("sig_leg_attempted_1", Integer)
    sig_leg_attempted_2 = Column("sig_leg_attempted_2", Integer)
    sig_distance_landed_1 = Column("sig_distance_landed_1", Integer)
    sig_distance_landed_2 = Column("sig_distance_landed_2", Integer)
    sig_distance_attempted_1 = Column("sig_distance_attempted_1", Integer)
    sig_distance_attempted_2 = Column("sig_distance_attempted_2", Integer)
    sig_clinch_landed_1 = Column("sig_clinch_landed_1", Integer)
    sig_clinch_landed_2 = Column("sig_clinch_landed_2", Integer)
    sig_clinch_attempted_1 = Column("sig_clinch_attempted_1", Integer)
    sig_clinch_attempted_2 = Column("sig_clinch_attempted_2", Integer)
    sig_ground_landed_1 = Column("sig_ground_landed_1", Integer)
    sig_ground_landed_2 = Column("sig_ground_landed_2", Integer)
    sig_ground_attempted_1 = Column("sig_ground_attempted_1", Integer)
    sig_ground_attempted_2 = Column("sig_ground_attempted_2", Integer)
