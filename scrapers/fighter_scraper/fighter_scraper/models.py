from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.engine.url import URL

from .settings import DATABASE

DeclarativeBase = declarative_base()

def db_connect():
    """
    Performs database connection using database settings from settings.py.
    Returns sqlalchemy engine instance
    """
    return create_engine(URL(**DATABASE), connect_args={'options': '-csearch_path={}'.format('ufc')})

def create_fighters_table(engine):
    """"""
    DeclarativeBase.metadata.create_all(engine)

class Fighters(DeclarativeBase):
    """Sqlalchemy fighter model"""
    __tablename__ = "fighter"

    id = Column(Integer, primary_key = True)
    fighter_name = Column("fighter_name", String)
    fighter_record = Column("fighter_record", String)
    height = Column("height", String)
    weight = Column("weight", String)
    reach = Column("reach", String)
    stance = Column("stance", String)
    date_of_birth = Column("date_of_birth", String)
    slpm = Column("slpm", Float(53))  # strikes landed per min stat
    td_avg = Column("td_avg", Float(53))  # takedown average
    strike_acc = Column("strike_acc", Float(53))  # striking accuracy
    td_acc = Column("td_acc", Float(53))  # takedown accuracy
    sapm = Column("sapm", Float(53))  # strikes absorbed per minute
    td_def = Column("td_def", Float(53))  # takedown defence
    strike_def = Column("strike_def", Float(53))  # striking defence
    sub_avg = Column("sub_avg", Float(53))  # submission average