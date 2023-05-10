from sqlalchemy import create_engine
from sqlalchemy.engine.base import Engine


def get_db_engine(
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
        f"{dbname}"
    )
    return engine
