import logging

from sqlalchemy.orm import sessionmaker

from .models import Bouts, create_bouts_table, db_connect


class BoutScraperPipeline:
    """boutScraper pipeline for storing scraped items in the database"""

    def __init__(self):
        """
        Initializes database connection and sessionmaker.
        Creates bouts table.
        """
        logging.info("Initializing BoutScraperPipeline")
        engine = db_connect()

        # Create ufc schema
        with engine.connect() as conn:
            conn.execute("CREATE SCHEMA IF NOT EXISTS ufc")

        # Create bouts table
        create_bouts_table(engine)

        self.Session = sessionmaker(bind=engine)

    def process_item(self, item, spider):
        """Save bouts in the database.

        This method is called for every item pipeline component.

        """
        session = self.Session()
        deal = Bouts(**item)

        try:
            logging.info(f"Inserting {deal} into ufc.bouts")
            session.add(deal)
            session.commit()
        except:
            session.rollback()
            raise
        finally:
            session.close()

        return item
