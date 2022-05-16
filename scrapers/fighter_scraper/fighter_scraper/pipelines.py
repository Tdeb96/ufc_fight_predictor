from sqlalchemy.orm import sessionmaker
from .models import Fighters, db_connect, create_fighters_table

class FighterScraperPipeline:
    """fighterScraper pipeline for storing scraped items in the database"""
    def __init__(self):
        """
        Initializes database connection and sessionmaker.
        Creates fighters table.
        """
        engine = db_connect()

        # Create ufc schema
        with engine.connect() as conn:
            conn.execute('CREATE SCHEMA IF NOT EXISTS ufc')

        # Create fighters table
        create_fighters_table(engine)
        
        self.Session = sessionmaker(bind=engine)
    
    def process_item(self, item, spider):
            """Save fighters in the database.

            This method is called for every item pipeline component.

            """
            session = self.Session()
            deal = Fighters(**item)

            try:
                session.add(deal)
                session.commit()
            except:
                session.rollback()
                raise
            finally:
                session.close()

            return item
