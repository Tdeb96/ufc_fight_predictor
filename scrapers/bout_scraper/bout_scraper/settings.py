import os

BOT_NAME = 'bout_scraper'

SPIDER_MODULES = ['bout_scraper.spiders']

DOWNLOAD_DELAY = 1.5
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 2
AUTOTHROTTLE_TARGET_CONCURRENCY = 6

DATABASE = {
    'drivername': 'postgresql',
    'host': 'timescale',
    'port': '5432',
    'username': os.environ.get("POSTGRES_USERNAME"),
    'password': os.environ.get("POSTGRES_PASSWORD"),
    'database': 'ufc'
}

# Obey robots.txt rules
ROBOTSTXT_OBEY = True

ITEM_PIPELINES = {
    'bout_scraper.pipelines.BoutScraperPipeline': 400,
}

