BOT_NAME = 'fighter_scraper'

SPIDER_MODULES = ['fighter_scraper.spiders']

DOWNLOAD_DELAY = 1.5
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 2
AUTOTHROTTLE_TARGET_CONCURRENCY = 6

DATABASE = {
    'drivername': 'postgresql',
    'host': 'localhost',
    'port': '5432',
    'username': 'postgres',
    'password': 'postgres',
    'database': 'ufc'
}

# Obey robots.txt rules
ROBOTSTXT_OBEY = True

ITEM_PIPELINES = {
    'fighter_scraper.pipelines.FighterScraperPipeline': 400,
}

