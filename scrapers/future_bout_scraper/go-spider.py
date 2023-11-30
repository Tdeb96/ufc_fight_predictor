from future_bout_scraper.spiders.bouts import FutureBouts
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

process = CrawlerProcess(get_project_settings())
process.crawl(FutureBouts)
process.start()
