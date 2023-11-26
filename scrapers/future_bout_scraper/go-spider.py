from bout_scraper.spiders.bouts import Bouts
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

process = CrawlerProcess(get_project_settings())
process.crawl(Bouts)
process.start()
