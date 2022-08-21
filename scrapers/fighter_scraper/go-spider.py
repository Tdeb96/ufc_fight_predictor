from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from fighter_scraper.spiders.fighters import Fighters

process = CrawlerProcess(get_project_settings())
process.crawl(Fighters)
process.start()