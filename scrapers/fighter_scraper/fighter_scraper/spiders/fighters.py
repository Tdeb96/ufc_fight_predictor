from __future__ import absolute_import

import logging
import string

import pandas as pd
import scrapy
from fighter_scraper.items import FightScraperItem
from scrapy import Selector
from scrapy.crawler import CrawlerProcess
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


def get_fighter_url(engine: Engine) -> pd.DataFrame:
    df = pd.read_sql("SELECT fighter_url FROM ufc.fighters", engine)
    return df


class Fighters(scrapy.Spider):
    name = "fighterSpider"

    def start_requests(self):
        start_urls = [
            "http://ufcstats.com/statistics/fighters?char=" + letter + "&page=all"
            for letter in string.ascii_lowercase
        ]
        for url in start_urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        previously_scraped_fighters = get_fighter_url(
            get_db_engine("postgres", "postgres")
        )
        links = response.xpath(
            "//td[@class ='b-statistics__table-col']//@href"
        ).extract()
        links = [
            link for link in links if link not in previously_scraped_fighters.values
        ]
        if len(links) == 0:
            logging.info("All fighters have been scraped")
        for link in links:
            yield scrapy.Request(
                link, callback=self.parse_fighter, cb_kwargs=dict(fighter_url=link)
            )

    def parse_fighter(self, response, fighter_url):
        sel = Selector(response)
        fighter_item = FightScraperItem()
        fighter_item["fighter_url"] = fighter_url
        fighter_item["fighter_name"] = (
            sel.xpath("//span[@class='b-content__title-highlight']//text()")
            .extract()[0]
            .strip()
        )
        fighter_item["fighter_record"] = (
            sel.xpath("//span[@class='b-content__title-record']//text()")
            .extract()[0]
            .strip()
        )
        for item in response.xpath('//ul[@class="b-list__box-list"]'):
            try:
                fighter_item["height"] = (
                    item.xpath("li[1]//text() ").extract()[2].strip().replace("\\", "")
                )
            except Exception:
                fighter_item["height"] = None
            try:
                fighter_item["weight"] = int(
                    item.xpath("li[2]//text()").extract()[2].replace("lbs.", "")
                )
            except Exception:
                fighter_item["weight"] = None
            try:
                fighter_item["reach"] = int(
                    item.xpath("li[3]//text()").extract()[2].replace('"', "")
                )
            except Exception:
                fighter_item["reach"] = None

            fighter_item["stance"] = item.xpath("li[4]//text()").extract()[2].strip()
            fighter_item["date_of_birth"] = (
                item.xpath("li[5]//text()").extract()[2].strip().replace(",", "")
            )

        for item in response.xpath('//div[@class="b-list__info-box-left"]//ul'):
            fighter_item["slpm"] = float(
                item.xpath("li[1]//text() ").extract()[2].strip()
            )
            fighter_item["strike_acc"] = int(
                item.xpath("li[2]//text()").extract()[2].replace("%", "")
            )
            fighter_item["sapm"] = float(item.xpath("li[3]//text()").extract()[2])
            fighter_item["strike_def"] = float(
                item.xpath("li[4]//text()").extract()[2].replace("%", "")
            )

        for item in response.xpath(
            '//div[@class="b-list__info-box-right b-list__info-box_style-margin-right"]//ul'
        ):
            fighter_item["td_avg"] = float(item.xpath("li[2]//text()").extract()[2])
            fighter_item["td_acc"] = int(
                item.xpath("li[3]//text()").extract()[2].replace("%", "")
            )
            fighter_item["td_def"] = int(
                item.xpath("li[4]//text()").extract()[2].replace("%", "")
            )
            fighter_item["sub_avg"] = float(item.xpath("li[5]//text()").extract()[2])

        yield fighter_item


if __name__ == "__main__":
    process = CrawlerProcess(
        {"USER_AGENT": "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)"}
    )

    process.crawl(Fighters)
    process.start()
