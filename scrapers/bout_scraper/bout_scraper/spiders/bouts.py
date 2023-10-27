from __future__ import absolute_import

import logging

import pandas as pd
import scrapy
from bout_scraper.items import BoutScraperItem
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


def get_event_url(engine: Engine) -> pd.DataFrame:
    df = pd.read_sql("SELECT event_url FROM ufc.bouts", engine)
    return df


class Bouts(scrapy.Spider):
    name = "boutSpider"

    def start_requests(self):
        start_urls = ["http://ufcstats.com/statistics/events/completed?page=all"]

        for url in start_urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        previously_scraped_urls = get_event_url(get_db_engine("postgres", "postgres"))
        links = response.xpath(
            "//td[@class='b-statistics__table-col']//a/@href"
        ).extract()
        links = [link for link in links if link not in previously_scraped_urls.values]
        if len(links) == 0:
            logging.info("No new events to scrape")
            return
        for link in links:
            logging.info(f"Scraping {link}")
            yield scrapy.Request(
                link, callback=self.parse_bouts, cb_kwargs=dict(event_url=link)
            )

    def parse_bouts(self, response, event_url):
        bouts = response.xpath(
            "//tr[@class='b-fight-details__table-row b-fight-details__table-row__hover js-fight-details-click']/@data-link"
        ).extract()
        event_date = response.xpath(
            '//li[1][@class="b-list__box-list-item"]/text()'
        ).extract()[1]
        for bout in bouts:
            yield scrapy.Request(
                bout,
                callback=self.parse_bouts_advanced,
                cb_kwargs=dict(event_url=event_url, event_date=event_date),
            )

    def parse_bouts_advanced(self, response, event_url, event_date):
        bout_item = BoutScraperItem()
        bout_item["event_url"] = event_url
        bout_item["event_name"] = (
            response.xpath('//h2[@class="b-content__title"]//a/text()')
            .extract()[0]
            .strip()
        )
        bout_item["event_date"] = event_date.strip().replace(",", "")

        # Extract winner
        try:
            response.xpath(
                '//i[@class="b-fight-details__person-status b-fight-details__person-status_style_green"]//text()'
            ).extract()[0].strip()
            bout_item["win"] = True
            bout_item["winner"] = (
                response.xpath(
                    '//i[@class="b-fight-details__person-status b-fight-details__person-status_style_green"]/../div//a/text()'
                )
                .extract()[0]
                .strip()
            )

        except:
            bout_item["win"] = False
            bout_item["winner"] = "--"

        bout_item["fighter1"] = (
            response.xpath(
                '//div[@class="b-fight-details__persons clearfix"]//a/text()'
            )
            .extract()[0]
            .strip()
        )
        bout_item["fighter2"] = (
            response.xpath(
                '//div[@class="b-fight-details__persons clearfix"]//a/text()'
            )
            .extract()[1]
            .strip()
        )

        if (
            response.xpath('//i[@class="b-fight-details__fight-title"]/text()')
            .extract()[0]
            .strip()
            != ""
        ):
            bout_item["weight_class"] = (
                response.xpath('//i[@class="b-fight-details__fight-title"]/text()')
                .extract()[0]
                .strip()
            )
            bout_item["title_fight"] = False
            bout_item["performance_bonus"] = False

        else:
            bout_item["weight_class"] = (
                response.xpath('//i[@class="b-fight-details__fight-title"]/text()')
                .extract()[1]
                .strip()
            )
            rewards = response.xpath(
                '//i[@class="b-fight-details__fight-title"]//img/@src'
            ).extract()
            if (
                "http://1e49bc5171d173577ecd-1323f4090557a33db01577564f60846c.r80.cf1.rackcdn.com/belt.png"
                in rewards
            ) & (len(rewards) > 1):
                bout_item["title_fight"] = True
                bout_item["performance_bonus"] = True
            elif (
                "http://1e49bc5171d173577ecd-1323f4090557a33db01577564f60846c.r80.cf1.rackcdn.com/belt.png"
                in rewards
            ):
                bout_item["title_fight"] = True
                bout_item["performance_bonus"] = False
            else:
                bout_item["title_fight"] = False
                bout_item["performance_bonus"] = True

        bout_item["win_method_type"] = (
            response.xpath('//p[@class="b-fight-details__text"]//i/text()')
            .extract()[3]
            .strip()
        )
        bout_item["round_"] = int(
            response.xpath('//p[@class="b-fight-details__text"]//i/text()')
            .extract()[7]
            .strip()
        )
        bout_item["time_minutes"] = int(
            response.xpath('//p[@class="b-fight-details__text"]//i/text()')
            .extract()[10]
            .strip()
            .split(":")[0]
        )
        bout_item["time_seconds"] = int(
            response.xpath('//p[@class="b-fight-details__text"]//i/text()')
            .extract()[10]
            .strip()
            .split(":")[1]
        )

        # Check if round by round stats are available
        try:
            totals = response.xpath('//tbody[@class="b-fight-details__table-body"]')[0]

            # Extract round by round stats
            bout_item["knock_down_1"] = int(
                totals.xpath("//td[2]//p/text()").extract()[0].strip()
            )
            bout_item["knock_down_2"] = int(
                totals.xpath("//td[2]//p/text()").extract()[1].strip()
            )
            bout_item["sig_strikes_1"] = int(
                totals.xpath("//td[3]//p/text()").extract()[0].strip().split(" of")[0]
            )
            bout_item["sig_strikes_2"] = int(
                totals.xpath("//td[3]//p/text()").extract()[1].strip().split(" of")[0]
            )
            bout_item["sig_strike_perc_1"] = (
                totals.xpath("//td[4]//p/text()")
                .extract()[0]
                .strip()
                .replace("%", "")
                .replace("---", "")
            )
            bout_item["sig_strike_perc_2"] = (
                totals.xpath("//td[4]//p/text()")
                .extract()[1]
                .strip()
                .replace("%", "")
                .replace("---", "")
            )
            bout_item["total_strike_1"] = int(
                totals.xpath("//td[5]//p/text()").extract()[0].strip().split(" of")[0]
            )
            bout_item["total_strike_2"] = int(
                totals.xpath("//td[5]//p/text()").extract()[1].strip().split(" of")[0]
            )
            bout_item["takedowns_1"] = int(
                totals.xpath("//td[6]//p/text()").extract()[0].strip().split(" of")[0]
            )
            bout_item["takedowns_2"] = int(
                totals.xpath("//td[6]//p/text()").extract()[1].strip().split(" of")[0]
            )
            bout_item["takedown_perc_1"] = (
                totals.xpath("//td[7]//p/text()")
                .extract()[0]
                .strip()
                .replace("%", "")
                .replace("---", "")
            )
            bout_item["takedown_perc_2"] = (
                totals.xpath("//td[7]//p/text()")
                .extract()[1]
                .strip()
                .replace("%", "")
                .replace("---", "")
            )
            bout_item["submission_attempt_1"] = int(
                totals.xpath("//td[8]//p/text()").extract()[0].strip()
            )
            bout_item["submission_attempt_2"] = int(
                totals.xpath("//td[8]//p/text()").extract()[1].strip()
            )
            bout_item["reversals_1"] = int(
                totals.xpath("//td[9]//p/text()").extract()[0].strip()
            )
            bout_item["reversals_2"] = int(
                totals.xpath("//td[9]//p/text()").extract()[1].strip()
            )
            bout_item["control_time_1"] = (
                totals.xpath("//td[10]//p/text()")
                .extract()[0]
                .strip()
                .replace("---", "")
            )
            bout_item["control_time_2"] = (
                totals.xpath("//td[10]//p/text()")
                .extract()[1]
                .strip()
                .replace("---", "")
            )

            # Extract significant strikes
            C = int(len(totals.xpath("//td[4]//p/text()").extract()) / 2)

            bout_item["sig_head_landed_1"] = int(
                totals.xpath("//td[4]//p/text()").extract()[C].strip().split(" of ")[0]
            )
            bout_item["sig_head_landed_2"] = int(
                totals.xpath("//td[4]//p/text()")
                .extract()[C + 1]
                .strip()
                .split(" of ")[0]
            )
            bout_item["sig_head_attempted_1"] = int(
                totals.xpath("//td[4]//p/text()").extract()[C].strip().split(" of ")[1]
            )
            bout_item["sig_head_attempted_2"] = int(
                totals.xpath("//td[4]//p/text()")
                .extract()[C + 1]
                .strip()
                .split(" of ")[1]
            )

            bout_item["sig_body_landed_1"] = int(
                totals.xpath("//td[5]//p/text()").extract()[C].strip().split(" of ")[0]
            )
            bout_item["sig_body_landed_2"] = int(
                totals.xpath("//td[5]//p/text()")
                .extract()[C + 1]
                .strip()
                .split(" of ")[0]
            )
            bout_item["sig_body_attempted_1"] = int(
                totals.xpath("//td[5]//p/text()").extract()[C].strip().split(" of ")[1]
            )
            bout_item["sig_body_attempted_2"] = int(
                totals.xpath("//td[5]//p/text()")
                .extract()[C + 1]
                .strip()
                .split(" of ")[1]
            )

            bout_item["sig_leg_landed_1"] = int(
                totals.xpath("//td[6]//p/text()").extract()[C].strip().split(" of ")[0]
            )
            bout_item["sig_leg_landed_2"] = int(
                totals.xpath("//td[6]//p/text()")
                .extract()[C + 1]
                .strip()
                .split(" of ")[0]
            )
            bout_item["sig_leg_attempted_1"] = int(
                totals.xpath("//td[6]//p/text()").extract()[C].strip().split(" of ")[1]
            )
            bout_item["sig_leg_attempted_2"] = int(
                totals.xpath("//td[6]//p/text()")
                .extract()[C + 1]
                .strip()
                .split(" of ")[1]
            )

            bout_item["sig_distance_landed_1"] = int(
                totals.xpath("//td[7]//p/text()").extract()[C].strip().split(" of ")[0]
            )
            bout_item["sig_distance_landed_2"] = int(
                totals.xpath("//td[7]//p/text()")
                .extract()[C + 1]
                .strip()
                .split(" of ")[0]
            )
            bout_item["sig_distance_attempted_1"] = int(
                totals.xpath("//td[7]//p/text()").extract()[C].strip().split(" of ")[1]
            )
            bout_item["sig_distance_attempted_2"] = int(
                totals.xpath("//td[7]//p/text()")
                .extract()[C + 1]
                .strip()
                .split(" of ")[1]
            )

            bout_item["sig_clinch_landed_1"] = int(
                totals.xpath("//td[8]//p/text()").extract()[C].strip().split(" of ")[0]
            )
            bout_item["sig_clinch_landed_2"] = int(
                totals.xpath("//td[8]//p/text()")
                .extract()[C + 1]
                .strip()
                .split(" of ")[0]
            )
            bout_item["sig_clinch_attempted_1"] = int(
                totals.xpath("//td[8]//p/text()").extract()[C].strip().split(" of ")[1]
            )
            bout_item["sig_clinch_attempted_2"] = int(
                totals.xpath("//td[8]//p/text()")
                .extract()[C + 1]
                .strip()
                .split(" of ")[1]
            )

            bout_item["sig_ground_landed_1"] = int(
                totals.xpath("//td[9]//p/text()").extract()[C].strip().split(" of ")[0]
            )
            bout_item["sig_ground_landed_2"] = int(
                totals.xpath("//td[9]//p/text()")
                .extract()[C + 1]
                .strip()
                .split(" of ")[0]
            )
            bout_item["sig_ground_attempted_1"] = int(
                totals.xpath("//td[9]//p/text()").extract()[C].strip().split(" of ")[1]
            )
            bout_item["sig_ground_attempted_2"] = int(
                totals.xpath("//td[9]//p/text()")
                .extract()[C + 1]
                .strip()
                .split(" of ")[1]
            )
        except:
            # Do nothing, statistics do not exist
            print("round by round stats not available")

        # Replace all '' items with None
        for key in bout_item.keys():
            bout_item[key] = None if bout_item[key] == "" else bout_item[key]

        yield bout_item


if __name__ == "__main__":
    process = CrawlerProcess(
        {"USER_AGENT": "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)"}
    )

    process.crawl(Bouts)
    process.start()
