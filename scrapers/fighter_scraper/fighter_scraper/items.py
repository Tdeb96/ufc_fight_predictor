from scrapy.item import Field, Item


class FightScraperItem(Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    fighter_url = Field()
    fighter_name = Field()
    fighter_record = Field()
    height = Field()
    weight = Field()
    reach = Field()
    stance = Field()
    date_of_birth = Field()
    slpm = Field()  # strikes landed per min stat
    td_avg = Field()  # takedown average
    strike_acc = Field()  # striking accuracy
    td_acc = Field()  # takedown accuracy
    sapm = Field()  # strikes absorbed per minute
    td_def = Field()  # takedown defence
    strike_def = Field()  # striking defence
    sub_avg = Field()  # submission average
