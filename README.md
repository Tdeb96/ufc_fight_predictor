# ufc_fight_predictor

## What does the ufc fight predictor do?
As the name suggests, the ufc fight predictor predicts ufc fights.

## Running the project
First need to create a docker network. To do this we run the command:
```
docker network create ufc
```
Afterwards, from the root of the folder, we can run
```
docker compose up -d --build
```
To bring up the postgres server and to populate the tables with pre-scraped data.

### Running the crawlers
If we want fresh data in the database we can run the crawlers. These crawlers will crawl all of the ufc fights and all of the fighters in the ufc. The crawlers are based on https://github.com/cdpierse/ufc_fight_predictor/tree/master.

We navigate to the root of the crawlers folder and run: 
```
docker compose up -d --build
```