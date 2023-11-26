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

## Running the crawlers
If we want fresh data in the database we can run the crawlers. The crawlers will crawl all fighters not in the db already, and all the fights after the last scraped date (last data updated in Oct 2023). The crawlers are based on https://github.com/cdpierse/ufc_fight_predictor/tree/master.

We navigate to the root of the scrapers folder and run: 
```
docker compose up -d --build
```

## Running the notebooks
The notebooks can all be run using the poetry environment. Run:
```
poetry install && poetry shell
```
To initialize the virtual environment.

## Odds
The odds requires a .env file in the odds folder with a valid API_KEY for the-odds-api.com