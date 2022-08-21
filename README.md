# ufc_fight_predictor

## What does the ufc fight predictor do?
As the name suggests, the ufc fight predictor predicts ufc fights.

## Running the project from scratch
The project has a lot of moving parts. For all of these parts to communicate with eachother we first need to create a docker network. To do this we run the command:
```
docker network create ufc
```
To bring up the project we need to run the folowing steps:<br />
1. Run the backend server for the data<br />
2. Either freshly scrape the data or use the data dump to quickly run it for yourself<br />
3. ...

### Backend
First we need to spin up the postgres service with the timescale pluging. To do so, navigate to the root of the backend folder and simply run:

```
docker-compose up -d --build
```
This should spin up a postgres database on your machine. For assesibility this project also comes with PgAdmin, so we can easily interact with the database through the url http://localhost:5050.

### Running the crawlers
If we want fresh data in the database we can run the crawlers. These crawlers will crawl all of the ufc fights and all of the fighters in the ufc. The crawlers are based on https://github.com/cdpierse/ufc_fight_predictor/tree/master.

Running the crawlers is equal to spinning up the backend. We navigate to the root of the crawlers folder and run: 
```
docker-compose up -d --build
```

### Skip the scraping
The scrapers will take some time to complete. To make it easier, and less time consuming, I also added csv files for both the fighters and the bouts. To populate db with these files we navigate to the root of the data folder and run: 
```
docker-compose up -d --build
```
