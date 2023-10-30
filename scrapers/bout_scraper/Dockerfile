# Use the official Python image as the base image
FROM python:3.8

# Set the working directory in the Docker container
WORKDIR /app

# Copy the requirements.txt file from the root into the current directory in the container
COPY ../requirements_scrapers.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements_scrapers.txt

# Copy the bout_scraper directory into the container
COPY bout_scraper/ bout_scraper/

# Set the PYTHONPATH to include the bout_scraper directory
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Command to run the spider
CMD ["python", "bout_scraper/go-spider.py"]
