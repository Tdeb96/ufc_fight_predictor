# Use the official Python image as the base image
FROM python:3.8

# Set the working directory in the Docker container
WORKDIR /app

# Copy the requirements.txt file from the root into the current directory in the container
COPY ../requirements_scrapers.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements_scrapers.txt

# Copy the fighter_scraper directory into the container
COPY fighter_scraper/ fighter_scraper/

# Set the PYTHONPATH to include the fighter_scraper directory
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Command to run the spider
CMD ["python", "fighter_scraper/go-spider.py"]
