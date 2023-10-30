# Use an official Python runtime as a parent image
FROM python:3.8

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY ./api /app/api
COPY ./data_processing /app/data_processing

# Install any needed packages specified in requirements_api.txt
RUN pip install --no-cache-dir -r /app/api/requirements_api.txt

# Set python path to be able to use utils module
ENV PYTHONPATH "${PYTHONPATH}:/app/data_processing"

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run main.py when the container launches
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
