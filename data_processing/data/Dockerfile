# pull official base image
FROM python:3.9.6-slim 

# Relevant files to the app folder
COPY ./data_processing/ /app/data_processing/

# set work directory
WORKDIR /app

# Add requirements.txt to the container
RUN pip install --no-cache-dir -r data_processing/data/requirements_add_data.txt

# Set python path to be able to use utils module
ENV PYTHONPATH "${PYTHONPATH}:/app/data_processing/utils"

# Set workdir to data module
WORKDIR /app/data_processing/data
CMD [ "python3", "add_data.py"]