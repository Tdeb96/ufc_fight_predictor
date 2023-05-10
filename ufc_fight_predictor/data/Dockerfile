# pull official base image
FROM python:3.9.6-slim 

# Relevant files to the app folder
COPY ./ufc_fight_predictor/ /app/ufc_fight_predictor/
COPY poetry.lock pyproject.toml README.md /app/

# Install poetry
RUN pip install poetry

# set work directory
WORKDIR /app

# Install packages and dependencies
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi

# Set python path to be able to use utils module
ENV PYTHONPATH "${PYTHONPATH}:/app/ufc_fight_predictor/utils"

# Set workdir to data module
WORKDIR /app/ufc_fight_predictor/data
CMD [ "python3", "add_data.py"]