# pull official base image
FROM python:3.9.6-slim

# set work directory
WORKDIR /usr/src/app

# install dependencies
RUN pip install --upgrade pip
COPY ./requirements.txt .
RUN pip install -r requirements.txt

# copy project
COPY . .

CMD [ "python3", "./add_data.py" ]