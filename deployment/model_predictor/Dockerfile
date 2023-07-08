FROM python:3.9-slim

RUN apt-get update

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY deployment/model_predictor/requirements.txt .
RUN pip install --upgrade wheel setuptools pip
RUN pip install -r requirements.txt
RUN apt-get install libgomp1
COPY ./src /sample_solution/src
# COPY ./checkpoints /sample_solution/checkpoints
WORKDIR /sample_solution