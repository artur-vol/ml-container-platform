FROM python:3.11

RUN apt-get update \
 && apt-get install -y gcc g++

COPY requirements-train.txt /tmp/requirements-train.txt
RUN pip install -r /tmp/requirements-train.txt

COPY src/train.py /app/src/train.py

WORKDIR /app
ENV CIFAR_DATA_DIR=/app/data \
    MLFLOW_TRACKING_URI=http://host.docker.internal:5000

ENTRYPOINT ["python", "/app/src/train.py"]
