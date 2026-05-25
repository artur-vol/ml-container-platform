FROM python:3.11

RUN apt-get update \
 && apt-get install -y gcc g++

COPY requirements-serve-cpu.txt /tmp/requirements-serve-cpu.txt
RUN pip install -r /tmp/requirements-serve-cpu.txt

COPY src/serve.py /app/src/serve.py

WORKDIR /app
ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
    MODEL_NAME=cifar10-mobilenet \
    MODEL_STAGE=Production

EXPOSE 8000
ENTRYPOINT ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
