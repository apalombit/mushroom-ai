FROM python:3.11-slim

RUN pip install --no-cache-dir mlflow==2.17.0

WORKDIR /mlflow
RUN mkdir -p /mlflow/data /mlflow/artifacts

EXPOSE 5000

CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "sqlite:////mlflow/data/mlflow.db", \
     "--default-artifact-root", "/mlflow/artifacts"]
