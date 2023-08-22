docker build -f deployment/mlflow/Dockerfile -t mlflow_server .
docker-compose -f deployment/mlflow/docker-compose.yml up -d