set IMAGE_NAME=model_predictor
set IMAGE_TAG=v1
set PORT=5040
set MODEL_CONFIG_PATH=configs/model_config.yml
docker build -f deployment/model_predictor/Dockerfile -t %IMAGE_NAME%:%IMAGE_TAG% .
docker-compose -f deployment/model_predictor/docker-compose.yml up -d