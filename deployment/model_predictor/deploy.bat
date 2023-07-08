docker build -f deployment/model_predictor/Dockerfile -t phase2:v1  .
set IMAGE_NAME phase2
set IMAGE_TAG v1
set PORT  5050
docker-compose -f deployment/model_predictor/docker-compose.yml up -d