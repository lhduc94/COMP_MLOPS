# run api
run_app:
	python src/app.py
train:
	python src/model_trainer.py
# run load test
load_test:
	locust -f tests/performance_tests/locust_test.py

predictor_restart:
	set PORT=5040
	set IMAGE_NAME=phase2
	set IMAGE_TAG=v1
	set MODEL_CONFIG_PATH=checkpoints
	docker-compose -f deployment/model_predictor/docker-compose.yml up -d start