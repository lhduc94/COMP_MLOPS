# run api
run_app:
	python src/app.py
train:
	python src/model_trainer.py
# run load test
load_test:
	locust -f tests/performance_tests/locust_test.py
