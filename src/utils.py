import os

def get_env():
    env = {
            'MLFLOW_TRACKING_URL': os.getenv('MLFLOW_TRACKING_URL'),
            'MODEL_CONFIG_PATH': os.getenv('MODEL_CONFIG_PATH')}
    return env