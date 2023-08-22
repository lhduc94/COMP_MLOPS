import os
import yaml
from yaml.loader import SafeLoader
def get_env():
    env = {
            'MLFLOW_TRACKING_URI': os.getenv('MLFLOW_TRACKING_URI'),
            'MODEL_CONFIG_PATH': os.getenv('MODEL_CONFIG_PATH')}
    return env

def load_model_config(env):
    with open(env['MODEL_CONFIG_PATH']) as f:
        model_config = yaml.load(f, Loader=SafeLoader)
    return model_config