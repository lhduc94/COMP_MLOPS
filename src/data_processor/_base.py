import pandas as pd
from abc import ABC, abstractmethod
import json


class BaseFeatureProcessor(ABC):
    @staticmethod
    def load_config(config_file):
        with open(config_file, 'r') as file:
            return json.load(file)

    def __init__(self):
        self.FEATURES = None
        self.CATEGORICAL_FEATURES = None

    @abstractmethod
    def fit(self, data: pd.DataFrame):
        pass

    @abstractmethod
    def transform(self, data:pd.DataFrame):
        pass

    @abstractmethod
    def fit_transform(self, data:pd.DataFrame):
        pass
    @property
    def data_features(self):
        return {'features': self.FEATURES,
                'categorical_features': self.CATEGORICAL_FEATURES}