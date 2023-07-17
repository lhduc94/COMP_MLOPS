from abc import ABC, abstractmethod
import json
import pandas as pd

class BaseFeatureProcessor(ABC):
    @classmethod
    def from_pretrained(cls,config_file):
        with open(config_file, 'r') as file:
            configs = json.load(file)
            return cls(features=configs['features'], categorical_features=configs['categorical_features'], agg_features=configs['agg_features'])

    def __init__(self, 
             features=None, 
             categorical_features=None, 
             agg_features=None):
        self.FEATURES = features
        self.CATEGORICAL_FEATURES = categorical_features
        self.agg_features = agg_features

    def flush(self):
        self.data = None
    @abstractmethod
    def fit(self, data: pd.DataFrame):
        pass

    @abstractmethod
    def transform(self, data:pd.DataFrame):
        pass

    def fit_transform(self, data:pd.DataFrame)-> pd.DataFrame:
        self.fit(data)
        return self.transform(data)
    
    def convert_to_categorical(self) -> pd.DataFrame:
        for col in self.CATEGORICAL_FEATURES:
            self.data[col] = self.data[col].astype('category')
    @property
    def data_features(self):
        return {'features': self.FEATURES,
                'categorical_features': self.CATEGORICAL_FEATURES,
                'agg_features': self.agg_features}


    def save_config(self,filename):
        with open(filename, 'w') as file:
            json.dump(self.data_features, file)