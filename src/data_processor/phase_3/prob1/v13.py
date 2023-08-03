from abc import ABC
from src.data_processor._base import BaseFeatureProcessor
import pandas as pd


class Phase2Prob1FeatureProcessor(BaseFeatureProcessor):
    def __init__(self, features=None, categorical_features=None, agg_features=None):
        super().__init__(features=features, 
                         categorical_features=categorical_features)
        if self.FEATURES is None:
            self.FEATURES = ['feature7', 'feature35', 'feature26', 'feature40', 'feature27',
       'feature11', 'feature29', 'feature9', 'feature15', 'feature23']
        if self.CATEGORICAL_FEATURES is None:
            self.CATEGORICAL_FEATURES = []

    def fit(self, data: pd.DataFrame):
        pass

    def transform(self, data:pd.DataFrame)-> pd.DataFrame:
        self.data = data.copy()
        self.convert_to_categorical()
        out = self.data.copy()
        self.flush()
        return out[self.FEATURES]
