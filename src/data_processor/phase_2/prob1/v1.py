from abc import ABC
from src.data_processor._base import BaseFeatureProcessor
import pandas as pd


class Phase2Prob1FeatureProcessor(BaseFeatureProcessor):
    def __init__(self, features=None, categorical_features=None, agg_features=None):
        super().__init__(features=features, 
                         categorical_features=categorical_features)
        if self.FEATURES is None:
            self.FEATURES = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6',
                         'feature7', 'feature8', 'feature9', 'feature10', 'feature11',
                         'feature12', 'feature13', 'feature14', 'feature15', 'feature16',
                         'feature17', 'feature18', 'feature19', 'feature20', 'feature21',
                         'feature22', 'feature23', 'feature24', 'feature25', 'feature26',
                         'feature27', 'feature28', 'feature29', 'feature30', 'feature31',
                         'feature32', 'feature33', 'feature34', 'feature35', 'feature36',
                         'feature37', 'feature38', 'feature39', 'feature40', 'feature41']
        if self.CATEGORICAL_FEATURES is None:
            self.CATEGORICAL_FEATURES = ["feature2",
                                     "feature3",
                                     "feature4"]

    def fit(self, data: pd.DataFrame):
        pass

    def transform(self, data:pd.DataFrame)-> pd.DataFrame:
        self.data = data.copy()
        self.convert_to_categorical()
        out = self.data.copy()
        self.flush()
        return out[self.FEATURES]
