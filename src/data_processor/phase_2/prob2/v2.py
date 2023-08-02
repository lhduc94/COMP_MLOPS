from src.data_processor._base import BaseFeatureProcessor
import pandas as pd
from abc import ABC


class Phase2Prob2FeatureProcessor(BaseFeatureProcessor, ABC):

    def __init__(self,features=None, categorical_features=None, agg_features=None):
        super().__init__(features=features, categorical_features=categorical_features, agg_features=agg_features)
        if self.FEATURES is None:
    #         self.FEATURES = ['feature26', 'feature7', 'feature11', 'feature27', 'feature8',
    #  'feature12',  'feature25', 'feature1',
    #    'feature24',  
    #    'feature17', 'feature18', 'feature16', 'feature40', 'feature5',
    #     'feature29', 'feature9',
    #     'feature31', 'feature38',
    #     'feature28', 
    #    'feature41', 'feature36']
            self.FEATURES = ['feature26', 'feature7', 'feature25', 'feature8', 'feature11',
       'feature27', 'feature24', 'feature1', 'feature17', 'feature12']
        if self.CATEGORICAL_FEATURES is None:
            self.CATEGORICAL_FEATURES = []


    
    def fit(self, data: pd.DataFrame):
        pass

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        self.data = data.copy()
        # self.CATEGORICAL_FEATURES = self.CATEGORICAL_FEATURES + ['9_10']
        # self.fillna()
        # self.cutoff_cat()
        self.convert_to_categorical()
        # self.FEATURES = [x for x in self.FEATURES if x not in ['feature22', 'feature37']]
        out = self.data.copy()
        self.flush()
        return out[self.FEATURES]

    # def fillna(self):
    #     self.data['feature3'] = self.data['feature3'].fillna('FIN')

    