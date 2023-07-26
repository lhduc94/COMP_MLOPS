from src.data_processor._base import BaseFeatureProcessor
import pandas as pd
from abc import ABC


class Phase2Prob2FeatureProcessor(BaseFeatureProcessor, ABC):

    def __init__(self,features=None, categorical_features=None, agg_features=None):
        super().__init__(features=features, categorical_features=categorical_features, agg_features=agg_features)
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

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        self.data = data.copy()
        self.data['9_10'] = self.data['feature9'].astype('int').astype('str') +'.' + self.data['feature10'].astype('int').astype('str')
        self.data['7+26'] = self.data['feature7']/3 + self.data['feature26']
        self.data['7-26'] = self.data['feature7']/3 - self.data['feature26']

        if '9_10' not in self.FEATURES:
            self.FEATURES = self.FEATURES + ['9_10', '7+26','7-26']
        if '9_10' not in self.CATEGORICAL_FEATURES:
            self.CATEGORICAL_FEATURES = self.CATEGORICAL_FEATURES + ['9_10']
        # self.CATEGORICAL_FEATURES = self.CATEGORICAL_FEATURES + ['9_10']
        # self.fillna()
        # self.cutoff_cat()
        self.convert_to_categorical()
        self.FEATURES = [x for x in self.FEATURES if x not in ['feature22', 'feature37']]
        out = self.data.copy()
        self.flush()
        return out[self.FEATURES]

    # def fillna(self):
    #     self.data['feature3'] = self.data['feature3'].fillna('FIN')

    def cutoff_cat(self):
        top_cat = ['tcp',
    'udp',
    'gre',
    'unas',
    'emcon',
    'arp',
    'ospf',
    'mtp',
    'any',
    'sctp',
    'tcf',
    'ip',
    'sun-nd',
    'visa',
    'mobile',
    'pim',
    'ipv6',
    'rsvp',
    'sat-expak',
    'pipe',
    'sep',
    'swipe',
    'narp']
        self.data['feature2'] = self.data['feature2'].apply(lambda x: 'Others' if x not in top_cat else x)