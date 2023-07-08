from src.data_processor._base import BaseFeatureProcessor
import pandas as pd
from abc import ABC


class Phase2Prob2FeatureProcessor(BaseFeatureProcessor, ABC):

    def __init__(self):
        super(Phase2Prob2FeatureProcessor, self).__init__()
        self.FEATURES = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6',
                         'feature7', 'feature8', 'feature9', 'feature10', 'feature11',
                         'feature12', 'feature13', 'feature14', 'feature15', 'feature16',
                         'feature17', 'feature18', 'feature19', 'feature20', 'feature21',
                         'feature22', 'feature23', 'feature24', 'feature25', 'feature26',
                         'feature27', 'feature28', 'feature29', 'feature30', 'feature31',
                         'feature32', 'feature33', 'feature34', 'feature35', 'feature36',
                         'feature37', 'feature38', 'feature39', 'feature40', 'feature41']
        self.CATEGORICAL_FEATURES = ["feature2",
        "feature3",
        "feature4"]


    def fit(self, data: pd.DataFrame):
        pass

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df['9_10'] = df['feature9'].astype('int').astype('str') +'.' + df['feature10'].astype('int').astype('str')
        df['7+26'] = df['feature7']/3 + df['feature26']
        # df['7-26'] = df['feature7']/3 - df['feature26']

        self.FEATURES = self.FEATURES + ['9_10','7+26']
        self.CATEGORICAL_FEATURES = self.CATEGORICAL_FEATURES + ['9_10']
        df = self.convert_to_categorical(df)
        self.FEATURES = [x for x in self.FEATURES if x not in ['feature22', 'feature37']]
        return df[self.FEATURES]

    def fit_transform(self, data:pd.DataFrame)-> pd.DataFrame:
        self.fit(data)
        return self.transform(data)

    def convert_to_categorical(self, data) -> pd.DataFrame:
        df = data.copy()
        for col in self.CATEGORICAL_FEATURES:
            try:
                df[col] = df[col].astype(int)
            except:
                df[col] = df[col].astype('category')
        return df
