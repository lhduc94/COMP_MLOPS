from abc import ABC
from src.data_processor._base import BaseFeatureProcessor
import pandas as pd


class Phase1Prob1FeatureProcessor(BaseFeatureProcessor, ABC):
    def __init__(self):
        super(Phase1Prob1FeatureProcessor, self).__init__()
        self.FEATURES = ["feature1",
                         "feature2",
                         "feature3",
                         "feature4",
                         "feature5",
                         "feature6",
                         "feature7",
                         "feature8",
                         "feature9",
                         "feature10",
                         "feature11",
                         "feature12",
                         "feature13",
                         "feature14",
                         "feature16"]
        self.CATEGORICAL_FEATURES = ["feature1", "feature2"]

    def fit(self, data: pd.DataFrame):
        pass

    def transform(self, data:pd.DataFrame) -> pd.DataFrame:
        # self.FEATURES = [x for x in data.columns if 'feature' in x not in 'feature15']
        df = self.convert_to_categorical(data)
        return df[self.FEATURES]

    def fit_transform(self, data:pd.DataFrame) -> pd.DataFrame:
        # self.fit(data)
        return self.transform(data)

    def convert_to_categorical(self, data: pd.DataFrame) -> pd.DataFrame:
        # df = data.copy()
        for col in self.CATEGORICAL_FEATURES:
            data[col] = data[col].astype('category')
        return data
