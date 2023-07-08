from src.data_processor._base import BaseFeatureProcessor
import pandas as pd
from abc import ABC


class Phase1Prob2FeatureProcessor(BaseFeatureProcessor, ABC):

    def __init__(self):
        super(Phase1Prob2FeatureProcessor, self).__init__()
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
                         "feature15",
                         "feature16",
                         "feature17",
                         "feature18",
                         "feature19",
                         "feature20"]
        self.CATEGORICAL_FEATURES = ["feature1",
                                     "feature3",
                                     "feature4",
                                     "feature6",
                                     "feature7",
                                     "feature8",
                                     "feature9",
                                     "feature10",
                                     "feature11",
                                     "feature12",
                                     "feature14",
                                     "feature15",
                                     "feature16",
                                     "feature17",
                                     "feature19",
                                     "feature20"]


    def fit(self, data: pd.DataFrame):
        pass

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        df = self.convert_to_categorical(data)
        return df[self.FEATURES]

    def fit_transform(self, data:pd.DataFrame)-> pd.DataFrame:
        self.fit(data)
        return self.transform(data)

    def convert_to_categorical(self, data) -> pd.DataFrame:
        df = data.copy()
        for col in self.CATEGORICAL_FEATURES:
            print(col)
            try:
                print(col)
                df[col] = df[col].astype(int)
            except:
                df[col] = df[col].astype('category')
        return df
