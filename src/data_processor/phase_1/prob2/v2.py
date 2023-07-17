from src.data_processor._base import BaseFeatureProcessor
import pandas as pd
from abc import ABC


class Phase1Prob2FeatureProcessor(BaseFeatureProcessor, ABC):

    def __init__(self,features=None, categorical_features=None, agg_features=None):
        super(Phase1Prob2FeatureProcessor, self).__init__(features=features, categorical_features=categorical_features, agg_features=agg_features)
        if self.FEATURES is None:
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
        if self.CATEGORICAL_FEATURES is None:
            self.CATEGORICAL_FEATURES = ["feature1",
                                        "feature3",
                                        "feature4",
                                        "feature6",
                                        "feature7",
                                        "feature9",
                                        "feature10",
                                        "feature12",
                                        "feature14",
                                        "feature15",
                                        "feature17",
                                        "feature19",
                                        "feature20"]


    def fit(self, data: pd.DataFrame):
        pass

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        self.data = data.copy()
        self.convert_to_categorical()
        out = self.data.copy()
        self.flush()
        return out[self.FEATURES]