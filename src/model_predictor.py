import pickle
from abc import ABC, abstractmethod


class BaseModelPredictor(ABC):
    def __init__(self, pretrained_model):
        self.model = pretrained_model

    @classmethod
    def from_pretrained(cls, model_path):
        with open(model_path, 'rb') as file:
            pretrained_model = pickle.load(file)
            return cls(pretrained_model)
    @abstractmethod
    def predict_proba(self, data):
        pass

class Phase1Prob1ModelPredictor(BaseModelPredictor):
    def __init__(self, model):
        super(Phase1Prob1ModelPredictor, self).__init__(model)

    def predict_proba(self, data):
        return self.model.predict_proba(data)[:, 1].tolist()


class Phase1Prob2ModelPredictor(BaseModelPredictor):
    def __init__(self, model):
        super(Phase1Prob2ModelPredictor, self).__init__(model)

    def predict_proba(self, data):
        return self.model.predict_proba(data)[:, 1].tolist()


class Phase2Prob1ModelPredictor(BaseModelPredictor):
    def __init__(self, model):
        super(Phase2Prob1ModelPredictor, self).__init__(model)

    def predict_proba(self, data):
        return self.model.predict_proba(data)[:, 1].tolist()


class Phase2Prob2ModelPredictor(BaseModelPredictor):
    def __init__(self, model):
        super(Phase2Prob2ModelPredictor, self).__init__(model)

    def predict_proba(self, data):
        return self.model.predict(data).tolist()
    
class Phase3Prob1ModelPredictor(BaseModelPredictor):
    def __init__(self, model):
        super(Phase3Prob1ModelPredictor, self).__init__(model)

    def predict_proba(self, data):
        return self.model.predict_proba(data)[:, 1].tolist()


class Phase3Prob2ModelPredictor(BaseModelPredictor):
    def __init__(self, model):
        super(Phase3Prob2ModelPredictor, self).__init__(model)

    def predict_proba(self, data):
        return self.model.predict(data).tolist()