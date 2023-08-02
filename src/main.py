import pandas as pd
import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel
from src.data_processor.phase_1.prob1.v1 import Phase1Prob1FeatureProcessor
from src.data_processor.phase_1.prob2.v2 import Phase1Prob2FeatureProcessor
from src.data_processor.phase_2.prob1.v13 import Phase2Prob1FeatureProcessor
from src.data_processor.phase_2.prob2.v1 import Phase2Prob2FeatureProcessor
from src.data_processor._base import BaseFeatureProcessor
from src.model_predictor import \
    (Phase1Prob1ModelPredictor, Phase1Prob2ModelPredictor, Phase2Prob1ModelPredictor, Phase2Prob2ModelPredictor)
import argparse
import pickle
from src.drift_detector import drift_psi
from collections import Counter

class InputData(BaseModel):
    id: str
    rows: list
    columns: list
DEFAULT_CONFIG_CHECKPOINTS = '././checkpoints'



class PredictorApi:
    def __init__(self, phase2_prob1_pretrained_model: Phase2Prob1ModelPredictor, 
                    phase2_prob2_pretrained_model: Phase2Prob2ModelPredictor,
                    phase2_prob1_feature_processor:BaseFeatureProcessor,
                    phase2_prob2_feature_processor: BaseFeatureProcessor):
        self.phase2_prob1_pretrained_model = phase2_prob1_pretrained_model
        self.phase2_prob2_pretrained_model = phase2_prob2_pretrained_model
        self.phase2_prob1_feature_processor = phase2_prob1_feature_processor
        self.phase2_prob2_feature_processor = phase2_prob2_feature_processor
        self.app = FastAPI()

        @self.app.get("/")
        async def root():
            return {"message": "hello"}

        @self.app.post("/phase-2/prob-1/predict")
        async def phase2_prob1(input_data: InputData, request: Request):
            df = pd.DataFrame(input_data.rows, columns=input_data.columns)
            data = phase2_prob1_feature_processor.transform(df)
            prediction = phase2_prob1_pretrained_model.predict_proba(data)
            # drift = int(df['feature28'].std() > 0.4)
            var_count_test_1 = Counter(df['feature3'])
            drift = drift_psi(var_count_train_1, var_count_test_1)
            # drift = 0
            # df.to_csv(f'test_phase1_prob1/mlops_phase1_prob1_{data.id}.csv', index=False)
            return {'id': input_data.id, 'predictions': prediction, 'drift': drift}
        @self.app.post("/phase-2/prob-2/predict")
        async def phase2_prob2(input_data: InputData, request: Request):
            df = pd.DataFrame(input_data.rows, columns=input_data.columns)
            data = phase2_prob2_feature_processor.transform(df)
            prediction = phase2_prob2_pretrained_model.predict_proba(data)
            drift = int(df['feature28'].std() > 0.4)
            var_count_test_1 = Counter(df['feature3'])
            # drift = 0
            # df.to_csv(f'test_phase1_prob1/mlops_phase1_prob1_{data.id}.csv', index=False)
            return {'id': input_data.id, 'predictions': prediction, 'drift': drift}

    @staticmethod
    def _log_request(request: Request):
        pass

    @staticmethod
    def _log_response(response: dict):
        pass




class MyAPI:
    def __init__(self):
        self.app = FastAPI()

    def configure_routes(self):
        @self.app.get("/")
        def read_root():
            return {"Hello": "World"}
    def run(self):
        uvicorn.run(self.app)
if __name__ == "__main__":


   
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-points", type=str, default=DEFAULT_CONFIG_CHECKPOINTS)
    parser.add_argument("--host",type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5040)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    var_count_train_1 = pickle.load(open(args.check_points + '/phase-2/prob-1/var_count.pkl','rb'))
    phase2_prob1_pretrained_model = Phase2Prob1ModelPredictor.from_pretrained(args.check_points + '/phase-2/prob-1/v1.pkl')
    phase2_prob1_feature_processor = Phase2Prob1FeatureProcessor()
    phase2_prob2_pretrained_model = Phase2Prob2ModelPredictor.from_pretrained(args.check_points + '/phase-2/prob-2/v1.pkl')
    phase2_prob2_feature_processor = Phase2Prob2FeatureProcessor()
    api = PredictorApi(phase2_prob1_pretrained_model,
                       phase2_prob2_pretrained_model,
                       phase2_prob1_feature_processor,
                       phase2_prob2_feature_processor)
    uvicorn.run("src:api:app", host=args.host, port=args.port, workers=args.workers, reload=False)
    
    