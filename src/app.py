import pandas as pd
import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel
from src.data_processor.phase_1.prob1.v1 import Phase1Prob1FeatureProcessor
from src.data_processor.phase_1.prob2.v2 import Phase1Prob2FeatureProcessor
from src.data_processor.phase_2.prob1.v1 import Phase2Prob1FeatureProcessor
from src.data_processor.phase_2.prob2.v1 import Phase2Prob2FeatureProcessor
from src.model_predictor import \
    (Phase1Prob1ModelPredictor, Phase1Prob2ModelPredictor, Phase2Prob1ModelPredictor, Phase2Prob2ModelPredictor)
import argparse

app = FastAPI()
DEFAULT_CONFIG_CHECKPOINTS = '././checkpoints'
parser = argparse.ArgumentParser()
parser.add_argument("--check-points", type=str, default=DEFAULT_CONFIG_CHECKPOINTS)
parser.add_argument("--port", type=int, default=5040)
parser.add_argument("--workers", type=int, default=8)
args = parser.parse_args()


class InputData(BaseModel):
    id: str
    rows: list
    columns: list


# phase1_prob1_pretrained_model = Phase1Prob1ModelPredictor.from_pretrained(args.check_points + '/phase-1/prob-1/v1.pkl')
# phase1_prob1_feature_processor = Phase1Prob1FeatureProcessor()
# phase1_prob2_pretrained_model = Phase1Prob2ModelPredictor.from_pretrained(args.check_points + '/phase-1/prob-2/catboost_v3.pkl')
# phase1_prob2_feature_processor = Phase1Prob2FeatureProcessor()

phase2_prob1_pretrained_model = Phase2Prob1ModelPredictor.from_pretrained(args.check_points + '/phase-2/prob-1/v1.pkl')
phase2_prob1_feature_processor = Phase2Prob1FeatureProcessor()
phase2_prob2_pretrained_model = Phase2Prob2ModelPredictor.from_pretrained(args.check_points + '/phase-2/prob-2/v1.pkl')
phase2_prob2_feature_processor = Phase2Prob2FeatureProcessor()
@app.get('/')
def home():
    return "The man Team"


@app.get('/health')
def health():
    return {'status': 200}


# @app.post('/phase-1/prob-1/predict')
# def phase1_prob1(input_data: InputData, request: Request):
#     df = pd.DataFrame(input_data.rows, columns=input_data.columns)
#     data = phase1_prob1_feature_processor.transform(df)
#     prediction = phase1_prob1_pretrained_model.predict_proba(data)
#     # df.to_csv(f'test_phase1_prob1/mlops_phase1_prob1_{data.id}.csv', index=False)
#     return {'id': input_data.id, 'predictions': prediction, 'drift': 0}
#
#
# @app.post('/phase-1/prob-2/predict')
# def phase1_prob2(input_data: InputData, request: Request):
#     df = pd.DataFrame(input_data.rows, columns=input_data.columns)
#     data = phase1_prob2_feature_processor.transform(df)
#     prediction = phase1_prob2_pretrained_model.predict_proba(data)
#     # df.to_csv(f'test_phase1_prob2/mlops_phase1_prob2_{data.id}.csv', index=False)
#     return {'id': input_data.id, 'predictions': prediction, 'drift': 0}

@app.post('/phase-2/prob-1/predict')
async def phase2_prob1(input_data: InputData, request: Request):
    df = pd.DataFrame(input_data.rows, columns=input_data.columns)
    data = phase2_prob1_feature_processor.transform(df)
    prediction = phase2_prob1_pretrained_model.predict_proba(data)
    # drift = int(df['feature28'].std() > 0.4)
    drift = 0
    # df.to_csv(f'test_phase1_prob1/mlops_phase1_prob1_{data.id}.csv', index=False)
    return {'id': input_data.id, 'predictions': prediction, 'drift': drift}


@app.post('/phase-2/prob-2/predict')
async def phase1_prob2(input_data: InputData, request: Request):
    df = pd.DataFrame(input_data.rows, columns=input_data.columns)
    data = phase2_prob2_feature_processor.transform(df)
    prediction = phase2_prob2_pretrained_model.predict_proba(data)
    drift = int(df['feature28'].std() > 0.4)
    # drift = 0
    # df.to_csv(f'test_phase1_prob2/mlops_phase1_prob2_{data.id}.csv', index=False)
    return {'id': input_data.id, 'predictions': prediction, 'drift': drift}
if __name__ == '__main__':
    uvicorn.run("src.app:app", host="0.0.0.0", port=args.port, workers=args.workers, reload=False)
