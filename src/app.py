import pandas as pd
import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel
from src.data_processor.phase_1.prob1.v1 import Phase1Prob1FeatureProcessor
from src.data_processor.phase_1.prob2.v2 import Phase1Prob2FeatureProcessor
from src.data_processor.phase_2.prob1.v13 import Phase2Prob1FeatureProcessor
from src.data_processor.phase_2.prob2.v2 import Phase2Prob2FeatureProcessor
from src.data_processor.phase_3.prob1.v1 import Phase3Prob1FeatureProcessor
from src.data_processor.phase_3.prob2.v1 import Phase3Prob2FeatureProcessor
import argparse
from src.drift_detector import drift_psi
from collections import Counter
import mlflow 
from src.utils import get_env, load_model_config
import pickle 
env = get_env()
model_config = load_model_config(env)
mlflow.set_tracking_uri(env['MLFLOW_TRACKING_URI'])
DEFAULT_CONFIG_CHECKPOINTS = 'checkpoints'

app = FastAPI()
parser = argparse.ArgumentParser()
parser.add_argument("--host",type=str, default="0.0.0.0")
parser.add_argument("--port", type=int, default=5040)
parser.add_argument("--workers", type=int, default=8)
args = parser.parse_args()
model_config = load_model_config(env)
PHASE3_PROB1 = model_config['phase3']['prob1']
PHASE3_PROB2 = model_config['phase3']['prob2']

phase3_prob1_pretrained_model = mlflow.sklearn.load_model(f"models:/{PHASE3_PROB1['model_name']}/{PHASE3_PROB1['model_version']}")
phase3_prob1_feature_processor = Phase3Prob1FeatureProcessor()
phase3_prob2_pretrained_model = mlflow.sklearn.load_model(f"models:/{PHASE3_PROB2['model_name']}/{PHASE3_PROB2['model_version']}")
phase3_prob2_feature_processor = Phase3Prob2FeatureProcessor()

with open(f'{DEFAULT_CONFIG_CHECKPOINTS}/phase-3/prob-1/varcount_v1.pkl','rb') as file:
    var_count_phase3_prob1 = pickle.load(file)
with open(f'{DEFAULT_CONFIG_CHECKPOINTS}/phase-3/prob-2/varcount_v1.pkl','rb') as file:
    var_count_phase3_prob2 = pickle.load(file)
@app.get('/')
def home():
    return "The man Team"


@app.get('/health')
def health():
    return {'status': 200}

class InputData(BaseModel):
    id: str
    rows: list
    columns: list

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

# @app.post('/phase-2/prob-1/predict')
# async def phase2_prob1(input_data: InputData, request: Request):
#     df = pd.DataFrame(input_data.rows, columns=input_data.columns)
#     data = phase2_prob1_feature_processor.transform(df)
#     prediction = phase2_prob1_pretrained_model.predict_proba(data)
#     drift = int(df['feature23'].mean() > 1.5)
#     # df.to_csv(f'test_phase1_prob1/mlops_phase1_prob1_{data.id}.csv', index=False)
#     return {'id': input_data.id, 'predictions': prediction, 'drift': drift}


# @app.post('/phase-2/prob-2/predict')
# async def phase1_prob2(input_data: InputData, request: Request):
#     df = pd.DataFrame(input_data.rows, columns=input_data.columns)
#     data = phase2_prob2_feature_processor.transform(df)
#     prediction = phase2_prob2_pretrained_model.predict_proba(data)
#     drift = int(df['feature23'].mean() > 1.5)
#     # df.to_csv(f'test_phase1_prob2/mlops_phase1_prob2_{data.id}.csv', index=False)
#     return {'id': input_data.id, 'predictions': prediction, 'drift': drift}



@app.post('/phase-3/prob-1/predict')
async def phase3_prob1(input_data: InputData, request: Request):
    df = pd.DataFrame(input_data.rows, columns=input_data.columns)
    data = phase3_prob1_feature_processor.transform(df)
    prediction = phase3_prob1_pretrained_model.predict_proba(data)[:,1].tolist()
    # drift = int(df['feature6'].std() > 200)
    drift = drift_psi(var_count_phase3_prob1, var_test=Counter(df['feature2']))

    # df.to_csv(f'test_phase1_prob1/mlops_phase1_prob1_{data.id}.csv', index=False)
    # drift = 0
    return {'id': input_data.id, 'predictions': prediction, 'drift': drift}


@app.post('/phase-3/prob-2/predict')
async def phase1_prob2(input_data: InputData, request: Request):
    df = pd.DataFrame(input_data.rows, columns=input_data.columns)
    data = phase3_prob2_feature_processor.transform(df)
    prediction = phase3_prob2_pretrained_model.predict(data)[:,0].tolist()
    # drift = int(df['feature6'].std() > 200)
    drift = drift_psi(var_count_phase3_prob2, var_test=Counter(df['feature2']))
    # df.to_csv(f'test_phase1_prob2/mlops_phase1_prob2_{data.id}.csv', index=False)
    # drift = 0
    return {'id': input_data.id, 'predictions': prediction, 'drift': drift}

if __name__ == '__main__':
    print('aaa')
    uvicorn.run("src.app:app", host=args.host, port=args.port, workers=args.workers, reload=False)