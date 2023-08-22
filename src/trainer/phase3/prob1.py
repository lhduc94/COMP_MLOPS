
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, f1_score, accuracy_score, multilabel_confusion_matrix, ConfusionMatrixDisplay, confusion_matrix
from catboost import CatBoostClassifier
from src.data_processor.phase_3.prob1.v1 import  Phase3Prob1FeatureProcessor
from src.data_processor._base import BaseFeatureProcessor
import numpy as np
import gc
from collections import Counter
import mlflow
import mlflow.catboost
from mlflow.models.signature import infer_signature
import argparse
from src.utils import get_env
env = get_env()
mlflow.set_tracking_uri(env['MLFLOW_TRACKING_URL'])
HyperParameters={
                    'iterations':1500, 
                    'learning_rate':0.7,
                    'max_depth':8, 
                    'random_state':42}

parser = argparse.ArgumentParser()
parser.add_argument("--train-data",type=str,required=True)
parser.add_argument("--test-tiny",type=str,default=None)
# parser.add_argument("--model-path",type=str,required=True)
# parser.add_argument("--processor-path", type=str,required=True)
parser.add_argument("--iterations", type=int, default=HyperParameters['iterations'])
parser.add_argument("--learning_rate", type=float, default=HyperParameters['learning_rate'])
parser.add_argument("--max_depth", type=int, default=HyperParameters['max_depth'])
parser.add_argument("--random_state", type=int, default=HyperParameters['random_state'])
parser.add_argument("--sampling", type=float, default=None)

args = parser.parse_args()

hyper_parameters = HyperParameters.copy()
hyper_parameters['iterations'] = args.iterations
hyper_parameters['learning_rate'] = args.learning_rate
hyper_parameters['max_depth'] = args.max_depth
hyper_parameters['random_state'] = args.random_state

def undersampling(df:pd.DataFrame, TARGET=None, downsample=0.5):
    _X_train_positive = df[df[TARGET]==1]
    _X_train_negative = df[df[TARGET]==0]
    _X_train_negative = _X_train_negative.sample(frac=0.1, random_state=42)
    _X_train = pd.concat([_X_train_negative, _X_train_positive], axis=0)
    _X_train = _X_train.sample(frac=downsample, random_state=123)
    return _X_train
# def sampling(X)
def cross_validate(hyper_parameters:HyperParameters, df:pd.DataFrame, processor:BaseFeatureProcessor, TARGET=None, sampling=None):
    kfold = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    models = []
    scores = []
    processors = []
    oofs = np.zeros(df.shape[0])
    for i, (train_idx, valid_idx) in enumerate(kfold.split(df, df[TARGET])):
        print(f"Fold {i}")
        _X_train = df.iloc[train_idx]
        if sampling is not None and sampling < 1:
            _X_train = undersampling(_X_train, TARGET, sampling)
        _X_valid = df.iloc[valid_idx]
        processor.fit(_X_train)
        X_train = processor.transform(_X_train)
        X_valid = processor.transform(_X_valid)
        y_train = df.iloc[train_idx][TARGET]
        y_train = _X_train[TARGET]
        y_valid = _X_valid[TARGET]

        model = CatBoostClassifier(**hyper_parameters,eval_metric='Accuracy', thread_count=-1)
        model.fit(X_train, y_train,
              cat_features=processor.data_features['categorical_features'],
              eval_set=[(X_train, y_train), (X_valid, y_valid)],
              verbose=50,
              )
        y_pred_proba = model.predict_proba(X_valid)[:,1]
        print(f"fold {i} : {roc_auc_score(y_valid, y_pred_proba)}")
        oofs[valid_idx] = y_pred_proba
        models.append(model)
        scores.append(roc_auc_score(y_valid, y_pred_proba))
        processors.append(processor)
    return models, scores, oofs, processors

def train_all(hyper_parameters:HyperParameters, df:pd.DataFrame, processor:BaseFeatureProcessor, TARGET=None,sampling=None):
    gmodel = CatBoostClassifier(**hyper_parameters, eval_metric='Accuracy', thread_count=-1)
    if sampling is not None and sampling < 1:
        df = undersampling(df, TARGET, downsample=sampling)
    processor.fit(df)
    new_df = processor.transform(df)
    gmodel.fit(new_df, df[TARGET],
           eval_set=[(new_df,df[TARGET])],
           cat_features=processor.data_features['categorical_features'],
           verbose=50)
    return gmodel, processor

if __name__ == '__main__':
    mlflow.set_experiment("phase3_prob1")
    print("Read file")
    df = pd.read_csv(args.train_data)
    # df = reduce_mem_usage(df)
    gc.collect()
    print("Cross validation")

    mlflow.log_params(hyper_parameters)
    mlflow.log_param('sampling',args.sampling)
    processor = Phase3Prob1FeatureProcessor()
    _, _, oofs, _ = cross_validate(hyper_parameters, df, processor, 'label', sampling=args.sampling)
    mlflow.log_metric('oof_AUC_score', roc_auc_score(df['label'], oofs))
    g_model, g_processor = train_all(hyper_parameters, df, processor, 'label')
    mlflow.sklearn.log_model(g_model, artifact_path='model', registered_model_name='phase3_prob1')
    mlflow.log_dict(g_processor.data_features, 'processor.json')
    var_count = Counter(df['feature2'])
    mlflow.log_dict(var_count, 'feature2_var_count.json')

    mlflow.end_run()