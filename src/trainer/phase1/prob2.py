import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score
from lightgbm import LGBMClassifier
from src.data_processor.phase_1.prob2.v2 import Phase1Prob2FeatureProcessor
from catboost import CatBoostClassifier
import argparse
from collections import Counter

HyperParameters={'n_estimators':100,
                 'learning_rate':0.05,
                 'max_depth':7,
                 'colsample_bytree':0.8,
                 'subsample':0.8,
                 'scale_pos_weight':1.5,
                 'random_state':42}
# HyperParameters = {'boosting_type': 'gbdt',
#                    'class_weight': None,
#                    'colsample_bytree': 1.0,
#                    'importance_type': 'split',
#                    'learning_rate': 0.1,
#                    'max_depth': 7,
#                    'min_child_samples': 20,
#                    'min_child_weight': 0.001,
#                    'min_split_gain': 0.0,
#                    'n_estimators': 100,
#                    'n_jobs': -1,
#                    'num_leaves': 31,
#                    'objective': None,
#                    'random_state': 123,
#                    'reg_alpha': 0.0,
#                    'reg_lambda': 0.0,
#                    'silent': 'warn',
#                    'subsample': 1.0,
#                    'subsample_for_bin': 200000,
#                    'subsample_freq': 0,
#                    'scale_pos_weight':1.5,
#                    }
parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int, default=HyperParameters['n_estimators'])
parser.add_argument("--learning_rate", type=float, default=HyperParameters['learning_rate'])
parser.add_argument("--max_depth", type=int, default=HyperParameters['max_depth'])
parser.add_argument("--colsample_bytree", type=float, default=HyperParameters['colsample_bytree'])
parser.add_argument("--subsample", type=float, default=HyperParameters['subsample'])
parser.add_argument("--scale_pos_weight", type=float, default=HyperParameters['scale_pos_weight'])
parser.add_argument("--random_state", type=int, default=HyperParameters['random_state'])
parser.add_argument("--model_version", type=str, default='v1')
args = parser.parse_args()

hyper_parameters = HyperParameters.copy()
hyper_parameters['n_estimators'] = args.n_estimators
hyper_parameters['learning_rate'] = args.learning_rate
hyper_parameters['max_depth'] = args.max_depth
hyper_parameters['colsample_bytree'] = args.colsample_bytree
hyper_parameters['subsample'] = args.subsample
hyper_parameters['scale_pos_weight'] = args.scale_pos_weight
hyper_parameters['random_state'] = args.random_state
df = pd.read_parquet("/mnt/d/Data/MLOPS_2023/data_phase-1/phase-1/prob-2/raw_train.parquet")
# test = pd.read_csv('/home/duclh3/Workspace/test_phase1_prob2/mlops_phase1_prob2_207c042d-7882-4831-b2bc-1740c5ed739e.csv')

processor = Phase1Prob2FeatureProcessor()
new_df = processor.fit_transform(df)

kfold = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
FEATURES = processor.data_features['features']
categorical = processor.data_features['categorical_features']
TARGET = 'label'

print(new_df.shape, df.shape)

models = []
scores = []
oofs = np.zeros(df.shape[0])
for i, (train_idx, valid_idx) in enumerate(kfold.split(new_df, df['label'])):
    X_train = new_df.iloc[train_idx][FEATURES]
    X_valid = new_df.iloc[valid_idx][FEATURES]
    y_train = df.iloc[train_idx][TARGET]
    y_valid = df.iloc[valid_idx][TARGET]
    # model = LGBMClassifier(**hyper_parameters)
    # model.fit(X_train, y_train,
    #           eval_set=[(X_train, y_train), (X_valid, y_valid)],
    #           eval_metric=["logloss", "auc"],
    #           categorical_feature=categorical,
    #           early_stopping_rounds=50,
    #           verbose=50)
    model = CatBoostClassifier(iterations=300, learning_rate=0.1, random_state=42, eval_metric='AUC')
    model.fit(X_train, y_train,
              cat_features=categorical,
              eval_set=[(X_train, y_train), (X_valid, y_valid)],
              verbose=100)
    models.append(model)
    y_pred_proba = model.predict_proba(X_valid)[:, 1]
    y_pred = (y_pred_proba>0.5).astype(int)
    oofs[valid_idx] = y_pred_proba
    # feats = pd.DataFrame({'feature_name':model.feature_names_,
    #                       'feature_score': model.get_feature_importance()}).sort_values('feature_score', ascending=False)
    # print(feats)
    print(f"fold {i} : {roc_auc_score(y_valid, y_pred_proba)}")
    print(classification_report(y_valid, y_pred))
    scores.append(roc_auc_score(y_valid, y_pred_proba))

print(np.mean(scores), np.std(scores))
gmodel = CatBoostClassifier(**models[0].get_params())
gmodel.fit(new_df[FEATURES], df[TARGET],
           eval_set=[(new_df[FEATURES],df[TARGET])],
           cat_features=categorical,
           verbose=50)

with open(f'././checkpoints/phase-1/prob-2/{args.model_version}.pkl','wb') as file:
    pickle.dump(gmodel, file)
var_count_train = Counter(df['feature11'])

with open(f'././checkpoints/phase-1/prob-2/var_count.pkl', 'wb') as file:
    pickle.dump(var_count_train, file)