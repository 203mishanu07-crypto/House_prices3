import torch
import torchvision

import torch
import torch.nn as nn

from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, random_split


from torchvision import transforms

from torchvision.transforms import v2

#################################################################
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


from matplotlib.ticker import AutoMinorLocator, MultipleLocator

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder

from sklearn.utils import check_random_state

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from pathlib import Path

import pandas as pd
import mlflow
from mlflow import MlflowClient
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn.model_selection

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.base import clone

#GridSearch
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV, ParameterSampler
import optuna

from sklearn.preprocessing import StandardScaler #уточнить
from sklearn.base import BaseEstimator,TransformerMixin
#Metrics
from sklearn.metrics import root_mean_squared_error
from sklearn.utils.validation import check_is_fitted
#Pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
#Models
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.ensemble import RandomForestRegressor

from catboost import CatBoostRegressor
from catboost import Pool

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import shap
import graphviz

import joblib

from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib

import mlflow
from mlflow import MlflowClient

import time
import subprocess
import tempfile
from pathlib import Path
from IPython.display import display
import json

import sys
import importlib

import src.modeling.train as train
from src.feature.features import FeatureTransformer
from src.feature.features import encoder_transformer
from config import Config
import src.utils.drawings as drawer
from src.utils.mymlflow import Mlflow_logger
from src.custom_models.myMLP import MyModel, Callback, EarlyStopping, CheckpointManager, RMSELoss
from src.feature.feature_selection import count_off_shap
from src.feature.features import DNN_encoder
from src.utils.model_saving import ModelSaver
from src.modeling.predict import Predictor


class MyStackingRidge:
    def __init__(self,meta_cols = None, groups = {}, return_mean = False):

        def standard_pipe(model):
            return Pipeline([
                ("feature_transform", FeatureTransformer()),
                ("encoder", clone(encoder_transformer)),
                ("scaler", StandardScaler()),
                ("model", model)
            ])

        self.pipes = {
            "xgb": standard_pipe(XGBRegressor()),

            "catboost": Pipeline([
                ("feature_transform", FeatureTransformer(cat_boosting=True)),
                ("model", CatBoostRegressor(allow_writing_files=False))
            ]),

            "lightgbm": Pipeline([
                ("feature_transform", FeatureTransformer(cat_boosting=True)),
                ("model", LGBMRegressor())
            ]),

            "rf": standard_pipe(RandomForestRegressor()),

            "dnn": Pipeline([
                ("feature_transform", FeatureTransformer()),
                ("encoder", DNN_encoder()),
                ("model", MyModel())
            ]),

            "ridgebase": standard_pipe(Ridge()),
        }

        self.meta_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge())
        ])

        self.return_mean = return_mean
        
        self.ensembles = {}
        self.oof = {}

        # Настройка дополнительных комбинаций прогнозов
        # 
        self.groups = groups
        self.meta_cols = meta_cols if meta_cols else list(self.pipes)
        
        # self.groups = {"avg_boost": ["xgb", "catboost", "lightgbm"]}
        # self.meta_cols = ["avg_boost", "rf", "dnn", "ridgebase"]

        # self.groups = {"avg_boost": ["xgb", "catboost", "lightgbm"]}
        # self.meta_cols = ["xgb", "catboost", "lightgbm", "rf", "dnn", "ridgebase", "avg_boost"]


    def fit_sub_models(
        self, X, y,
        rf_params, dnn_params, cat_params,
        lgbm_params, xgb_params, ridge_params,
        cv_folds, random_state
    ):

        configs = {
            "xgb": xgb_params,
            "catboost": cat_params,
            "lightgbm": lgbm_params,
            "rf": rf_params,
            "dnn": dnn_params,
            "ridgebase": ridge_params
        }

        for name, config in configs.items():

            ensemble, results = train.Trainer.cross_validate(
                X=X,
                y=y,
                estimator=self.pipes[name],
                params=config["parameters"],
                fit_params=config["fit_param"],
                cv_folds=cv_folds,
                random_state=random_state
            )

            self.ensembles[name] = ensemble
            self.oof[name] = results["oof_pred"]
            print(f"Обучился {name}")

        return self


    def _meta_features(self, predictions):

        predictions = predictions.copy()

        for name, models in self.groups.items():
            predictions[name] = predictions[list(models)].mean(axis=1)

        return predictions[self.meta_cols]


    def fit(self, X, y, ridge_meta_params, cv_folds, random_state):

        oof_df = pd.DataFrame(self.oof, index=X.index)
        self.meta_data = self._meta_features(oof_df)

        if self.return_mean:
            oof_pred = self.meta_data.to_numpy().mean(axis=1)
            return {
                    "oof_rmse": root_mean_squared_error(y, oof_pred),
                    "oof_pred": oof_pred,
                    }

        self.meta_ensemble, meta_res = train.Trainer.cross_validate(
            X=self.meta_data,
            y=y,
            estimator=self.meta_pipe,
            params=ridge_meta_params["parameters"],
            fit_params=ridge_meta_params["fit_param"],
            cv_folds=cv_folds,
            random_state=random_state
        )   

        return meta_res

    def predict(self, X):

        predictions = pd.DataFrame({
            name: np.mean(
                [pipe.predict(X) for pipe in ensemble],
                axis=0
            ).reshape(-1)
            for name, ensemble in self.ensembles.items()
        }, index=X.index)

        meta_data = self._meta_features(predictions)

        if self.return_mean:
            return meta_data.to_numpy().mean(axis=1)

        pred_log = np.mean(
            [pipe.predict(meta_data) for pipe in self.meta_ensemble],
            axis=0
        ).reshape(-1)

        return pred_log
