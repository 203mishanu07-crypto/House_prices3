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
import patchworklib as pw

#main будет брать лучшие параметры модели и обучать их на лучшем наборе параметров и выводить результат
#run_inference скаичивает все модели и проводит инференс
from config import Config
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
if 'src.core_funcs' in sys.modules:
    del sys.modules['src.core_funcs']
from src.core_funcs import *
from src.custom_models.meta_model import MyStackingRidge

def run_meta_model(X_train,y_train):
    ridge_params = {
        "parameters": {
            "model__alpha": np.float64(0.11144234247204798),
            "model__solver": "saga",
            "model__max_iter": 1000,
            "model__random_state": Config.seed,
        },
        "fit_param": {'feature_transform__selected_columns':[
        "BldgType",
        "Condition1",
        "Exterior1st",
        "Exterior2nd",
        "Fence",
        "Foundation",
        "GarageType",
        "HouseStyle",
        "LotConfig",
        "MSSubClass",
        "MSZoning",
        "MiscFeature",
        "Neighborhood",
        "SaleCondition",
        "SaleType",
        "1stFlrSF",
        "1stFlrSF_log",
        "2ndFlrSF",
        "2ndFlrSF_log",
        "Age",
        "Age_from_Remod",
        "BsmtExposure",
        "BsmtFinSF1",
        "BsmtFinSF1_log",
        "BsmtFinType1*BsmtFinSF1*BsmtQual",
        "CentralAir",
        "Functional",
        "GarageArea",
        "GarageArea_log",
        "GarageCars",
        "KitchenQual",
        "LotArea_log",
        "MasVnrArea",
        "MasVnrArea_log",
        "OpenPorch",
        "OpenPorchSF",
        "OpenPorchSF_log",
        "OverallBath",
        "OverallCond",
        "OverallQual",
        "OverallQual*OverallCond",
        "Overall_Fireplaces",
        "TotRmsAbvGrd",
        "TotalBsmtSF_log",
        "TotalSqFeet",
        "TotalSqFeet_log",
        "WoodDeckSF"
    ]}
    }

    xgb_params = {
        "parameters": {
            "model__device": "cuda",
            "model__booster": "gbtree",
            "model__tree_method": "hist",
            "model__n_estimators": 1000,
            "model__learning_rate":np.float64(0.013190161120540767),
            "model__max_depth": 6,
            "model__min_child_weight": 0.03794613934217169,
            "model__n_jobs": 1,
            "model__subsample": 0.4,
            "model__colsample_bytree": 0.75,
            "model__reg_alpha": 0.0193336968922133,
            "model__reg_lambda": 0.032333649917330995,
            "model__gamma": 0.017362357647552687,
            "model__objective": "reg:squarederror",
            "model__eval_metric": "rmse",
            "model__random_state": Config.seed,
        },
        "fit_param": {'feature_transform__selected_columns':[
            "BldgType",
            "Condition1",
            "Exterior1st",
            "Exterior2nd",
            "Fence",
            "Foundation",
            "GarageType",
            "HouseStyle",
            "LandContour",
            "LotConfig",
            "LotShape",
            "MSSubClass",
            "MSZoning",
            "Neighborhood",
            "PavedDrive",
            "RoofStyle",
            "SaleCondition",
            "SaleType",
            "1stFlrSF",
            "1stFlrSF_log",
            "2ndFlrSF",
            "2ndFlrSF_log",
            "Age",
            "Age_from_Remod",
            "BedroomAbvGr",
            "BsmtExposure",
            "BsmtFinSF1",
            "BsmtFinSF1_log",
            "BsmtFinType1*BsmtFinSF1*BsmtQual",
            "BsmtQual",
            "BsmtUnfSF",
            "BsmtUnfSF_log",
            "CentralAir",
            "ExterQual*ExterCond",
            "FullBath",
            "Functional",
            "GarageArea",
            "GarageArea_log",
            "GarageCars",
            "GarageFinish",
            "HalfBath",
            "HeatingQC",
            "KitchenAbvGr",
            "KitchenQual",
            "LotArea",
            "LotArea_log",
            "LotFrontage",
            "MasVnrArea",
            "MoSold",
            "OpenPorchSF",
            "OpenPorchSF_log",
            "OverallBath",
            "OverallCond",
            "OverallQual",
            "OverallQual*OverallCond",
            "Overall_Fireplaces",
            "ScreenPorch",
            "TotRmsAbvGrd",
            "TotalBsmtSF",
            "TotalSqFeet",
            "TotalSqFeet_log",
            "WoodDeckSF",
            "YrSold"
    ]}
    }

    rf_params = {
        "parameters": {
            "model__n_estimators": 1000,
            "model__max_depth": 14,
            "model__criterion": "squared_error",
            "model__n_jobs": -1,
            "model__min_samples_split": 10,
            "model__min_samples_leaf": 5,
            "model__max_leaf_nodes": 100,
            "model__min_impurity_decrease": 0,
            "model__max_features": None,
            "model__bootstrap": True,
            "model__max_samples": 0.8,
            "model__oob_score": True,
            "model__random_state": Config.seed,
        },
        "fit_param": {'feature_transform__selected_columns':
        [
        "GarageType",
        "MSZoning",
        "2ndFlrSF",
        "2ndFlrSF_log",
        "Age",
        "Age_from_Remod",
        "BsmtFinType1*BsmtFinSF1*BsmtQual",
        "BsmtQual",
        "BsmtUnfSF_log",
        "CentralAir",
        "GarageArea",
        "GarageArea_log",
        "GarageCars",
        "GarageFinish",
        "KitchenQual",
        "LotArea",
        "LotArea_log",
        "OverallBath",
        "OverallQual",
        "OverallQual*OverallCond",
        "Overall_Fireplaces",
        "TotalSqFeet",
        "TotalSqFeet_log"
    ]}
    }

    lgbm_params = {
        "parameters": {
            "model__device": "cpu",
            "model__boosting_type": "gbdt",
            "model__n_estimators": 2000,
            "model__learning_rate": 0.098,
            "model__max_depth": 4,
            "model__subsample": 0.4,
            "model__colsample_bytree": 0.8,
            "model__reg_alpha": 0.008,
            "model__reg_lambda": 0.04000000000000001,
            "model__objective": "regression",
            "model__metric": "rmse",
            "model__min_child_samples": 40,
            "model__min_child_weight": 1,
            "model__min_split_gain": 0.0005,
            "model__n_jobs": -1,
            "model__random_state": Config.seed,
            "model__verbosity": -1,
            "model__early_stopping_round": 30,
        },
        "fit_param": {'feature_transform__selected_columns':[
        "1stFlrSF",
        "2ndFlrSF",
        "Age",
        "Age_from_Remod",
        "BsmtExposure",
        "BsmtFinSF1",
        "BsmtFinType1*BsmtFinSF1*BsmtQual",
        "BsmtQual",
        "BsmtUnfSF",
        "CentralAir",
        "Condition1",
        "Exterior1st",
        "Functional",
        "GarageArea",
        "GarageArea_log",
        "GarageCars",
        "GarageFinish",
        "KitchenQual",
        "LotArea",
        "LotArea_log",
        "MSSubClass",
        "MoSold",
        "Neighborhood",
        "OpenPorchSF",
        "OverallBath",
        "OverallCond",
        "OverallQual",
        "OverallQual*OverallCond",
        "Overall_Fireplaces",
        "TotalSqFeet",
        "TotalSqFeet_log"
    ]}
    }

    cat_params = {
        "parameters": {
            "model__boosting_type": "Plain",
            "model__iterations": 1000,
            "model__learning_rate": 0.05211448378065621,
            "model__depth": 4,
            "model__bootstrap_type": "Bernoulli",
            "model__thread_count": -1,
            "model__l2_leaf_reg": 0.01727277227745447,
            "model__rsm": 1.0,
            "model__subsample": 0.8,
            "model__border_count": 256,
            "model__min_data_in_leaf": 50,
            "model__max_ctr_complexity": 2,
            "model__one_hot_max_size": 6,
            "model__eval_metric": "RMSE",
            "model__loss_function": "RMSE",
            "model__random_strength": 1.7444045228543141,
            "model__verbose": False,
            "model__random_seed": Config.seed,
            "model__early_stopping_rounds": 50,
        },
        "fit_param": {
            "model__cat_features": FeatureTransformer.cat_cols
        }
    }

    dnn_params = {
        "parameters": {
            "model__cat_cols": FeatureTransformer.cat_cols,
            "model__num_cols": FeatureTransformer.num_cols,
            "model__epochs": 1000,
            "model__device": Config.device,
            "model__emb_dim": 12,
            "model__inner_n": 512,
            "model__drop_n": 0.1,
            "model__n_layers": 1,
            "model__use_batchnorm": True,
            "model__loss_reg": RMSELoss,
            "model__l1_lambda": 0.003433628682365358,
            "model__optim": torch.optim.Adam,
            "model__optim_lr": 0.08686979995821902,
            "model__scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau,
            "model__lr_scheduler": True,
            "model__factor": 0.08066252499544253,
            "model__patience": 15,
            "model__threshold": 0.01,
            "model__min_lr": 1e-10,
            "model__early_stopping_patience": 45,
            "model__early_stopping_threshold": 0.001,
            "model__batch_size": 64,
            "model__saving_threshold": 0.001,
            "model__save_path": "checkpoints",
        },
        "fit_param": {
            "model__callbacks": [
                CheckpointManager(),
                EarlyStopping()
            ]
        }
    }

    ridge_meta_params = {
        "parameters": {
            "model__alpha": np.float64(0.11144234247204798),
            "model__solver": "saga",
            "model__max_iter": 1000,
            "model__random_state": Config.seed,
        },
        "fit_param": {}
    }
    #------------------------------------------------------------------------------------------------------------------------------------
    kf = KFold(n_splits=5, shuffle=True, random_state=Config.seed)
    cv_folds = list(kf.split(X_train, y_train))

    meta_ridge = MyStackingRidge(groups = {"avg_boost": ["xgb", "catboost", "lightgbm"]},meta_cols=["avg_boost", "rf", "dnn", "ridgebase"]) 
    meta_ridge.fit_sub_models(X_train, y_train,
            rf_params, dnn_params, cat_params,
            lgbm_params, xgb_params, ridge_params,
            cv_folds, Config.seed)

    meta_res = meta_ridge.fit(X_train, y_train, ridge_meta_params, cv_folds, Config.seed)
    ModelSaver.save_ensemble([meta_ridge],'meta_ridge')
    return meta_res['oof_rmse']

def run_cv_all_models():
    "Cудя идут лучшие параметры которые я отобрал во время экспериментов"
    all_pipelines = {
                "ridge":{"ridge": {
                        "pipeline": Pipeline([
                            ("feature_transform", FeatureTransformer()),
                            ("encoder", clone(encoder_transformer)),
                            ("scaler", StandardScaler()),
                            ("model", Ridge()),
                            ]),
                        "parameters": {
                            "model__alpha": [np.float64(0.11144234247204798)],  # np.float64(0.11144234247204798)
                            "model__solver": ["saga"],
                            "model__max_iter": [1000],
                            "model__random_state": [Config.seed],
                        },
                        "search_type": "grid",
                        "fit_param": {},
                        "n_iter": 1,
                        "plot_param": "model__alpha",
                        "heatmap_params": ("model__alpha", "model__max_iter"),
                        "shap_mode":"linear",
                        "cv": 5,
                        "n_jobs":-1
                    }
                },

                "dnn":{"dnn": {
                        "pipeline": Pipeline([
                            ("feature_transform", FeatureTransformer()),
                            ("encoder", DNN_encoder()),
                            ("model", MyModel()),
                        ]),

                        "parameters": {
                            # Данные
                            "model__cat_cols": [FeatureTransformer.cat_cols],
                            "model__num_cols": [FeatureTransformer.num_cols],
                            # Обучение
                            "model__epochs": [1000],
                            "model__device": [Config.device],
                            # Архитектура
                            "model__emb_dim": [12],
                            "model__inner_n": [512],
                            "model__drop_n": [0.1],
                            "model__n_layers": [1],
                            "model__use_batchnorm": [True],
                            # Loss
                            "model__loss_reg": [RMSELoss],
                            #Regularization
                            "model__l1_lambda":[0.003433628682365358],
                            # Optimizer
                            "model__optim": [torch.optim.Adam],
                            "model__optim_lr": [0.08686979995821902],
                            # Scheduler
                            "model__scheduler": [
                                torch.optim.lr_scheduler.ReduceLROnPlateau
                            ],
                            "model__lr_scheduler": [True],
                            "model__factor": [0.08066252499544253],
                            "model__patience": [15],
                            "model__threshold": [0.01],
                            "model__min_lr": [1e-10],
                            # Early Stopping
                            "model__early_stopping_patience": [45],
                            "model__early_stopping_threshold": [0.001],
                            # DataLoader
                            "model__batch_size": [64],
                            # Checkpoint
                            "model__saving_threshold": [0.001],
                            "model__save_path": ["checkpoints"],
                        },
                        "search_type": "grid",
                        "fit_param": {"model__callbacks":[CheckpointManager(),EarlyStopping()]}, #,MLflow_logger()
                        "n_iter": 1, #для рандом серча
                        "plot_param": "model__optim_lr",
                        "heatmap_params": ("model__optim_lr", "model__n_layers"),
                        "shap_mode": "dnn",
                        "cv": 5,
                        "n_jobs": 1,
                }
            },

        "catboost":{"catboost": {
                "pipeline": Pipeline([
                    ("feature_transform", FeatureTransformer(cat_boosting=True)),
                    ("model", CatBoostRegressor(allow_writing_files=False)),
                ]),
                "parameters": {
                    "model__boosting_type": ["Plain"],  # ["Plain", "Ordered"]
                    "model__iterations": [1000],  # np.linspace(1, 100, 5, dtype=int)
                    "model__learning_rate": [np.float64(0.08)],  # [0.08]
                    "model__depth": [6],  # [4, 5, 6, 8]
                    "model__bootstrap_type": ["Bernoulli"],
                    "model__thread_count": [-1],
                    "model__l2_leaf_reg": [np.float64(0.0098)],  # np.float64(0.0098)
                    "model__rsm": [0.8],  # [0.6, 0.8, 1.0]
                    "model__subsample": [1.0],
                    "model__border_count": [32],
                    "model__min_data_in_leaf": [45],
                    "model__max_ctr_complexity": [2],
                    "model__one_hot_max_size": [5],
                    "model__eval_metric": ["RMSE"],
                    "model__loss_function": ["RMSE"],
                    "model__random_strength": [1],  # [0, 0.5, 1, 2]
                    "model__verbose": [False],
                    "model__random_seed": [Config.seed],
                },
                'fit_param': {"model__cat_features": FeatureTransformer.cat_cols},
                "n_iter": 1,
                "plot_param": "model__learning_rate",
                "heatmap_params": ("model__learning_rate", "model__l2_leaf_reg"),
                "shap_mode": "catboost",
                "cv": 5,
                "n_jobs": -1,
            }
        },

        "lightgbm":{"lightgbm": {
                "pipeline": Pipeline([
                    ("feature_tranform", FeatureTransformer(cat_boosting=True)),
                    ("model", LGBMRegressor()),
                ]),
                "parameters": {
                    "model__device": ["cpu"],
                    "model__boosting_type": ["gbdt"],
                    "model__n_estimators": [200],  # np.linspace(1, 100, 5, dtype=int)
                    "model__learning_rate": [0.098],  # np.linspace(0.01, 0.1, 5); [np.float64(0.0775)]
                    "model__max_depth": [4],  # [4, 5, 6, 8]
                    "model__subsample": [0.4],  # [0.6, 0.8, 1.0]
                    "model__colsample_bytree": [0.8],  # [0.6, 0.8, 1.0]
                    "model__reg_alpha": [0.008],  # [0.1, 0.2, 0.4, 0.5, 1]
                    "model__reg_lambda": [0.04000000000000001],  # [0.01, 0.1, 0.5, 5, 10]
                    "model__objective": ["regression"],
                    "model__metric": ["rmse"],
                    "model__min_child_samples": [40],
                    "model__min_child_weight": [1],  # np.linspace(1, 5, 3, dtype=int)
                    "model__min_split_gain": [0.0005],  # np.linspace(0.01, 0.1, 5)
                    "model__n_jobs": [-1],
                    "model__random_state": [Config.seed],
                    "model__verbosity": [-1]
                },
                "fit_param" : {},
                "n_iter": 1,
                "plot_param": "model__learning_rate",
                "heatmap_params": ("model__learning_rate", "model__max_depth"),
                'shap_mode':'lightgbm',
                "cv": 5,
                "n_jobs": 1,
            }
        },

        "knn":{"knn": {
            "pipeline": Pipeline([
                    ("feature_transform", FeatureTransformer()),
                    ("encoder", clone(encoder_transformer)),
                    ("scaler", StandardScaler()),
                    ("model", KNeighborsRegressor()),
                ]),
            "parameters": {
                    "model__n_neighbors": [9],
                    "model__weights": ["distance"],
                    "model__p": [1],
                    "model__n_jobs": [-1],
                },
            "search_type": "grid",
            "fit_param": {},
            "n_iter": 1,
            "plot_param": "model__n_neighbors",
            "heatmap_params": ("model__n_neighbors", "model__p"),
            "shap_mode": "dnn",
            "cv": 5,
            "n_jobs": -1,
            }
        },

        "decision_tree":{"decision_tree": {
            "pipeline": Pipeline([
                    ("feature_transform", FeatureTransformer()),
                    ("encoder", clone(encoder_transformer)),
                    ("scaler", StandardScaler()),
                    ("model", DecisionTreeRegressor()),
                ]),
            "parameters": {
                    "model__max_depth": [6],
                    "model__criterion": ["squared_error"],
                    "model__splitter": ["best"],
                    "model__min_samples_split": [36],
                    "model__min_samples_leaf": [11],
                    "model__max_leaf_nodes": [50],
                    "model__min_impurity_decrease": [0],
                    "model__random_state": [Config.seed],
                },
            "search_type": "grid",
            "fit_param": {},
            "n_iter": 1,
            "plot_param": "model__max_depth",
            "heatmap_params": ("model__max_depth", "model__min_samples_leaf"),
            "shap_mode": "tree",
            "cv": 5,
            "n_jobs": -1,
            }
        },

        "random_forest":{"random_forest": {
            "pipeline": Pipeline([
                    ("feature_transform", FeatureTransformer()),
                    ("encoder", clone(encoder_transformer)),
                    ("scaler", StandardScaler()),
                    ("model", RandomForestRegressor()),
                ]),
            "parameters": {
                    # bootstrap=True
                    "model__n_estimators": [1000],  # np.linspace(1, 100, 5, dtype=int)
                    "model__max_depth": [13],  # np.linspace(1, 10, 5, dtype=int)
                    "model__criterion": ["squared_error"],
                    "model__n_jobs": [-1],
                    "model__min_samples_split": [10],  # np.linspace(10, 100, 5, dtype=int)
                    "model__min_samples_leaf": [5],  # np.linspace(5, 15, 5, dtype=int); min_samples_leaf: np.int64(6)
                    "model__max_leaf_nodes": [75],  # np.linspace(50, 70, 5, dtype=int)
                    "model__min_impurity_decrease": [0],
                    "model__max_features": [None],  # "sqrt", "log2"
                    "model__bootstrap": [True],
                    "model__max_samples": [0.8],
                    "model__oob_score": [True],
                    "model__random_state": [Config.seed],
                },
            "search_type": "grid",
            "fit_param": {},
            "n_iter": 1,
            "plot_param": "model__n_estimators",
            "heatmap_params": ("model__n_estimators", "model__max_depth"),
            "shap_mode": "tree",
            "cv": 5,
            "n_jobs": -1,
        }
        },

        "xgboost":{"xgboost": {
            "pipeline": Pipeline([
                    ("feature_transform", FeatureTransformer()),
                    ("encoder", clone(encoder_transformer)),
                    ("scaler", StandardScaler()),
                    ("model", XGBRegressor()),
                ]),
            "parameters": {
                    "model__device": ["cuda"],
                    "model__booster": ["gbtree"],
                    "model__tree_method": ["hist"],
                    "model__n_estimators": [1000],  # np.linspace(1, 100, 5, dtype=int); лучшее 12
                    "model__learning_rate": [np.float64(0.013000000000000001)],  # [np.float64(0.0775)]
                    "model__max_depth": [6],  # [6, 8, 10, 12, 14]; лучшее 10 для случайного леса
                    "model__min_child_weight": [5],  # np.linspace(1, 10, 5, dtype=int)
                    "model__n_jobs": [-1],
                    "model__subsample": [0.5],  # [0.6, 0.8, 1.0]
                    "model__colsample_bytree": [0.4793103448275863],  # [0.65, 0.7, 0.75]
                    "model__reg_alpha": [0.003666666666666667],  # [0, 0.01, 0.1, 1]
                    "model__reg_lambda": [0.04555555555555556],  # [0.01, 0.1, 0.5, 5, 10]
                    "model__gamma": [0],  # np.linspace(0.01, 0.1, 10)
                    "model__objective": ["reg:squarederror"],
                    "model__eval_metric": ["rmse"],  # RMSE для регрессии
                    "model__random_state": [Config.seed],
                },
            "search_type": "grid",
            "fit_param": {},
            "n_iter": 1,
            "plot_param": "model__learning_rate",
            "heatmap_params": ("model__learning_rate", "model__n_estimators"),
            "shap_mode": "tree",
            "cv": 5,
            "n_jobs": 1,
        }
        }
    }
    #cmd - python -m mlflow server --port 5000
    logger = Mlflow_logger("House_Prices_Kaggle")
    logger.setup_connection()

    X_train, X_holdout, y_train, y_holdout = setup_validation(Config.path_train_data, test_size = 0.2)

    all_results = {'model':[],
                   'oof_score':[]}
    
    for model_name, pipeline in all_pipelines.items():
        model_results = run_experiment(
                X_train, y_train,
                model_name,
                pipeline,
                logger
            )
        all_results['model'].append(model_name)
        all_results['oof_score'].append(model_results["best_value"])
        ModelSaver.save_ensemble(model_results['best_ensemble'],model_name)


    stacking_oof = run_meta_model(X_train,y_train)
    all_results['model'].append('stacking_ridge')
    all_results['oof_score'].append(stacking_oof)

    results = pd.DataFrame(all_results)
    print(results.sort_values('oof_score',ascending=True))


if __name__ == '__main__':
    run_cv_all_models()