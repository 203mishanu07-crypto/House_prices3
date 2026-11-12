import sys

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor

# Models
from sklearn.linear_model import Ridge

#################################################################
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor

# Metrics
# Pipeline
from sklearn.pipeline import Pipeline

# GridSearch
from sklearn.preprocessing import StandardScaler  # уточнить
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# main будет брать лучшие параметры модели и обучать их на лучшем наборе параметров и выводить результат
# run_inference скаичивает все модели и проводит инференс
from config import Config
from src.custom_models.myMLP import CheckpointManager, EarlyStopping, MyModel, RMSELoss
from src.feature.features import DNN_encoder, FeatureTransformer, encoder_transformer
from src.modeling.predict import Predictor
from src.utils.model_saving import ModelSaver
from src.utils.mymlflow import Mlflow_logger

if "src.core_funcs" in sys.modules:
    del sys.modules["src.core_funcs"]
from src.core_funcs import *
from src.custom_models.meta_model import MyStackingRidge


def run_meta_model(X_train, y_train):
    ridge_params = {
        "parameters": {
            "model__alpha": np.float64(0.11144234247204798),
            "model__solver": "saga",
            "model__max_iter": 1000,
            "model__random_state": Config.seed,
        },
        "fit_param": {},
    }

    xgb_params = {
        "parameters": {
            "model__device": "cuda",
            "model__booster": "gbtree",
            "model__tree_method": "hist",
            "model__n_estimators": 1000,
            "model__learning_rate": np.float64(0.013190161120540767),
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
        "fit_param": {},
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
        "fit_param": {},
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
        "fit_param": {},
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
        "fit_param": {"model__cat_features": FeatureTransformer.cat_cols},
    }

    dnn_params = {
        "parameters": {
            "model__cat_cols": FeatureTransformer.cat_cols,
            "model__num_cols": FeatureTransformer.num_cols,
            "model__epochs": 1000,
            "model__device": Config.device,
            "model__emb_dim": 24,
            "model__inner_n": 256,
            "model__drop_n": 0.2,
            "model__n_layers": 1,
            "model__use_batchnorm": True,
            "model__loss_reg": RMSELoss,
            "model__l1_lambda": 0.023081011668103977,
            "model__optim": torch.optim.Adam,
            "model__optim_lr": 0.06343656703028737,
            "model__scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau,
            "model__lr_scheduler": True,
            "model__factor": 0.09023739101987314,
            "model__patience": 20,
            "model__threshold": 0.01,
            "model__min_lr": 1e-10,
            "model__early_stopping_patience": 45,
            "model__early_stopping_threshold": 0.001,
            "model__batch_size": 64,
            "model__saving_threshold": 0.001,
            "model__save_path": "checkpoints",
        },
        "fit_param": {"model__callbacks": [CheckpointManager(), EarlyStopping()]},
    }

    ridge_meta_params = {
        "parameters": {
            "model__alpha": np.float64(0.11144234247204798),
            "model__solver": "saga",
            "model__max_iter": 1000,
            "model__random_state": Config.seed,
        },
        "fit_param": {},
    }
    # ------------------------------------------------------------------------------------------------------------------------------------
    kf = KFold(n_splits=5, shuffle=True, random_state=Config.seed)
    cv_folds = list(kf.split(X_train, y_train))

    meta_ridge = MyStackingRidge(
        groups={"avg_boost": ["xgb", "catboost", "lightgbm"]},
        meta_cols=["avg_boost", "rf", "dnn", "ridgebase"],
    )
    meta_ridge.fit_sub_models(
        X_train,
        y_train,
        rf_params,
        dnn_params,
        cat_params,
        lgbm_params,
        xgb_params,
        ridge_params,
        cv_folds,
        Config.seed,
    )

    meta_res = meta_ridge.fit(
        X_train, y_train, ridge_meta_params, cv_folds, Config.seed
    )
    ModelSaver.save_ensemble([meta_ridge], "meta_ridge")
    return meta_res["oof_rmse"]


def run_cv_all_models():
    "Cудя идут лучшие параметры которые я отобрал во время экспериментов"
    all_pipelines = {
        "ridge": {
            "ridge": {
                "pipeline": Pipeline(
                    [
                        ("feature_transform", FeatureTransformer()),
                        ("encoder", clone(encoder_transformer)),
                        ("scaler", StandardScaler()),
                        ("model", Ridge()),
                    ]
                ),
                "parameters": {
                    "model__alpha": [
                        np.float64(0.11144234247204798)
                    ],  # np.float64(0.11144234247204798)
                    "model__solver": ["saga"],
                    "model__max_iter": [1000],
                    "model__random_state": [Config.seed],
                },
                "search_type": "grid",
                "fit_param": {},
                "n_iter": 1,
                "plot_param": "model__alpha",
                "heatmap_params": ("model__alpha", "model__max_iter"),
                "shap_mode": "linear",
                "cv": 5,
                "n_jobs": -1,
            }
        },
        "dnn": {
            "dnn": {
                "pipeline": Pipeline(
                    [
                        ("feature_transform", FeatureTransformer()),
                        ("encoder", DNN_encoder()),
                        ("model", MyModel()),
                    ]
                ),
                "parameters": {
                    "model__cat_cols": [FeatureTransformer.cat_cols],
                    "model__num_cols": [FeatureTransformer.num_cols],
                    "model__epochs": [1000],
                    "model__device": [Config.device],
                    "model__emb_dim": [24],
                    "model__inner_n": [256],
                    "model__drop_n": [0.2],
                    "model__n_layers": [1],
                    "model__use_batchnorm": [True],
                    "model__loss_reg": [RMSELoss],
                    "model__l1_lambda": [0.023081011668103977],
                    "model__optim": [torch.optim.Adam],
                    "model__optim_lr": [0.06343656703028737],
                    "model__scheduler": [torch.optim.lr_scheduler.ReduceLROnPlateau],
                    "model__lr_scheduler": [True],
                    "model__factor": [0.09023739101987314],
                    "model__patience": [20],
                    "model__threshold": [0.1],
                    "model__min_lr": [1e-10],
                    "model__early_stopping_patience": [45],
                    "model__early_stopping_threshold": [0.001],
                    "model__batch_size": [32],
                    "model__saving_threshold": [0.001],
                    "model__save_path": ["checkpoints"],
                },
                "search_type": "grid",
                "fit_param": {
                    "model__callbacks": [CheckpointManager(), EarlyStopping()]
                },  # ,MLflow_logger()
                "n_iter": 1,  # для рандом серча
                "plot_param": "model__optim_lr",
                "heatmap_params": ("model__optim_lr", "model__n_layers"),
                "shap_mode": "dnn",
                "cv": 5,
                "n_jobs": 1,
            }
        },
        "catboost": {
            "catboost": {
                "pipeline": Pipeline(
                    [
                        ("feature_transform", FeatureTransformer(cat_boosting=True)),
                        ("model", CatBoostRegressor(allow_writing_files=False)),
                    ]
                ),
                "parameters": {
                    "model__boosting_type": ["Plain"],
                    "model__iterations": [1000],
                    "model__learning_rate": [0.05211448378065621],
                    "model__depth": [4],
                    "model__bootstrap_type": ["Bernoulli"],
                    "model__thread_count": [-1],
                    "model__l2_leaf_reg": [0.01727277227745447],
                    "model__rsm": [1.0],
                    "model__subsample": [0.8],
                    "model__border_count": [256],
                    "model__min_data_in_leaf": [50],
                    "model__max_ctr_complexity": [2],
                    "model__one_hot_max_size": [6],
                    "model__eval_metric": ["RMSE"],
                    "model__loss_function": ["RMSE"],
                    "model__random_strength": [1.7444045228543141],
                    "model__verbose": [False],
                    "model__random_seed": [Config.seed],
                    "model__early_stopping_rounds": [50],
                },
                "fit_param": {"model__cat_features": FeatureTransformer.cat_cols},
                "n_iter": 1,
                "plot_param": "model__learning_rate",
                "heatmap_params": ("model__learning_rate", "model__l2_leaf_reg"),
                "shap_mode": "catboost",
                "cv": 5,
                "n_jobs": -1,
            }
        },
        "lightgbm": {
            "lightgbm": {
                "pipeline": Pipeline(
                    [
                        ("feature_tranform", FeatureTransformer(cat_boosting=True)),
                        ("model", LGBMRegressor()),
                    ]
                ),
                "parameters": {
                    "model__device": ["cpu"],
                    "model__boosting_type": ["gbdt"],
                    "model__n_estimators": [2000],
                    "model__learning_rate": [0.098],
                    "model__max_depth": [4],
                    "model__subsample": [0.4],
                    "model__colsample_bytree": [0.8],
                    "model__reg_alpha": [0.008],
                    "model__reg_lambda": [0.04000000000000001],
                    "model__objective": ["regression"],
                    "model__metric": ["rmse"],
                    "model__min_child_samples": [40],
                    "model__min_child_weight": [1],
                    "model__min_split_gain": [0.0005],
                    "model__n_jobs": [-1],
                    "model__random_state": [Config.seed],
                    "model__verbosity": [-1],
                    "model__early_stopping_round": [30],
                },
                "fit_param": {},
                "n_iter": 1,
                "plot_param": "model__learning_rate",
                "heatmap_params": ("model__learning_rate", "model__max_depth"),
                "shap_mode": "lightgbm",
                "cv": 5,
                "n_jobs": 1,
            }
        },
        "knn": {
            "knn": {
                "pipeline": Pipeline(
                    [
                        ("feature_transform", FeatureTransformer()),
                        ("encoder", clone(encoder_transformer)),
                        ("scaler", StandardScaler()),
                        ("model", KNeighborsRegressor()),
                    ]
                ),
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
        "decision_tree": {
            "decision_tree": {
                "pipeline": Pipeline(
                    [
                        ("feature_transform", FeatureTransformer()),
                        ("encoder", clone(encoder_transformer)),
                        ("scaler", StandardScaler()),
                        ("model", DecisionTreeRegressor()),
                    ]
                ),
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
        "random_forest": {
            "random_forest": {
                "pipeline": Pipeline(
                    [
                        ("feature_transform", FeatureTransformer()),
                        ("encoder", clone(encoder_transformer)),
                        ("scaler", StandardScaler()),
                        ("model", RandomForestRegressor()),
                    ]
                ),
                "parameters": {
                    "model__n_estimators": [1000],
                    "model__max_depth": [14],
                    "model__criterion": ["squared_error"],
                    "model__n_jobs": [-1],
                    "model__min_samples_split": [10],
                    "model__min_samples_leaf": [5],
                    "model__max_leaf_nodes": [100],
                    "model__min_impurity_decrease": [0],
                    "model__max_features": ["sqrt"],
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
        "xgboost": {
            "xgboost": {
                "pipeline": Pipeline(
                    [
                        ("feature_transform", FeatureTransformer()),
                        ("encoder", clone(encoder_transformer)),
                        ("scaler", StandardScaler()),
                        ("model", XGBRegressor()),
                    ]
                ),
                "parameters": {
                    "model__device": ["cuda"],
                    "model__booster": ["gbtree"],
                    "model__tree_method": ["hist"],
                    "model__n_estimators": [1000],
                    "model__learning_rate": [np.float64(0.013190161120540767)],
                    "model__max_depth": [6],
                    "model__min_child_weight": [0.03794613934217169],
                    "model__n_jobs": [1],
                    "model__subsample": [0.4],
                    "model__colsample_bytree": [0.75],
                    "model__reg_alpha": [0.0193336968922133],
                    "model__reg_lambda": [0.032333649917330995],
                    "model__gamma": [0.017362357647552687],
                    "model__objective": ["reg:squarederror"],
                    "model__eval_metric": ["rmse"],
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
        },
    }
    # cmd - python -m mlflow server --port 5000
    logger = Mlflow_logger("House_Prices_Kaggle")
    logger.setup_connection()

    X_train, X_holdout, y_train, y_holdout = setup_validation(
        Config.path_train_data, test_size=0.2
    )

    all_results = {"model": [], "oof_score": []}

    for model_name, pipeline in all_pipelines.items():
        model_results = run_experiment(X_train, y_train, model_name, pipeline, logger)
        all_results["model"].append(model_name)
        all_results["oof_score"].append(model_results["best_value"])
        ModelSaver.save_ensemble(model_results["best_ensemble"], model_name)

    stacking_oof = run_meta_model(X_train, y_train)
    all_results["model"].append("meta_ridge")
    all_results["oof_score"].append(stacking_oof)

    results = pd.DataFrame(all_results)
    print(results.sort_values("oof_score", ascending=True))

    saved_models = ModelSaver.upload_saved_models(Config.path_to_save_models)
    hold_out_res = Predictor.run_test(saved_models, X_holdout, y_holdout)

    results["holdout_rmse"] = results["model"].map(hold_out_res.set_index("model")["holdout_log_rmse"])
    results = results.sort_values("oof_score", ascending=True)

    print("\nРезультаты моделей\n")
    print(
        results.to_markdown(
            index=False,
            tablefmt="rounded_outline",
            floatfmt=".5f",
        )
)



if __name__ == "__main__":
    run_cv_all_models()
