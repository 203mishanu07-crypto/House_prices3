# -*- coding: utf-8 -*-
import random
from copy import deepcopy

import mlflow

# from config import Config
import numpy as np
import optuna
import pandas as pd
import torch
from catboost import CatBoostRegressor
from joblib import Parallel, delayed
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold, ParameterGrid
from tqdm.auto import tqdm

from src.custom_models.myMLP import MyModel
from src.feature.features import FeatureTransformer


class Trainer:
    @staticmethod
    def cross_validate(estimator, params, fit_params, X, y, cv_folds, random_state):
        """Обучает отдельную копию пайплайна на каждом фолде и собирает OOF-прогнозы.

        Предобработка обучается на тренировочной части фолда. Для моделей,
        поддерживаемых специальной веткой обучения, преобразованная валидационная
        часть также передаётся в eval_set.

        Args:
            estimator: Пайплайн предобработки и модели.
            params: Параметры текущего эксперимента в формате set_params.
            fit_params: Фиксированные параметры в том же формате.
            X: DataFrame с признаками.
            y: Series с целевой переменной в том же порядке строк, что и X.
            cv_folds: Список пар позиционных индексов (train_idx, val_idx).
                Для обычного OOF каждая строка должна попасть в валидацию один раз.
            random_state: Seed для воспроизводимости обучения.

        Returns:
            tuple: Список обученных пайплайнов и словарь результатов:
                OOF-прогнозы, итоговая OOF-метрика, метрики отдельных фолдов,
                их среднее и стандартное отклонение, признаки и история нейронки.
                В регрессии OOF содержит прогнозы таргета.
        """
        fold_scores = []
        ensemble = []
        fold_histories = []
        fold_idx = 0

        for train_idx, test_idx in tqdm(cv_folds, desc="Training", leave=False):
            random.seed(random_state)
            np.random.seed(random_state)
            torch.manual_seed(random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(random_state)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)

            X_train = X.iloc[train_idx].copy()
            y_train = y.iloc[train_idx].copy()

            X_val = X.iloc[test_idx].copy()
            y_val = y.iloc[test_idx].copy()

            # Эксперимент с двумя ранее найденными выбросами. Убираем тут, так как обнаружил поздно, чтобы не менять CV
            if "Id" in X_train.columns:
                keep = ~X_train["Id"].isin([524, 1299])

                X_train = X_train.loc[keep].copy()
                y_train = y_train.loc[keep].copy()

            model = clone(estimator)
            model.set_params(**params, **fit_params)

            if (
                isinstance(model["model"], CatBoostRegressor)
                | isinstance(model["model"], LGBMRegressor)
                | isinstance(model["model"], MyModel)
            ):
                preprocessor = model[:-1]
                X_train_t = preprocessor.fit_transform(X_train, y_train)
                X_val_t = preprocessor.transform(X_val)

                if isinstance(model["model"], CatBoostRegressor):
                    cat_features = [
                        col
                        for col in FeatureTransformer.cat_cols
                        if col in X_train_t.columns
                    ]
                    model["model"].set_params(cat_features=cat_features)

                if isinstance(model["model"], MyModel):
                    model["model"].fit(
                        X_train_t, y_train, eval_set=(X_val_t, y_val), fold_idx=fold_idx
                    )
                else:
                    model["model"].fit(X_train_t, y_train, eval_set=(X_val_t, y_val))
            else:
                model.fit(X_train, y_train)

            preds = model.predict(X_val)
            score = root_mean_squared_error(y_val, preds)
            fold_scores.append(score)
            ensemble.append(model)

            if isinstance(model["model"], MyModel):
                fold_histories.append(
                    {
                        "fold": fold_idx,
                        "val_rmse": model["model"].epochs_val_RSME.copy(),
                        "train_rmse": model["model"].epochs_train_RSME.copy(),
                        "learning_rate": model["model"].learning_rate_epoch.copy(),
                        "selected_epoch": model["model"].best_epoch,
                        "selected_val_rmse": score,
                    }
                )

            fold_idx += 1

        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)

        oof_pred = np.full(len(y), np.nan)

        for pipeline, (_, val_idx) in zip(ensemble, cv_folds):
            oof_pred[val_idx] = pipeline.predict(X.iloc[val_idx])
        oof_rmse = np.sqrt(np.mean((np.asarray(y) - oof_pred) ** 2))

        all_features = sorted(
            set().union(
                *[pipeline[:-1].get_feature_names_out() for pipeline in ensemble]
            )
        )

        return ensemble, {
            "params": params,
            "oof_rmse": oof_rmse,
            "mean_score": mean_score,
            "std_score": std_score,
            "fold_scores": fold_scores,
            "fold_histories": fold_histories,
            "selected_cols": all_features,
            "oof_pred": oof_pred,
        }

    @staticmethod
    def create_result_table(cv, param_distributions, raw_results):
        # Инициализация структуры для хранения результатов
        cv_results = {
            "params": [],
            "off_score": [],
            "mean_val_score": [],
            "std_val_score": [],
            "feature_count": [],
        }
        best_history = []

        # Создаем динамические списки для каждого фолда: split0_val_score, split1_val_score и т.д.
        for split_idx in range(cv):
            cv_results[f"split{split_idx}_val_score"] = []

        # Создаем динамические списки для каждого параметра (param_имя_параметра)
        for param_name in param_distributions.keys():
            cv_results[f"param_{param_name}"] = []

        ensembles, param_results = map(list, zip(*raw_results))

        best_score = +float("inf")
        best_params = None

        cv_results["candidate_id"] = []
        histories_by_candidate = {}

        candidate_id = 0

        for res, ensemble in zip(param_results, ensembles):
            if res["fold_histories"]:
                histories_by_candidate[str(candidate_id)] = res["fold_histories"]

            params = res["params"]
            oof_score = res["oof_rmse"]
            mean_score = res["mean_score"]
            std_score = res["std_score"]
            fold_scores = res["fold_scores"]

            cv_results["params"].append(params)
            cv_results["off_score"].append(oof_score)
            cv_results["mean_val_score"].append(mean_score)
            cv_results["std_val_score"].append(std_score)
            cv_results["feature_count"].append(len(res["selected_cols"]))

            for split_idx, score in enumerate(fold_scores):
                cv_results[f"split{split_idx}_val_score"].append(score)

            for param_name, param_value in params.items():
                cv_results[f"param_{param_name}"].append(param_value)

            if oof_score < best_score:
                best_score = oof_score
                best_mean = mean_score
                best_std = std_score
                best_params = params
                best_ensemble = ensemble
                best_history = res["fold_histories"]
                best_cols = res["selected_cols"]

            cv_results["candidate_id"].append(candidate_id)
            candidate_id += 1

        results_df = pd.DataFrame(cv_results)

        return (
            results_df,
            best_ensemble,
            best_score,
            {
                "best_oof_score": best_score,
                "best_mean_val_score": best_mean,
                "best_val_std": best_std,
                "best_params": best_params,
                "best_cols": best_cols,
            },
            best_history,
        )

    @staticmethod
    def evaluate_model(
        X, y, estimator, param_distributions, fit_params, cv, n_jobs, random_state
    ):
        """Перебирает сетку параметров и выбирает лучший результат по OOF-метрике.

        Все комбинации оцениваются на одинаковых фолдах. Для каждой комбинации
        сохраняются обученные пайплайны и результаты кросс-валидации.

        Args:
            X: DataFrame с признаками.
            y: Series с целевой переменной.
            estimator: Пайплайн предобработки и модели.
            param_distributions: Сетка параметров для ParameterGrid.
                Каждое значение должно быть последовательностью вариантов.
            fit_params: Фиксированные параметры пайплайна.
            cv: Количество фолдов.
            n_jobs: Число параллельно проверяемых комбинаций.
            random_state: Seed для разбиения данных и обучения.

        Returns:
            tuple: Таблица экспериментов, лучшая OOF-метрика, описание лучшего
                результата, пара (лучший ансамбль, разбиения) для SHAP
                и история обучения лучшей конфигурации для нейросети.
        """
        kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
        cv_folds = list(kf.split(X, y))

        all_experiments_params = list(ParameterGrid(param_distributions))
        print(f"Начинаем обучение {len(all_experiments_params)} комбинаций")

        raw_results = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(Trainer.cross_validate)(
                estimator, params, fit_params, X, y, cv_folds, random_state
            )
            for params in all_experiments_params
        )

        results_df, best_ensemble, best_oof_score, best_results, best_history = (
            Trainer.create_result_table(cv, param_distributions, raw_results)
        )

        return (
            results_df,
            best_oof_score,
            best_results,
            (best_ensemble, cv_folds),
            best_history,
        )  # Это для нейроночки

    @staticmethod
    def searh_params(
        X,
        y,
        estimator,
        param_distributions,
        fit_params,
        cv,
        random_state,
        n_trials,
        experiment_name,
    ):
        """Подбирает параметры через Optuna на фиксированных фолдах.

        Каждый trial формирует параметры, запускает кросс-валидацию и возвращает
        OOF-метрику.
        Args:
            X: DataFrame с признаками.
            y: Series с целевой переменной.
            estimator: Пайплайн предобработки и модели.
            param_distributions: Функция, принимающая trial и возвращающая
                словарь параметров пайплайна.
            fit_params: Фиксированные параметры пайплайна.
            cv: Количество фолдов.
            random_state: Seed для разбиений, обучения и sampler.
            n_trials: Максимальное число экспериментов Optuna.
            experiment_name: Название эксперимента MLflow.

        Returns:
            tuple: Таблица экспериментов, лучшая OOF-метрика, описание лучшего
                результата, пара (лучший ансамбль, разбиения) для SHAP
                и история обучения лучшей конфигурации для нейросети.
        """
        mlflow.set_experiment(experiment_name)

        kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
        cv_folds = list(kf.split(X, y))

        result_compilation = []

        params_func = param_distributions

        def objective(trial):
            sampled_params = params_func(trial)
            param_distributions = {**sampled_params}

            ensembles, param_results = Trainer.cross_validate(
                estimator, param_distributions, fit_params, X, y, cv_folds, random_state
            )
            result_compilation.append((ensembles, param_results))

            oof_rmse = param_results["oof_rmse"]

            return oof_rmse

        study = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_state)
        )

        study.optimize(objective, n_trials=n_trials, n_jobs=1)

        best_value = study.best_value
        print(f"Лучший скор: {best_value}")

        best_params = study.best_params
        print(f"Лучшие параметры: {best_params}")

        results_df, best_ensemble, best_oof_score, best_results, best_history = (
            Trainer.create_result_table(cv, best_params, result_compilation)
        )

        return (
            results_df,
            best_oof_score,
            best_results,
            (best_ensemble, cv_folds),
            best_history,
        )

    @staticmethod
    def feature_selection(
        X,
        y,
        cv,
        feature_importance,
        estimator,
        params,
        fit_params,
        model_name,
        n_jobs,
        random_state,
    ):
        """Сравнивает наборы признаков, последовательно исключая наименее важные.

        Использует готовый рейтинг важности и проверяет несколько размеров
        набора на одинаковых фолдах. Гиперпараметры модели при этом фиксированы.
        Метод не пересчитывает важности после каждого сокращения.

        Args:
            X: DataFrame с исходными признаками.
            y: Series с целевой переменной.
            cv: Готовый список разбиений, а не количество фолдов.
            feature_importance: Таблица с колонками feature и importance.
            estimator: Пайплайн с шагом feature_transform.
            params: Зафиксированные гиперпараметры модели.
            fit_params: Дополнительные фиксированные параметры пайплайна.
            model_name: Название модели для специальных настроек категорий.
            n_jobs: Число параллельно проверяемых наборов.
            random_state: Seed обучения.

        Returns:
            tuple: Таблица проверенных наборов, лучший ансамбль, его OOF-метрика,
                описание лучшего результата и история обучения для нейросети.
        """

        all_params = []

        best_oof = float("inf")
        best_ensemble = None
        selection_results = []

        # Оъеденяем ванхот фичи и приводим значения наименований к изначальному(только если в пайплайне есть ванхот энкодер)
        if "encoder" in estimator.named_steps:
            feature_importance["feature"] = feature_importance["feature"].str.replace(
                r"^one_hot__(.*?)_.*$", r"\1", regex=True
            )
            feature_importance["feature"] = feature_importance[
                "feature"
            ].str.removeprefix("remainder__")
            feature_importance = (
                feature_importance.groupby("feature")["importance"]
                .sum()
                .reset_index()
                .sort_values("importance", ascending=False)
            )

        n_total = len(feature_importance)
        step = max(1, int(n_total * 0.1))

        # Создаем комбинаций параметров
        for k in range(step, int(n_total * 0.9), step):
            # Оставляем n - k наиболее важных признаков
            selected_features = (feature_importance.head(n_total - k))[
                "feature"
            ].tolist()
            # Обновляем категориальные признаки CatBoost
            if model_name == "catboost":
                selected_cat_features = [
                    col
                    for col in selected_features
                    if col in FeatureTransformer.cat_cols
                ]

            fit_params_ = deepcopy(fit_params)

            if model_name == "catboost":
                fit_params_["model__cat_features"] = selected_cat_features

            fit_params_["feature_transform__selected_columns"] = selected_features

            all_params.append(fit_params_)

        raw_results = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(Trainer.cross_validate)(
                estimator, params, fit_params_, X, y, cv, random_state
            )
            for fit_params_ in all_params
        )

        results_df, best_ensemble, best_oof_score, best_results, best_history = (
            Trainer.create_result_table(len(cv), params, raw_results)
        )

        return results_df, best_ensemble, best_oof_score, best_results, best_history
