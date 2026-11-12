import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split

import src.modeling.train as train
import src.utils.drawings as drawer
from config import Config
from src.feature.feature_selection import count_off_shap
from src.feature.features import FeatureTransformer


def setup_validation(path_to_data=Config.path_train_data, test_size=0.2):
    data = pd.read_csv(path_to_data)

    # Prepare data
    y_train = np.log1p(data["SalePrice"])
    data = data.drop("SalePrice", axis=1)

    X_train, X_holdout, y_train, y_holdout = train_test_split(
        data,
        y_train,
        test_size=test_size,  # 10% данных уйдет в тест, 90% — в train
        random_state=Config.seed,  # фиксирует случайность для воспроизводимости
        shuffle=True,  # перемешивать данные перед разделением (по умолчанию True)
    )

    return X_train, X_holdout, y_train, y_holdout


def run_experiment(X, y, model_name, model_config, logger):
    """Запускает перебор параметров модели и сохраняет результаты в MLflow.

    Возвращает словарь с идентификатором запуска, таблицей экспериментов,
    лучшим ансамблем, метриками, разбиениями для SHAP и историей обучения.
    """

    config = model_config[model_name]

    with mlflow.start_run(run_name=model_name) as run:
        run_id = run.info.run_id

        # Обучение
        results_df, oof_rmse, best_results, for_shap, best_history = (
            train.Trainer.evaluate_model(
                X=X,
                y=y,
                estimator=config["pipeline"],
                param_distributions=config["parameters"],
                fit_params=config["fit_param"],
                cv=config["cv"],
                n_jobs=config["n_jobs"],
                random_state=Config.seed,
            )
        )
        pipelines, cv_folds = for_shap
        # Графики
        double_fig = drawer.Drawer.draw_graphics(
            results_df, oof_rmse, model_config, best_results, model_name
        )
        if best_history:
            curves_fig = drawer.Drawer.draw_lerning_curves(best_history)
            logger.log_learning_curve(curves_fig, "learning curve", run_id)
            # display(curves_fig)

        # Отображение результатов
        # display(results_df)
        # print("Best results:", best_results)
        # display(figures)

        # MLflow
        mlflow.log_table(results_df, "cv_results.json")
        mlflow.log_params(best_results["best_params"])
        logger.log_fig(double_fig, "heatmap + plot", run_id)

        mlflow.log_metrics(
            {
                "Best OOF": oof_rmse,
                "best_cv_rmse_std": best_results["best_val_std"],
                "n_candidates": len(results_df),
            }
        )

        mlflow.log_params(best_results["best_params"])

        return {
            "run_id": run_id,
            "results_df": results_df,
            "best_value": oof_rmse,
            "best_ensemble": pipelines,
            "for_shap": for_shap,
            "best_results": best_results,
            "best_history": best_history,
            "cv_folds": cv_folds,
        }


def run_optuna_experiment(X, y, model_name, model_configs, logger):
    """Запускает подбор параметров через Optuna и логирует результаты в MLflow.

    Возвращает словарь с идентификатором запуска, таблицей экспериментов,
    лучшим ансамблем, метриками, разбиениями для SHAP и историей обучения.
    """
    config = model_configs[model_name]

    with mlflow.start_run(run_name=model_name) as run:
        run_id = run.info.run_id

        results_df, oof_rmse, best_results, for_shap, best_history = (
            train.Trainer.searh_params(
                X=X,
                y=y,
                estimator=config["pipeline"],
                param_distributions=config["parameters_to_search"],
                fit_params=config["fit_param"],
                cv=config["cv"],
                random_state=Config.seed,
                n_trials=config["n_iter"],
                experiment_name=Config.experiments_name,
            )
        )
        pipelines, cv_folds = for_shap

        double_fig = drawer.Drawer.draw_graphics(
            results_df, oof_rmse, model_configs, best_results, model_name
        )

        if best_history:
            curves_fig = drawer.Drawer.draw_lerning_curves(best_history)
            logger.log_learning_curve(curves_fig, "learning curve", run_id)
            # display(curves_fig)

        # Вывод результатов
        # display(results_df)
        # print("Best value:", oof_rmse)
        # print("Best results:", best_results)
        # display(best_history)
        # display(double_fig)

        # Логирование в MLflow
        mlflow.log_table(results_df, "cv_results.json")
        mlflow.log_params(best_results["best_params"])

        logger.log_fig(double_fig, "heatmap + plot", run_id)

        mlflow.log_metrics(
            {
                "Best OOF": oof_rmse,
                "best_cv_rmse_std": best_results["best_val_std"],
                "n_candidates": len(results_df),
            }
        )

        return {
            "run_id": run_id,
            "results_df": results_df,
            "best_value": oof_rmse,
            "best_ensemble": pipelines,
            "for_shap": for_shap,
            "best_results": best_results,
            "best_history": best_history,
            "cv_folds": cv_folds,
        }


def run_shap(X, experiment_results, mode, logger, background_size=50, n_permutations=5):

    pipelines, cv_folds = experiment_results["for_shap"]
    model_run_id = experiment_results["run_id"]

    # Продолжаем существующий MLflow run
    with mlflow.start_run(run_id=model_run_id):
        importance, oof_explanation = count_off_shap(
            pipelines,
            mode,
            X,
            model_run_id,
            cv_folds,
            background_size=background_size,
            n_permutations=n_permutations,
            seed=Config.seed,
        )

        figures = drawer.Drawer.draw_shap_plots(oof_explanation)

        # Вывод результатов
        # display(importance)

        # for figure in figures.values():
        #     display(figure)

        # Логирование в MLflow
        mlflow.log_table(importance, "feature_importance.json")
        logger.log_shap_plots(figures)

        return {
            "importance": importance,
            "oof_explanation": oof_explanation,
            "figures": figures,
        }


def run_feature_selection(
    X,
    y,
    cv_folds,
    importance,
    search_results,
    model_name,
    model_configs,
    model_run_id,
    logger,
):
    config = model_configs[model_name]

    with mlflow.start_run(run_id=model_run_id):
        results_df, best_ensemble, best_oof_score, fs_best_results, best_history = (
            train.Trainer.feature_selection(
                X,
                y,
                cv_folds,
                importance,
                config["pipeline"],
                search_results["best_params"],
                config["fit_param"],
                model_name,
                config["n_jobs"],
                Config.seed,
            )
        )

        cleaned_features = clean_features(fs_best_results["best_cols"])
        fs_best_results["best_cols"] = cleaned_features
        # Вывод результатов
        print(
            f"Best OOF: {best_oof_score}, "
            f"best mean val RMSE: {fs_best_results['best_mean_val_score']}, "
            f"best val std: {fs_best_results['best_val_std']}, "
            f"best feature amount: {len(cleaned_features)}"
        )

        print("Best searched params".center(30, "-"))

        best_results_df = pd.DataFrame([fs_best_results["best_params"]])
        print(best_results_df.T.to_string(header=False))

        display(results_df.sort_values("off_score", ascending=True))
        # display(best_history)

        # График
        fig, ax = plt.subplots()

        ax.plot(results_df["feature_count"], results_df["off_score"])

        ax.set_xlabel("Feature count")
        ax.set_ylabel("OOF RMSE")
        ax.set_title(f"{model_name}: feature selection")

        # display(fig)

        # Логирование в MLflow
        mlflow.log_table(results_df, "feature_selection_results.json")

        mlflow.log_table(best_results_df, "feature_selection_best_params.json")

        mlflow.log_metrics(
            {
                "fs_best_oof_score": float(best_oof_score),
                "fs_best_mean_val_rmse": float(fs_best_results["best_mean_val_score"]),
                "fs_best_val_std": float(fs_best_results["best_val_std"]),
                "fs_best_feature_count": len(fs_best_results["best_cols"]),
            }
        )

        logger.log_fig(fig, "feature_selection", model_run_id)

        return {
            "results_df": results_df,
            "best_ensemble": best_ensemble,
            "best_oof_score": best_oof_score,
            "best_results": fs_best_results,
            "best_history": best_history,
            "figures": fig,
        }


def clean_features(features):
    cat_cols = sorted(FeatureTransformer.cat_cols, key=len, reverse=True)

    cleaned = []

    for feature in features:
        if feature.startswith("one_hot__"):
            name = feature.removeprefix("one_hot__")
            feature = next(
                (col for col in cat_cols if name.startswith(col + "_")), name
            )
        else:
            feature = feature.removeprefix("remainder__")

        cleaned.append(feature)

    return list(dict.fromkeys(cleaned))
