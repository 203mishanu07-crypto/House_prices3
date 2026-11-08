import json
from pathlib import Path
import mlflow
import numpy as np
import pandas as pd
import joblib
from config import Config


class Predictor:    
    @staticmethod
    def run_test(pipelines, X_holdout, y_holdout):
        scores = {}

        for model_name, fold_pipelines in pipelines.items():
            fold_pipelines = fold_pipelines['models']
            pred_log = np.mean(
                [pipeline.predict(X_holdout) for pipeline in fold_pipelines],
                axis=0,
            ).reshape(-1)

            rmse = np.sqrt(
                np.mean((np.asarray(y_holdout).reshape(-1) - pred_log) ** 2)
            )
            scores[model_name] = rmse

        results = (
            pd.DataFrame(scores.items(), columns=["model", "holdout_log_rmse"])
            .sort_values("holdout_log_rmse")
            .reset_index(drop=True)
        )

        print(results.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
        return results


    @staticmethod
    # Предсказания на Kaggle test и отдельный CSV для каждой модели
    def run_inference(pipelines, path_to_test = Config.path_test_data, output_dir=Config.path_to_submission):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        data_test = pd.read_csv(path_to_test)

        predictions = {}

        for model_name, fold_pipelines in pipelines.items():
            fold_pipelines = fold_pipelines['models']
            pred_log = np.mean(
                [pipeline.predict(data_test) for pipeline in fold_pipelines],
                axis=0,
            ).reshape(-1)

            pred_price = np.expm1(pred_log)
            predictions[model_name] = pred_price

            submission = pd.DataFrame({
                "Id": data_test["Id"].to_numpy(),
                "SalePrice": pred_price,
            })

            path = output_dir / f"submission_{model_name}.csv"
            submission.to_csv(path, index=False)

            print(f"{model_name}: {len(submission)} строк → {path}")

        return predictions