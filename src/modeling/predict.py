from pathlib import Path

import numpy as np
import pandas as pd

from config import Config


class Predictor:
    @staticmethod
    def run_test(pipelines, X_holdout, y_holdout):
        """Оценивает сохранённые ансамбли на отложенной выборке.

        В регрессии усредняет числовые прогнозы фолдовых моделей.

        Args:
            pipelines: Словарь ансамблей с обученными объектами в поле models.
            X_holdout: Признаки отложенной выборки.
            y_holdout: Таргет в той же шкале и порядке строк, что при обучении.

        Returns:
            DataFrame: Названия ансамблей и значения holdout-метрики.
        """
        scores = {}

        for model_name, fold_pipelines in pipelines.items():
            fold_pipelines = fold_pipelines["models"]
            pred_log = np.mean(
                [pipeline.predict(X_holdout) for pipeline in fold_pipelines],
                axis=0,
            ).reshape(-1)

            rmse = np.sqrt(np.mean((np.asarray(y_holdout).reshape(-1) - pred_log) ** 2))
            scores[model_name] = rmse

        results = (
            pd.DataFrame(scores.items(), columns=["model", "holdout_log_rmse"])
            .sort_values("holdout_log_rmse")
            .reset_index(drop=True)
        )

        print(results.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
        return results

    @staticmethod
    def run_inference(
        pipelines,
        path_to_test=Config.path_test_data,
        output_dir=Config.path_to_submission,
    ):
        """Создаёт и сохраняет отдельный файл предсказаний для каждого ансамбля.

        Загружает тестовые данные и объединяет прогнозы фолдовых моделей.
        Преобразование выхода модели и формат CSV определяются задачей:
        для регрессии восстанавливается нужная шкала таргета.

        Args:
            pipelines: Словарь ансамблей с обученными объектами в поле models.
            path_to_test: Путь к CSV с тестовыми признаками.
            output_dir: Папка для файлов submission_<название>.csv.

        Notes:
            Результат работы метода — сохранённые CSV-файлы.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        data_test = pd.read_csv(path_to_test)

        predictions = {}

        for model_name, fold_pipelines in pipelines.items():
            fold_pipelines = fold_pipelines["models"]
            pred_log = np.mean(
                [pipeline.predict(data_test) for pipeline in fold_pipelines],
                axis=0,
            ).reshape(-1)

            pred_price = np.expm1(pred_log)
            predictions[model_name] = pred_price

            submission = pd.DataFrame(
                {
                    "Id": data_test["Id"].to_numpy(),
                    "SalePrice": pred_price,
                }
            )

            path = output_dir / f"submission_{model_name}.csv"
            submission.to_csv(path, index=False)

            print(f"{model_name}: {len(submission)} строк → {path}")

        return predictions
