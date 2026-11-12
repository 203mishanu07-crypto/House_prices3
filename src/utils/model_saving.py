import json
import shutil
from pathlib import Path

import joblib

from config import Config


class ModelSaver:
    @staticmethod
    def save_ensemble(
        ensemble, model_name, save_dir=Config.path_to_save_models, save_fs=None
    ):
        """Сохраняет обученные модели ансамбля и необязательный список признаков.

        Каждая модель записывается отдельным файлом fold_<номер>.pkl.
        Существующая папка model_name удаляется перед сохранением.

        Args:
            ensemble: Последовательность обученных объектов моделей или
                пайплайнов. Словарь результатов эксперимента сюда не передаётся.
            model_name: Название подпапки ансамбля.
            save_dir: Корневая папка сохранения.
            save_fs: Необязательный список признаков для записи в JSON.
        """
        path = Path(save_dir) / model_name

        if path.exists():
            shutil.rmtree(path)

        path.mkdir(parents=True, exist_ok=True)

        for fold_idx, pipeline in enumerate(ensemble):
            joblib.dump(pipeline, path / f"fold_{fold_idx}.pkl")

        if save_fs:
            with open(path / f"{model_name}_features.json", "w", encoding="utf-8") as f:
                json.dump(save_fs, f, ensure_ascii=False, indent=4)

    @staticmethod
    def upload_saved_models(
        models_dir=Config.path_to_save_models, apply_saved_features=True
    ):
        """Загружает сохранённые ансамбли из подпапок models_dir.

        Файлы fold_*.pkl загружаются в порядке номеров фолдов.
        При наличии JSON со списком признаков он добавляется в результат
        как метаданные.

        Args:
            models_dir: Корневая папка с сохранёнными ансамблями.
            apply_saved_features: В текущей реализации не используется;
                сохранённые признаки не применяются повторно.

        Returns:
            dict: Словарь по названиям ансамблей. Поле models содержит список
                загруженных объектов, feature — сохранённый список признаков.
        """
        models_dir = Path(models_dir)

        models = {}

        for model_dir in sorted(models_dir.iterdir()):
            if not model_dir.is_dir():
                continue

            model_name = model_dir.name
            print(f"\n{'=' * 40}")
            print(f"Model: {model_name}")

            model_paths = sorted(
                model_dir.glob("fold_*.pkl"), key=lambda p: int(p.stem.split("_")[-1])
            )
            if model_paths:
                print(f"Найдены модели")
            else:
                continue

            pipelines = []

            for path in model_paths:
                try:
                    pipeline = joblib.load(path)
                except EOFError as exc:
                    raise RuntimeError(
                        f"Не удалось прочитать модель: {path}\n"
                        f"Размер файла: {path.stat().st_size} байт. "
                        "Вероятно, сохранение не завершилось."
                    ) from exc

                pipelines.append(pipeline)

            models[model_name] = {}
            models[model_name]["models"] = pipelines

            features_path = model_dir / f"{model_name}_features.json"
            selected_features = None

            if features_path.exists():
                with open(features_path, "r", encoding="utf-8") as f:
                    feature_data = json.load(f)

                selected_features = feature_data
                models[model_name]["feature"] = selected_features

                print(
                    f"Selected features: {len(selected_features) if selected_features is not None else 'unknown'}"
                )
            else:
                print("Selected features: all")
                models[model_name]["feature"] = []

        return models
