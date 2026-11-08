import joblib
from pathlib import Path
from config import Config
import json

class ModelSaver:
    @staticmethod
    def save_ensemble(ensemble, model_name, save_dir=Config.path_to_save_models, save_fs = None):
        path = Path(save_dir) / model_name
        path.mkdir(parents=True, exist_ok=True)

        for fold_idx, pipeline in enumerate(ensemble):
            joblib.dump(pipeline,path / f"fold_{fold_idx}.pkl")

        if save_fs:
            with open(path / f"{model_name}_features.json", "w", encoding="utf-8") as f:
                json.dump(save_fs, f, ensure_ascii=False, indent=4)


    @staticmethod
    def upload_saved_models(
        models_dir=Config.path_to_save_models,
        apply_saved_features=True
    ):
        models_dir = Path(models_dir)

        models = {}

        for model_dir in sorted(models_dir.iterdir()):
            if not model_dir.is_dir():
                continue
                
            model_name = model_dir.name
            print(f"\n{'=' * 40}")
            print(f"Model: {model_name}")

            models[model_name] = {}

            model_paths = sorted(
                model_dir.glob("fold_*.pkl"),
                key=lambda p: int(p.stem.split("_")[-1])
            )
            if model_paths:
                print(f"Найдены модели")
            else:
                continue
            
            pipelines = [joblib.load(path) for path in model_paths]

            models[model_name]['models'] = pipelines

            features_path = model_dir / f"{model_name}_features.json"
            selected_features = None

            if features_path.exists():
                with open(features_path, "r", encoding="utf-8") as f:
                    feature_data = json.load(f)

                selected_features = feature_data
                models[model_name]['feature'] = selected_features

                print(f"Selected features: {len(selected_features) if selected_features is not None else 'unknown'}")
            else:
                print("Selected features: all")
                models[model_name]['feature'] = []

        return models
