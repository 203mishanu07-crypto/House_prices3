from config import Config
from src.core_funcs import *
from src.modeling.predict import Predictor
from src.utils.model_saving import ModelSaver

if __name__ == "__main__":
    saved_models = ModelSaver.upload_saved_models(Config.path_to_save_models)
    Predictor.run_inference(
        saved_models, Config.path_test_data, Config.path_to_submission
    )
    print("Закончил")
