from src.core_funcs import *
from src.utils.model_saving import ModelSaver
from src.modeling.predict import Predictor
from config import Config


if __name__ == '__main__':
    saved_models = ModelSaver.upload_saved_models(Config.path_to_save_models)
    Predictor.run_inference(saved_models,Config.path_test_data,Config.path_to_submission)
    print("Закончил")