from pathlib import Path

import torch


class Config:
    """Config."""

    # Seed
    seed = 42

    experiment_uri = "http://127.0.0.1:5000"
    experiments_name = "House_Prices_Kaggle"

    ROOT = Path(__file__).resolve().parent
    
    path_train_data = ROOT / "data/raw/train.csv"
    path_test_data = ROOT / "data/raw/test.csv"
    path_to_save_models = ROOT / "Models"
    path_to_submission = ROOT / "data/submission"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    cv = 5
