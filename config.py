from pathlib import Path
import torch

class Config:
    """Config."""
    # Seed
    seed = 42

    experiment_uri = "http://localhost:5000"
    experiments_name = "House_Prices_Kaggle"

    path_train_data = Path(r"C:\Users\user\3D Objects\code 2\Mentor_python\Block 2\Final_prodject\House-prices3\data\raw\train.csv")
    path_test_data = Path(r"C:\Users\user\3D Objects\code 2\Mentor_python\Block 2\Final_prodject\House-prices3\data\raw\test.csv")

    path_to_save_models = Path(r"C:\Users\user\3D Objects\code 2\Mentor_python\Block 2\Final_prodject\House-prices3\Models")

    path_to_submission = Path(r"C:\Users\user\3D Objects\code 2\Mentor_python\Block 2\Final_prodject\House-prices3\data\submission")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cv = 5