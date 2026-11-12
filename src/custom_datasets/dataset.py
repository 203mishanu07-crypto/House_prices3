import torch
from torch.utils.data import DataLoader, Dataset


class TableDataset(Dataset):
    def __init__(self, X_cat, X_num, y):
        self.X_cat = torch.tensor(X_cat.to_numpy(), dtype=torch.long)

        self.X_num = torch.tensor(X_num.to_numpy(), dtype=torch.float32)

        self.y = torch.tensor(y, dtype=torch.float32).reshape(-1)

    def __len__(self):
        # Возвращает общее количество строк в таблице
        return len(self.X_num)

    def __getitem__(self, idx):
        # Возвращает одну строку (признаки и таргет) по индексу
        return self.X_num[idx], self.X_cat[idx], self.y[idx]


def transform_data(
    X_train_cat,
    X_train_num,
    X_val_cat,
    X_val_num,
    y_train,
    y_val,
    cat_cols,
    num_cols,
    batch_size,
):
    # Создаем датасет и даталоадер
    train_data = TableDataset(X_cat=X_train_cat, X_num=X_train_num, y=y_train.values)
    val_data = TableDataset(X_cat=X_val_cat, X_num=X_val_num, y=y_val.values)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader
