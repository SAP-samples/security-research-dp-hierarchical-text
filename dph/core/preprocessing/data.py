from pathlib import Path

import pandas as pd


class Dataset:

    def __init__(self, data_folder: str):

        self.data_folder = data_folder

        local_path = Path('./data', data_folder)
        efs_path = Path('/mnt/efs/DPTextHierarchy/datasets', data_folder)

        if local_path.exists():
            train_path = Path(local_path, "train.csv")
            val_path = Path(local_path, "val.csv")
            test_path = Path(local_path, "test.csv")

        elif efs_path.exists():
            train_path = Path(efs_path, "train.csv")
            val_path = Path(efs_path, "val.csv")
            test_path = Path(efs_path, "test.csv")

        else:
            raise ValueError(f'This dataset does not exist: {data_folder}')

        self.train = pd.read_csv(train_path)
        self.val = pd.read_csv(val_path)
        self.test = pd.read_csv(test_path)
