from pathlib import Path
import torch as th
import pytorch_lightning as pl
import pandas as pd
#from sklearn import preprocessing
from torch.utils.data import TensorDataset, DataLoader, random_split


class PlDataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super(PlDataModule, self).__init__()
        self.hparams = hparams
        self.train_dataset = None
        self.valid_dataset = None

    def load_data(self):
        input_file = Path(self.hparams.data_folder).joinpath(self.hparams.train_input).as_posix()
        df = pd.read_csv(input_file)
        # commented code for normalization options. minmax or mean-std approach
        #min_max_scaler = preprocessing.MinMaxScaler()
        #scaled_input = min_max_scaler.fit_transform(df.values)
        #normalized_df = (df-df.mean())/df.std()
        df = pd.DataFrame(df.values)
        x = df.loc[:, :9].values
        y = df.loc[:, 10:].values
        return x, y

    def setup(self, stage=None):
        x, y = self.load_data()
        dataset = TensorDataset(
            th.tensor(x, dtype=th.float), th.tensor(y, dtype=th.float))
        train_size = int(0.90 * len(dataset))
        val_size = int(0.1 * len(dataset))
        self.train_dataset, self.valid_dataset = random_split(dataset, (train_size, val_size))

    def train_dataloader(self):
        dataloader = DataLoader(self.train_dataset,
                                batch_size=self.hparams.batch_size,
                                shuffle=True,
                                num_workers=self.hparams.num_workers,
                                pin_memory=True,
                                drop_last=True)
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(self.valid_dataset,
                                batch_size=self.hparams.batch_size,
                                num_workers=self.hparams.num_workers,
                                pin_memory=True,
                                drop_last=True)
        return dataloader
