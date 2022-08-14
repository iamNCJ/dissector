import numpy as np
import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl


class NPYDataModule(pl.LightningDataModule):
    def __init__(self, image_npy_file, label_npy_file, batch_size: int = 32, num_workers: int = 4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.images = torch.from_numpy(np.load(image_npy_file))
        self.labels = torch.from_numpy(np.load(label_npy_file))
        self.datasets = torch.utils.data.TensorDataset(self.images, self.labels)

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(self.datasets, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.datasets, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.datasets, batch_size=self.batch_size, num_workers=self.num_workers)
