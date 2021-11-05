from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader

# Note - you must have torchvision installed for this example
from torchvision.datasets import CIFAR10
from torchvision import transforms


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, split_ratio: float, num_workers: int):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.split_ratio = split_ratio
        self.num_workers = num_workers
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()

    def prepare_data(self):
        # download
        CIFAR10(root=self.data_dir, train=True, download=True)
        CIFAR10(root=self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            cifar10_full = CIFAR10(self.data_dir, train=True, transform=self.transform)
            # train_size = int(cifar10_full.data.shape[0] * self.split_ratio)
            train_size = 5
            # val_size = cifar10_full.data.shape[0] - train_size
            val_size = 5
            self.cifar10_train, self.cifar10_val, _ = random_split(cifar10_full, [train_size, val_size, cifar10_full.data.shape[0] - train_size - val_size])
            self.dims = tuple(self.cifar10_train[0][0].shape)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.cifar10_test = CIFAR10(self.data_dir, train=False, transform=self.transform)
            self.dims = tuple(self.cifar10_test[0][0].shape)

    def train_dataloader(self):
        return DataLoader(self.cifar10_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.cifar10_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.cifar10_test, batch_size=self.batch_size, num_workers=self.num_workers)
