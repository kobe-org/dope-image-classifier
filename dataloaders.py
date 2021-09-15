from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader

# Note - you must have torchvision installed for this example
from torchvision.datasets import CIFAR10
from torchvision import transforms


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        """
        The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. 
        There are 50000 training images and 10000 test images. 
        The dataset is divided into five training batches and one test batch, each with 10000 images.
        """
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        self.dims = (1, 32, 32, 3)

    def prepare_data(self):
        # download
        CIFAR10(root=self.data_dir, train=True, download=True)
        CIFAR10(root=self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            cifar10_full = CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.cifar10_train, self.cifar10_val = random_split(cifar10_full, [55000, 5000])

            # Optionally...
            # self.dims = tuple(self.cifar10_train[0][0].shape)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.cifar10_test = CIFAR10(self.data_dir, train=False, transform=self.transform)

            # Optionally...
            # self.dims = tuple(self.mnist_test[0][0].shape)

    def train_dataloader(self):
        return DataLoader(self.cifar10_train, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.cifar10_val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.cifar10_test, batch_size=32)
