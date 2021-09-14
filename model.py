import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import pytorch_lightning as pl


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LitNet(pl.LightningModule):
    def __init__(self, learning_rate: float=0.001, momentum: float=0.9):
        super().__init__()
        self.save_hyperparameters()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    """This was in some tutorial code, not sure if we need"""
    #TODO: Look at what this is for
    def split_batch(self, batch):
        return batch[0]["data"], batch[0]["label"].squeeze().long()

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        inputs, labels = batch
        # inputs, labels = self.split_batch(batch)
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        # inputs, labels = self.split_batch(batch)
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, labels)
        self.log("valid loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        # inputs, labels = self.split_batch(batch)
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, labels)
        self.log("test loss", loss)
        return loss

    def configure_optimizers(self):
        # return optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optim.SGD(self.parameters(), lr=self.hparams.learning_rate, momentum=self.hparams.momentum)
