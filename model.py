import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import pytorch_lightning as pl
import torchmetrics
from torchmetrics.classification import accuracy
from torchmetrics.classification.f_beta import F1

from loguru import logger


class LitNet(pl.LightningModule):
    def __init__(self, learning_rate: float, momentum: float):
        super().__init__()
        # This saves all constructor arguments as items in the hparams dictionary
        self.save_hyperparameters()
        
        # model architecture
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        # metrics
    
        self.accuracy = torchmetrics.Accuracy(num_classes=10)
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    @staticmethod
    def cross_entropy_loss(outputs, labels):
        return F.cross_entropy(outputs, labels)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        inputs, labels = batch
        outputs = self.forward(inputs)
        
        loss = self.cross_entropy_loss(outputs, labels)
        
        preds = torch.argmax(outputs, dim=1)
        acc = self.accuracy(preds, labels)

        # print(f'Outputs:\n{outputs}\nLoss:\n{loss}\nY_hat:\n{preds}\nLabels:{labels}\nacc:{acc}')
        
        self.log("train loss", loss, on_step=True, on_epoch=True)
        self.log("train acc", acc, on_step=True, on_epoch=True)

        return loss

    """
    def training_epoch_end(self, outs):
        # log epoch metric
        self.log('train_acc_epoch', self.accuracy.compute())
    """

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.forward(inputs)
        loss = self.cross_entropy_loss(outputs, labels)
        
        preds = torch.argmax(outputs, dim=1)
        acc = self.accuracy(preds, labels)

        self.log("valid loss", loss, on_step=True, on_epoch=True)
        self.log("valid acc", acc, on_step=True, on_epoch=True)
        
        return loss
    
    """
    def validation_epoch_end(self, outs):
        # log epoch metric
        self.log('valid_acc_epoch', self.accuracy.compute())
        # pass
    """

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.forward(inputs)
        loss = self.cross_entropy_loss(outputs, labels)
        
        preds = torch.argmax(outputs, dim=1)
        acc = self.accuracy(preds, labels)

        self.log("test loss", loss, on_step=True, on_epoch=True)
        self.log("test acc", acc, on_step=True, on_epoch=True)

        return loss

    """
    def test_epoch_end(self, outputs):
        self.log('test_acc_epoch', self.accuracy.compute())
        # return super().test_epoch_end(outputs)
        # pass
    """

    def configure_optimizers(self):
        # return optim.SGD(self.parameters(), lr=self.hparams.learning_rate, momentum=self.hparams.momentum)
        return optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
