import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import pytorch_lightning as pl
import torchmetrics
from torchmetrics.classification import accuracy
from torchmetrics.classification.f_beta import F1


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
        self.precision = torchmetrics.Precision(num_classes=10)
        self.recall = torchmetrics.Recall(num_classes=10)
        self.f1 = torchmetrics.F1(num_classes=10)

        self.train_acc = torchmetrics.Accuracy(num_classes=10)
        self.train_precision = torchmetrics.Precision(num_classes=10)
        self.train_recall = torchmetrics.Recall(num_classes=10)
        self.train_f1 = torchmetrics.F1(num_classes=10)

        self.valid_acc = torchmetrics.Accuracy(num_classes=10)
        self.valid_precision = torchmetrics.Precision(num_classes=10)
        self.valid_recall = torchmetrics.Recall(num_classes=10)
        self.valid_f1 = torchmetrics.F1(num_classes=10)

        self.test_acc = torchmetrics.Accuracy()
        self.test_precision = torchmetrics.Precision(num_classes=10)
        self.test_recall = torchmetrics.Recall(num_classes=10)
        self.test_f1 = torchmetrics.F1(num_classes=10)

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
        # self.cross_entropy_loss(outputs, labels)
        self.log("train loss", loss, on_step=True, on_epoch=True)

        self.log("train performance", {
            "train_acc": self.accuracy(outputs, labels),
            "train_precision": self.precision(outputs, labels),
            "train_recall": self.recall(outputs, labels),
            "train_f1": self.f1(outputs, labels)
        }, on_step=True, on_epoch=True)
        return loss

    
    def training_epoch_end(self, outs):
        # log epoch metric
        self.log('train_acc_epoch', self.accuracy.compute())
        self.log('train_precision_epoch', self.precision.compute())
        self.log('train_recall_epoch', self.recall.compute())
        self.log('train_f1_epoch', self.f1.compute())

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.forward(inputs)
        
        loss = self.cross_entropy_loss(outputs, labels)
        
        self.valid_acc(outputs, labels)
        self.valid_precision(outputs, labels)
        self.valid_recall(outputs, labels)
        self.valid_f1(outputs, labels)

        self.log("valid loss", loss, on_step=True, on_epoch=True)
        self.log("valid performance", {
            "valid_acc": self.valid_acc,
            "valid_precision": self.valid_precision,
            "valid_recall": self.valid_recall,
            "valid_f1": self.valid_f1
        }, on_step=True, on_epoch=True)
        
        return loss


    def validation_epoch_end(self, outs):
        # log epoch metric
        self.log('train_acc_epoch', self.accuracy.compute())
        self.log('train_precision_epoch', self.precision.compute())
        self.log('train_recall_epoch', self.recall.compute())
        self.log('train_f1_epoch', self.f1.compute())

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.forward(inputs)

        loss = self.cross_entropy_loss(outputs, labels)
        
        self.test_acc(outputs, labels)
        self.test_precision(outputs, labels)
        self.test_recall(outputs, labels)
        self.test_f1(outputs, labels)
        
        self.log("test loss", loss, on_step=True, on_epoch=True)
        self.log("test performance", {
            "test_acc": self.test_acc,
            "test_precision": self.test_precision,
            "test_recall": self.test_recall,
            "test_f1": self.test_f1
        }, on_step=True, on_epoch=True)

        return loss

    def test_epoch_end(self, outputs):
        return super().test_epoch_end(outputs)

    def configure_optimizers(self):
        # return optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optim.SGD(self.parameters(), lr=self.hparams.learning_rate, momentum=self.hparams.momentum)

