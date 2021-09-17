import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import pytorch_lightning as pl
import torchmetrics
from torchmetrics.classification import accuracy
from torchmetrics.classification.f_beta import F1


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
        self.train_acc = torchmetrics.Accuracy()
        self.train_precision = torchmetrics.Precision()
        self.train_recall = torchmetrics.Recall()
        self.train_f1 = torchmetrics.F1()

        self.valid_acc = torchmetrics.Accuracy()
        self.valid_precision = torchmetrics.Precision()
        self.valid_recall = torchmetrics.Recall()
        self.valid_f1 = torchmetrics.F1()

        self.test_acc = torchmetrics.Accuracy()
        self.test_precision = torchmetrics.Precision()
        self.test_recall = torchmetrics.Recall()
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
        acc = self.train_acc(outputs, labels)
        precision = self.train_precision(outputs, labels)
        recall = self.train_recall(outputs, labels)
        f1_score = self.train_f1(outputs, labels)

        self.log("train loss", loss)
        self.log("train performance", {
            "train_acc": acc,
            "train_precision": precision,
            "train_recall": recall,
            "train_f1": f1_score
        })
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.forward(inputs)
        
        loss = self.cross_entropy_loss(outputs, labels)
        acc = self.valid_acc(outputs, labels)
        precision = self.valid_precision(outputs, labels)
        recall = self.valid_recall(outputs, labels)
        f1_score = self.valid_f1(outputs, labels)

        self.log("valid loss", loss)
        self.log("valid performance", {
            "valid_acc": acc,
            "valid_precision": precision,
            "valid_recall": recall,
            "valid_f1": f1_score
        })
        
        return loss

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.forward(inputs)

        loss = self.cross_entropy_loss(outputs, labels)
        acc = self.test_acc(outputs, labels)
        precision = self.test_precision(outputs, labels)
        recall = self.test_recall(outputs, labels)
        f1_score = self.test_f1(outputs, labels)
        
        self.log("test loss", loss)
        self.log("test performance", {
            "test_acc": acc,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1_score
        })

        return loss

    def configure_optimizers(self):
        # return optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optim.SGD(self.parameters(), lr=self.hparams.learning_rate, momentum=self.hparams.momentum)

