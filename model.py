import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch import Tensor
import pytorch_lightning as pl
import torchmetrics
from torchmetrics.classification.f_beta import F1

from loguru import logger
from torchmetrics import Accuracy, MetricCollection, Precision, Recall, F1


class LiNet(pl.LightningModule):
    def __init__(self, learning_rate: float, momentum: float, batch_size: int, dropout_rate: float):
        super().__init__()
        # This saves all constructor arguments as items in the hparams dictionary
        self.save_hyperparameters()

        # model architecture
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate, inplace=False)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        # metrics

        # self.accuracy = torchmetrics.Accuracy(num_classes=10)
        # self.precision_ = torchmetrics.Precision(average='micro', num_classes=10)
        # self.recall_ = torchmetrics.Recall(average='micro', num_classes=10)
        # self.f1_ = torchmetrics.F1(average='micro', num_classes=10, multiclass=True)

        # https://githubmemory.com/repo/PyTorchLightning/metrics/issues/298
        self.average = 'weighted'

        metrics = MetricCollection(
            [
                Accuracy(),
                Precision(num_classes=10, multiclass=True, average=self.average),
                Recall(num_classes=10, multiclass=True, average=self.average),
                F1(num_classes=10, multiclass=True, average=self.average)
            ])
        self.train_metrics = metrics.clone(prefix='train/')
        self.valid_metrics = metrics.clone(prefix='valid/')
        self.test_metrics = metrics.clone(prefix='test/')

    def forward(self, x: Tensor) -> Tensor:
        x = self.dropout(self.pool(F.relu(self.conv1(x))))
        x = self.dropout(self.pool(F.relu(self.conv2(x))))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    @staticmethod
    def cross_entropy_loss(outputs, labels):
        return F.cross_entropy(outputs, labels)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.forward(inputs)
        loss = self.cross_entropy_loss(outputs, labels)

        preds = torch.argmax(outputs, dim=1)
        output = self.train_metrics(preds, labels)
        # use log_dict instead of log
        # metrics are logged with keys: train_Accuracy, train_Precision and train_Recall
        self.log_dict(output)
        self.log("train/loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.forward(inputs)
        loss = self.cross_entropy_loss(outputs, labels)

        preds = torch.argmax(outputs, dim=1)
        output = self.valid_metrics(preds, labels)
        # use log_dict instead of log
        # metrics are logged with keys: val_Accuracy, val_Precision and val_Recall
        self.log_dict(output)

        self.log("valid/loss", loss)

        return loss

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.forward(inputs)
        loss = self.cross_entropy_loss(outputs, labels)

        preds = torch.argmax(outputs, dim=1)
        output = self.test_metrics(preds, labels)
        # use log_dict instead of log
        # metrics are logged with keys: train_Accuracy, train_Precision and train_Recall
        self.log_dict(output)
        self.log("test/loss", loss)

        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
