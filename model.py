import pytorch_lightning as pl
import numpy as np
from torchvision import transforms
import torch.utils.data as data
import torchvision
import torch
import torch.nn.functional as F


class Resnet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18()
        self.output = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Linear(1000, 10))

    def forward(self, x):
        prediction = self.output(self.model(x))
        return prediction

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_hat = self.model(x)
        y_hat = self.output(y_hat)
        loss_function = torch.nn.BCEWithLogitsLoss()
        label = F.one_hot(y, num_classes=10).float()
        loss = loss_function(y_hat, label)

        self.log('train_loss:', loss)
        return loss

    def validation_step(self, val_batch, val_idx):
        x, y = val_batch
        y_hat = self.model(x)
        y_hat = self.output(y_hat)
        loss_function = torch.nn.BCEWithLogitsLoss()
        loss = loss_function(y_hat, y)

        self.log('val_loss', loss)
