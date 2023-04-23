import pytorch_lightning as pl
import torchvision
import torch
import torch.nn.functional as F
from mixup import mixup_data, mixup_criterion

class Resnet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.model = torchvision.models.resnet18()
        self.output = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Linear(1000, 100))

    def forward(self, x):
        prediction = self.output(self.model(x))
        return prediction

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-2, momentum=0.9)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_hat = self.model(x)
        y_hat = self.output(y_hat)
        loss_function = torch.nn.CrossEntropyLoss()
        label = F.one_hot(y, num_classes=100).float()
        loss = loss_function(y_hat, label)

        _, predicted = torch.max(y_hat, 1)
        accuracy = torch.sum(predicted == y).item() / len(y)

        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('train_accuracy', accuracy, on_step=True, on_epoch=True)
        
        return loss

    def validation_step(self, val_batch, val_idx):
        x, y = val_batch
        y_hat = self.model(x)
        y_hat = self.output(y_hat)
        loss_function = torch.nn.CrossEntropyLoss()
        label = F.one_hot(y, num_classes=100).float()
        loss = loss_function(y_hat, label)

        _, predicted = torch.max(y_hat, 1)
        accuracy = torch.sum(predicted == y).item() / len(y)

        self.log('val_loss', loss, on_step=True, on_epoch=True)
        self.log('val_accuracy', accuracy, on_step=True, on_epoch=True)