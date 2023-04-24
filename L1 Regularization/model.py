import pytorch_lightning as pl
import torchvision
import torch
import torch.nn.functional as F


class Resnet(pl.LightningModule):
    def __init__(self, l1_strength=0.0):
        super().__init__()
        self.save_hyperparameters()
        self.model = torchvision.models.resnet18()
        self.output = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Linear(1000, 100))
        self.l1_strength = l1_strength

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

        if self.l1_strength > 0:
            l1_loss = torch.tensor(0.0, requires_grad=True).to(self.device)
            for param in self.parameters():
                l1_loss += torch.norm(param, 1)
            loss += self.l1_strength * l1_loss

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

        if self.l1_strength > 0:
            l1_loss = torch.tensor(0.0, requires_grad=True).to(self.device)
            for param in self.parameters():
                l1_loss += torch.norm(param, 1)
            loss += self.l1_strength * l1_loss

        _, predicted = torch.max(y_hat, 1)
        accuracy = torch.sum(predicted == y).item() / len(y)

        self.log('val_loss', loss, on_step=True, on_epoch=True)
        self.log('val_accuracy', accuracy, on_step=True, on_epoch=True)
