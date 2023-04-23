# %%
import pytorch_lightning as pl
import torch
import torchvision
from torch.utils.data import DataLoader
import numpy as np
from torchvision import transforms
import torch.utils.data as data
from model import Resnet
from pytorch_lightning.loggers import WandbLogger



# %%
np.random.seed(123)

# Data Preprocessing
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                         std=[0.2675, 0.2565, 0.2761])
])

tempset = torchvision.datasets.CIFAR100(
    root='./data', train=True, download=True, transform=transform)
trainset, valset = data.random_split(tempset, [40000, 10000], generator=torch.Generator().manual_seed(0))

train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

valid_loader = torch.utils.data.DataLoader(
    valset, batch_size=128, shuffle=True, num_workers=2)

test_set = torchvision.datasets.CIFAR100(
    root='./data', train=False, download=False, transform=transform)

test_dataloader = torch.utils.data.DataLoader(
    test_set, batch_size=256, shuffle=False, num_workers=2)

# %%

wandb_logger = WandbLogger(project="Baseline", name="Data Preprocessing")
model = Resnet()
train = pl.Trainer(logger=wandb_logger, max_epochs=300, log_every_n_steps=1)
train.fit(model,train_loader, valid_loader)