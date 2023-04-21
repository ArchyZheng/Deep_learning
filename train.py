import pytorch_lightning as pl
import torch
import torchvision
from torch.utils.data import DataLoader
import numpy as np
from torchvision import transforms
import torch.utils.data as data
from model import Resnet
# %%
np.random.seed(123)
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
tempset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)

trainset, valset = data.random_split(tempset, [40000, 10000], generator=torch.Generator().manual_seed(0))

train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

valid_loader = torch.utils.data.DataLoader(
    valset, batch_size=128, shuffle=True, num_workers=2)

test_set = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=False, transform=transform)

# we can use a larger batch size during test, because we do not save
# intermediate variables for gradient computation, which leaves more memory
test_dataloader = torch.utils.data.DataLoader(
    test_set, batch_size=256, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
# %%
train = pl.Trainer(max_epochs=150)
model = Resnet()
train.fit(model=model, train_dataloaders=train_loader)
