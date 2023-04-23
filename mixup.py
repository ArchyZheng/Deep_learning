import torch
import numpy as np

def mixup_data(x, y, alpha=0.2):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size) # permute within batch
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index] # y and permuted y
    return mixed_x, y_a, y_b, lam

def mixup_criterion(pred, y_a, y_b, lam):
    criterion = torch.nn.CrossEntropyLoss()
    return lam * criterion(pred, y_a)+(1 - lam)*criterion(pred,y_b)
