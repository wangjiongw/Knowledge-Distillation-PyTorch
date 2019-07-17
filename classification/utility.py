import torch
import torch.nn as nn
import torch.nn.functional as F


def attention(x):
    return F.normalize(x.pow(2).mean(1).view(x.shape[0], -1))


def at_loss(x, y):
    return (attention(x) - attention(y)).pow(2).mean()

