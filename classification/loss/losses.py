import torch
import torch.nn as nn
import torch.nn.functional as F


# https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
# Total loss = kd_loss * alpha + CELoss * (1 - alpha)
def kd_loss(s, t, args):
    """
    Knowledge Distillation Loss
    :param s:
    :param t:
    :param args:
    :return:
    """
    T = args.temperature
    kd_loss = nn.KLDivLoss()(F.log_softmax(s / T, dim=1), F.softmax(t / T, dim=1)) * (T * T)
    return kd_loss


# attention transfer loss
def attention(x):
    return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))


def at_loss(s, t, args):
    """
    Attention Transfer Loss
    :param s:
    :param t:
    :param args:
    :return:
    """
    n, c, h, w = s.shape
    _n, _c, _h, _w = t.shape
    assert h == _h and w == _w
    loss = (attention(s) - attention(t)).pow(2).mean()
    return loss


def nst_loss(s, t):
    """
    Neural Selectivity Transfer Loss
    :param s:
    :param t:
    :return:
    """
    n, c, h, w = s.shape
    s = F.normalize(s.view(n, c, -1), dim=1)           # N, C, H*W
    gram_s = s.transpose(1, 2).bmm(s)
    assert s.shape[2] == s.shape[3]
    t = F.normalize(t.view(n, c, -1), dim=1)
    gram_t = t.transpose(1, 2).bmm(t)
    loss = ()



def similarity_loss(s, t):
    """
    Similarity Transfer Loss
    :param s:
    :param t:
    :return:
    """


def fsp_loss(s, t):
    """
    FSP matrix loss
    :param s:
    :param t:
    :return:
    """