import torch
import torch.nn as nn
import torch.nn.functional as F


# https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
# Total loss = kd_loss * alpha + CELoss * (1 - alpha)
def kd_loss(outputs, teacher_outputs, args):
    """
    Knowledge Distillation Loss
    :param outputs:
    :param teacher_outputs:
    :param args:
    :return:
    """
    T = args.temperature
    kd_loss = nn.KLDivLoss()(F.log_softmax(outputs / T, dim=1), F.softmax(teacher_outputs / T, dim=1)) * (T * T)
    return kd_loss


# attention transfer loss
def at_loss(student_feature, teacher_feature):
    """
    Attention Transfer Loss
    :param student_feature:
    :param teacher_feature:
    :return:
    """
    n, c, h, w = student_feature.shape
    _n, _c, _h, _w = teacher_feature.shape
    assert h == _h and w == _w
    student_at = F.normalize(student_feature.pow(pow).mean(1).view(n, -1))
    teacher_at = F.normalize(teacher_feature.pow(pow).mean(1).view(n, -1))
    loss = (student_at - teacher_at).pow(2).mean()
    return loss


def nst_loss(student_feature, teacher_feature):
    """
    Neural Selectivity Transfer Loss
    :param student_feature:
    :param teacher_feature:
    :return:
    """


def similarity_loss(student_feature, teacher_feature):
    """
    Similarity Transfer Loss
    :param student_feature:
    :param teacher_feature:
    :return:
    """


def fsp_loss(student_features, teacher_features):
    """
    Flow of Solution Procedure Loss
    :param student_features:
    :param teacher_features:
    :return:
    """