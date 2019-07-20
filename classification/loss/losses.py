import torch
import torch.nn as nn
import torch.nn.functional as F


# https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
# Total loss = kd_loss * alpha + CELoss * (1 - alpha)
def kd_loss(s, t, args=None):
    """
    Knowledge Distillation Loss
    :param s:
    :param t:
    :param args:
    :return:
    """
    T = args.temperature if args and hasattr(args, 'temperature') else 4
    kd_loss = nn.KLDivLoss()(F.log_softmax(s / T, dim=1), F.softmax(t / T, dim=1)) * (T * T)
    return kd_loss


# attention transfer loss
def attention(x):
    return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))


def at_loss(s, t, args=None):
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


def nst_loss(s, t, args=None):
    """
    Neural Selectivity Transfer Loss (paper)
    :param s:
    :param t:
    :param args:
    :return:
    """
    s = F.normalize(s.view(s.shape[0], s.shape[1], -1), dim=1)            # N, C, H * W
    gram_s = s.transpose(1, 2).bmm(s)                                     # H * W, H * W
    assert gram_s.shape[1] == gram_s.shape[2], print("gram_student's shape: {}".format(gram_s.shape))

    t = F.normalize(t.view(t.shape[0], t.shape[1], -1), dim=1)
    gram_t = t.transpose(1, 2).bmm(t)
    assert gram_t.shape[1] == gram_t.shape[2], print("gram_teacher's shape: {}".format(gram_t.shape))
    loss = (gram_s - gram_t).pow(2).mean()
    return loss


def mmd_loss(s, t, args=None):
    """
    Maximum Mean Discrepancy Loss (NST Project)
    :param s:
    :param t:
    :return:
    """
    s = F.normalize(s.view(s.shape[0], s.shape[1], -1), dim=1)              # N, C, H * W
    mmd_s = s.bmm(s.transpose(2, 1))                                        # N, C, C
    mmd_s_mean = mmd_s.pow(2).mean()
    t = F.normalize(t.view(t.shape[0], t.shape[1], -1), dim=1)
    mmd_st = s.bmm(t.transpose(2, 1))
    mmd_st_mean = mmd_st.pow(2).mean()
    return mmd_s_mean * 2 * mmd_st_mean


def similarity_loss(s, t, args):
    """
    Similarity Transfer Loss
    :param s:
    :param t:
    :param args:
    :return:
    """
    return


def fsp_loss(s1, s2, t1, t2):
    """
    Flow of Solving Problem Loss
    :param s1: F1 from student
    :param s2: F2 from student
    :param t1: F1 from teacher
    :param t2: F2 from teacher
    :return:
    """
    s1 = s1.view(s1.shape[0], s1.shape[1], -1)                          # N, C1, H * W
    s2 = s2.view(s2.shape[0], s2.shape[1], -1)                          # N, C2, H * W
    fsp_s = s1.bmm(s2.transpose(1, 2)) / s1.shape[2]                    # N, C1, C2

    t1 = t1.view(t1.shape[0], t1.shape[1], -1)
    t2 = t2.view(t2.shape[0], t2.shape[1], -1)
    fsp_t = t1.bmm(t2.transpose(1, 2)) / t1.shape[2]                    # N, C1, C2
    loss = (fsp_s - fsp_t).pow(2).mean()
    return loss
