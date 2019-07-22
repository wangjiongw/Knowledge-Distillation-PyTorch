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


def at_loss(student, teacher, args=None):
    """
    Attention Transfer Loss
    :param student:
    :param teacher:
    :param args:
    :return:
    """
    losses = []
    assert len(args.mimic_position) == len(args.mimic_lambda)
    for i, p in enumerate(args.mimic_position):
        s = student[p]
        t = teacher[p]
        n, c, h, w = s.shape
        _n, _c, _h, _w = t.shape
        assert h == _h and w == _w
        loss = (attention(s) - attention(t)).pow(2).mean()
        losses.append(args.mimic_lambda[i] * loss)
    return losses


def fm_loss(student, teacher, args=None):
    """
    Feature Mimic Loss
    :param student:
    :param teacher:
    :param args:
    :return:
    """
    losses = []
    assert len(args.mimic_position) == len(args.mimic_lambda)
    for i, p in enumerate(args.mimic_position):
        s = student[p]
        t = teacher[p]
        loss = (s - t).pow(2).mean()
        losses.append(args.mimic_lambda[i] * loss)
    return losses


def nst_loss(student, teacher, args=None):
    """
    Neural Selectivity Transfer Loss (paper)
    :param student:
    :param teacher:
    :param args:
    :return:
    """
    losses = []
    assert len(args.mimic_position) == len(args.mimic_lambda)
    for i, p in enumerate(args.mimic_position):
        s = student[p]
        t = teacher[p]
        s = F.normalize(s.view(s.shape[0], s.shape[1], -1), dim=1)            # N, C, H * W
        gram_s = s.transpose(1, 2).bmm(s)                                     # H * W, H * W
        assert gram_s.shape[1] == gram_s.shape[2], print("gram_student's shape: {}".format(gram_s.shape))

        t = F.normalize(t.view(t.shape[0], t.shape[1], -1), dim=1)
        gram_t = t.transpose(1, 2).bmm(t)
        assert gram_t.shape[1] == gram_t.shape[2], print("gram_teacher's shape: {}".format(gram_t.shape))
        loss = (gram_s - gram_t).pow(2).mean()
        losses.append(args.mimic_lambda[i] * loss)
    return losses


def mmd_loss(student, teacher, args=None):
    """
    Maximum Mean Discrepancy Loss (NST Project)
    :param student:
    :param teacher:
    :param args:
    :return:
    """
    losses = []
    assert len(args.mimic_position) == len(args.mimic_lambda)
    for i, p in enumerate(args.mimic_position):
        s = student[p]
        t = teacher[p]
        s = F.normalize(s.view(s.shape[0], s.shape[1], -1), dim=1)              # N, C, H * W
        mmd_s = s.bmm(s.transpose(2, 1))                                        # N, C, C
        mmd_s_mean = mmd_s.pow(2).mean()
        t = F.normalize(t.view(t.shape[0], t.shape[1], -1), dim=1)
        mmd_st = s.bmm(t.transpose(2, 1))
        mmd_st_mean = mmd_st.pow(2).mean()
        loss = mmd_s_mean * 2 * mmd_st_mean
        losses.append(args.mimic_lambda[i] * loss)
    return losses


def similarity_loss(student, teacher, args=None):
    """
    Similarity Transfer Loss
    :param student:
    :param teacher:
    :param args:
    :return:
    """
    return


def fsp_loss(student, teacher, args=None):
    """
    Flow of Solving Problem Loss
    :param student:
    :param teacher:
    :param args:
    :return:
    """
    losses = []
    assert len(args.mimic_position) == 2 * len(args.mimic_lambda)
    for i in range(len(args.mimic_theta)):
        s1 = student[args.mimic_position[2 * i]]
        s2 = student[args.mimic_position[2 * i + 1]]
        t1 = teacher[args.mimic_position[2 * i]]
        t2 = teacher[args.mimic_position[2 * i + 1]]
        s1 = s1.view(s1.shape[0], s1.shape[1], -1)                          # N, C1, H * W
        s2 = s2.view(s2.shape[0], s2.shape[1], -1)                          # N, C2, H * W
        fsp_s = s1.bmm(s2.transpose(1, 2)) / s1.shape[2]                    # N, C1, C2

        t1 = t1.view(t1.shape[0], t1.shape[1], -1)
        t2 = t2.view(t2.shape[0], t2.shape[1], -1)
        fsp_t = t1.bmm(t2.transpose(1, 2)) / t1.shape[2]                    # N, C1, C2
        loss = (fsp_s - fsp_t).pow(2).mean()
        losses.append(args.mimic_lambda[i] * loss)
    return losses
