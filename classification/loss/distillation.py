import logging

# from utility import vis_feat
# from loss import adversarial
# from loss import discriminator

import torch
import torch.nn as nn
import torch.nn.functional as F

# logger = logging.getLogger('SR')


def input2feature(x, normalize=-1, pow=1):
    return x


def input2attention(x, normalize=-1, pow=2):
    """
    :param x:
    :param normalize: axis to normalize along with
    :param pow: order of power
    :return:
    """
    # follows Attention Transfer
    if normalize < 0:
        return x.pow(pow).mean(1).view(x.size(0), -1)
    else:
        return F.normalize(x.pow(pow).mean(1).view(x.size(0), -1))


def gram(x, norm=-1):
    """
    :param x:
    :param norm: normalization axis
    :return:
    """
    n, c, h, w = x.size()
    x = x.view(n, c, -1)
    if norm > 0:
        x = F.normalize(x, dim=norm)
    return x.transpose(1, 2).bmm(x)


def input2gram(x, normalize=2, power=1):
    return gram(x.pow(power), norm=normalize)


def input2similarity(x, normalize=1, power=1):
    return gram(x.pow(power), norm=normalize)


def L1Loss(x, y):
    return torch.abs(x - y).mean()


def L2Loss(x, y):
    return (x - y).pow(2).mean()


class Distillation(nn.Module):
    def __init__(self, supervision='attention', function='L2', normalize=False):
        super(Distillation, self).__init__()
        if supervision == 'feature':
            self.process = input2feature
        elif supervision == 'attention':
            self.process = input2attention
        elif supervision == 'gram':
            self.process = input2gram
        elif supervision =='similarity':
            self.process = input2similarity
        else:
            logger.info('Supervision type [{}] is not implemented'.format(args.distill_supervision))

        self.norm = normalize

        if function == 'L1':
            self.function = nn.L1Loss()
            # self.function = L1Loss
        elif function == 'L2':
            # self.function = nn.MSELoss()
            self.function = L2Loss
        else:
            logger.info('Choose L1, L2 loss, rather than {}'.format(function))

    def forward(self, student, teacher, assistant=None, writer=None, batch=None):
        """
        :param student: dict of student feature to distillate
        :param teacher: dict of teacher feature to distillate
        :param assistant: dict of assistant feature to distillate
        :param label: label in tensorboard
        :param writer: tensorboard writer
        :return:
        """
        losses = list()
        # k: feature position; fs: student feature
        for k, fs in student.items():
            if fs is not None:
                ft = teacher[k] if teacher is not None else None
                fa = assistant[k] if assistant is not None else None
                if ft is None:
                    # logger.info('teacher feature {} is None'.format(k))
                    continue
                # assistant provides feature residual
                fs = fs + fa if fa is not None else fs
                # logger.info('{} vs {}'.format(fs.shape, ft.shape))
                # map input features to supervision
                fs = self.process(fs, normalize=self.norm) if fs is not None else None
                ft = self.process(ft, normalize=self.norm) if ft is not None else None
                # fa = self.process(fa) if fa else None
                loss = self.function(fs, ft)
                losses.append(loss)
                if writer and batch % 10 == 0:
                    name = 'Mimic Loss feat{}'.format(k)
                    writer.add_scalar('Distill_loss_batch/{}'.format(name), loss, batch + 1)
            else:
                # logger.info('{}th feature of student: {}'.format(k, fs))
                continue
        return torch.sum(torch.stack(losses, dim=0)) if len(losses) > 0 else torch.tensor(0).float().cuda()
