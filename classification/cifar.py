"""
Training script for CIFAR-10/100
Copyright (c) Wei YANG, 2017
"""

from __future__ import print_function

import argparse
import os
import json
import shutil
import time
import random
from tqdm import tqdm
from .loss.distillation import Distillation
from .loss.losses import *
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.cifar as models
from models.cifar.resnet_vision import BasicBlock, Bottleneck

from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('--data_dir', default='./data', type=str,
                    help='path of directory to store data')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('--reset', action='store_true',
                    help='clear all files to reset experiment')
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--block-name', type=str, default='BasicBlock',
                    help='the building block for Resnet and Preresnet: BasicBlock, Bottleneck '
                         '(default: Basicblock for cifar10 / cifar100)')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
# Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
# Teacher Model setting
parser.add_argument('--teacher', type=str, default='False', choices=('True', 'False'),
                    help='use teacher model')
parser.add_argument('--teacher_arch', type=str, default='resnet50', choices=model_names,
                    help='teacher model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
parser.add_argument('--teacher_depth', type=int, default=50, help='Teacher Model Depth')
parser.add_argument('--teacher_block_name', type=str, default='BasicBlock',
                    help='the building block for Resnet and Preresnet: BasicBlock, Bottleneck '
                         '(default: BasicBlock for cifar10 / cifar100)')
parser.add_argument('--teacher_cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--teacher_widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--teacher_growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--teacher_compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
parser.add_argument('--teacher_checkpoint', default='', type=str, metavar='PATH',
                    help='path to teacher checkpoint (default: none)')
parser.add_argument('--mutual_learning', type=bool, default=False,
                    help='Co-train teacher model from scratch')
# Mimic setting
parser.add_argument('--mimic_loss', type=str, default='attention',
                    choices=('at_loss', 'ft_loss', 'nst_loss', 'mmd_loss'),
                    help='means of feature mimicking')
parser.add_argument('--mimic_position', type=int, nargs='+',
                    help='which positions to be mimicked')
parser.add_argument('--temperature', type=float, default=4,
                    help='Temperature for soft labels')

parser.add_argument('--task_theta', type=float, default=1,
                    help='weight of main task loss (classification)')
parser.add_argument('--kd_theta', type=float, default=0,
                    help='weight of ')
parser.add_argument('--mimic_theta', type=float, default=0,
                    help='weight of total mimic loss')
parser.add_argument('--mimic_lambda', type=float, nargs='+',
                    help='importance term for each loss term')

args = parser.parse_args()
# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()
# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)

random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
# fsp loss requires feature pairs
assert len(args.mimic_position) % len(args.mimic_lambda) == 0

state = {k: v for k, v in args._get_kwargs()}
print('Input Arguments: \n{}'.format(json.dumps(state, indent=4)))
best_acc = 0        # best test accuracy
t_best_acc = 0      # best test accuracy of teacher


def main():
    global best_acc, t_best_acc, writer
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)
    if not args.evaluate and args.reset:
        # remove previous files if reset and not test
        file_list = os.listdir(args.checkpoint)
        for item in file_list:
            # except logger
            if os.path.splitext(item)[-1] != '.log':
                os.system('rm -rf ' + os.path.join(args.checkpoint, item))
    # open tensorboard writer
    writer = SummaryWriter(args.checkpoint)
    # Data
    print('==> Preparing dataset {}'.format(args.dataset))
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100

    trainset = dataloader(root=args.data_dir, train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

    testset = dataloader(root=args.data_dir, train=False, download=True, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    print('==> Train: {} batches/epoch; Test: {} batches'.format(len(trainloader), len(testloader)))

    # Model
    print("==> creating model '{}{}'".format(args.arch, args.depth))
    if args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
                    cardinality=args.cardinality,
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('densenet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    growthRate=args.growthRate,
                    compressionRate=args.compressionRate,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('wrn'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.endswith('resnet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    block_name=args.block_name,
                )
    elif args.arch.endswith('vision'):
        layers = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3],
                  101: [3, 4, 23, 3], 152: [3, 8, 36, 3]}
        if args.block_name == 'BasicBlock':
            block = BasicBlock
        elif args.block_name == 'Bottleneck':
            block = Bottleneck
        else:
            raise ValueError('block name should be either Basicblock or Bottleneck')
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    block=block,
                    layers=layers[args.depth],
        )
    else:
        model = models.__dict__[args.arch](num_classes=num_classes)

    model = torch.nn.DataParallel(model).cuda()
    print('Model Architecture: {}'.format(model))
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    criterion = nn.CrossEntropyLoss()
    loss_info = 'Total loss = {} * Cross Entropy'.format(args.task_theta)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Teacher Model: teacher model, teacher_crit, teacher_opt, teacher_acc
    if eval(args.teacher):
        print("==> creating Teacher Model '{}{}'".format(args.teacher_arch, args.teacher_depth))
        if args.teacher_arch.startswith('resnext'):
            teacher = models.__dict__[args.teacher_arch](
                cardinality=args.teacher_cardinality,
                num_classes=num_classes,
                depth=args.teacher_depth,
                widen_factor=args.teacher_widen_factor,
                dropRate=args.drop,
            )
        elif args.teacher_arch.startswith('densenet'):
            teacher = models.__dict__[args.teacher_arch](
                num_classes=num_classes,
                depth=args.teacher_depth,
                growthRate=args.teacher_growthRate,
                compressionRate=args.teacher_compressionRate,
                dropRate=args.drop,
            )
        elif args.teacher_arch.startswith('wrn'):
            teacher = models.__dict__[args.teacher_arch](
                num_classes=num_classes,
                depth=args.teacher_depth,
                widen_factor=args.teacher_widen_factor,
                dropRate=args.drop,
            )
        elif args.teacher_arch.endswith('resnet'):
            teacher = models.__dict__[args.teacher_arch](
                num_classes=num_classes,
                depth=args.teacher_depth,
                block_name=args.teacher_block_name,
            )
        elif args.teacher_arch.endswith('vision'):
            layers = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3],
                      101: [3, 4, 23, 3], 152: [3, 8, 36, 3]}
            if args.block_name == 'BasicBlock':
                block = BasicBlock
            elif args.block_name == 'Bottleneck':
                block = Bottleneck
            else:
                raise ValueError('block name should be either Basicblock or Bottleneck')
            teacher = models.__dict__[args.teacher_arch](
                num_classes=num_classes,
                block=block,
                layers=layers[args.teacher_depth],
            )
        else:
            teacher = models.__dict__[args.teacher_arch](num_classes=num_classes)
        teacher = torch.nn.DataParallel(teacher).cuda()
        print('Teacher Model Architecture: {}'.format(teacher))
        # teacher checkpoint
        if os.path.isfile(args.teacher_checkpoint):
            print('==> Loading Teacher checkpoint at {}...'.format(args.teacher_checkpoint))
            checkpoint = torch.load(args.teacher_checkpoint)
            t_best_acc = checkpoint['best_acc']
            teacher.load_state_dict(checkpoint['state_dict'], strict=True)
            teacher_opt = None
            args.mutual_learning = False
        else:
            print('==> No checkpoint for teacher found...')
            t_best_acc = None
            teacher_opt = optim.SGD(teacher.parameters(), lr=args.lr,
                                    momentum=args.momentum, weight_decay=args.weight_decay)
            args.mutual_learning = True

        print('    Total params of main model: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
        print('    Total params of Teacher model: %.2fM' % (sum(p.numel() for p in teacher.parameters()) / 1000000.0))
        # mimic_crit = Distillation(supervision=args.mimic_mean, function=args.mimic_function, normalize=args.normalize)
        mimic_crit = eval(args.mimic_loss) if len(args.mimic_loss) > 0 and args.mimic_theta > 0 else None
        loss_info += ' + {} * kd_loss + {} * {}'.format(args.kd_theta, args.mimic_theta, args.mimic_loss)
    else:
        teacher = None
        t_best_acc = None
        mimic_crit = None
        teacher_opt = None

    print('    Loss info: [ {} ]'.format(loss_info))
    # Resume
    if eval(args.teacher):
        if args.mutual_learning is True:
            title = '{}-[s]{}{}[M]'.format(args.dataset, args.arch, args.depth)
        else:
            title = '{}-[s]{}{}'.format(args.dataset, args.arch, args.depth)
    else:
        title = '{}-{}{}'.format(args.dataset, args.arch, args.depth)
    if args.resume:
        # Load checkpoint & logger
        print('==> Resuming from checkpoint at {}...'.format(args.resume))
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        # start new logger
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Error Rate', 'Valid Error Rate'])

    # Only eval student(model)
    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc1, test_acc5 = test(testloader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  {:.8f}, Test Acc:  {:.3f}[{:.3f}]'.format(test_loss, test_acc1, test_acc5))
        return

    # Train and val: T == teacher, S == student
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        print('\nEpoch: [{} | {}] | LR: {:.5f} '.format(epoch + 1, args.epochs, state['lr']))

        # TO DO: Train teacher model
        """
        if teacher and teacher_opt:
            t_train_loss, t_train_acc1, t_train_acc5 = train(trainloader, teacher, criterion,
                                                             teacher_opt, epoch, use_cuda, name='T')
            t_test_loss, t_test_acc1, t_test_acc5 = test(testloader, teacher, criterion, epoch, use_cuda, name='T')
            print('[T] Loss: [train: {:.4f} | test: {:.4f}] Acc(1[5]): [train: {:.3f}[{:.3f}] | test: {:.3f}[{:.3f}]]'
                  .format(t_train_loss, t_test_loss, t_train_acc1, t_train_acc5, t_test_acc1, t_test_acc5))

            # write tensorboard
            writer.add_scalar('Train_epoch/Teacher Loss', t_train_loss, epoch + 1)
            writer.add_scalar('Train_epoch/Teacher Error Rate', 100 - t_train_acc1, epoch + 1)
            writer.add_scalar('Train_epoch/Teacher Top1 Accuracy', t_train_acc1, epoch + 1)
            writer.add_scalar('Train_epoch/Teacher Top5 Accuracy', t_train_acc5, epoch + 1)
            writer.add_scalar('Test_epoch/Teacher Loss', t_test_loss, epoch + 1)
            writer.add_scalar('Test_epoch/Teacher Error Rate', 100 - t_test_acc1, epoch + 1)
            writer.add_scalar('Test_epoch/Teacher Top1 Accuracy', t_test_acc1, epoch + 1)
            writer.add_scalar('Test_epoch/Teacher Top5 Accuracy', t_test_acc5, epoch + 1)

            # save teacher model
            t_is_best = t_test_acc1 > t_best_acc
            t_test_acc1 = max(t_test_acc1, t_best_acc)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': teacher.state_dict(),
                'acc': t_test_acc1,
                'best_acc': t_best_acc,
                'optimizer': teacher_opt.state_dict(),
            }, t_is_best, checkpoint=args.checkpoint)
        """

        train_loss, train_acc1, train_acc5 = train(trainloader, model, criterion, optimizer, epoch, use_cuda, name='S',
                                                   teacher=teacher, teacher_opt=teacher_opt, mimic_criterion=mimic_crit)
        test_loss, test_acc1, test_acc5 = test(testloader, model, criterion, epoch, use_cuda, name='S', teacher=teacher)

        # append logger file
        print('[S] Loss: [train: {:.4f} | test: {:.4f}] Acc(1[5]): [train: {:.3f}[{:.3f}] | test: {:.3f}[{:.3f}]]'.
              format(train_loss, test_loss, train_acc1, train_acc5, test_acc1, test_acc5))

        logger.append([state['lr'], train_loss, test_loss, 100 - train_acc1, 100 - test_acc1])

        # write tensorboard
        writer.add_scalar('Train_epoch/LR', state['lr'], epoch + 1)
        writer.add_scalar('Train_epoch/Loss', train_loss, epoch + 1)
        writer.add_scalar('Train_epoch/Error Rate', 100 - train_acc1, epoch + 1)
        writer.add_scalar('Train_epoch/Top1 Accuracy', train_acc1, epoch + 1)
        writer.add_scalar('Train_epoch/Top5 Accuracy', train_acc5, epoch + 1)
        writer.add_scalar('Test_epoch/Loss', test_loss, epoch + 1)
        writer.add_scalar('Test_epoch/Error Rate', 100 - test_acc1, epoch + 1)
        writer.add_scalar('Test_epoch/Top1 Accuracy', test_acc1, epoch + 1)
        writer.add_scalar('Test_epoch/Top5 Accuracy', test_acc5, epoch + 1)

        # save model
        is_best = test_acc1 > best_acc
        best_acc = max(test_acc1, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc1,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint)

    writer.close()
    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.pdf'))

    print('\n{}{}[{}] | Best acc:'.format(args.arch, args.depth, args.dataset))
    print(best_acc)


def train(trainloader, model, criterion, optimizer, epoch, use_cuda, name='S',
          teacher=None, teacher_opt=None, mimic_criterion=None):
    # switch to train mode
    model.train()
    teacher.train()

    end = time.time()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # Average Meters for teacher
    t_losses = AverageMeter()
    t_top1 = AverageMeter()
    t_top5 = AverageMeter()

    tbar = tqdm(trainloader, ncols=80)
    # Consider Task Loss & KD Loss & Mimic Loss
    for batch_idx, (inputs, targets) in enumerate(tbar):
        # measure data loading time
        data_time.update(time.time() - end)
        curr_batch = epoch * len(trainloader) + batch_idx + 1

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(async=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # model output
        outputs, features = model(inputs)
        # Task Loss
        task_loss = criterion(outputs, targets)
        effective_task_loss = args.task_theta * task_loss

        # output from teacher
        if teacher and teacher_opt:
            t_outputs, t_features = teacher(inputs)
            t_loss = criterion(t_outputs, targets)

            # measure accuracy and record loss
            t_prec1, t_prec5 = accuracy(t_outputs.data, targets.data, topk=(1, 5))
            t_losses.update(t_loss.item(), inputs.size(0))
            t_top1.update(t_prec1.item(), inputs.size(0))
            t_top5.update(t_prec5.item(), inputs.size(0))

            # compute gradient and do SGD step
            teacher_opt.zero_grad()
            t_loss.backward()
            teacher_opt.step()

            writer.add_scalar('Train_batch_batch/Teacher_Loss', t_loss.item(), curr_batch)
        elif teacher and teacher_opt is None:
            # teacher.eval()
            with torch.no_grad():
                t_outputs, t_features = teacher(inputs)
        else:
            t_outputs, t_features = None, None

        # KD Loss
        if t_outputs is not None and args.kd_theta > 0:
            soft_loss = kd_loss(outputs, t_outputs, args)
            effective_soft_loss = args.kd_theta * soft_loss
        else:
            soft_loss = torch.tensor(0).float().to(inputs.device)
            effective_soft_loss = args.kd_theta * soft_loss

        # Mimic Loss
        if mimic_criterion and t_features and len(args.mimic_position) > 0:
            mimic_losses = mimic_criterion(features, t_features, args)
            mimic_loss = sum(mimic_losses)
            effective_mimic_loss = args.mimic_theta * mimic_loss
        else:
            mimic_loss = torch.tensor(0).float().to(inputs.device)
            effective_mimic_loss = args.mimic_theta * mimic_loss

        total_loss = effective_task_loss + effective_soft_loss + effective_mimic_loss

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(task_loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot tensorboard;
        writer.add_scalar('Train_batch/Loss', task_loss.item(), curr_batch)
        writer.add_scalar('Train_batch/Total Loss', total_loss.item(), curr_batch)
        writer.add_scalar('Train_batch/Error Rate', 100 - top1.val, curr_batch)
        writer.add_scalar('Train_batch/Top1 Accuracy', top1.val, curr_batch)
        writer.add_scalar('Train_batch/Top5 Accuracy', top5.val, curr_batch)
        # distillation; each term of distillation loss
        writer.add_scalar('Train_batch_distill/task_loss*{}'.format(args.task_theta),
                          effective_task_loss.item(), curr_batch)
        writer.add_scalar('Train_batch_distill/kd_loss*{}'.format(args.kd_theta),
                          effective_soft_loss.item(), curr_batch)
        writer.add_scalar('Train_batch_distill/{}*{}'.format(args.mimic_loss, args.mimic_theta),
                          effective_mimic_loss.item(), curr_batch)
        # plot progress
        tbar.set_description('[{}]DT:{data:.3f}s|BT:{bt:.3f}s|tL:{loss:.4f}|kL:{kloss:.4f}'
                             'mL: {mloss:.4f}|top1:{top1:.4f}|top5:{top5:.4f}'.
                             format(name, data=data_time.avg, bt=batch_time.avg, loss=effective_task_loss.item(),
                                    kloss=effective_soft_loss.item(), mloss=effective_mimic_loss.item(),
                                    top1=top1.avg, top5=top5.avg))
    return losses.avg, top1.avg, top5.avg


def test(testloader, model, criterion, epoch, use_cuda, name='S', teacher=None):
    global best_acc, t_best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    tbar = tqdm(testloader, ncols=80)
    for batch_idx, (inputs, targets) in enumerate(tbar):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        with torch.no_grad():
            outputs, _ = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        tbar.set_description('[{name}]DT:{data:.3f}s|BT:{bt:.3f}s|Loss:{loss:.4f}|top1:{top1:.4f}|top5:{top5:.4f}'.
                             format(name=name, data=data_time.avg, bt=batch_time.avg,
                                    loss=losses.avg, top1=top1.avg, top5=top5.avg
                                    ))
    return losses.avg, top1.avg, top5.avg


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']


if __name__ == '__main__':
    main()
