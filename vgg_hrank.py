import time
import queue
import torch
import argparse
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

from utils import *
from utils.utils import *
from model2.drives import *






import os
import sys
import shutil
import numpy as np
import time, datetime
import torch
import random
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data.distributed
from tqdm import tqdm

from mask import *
import utils
sys.path.append("../../")
from engine import *

from torchvision import datasets, transforms
from torch.autograd import Variable
import torchvision.models as models
from model.vgg16_hrank import *
from model.resnet56_hrank import *
from model.vgg_cifar_qt1 import *
from model.vgg_cifar_qt2 import *




parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# parser.add_argument(
#     '--data_dir',
#     type=str,
#     default='/home/fanxiaoyi/HRank-master/datasets',
#     help='dataset path')
# parser.add_argument(
#     '--dataset',
#     type=str,
#     default='cifar10',
#     choices=('cifar10','imagenet'),
#     help='dataset')
parser.add_argument(
    '--lr',
    default=0.01,
    type=float,
    help='initial learning rate')
parser.add_argument(
    '--lr_decay_step',
    default='5,10',
    type=str,
    help='learning rate decay step')
parser.add_argument(
    '--resume',
    type=str,
    default='./model2/ckpt/resnet56_UCML.t7',
    help='load the model from the specified checkpoint')
parser.add_argument(
    '--resume_mask',
    type=str,
    default=None,
    help='mask loading')
parser.add_argument(
    '--gpu',
    type=str,
    default='0',
    help='Select gpu to use')
parser.add_argument(
    '--job_dir',
    type=str,
    default='./result/tmp/',
    help='The directory where the summaries will be stored.')
parser.add_argument(
    '--epochs',
    type=int,
    default=15,
    help='The num of epochs to train.')
parser.add_argument(
    '--train_batch_size',
    type=int,
    default=16,
    help='Batch size for training.')
parser.add_argument(
    '--eval_batch_size',
    type=int,
    default=16,
    help='Batch size for validation.')
parser.add_argument(
    '--start_cov',
    type=int,
    default=0,
    help='The num of conv to start prune')
parser.add_argument(
    '--compress_rate',
    type=str,
    default='[0.50]*55',
    help='compress rate of each conv')
parser.add_argument(
    '--arch',
    type=str,
    default='resnet_56',
    choices=('resnet_50','vgg_16_bn','resnet_56','resnet_110','densenet_40','googlenet'),
    help='The architecture to prune')
parser.add_argument("--savename", type=str, default='resnet_56_UCML')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

if len(args.gpu)==1:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
lr_decay_step = list(map(int, args.lr_decay_step.split(',')))

ckpt = checkpoint(args)
print_logger = get_logger(os.path.join(args.job_dir, "logger.log"))
print_params(vars(args), print_logger.info)

# Data
print_logger.info('==> Preparing data..')

trainloader, testloader = get_ucml_dataloader(args.train_batch_size)

if args.compress_rate:
    import re
    cprate_str=args.compress_rate
    cprate_str_list=cprate_str.split('+')
    pat_cprate = re.compile(r'\d+\.\d*')
    pat_num = re.compile(r'\*\d+')
    cprate=[]
    for x in cprate_str_list:
        num=1
        find_num=re.findall(pat_num,x)
        if find_num:
            assert len(find_num) == 1
            num=int(find_num[0].replace('*',''))
        find_cprate = re.findall(pat_cprate, x)
        assert len(find_cprate)==1
        cprate+=[float(find_cprate[0])]*num

    compress_rate=cprate

# Model
device_ids=list(map(int, args.gpu.split(',')))
print_logger.info('==> Building model..')
net = eval(args.arch)(compress_rate=compress_rate)
net = net.to(device)

if len(args.gpu)>1 and torch.cuda.is_available():
    device_id = []
    for i in range((len(args.gpu) + 1) // 2):
        device_id.append(i)
    net = torch.nn.DataParallel(net, device_ids=device_id)

cudnn.benchmark = True
print(net)

if len(args.gpu)>1:
    m = eval('mask_'+args.arch)(model=net, compress_rate=net.module.compress_rate, job_dir=args.job_dir, device=device)
else:
    m = eval('mask_' + args.arch)(model=net, compress_rate=net.compress_rate, job_dir=args.job_dir, device=device)

criterion = nn.CrossEntropyLoss()

# Training
def train(epoch, cov_id, optimizer, scheduler, pruning=True):
    print_logger.info('\nEpoch: %d' % epoch)
    net.train()

    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        with torch.cuda.device(device):
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            optimizer.step()

            if pruning:
                m.grad_mask(cov_id)

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx,len(trainloader),
                         'Cov: %d | Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (cov_id, train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

def test(epoch, cov_id, optimizer, scheduler):
    top1 = AverageMeter_rank()
    top5 = AverageMeter_rank()

    global best_acc
    net.eval()
    num_iterations = len(testloader)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            top1.update(prec1[0], inputs.size(0))
            top5.update(prec5[0], inputs.size(0))

        print_logger.info(
            'Epoch[{0}]({1}/{2}): '
            'Prec@1(1,5) {top1.avg:.2f}, {top5.avg:.2f}'.format(
                epoch, batch_idx, num_iterations, top1=top1, top5=top5))

    if top1.avg > best_acc:
        print_logger.info('Saving to '+args.arch+'_cov'+str(cov_id)+'.pt')
        state = {
            'state_dict': net.state_dict(),
            'best_prec1': top1.avg,
            'epoch': epoch,
            'scheduler':scheduler.state_dict(),
            'optimizer': optimizer.state_dict() 
        }
        if not os.path.isdir(args.job_dir+'/pruned_checkpoint'):
            os.mkdir(args.job_dir+'/pruned_checkpoint')
        best_acc = top1.avg
        torch.save(state, args.job_dir+'/pruned_checkpoint/'+args.arch+'_cov'+str(cov_id)+'.pt')

    print_logger.info("=>Best accuracy {:.3f}".format(best_acc))


if len(args.gpu)>1:
    convcfg = net.module.covcfg
else:
    convcfg = net.covcfg

param_per_cov_dic={
    'vgg_16_bn': 4,
    'densenet_40': 3,
    'googlenet': 28,
    'resnet_50':3,
    'resnet_56':3,
    'resnet_110':3
}

if len(args.gpu)>1:
    print_logger.info('compress rate: ' + str(net.module.compress_rate))
else:
    print_logger.info('compress rate: ' + str(net.compress_rate))

for cov_id in range(args.start_cov, len(convcfg)):##对应resnet56的每一层
    # Load pruned_checkpoint
    print_logger.info("cov-id: %d ====> Resuming from pruned_checkpoint..." % (cov_id))

    m.layer_mask(cov_id + 1, resume=args.resume_mask, param_per_cov=param_per_cov_dic[args.arch], arch=args.arch)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_step, gamma=0.1)

    if cov_id == 0:  ##第0个卷积层不是残差块，单独写

        pruned_checkpoint = torch.load(args.resume, map_location='cuda:0')
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        if args.arch == 'resnet_50':
            tmp_ckpt = pruned_checkpoint
        else:
            # tmp_ckpt = pruned_checkpoint['state_dict']
            tmp_ckpt = pruned_checkpoint

        if len(args.gpu) > 1:
            for k, v in tmp_ckpt.items():
                new_state_dict['module.' + k.replace('module.', '')] = v
        else:
            for k, v in tmp_ckpt.items():
                new_state_dict[k.replace('module.', '')] = v

        net.load_state_dict(new_state_dict)#'''
    else:
        if args.arch=='resnet_50':
            skip_list=[1,5,8,11,15,18,21,24,28,31,34,37,40,43,47,50,53]
            if cov_id+1 not in skip_list:
                continue
            else:
                pruned_checkpoint = torch.load(
                    args.job_dir + "/pruned_checkpoint/" + args.arch + "_cov" + str(skip_list[skip_list.index(cov_id+1)-1]) + '.pt')
                net.load_state_dict(pruned_checkpoint['state_dict'])
        else:
            if len(args.gpu) == 1:
                pruned_checkpoint = torch.load(args.job_dir + "/pruned_checkpoint/" + args.arch + "_cov" + str(cov_id) + '.pt', map_location='cuda:' + args.gpu)
            else:
                pruned_checkpoint = torch.load(args.job_dir + "/pruned_checkpoint/" + args.arch + "_cov" + str(cov_id) + '.pt')
            net.load_state_dict(pruned_checkpoint['state_dict'])

    best_acc=0.
    for epoch in range(0, args.epochs):
        train(epoch, cov_id + 1, optimizer, scheduler)
        scheduler.step()
        test(epoch, cov_id + 1, optimizer, scheduler)

# epoch_qt = 300
# bit = [8]*13
# bit_all = []
# for i in range(len(bit)):
#     bit_all.append(bit[i])
#     if i == 1 or i == 3 or i == 6 or i == 9:
#         bit_all.append('M')
#     elif  i == 12:
#         bit_all.append(32)
# # sparse_channel = [32, 32, 64, 64, 128, 128, 128, 256, 256, 256, 256, 256, 256]
# sparse_channel = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
# model_qt1 = qvgg_16_bn_A(filters_left=sparse_channel, bit=bit_all).cuda()
# # model_qt1.load_state_dict(torch.load('./ckpt_{}/'.format('hrank') + '{}_best.t7'.format(args.savename)))
# model_qt1.load_state_dict(torch.load('./result/tmp/pruned_checkpoint/vgg_16_bn_cov12.pt')['state_dict'])

# model_qt1.to(device)
# optimizer = optim.SGD(model_qt1.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [int(epoch_qt*0.3), int(epoch_qt*0.6), int(epoch_qt*0.8)], gamma=0.2)
# train(model_qt1, trainloader, testloader, optimizer, scheduler, epoch_qt, args.arch,  train_model_Running=False, quant=True, name=args.savename, ratio = 'hrank', device=device)

# model_qt2 = qvgg_16_bn_A_W(filters_left=sparse_channel, bit=bit_all).cuda()
# model_qt2.load_state_dict(torch.load('./ckpt_{}/'.format('hrank') + '{}_quant_best.t7'.format(args.savename)))
# model_qt2.to(device)
# optimizer = optim.SGD(model_qt2.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [int(epoch_qt*0.3), int(epoch_qt*0.6), int(epoch_qt*0.8)], gamma=0.2)
# train(model_qt2, trainloader, testloader, optimizer, scheduler, epoch_qt, args.arch,  train_model_Running=False, quant=True, name=args.savename, ratio = 'hrank', device=device)

