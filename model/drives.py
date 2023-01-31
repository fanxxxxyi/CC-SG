import os
import numpy as np
import time
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import copy

import sys
import shutil
import numpy as np
import time, datetime
import random
import logging
import argparse
import torch.utils
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data.distributed
from tqdm import tqdm
from utils import *
 
def sgd_optimizer(args, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = args.base_lr
        weight_decay = args.weight_decay
        if "bias" in key or "bn" in key or "BN" in key:
            # lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = args.weight_decay
            print('set weight_decay={} for {}'.format(weight_decay, key))
        if 'bias' in key:
            apply_lr = 2 * lr
            print('set lr={} for {}'.format(apply_lr, key))
        else:
            apply_lr = lr

        params += [{"params": [value], "lr": apply_lr, "weight_decay": weight_decay}]
    # optimizer = torch.optim.Adam(params, lr)
    optimizer = torch.optim.SGD(params, lr, momentum=args.momentum)
    return optimizer

def get_optimizer(args, model):
    return sgd_optimizer(args, model)
# Dataloader
def get_ucml_dataloader(batch_size):
    ROOT_TRAIN = r'/home/fanxiaoyi/UCML_0.8/train'
    ROOT_TEST = r'/home/fanxiaoyi/UCML_0.8/val'
    # 将图像的像素值归一化到[-1， 1]之间
    normalize = transforms.Normalize(mean=[0.48422759, 0.49005176, 0.45050278],
                                std=[0.17348298, 0.16352356, 0.15547497])
    # 将训练集输入图像进行预处理，重映射尺寸、随机旋转、转化张量
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])
    # 将验证集输入图像进行预处理，重映射尺寸、随机旋转、转化张量
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        normalize])
    # 设置训练集、验证集加载路径
    train_dataset = ImageFolder(ROOT_TRAIN, transform=train_transform)
    val_dataset = ImageFolder(ROOT_TEST, transform=val_transform)
    # 加载训练集、验证集
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,num_workers=8)

    return train_dataloader, val_dataloader

def get_dataloader(img_size, dataset, datapath, batch_size, no_val):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if img_size == 32:
        train_set = eval(dataset)(datapath, True, transforms.Compose([
                transforms.RandomCrop(img_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]), download=True)
        val_set = eval(dataset)(datapath, True, transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]), download=True)

        num_train = len(train_set)
        indices = list(range(num_train))
        split = int(np.floor(0.1 * num_train))

        np.random.seed(int(time.time()))
        np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]                                             
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        test_set = eval(dataset)(datapath, False, transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]), download=True)

        if no_val:
            train_loader = torch.utils.data.DataLoader(
                train_set, batch_size=batch_size, shuffle=True,
                num_workers=0, pin_memory=True
            )
            val_loader = torch.utils.data.DataLoader(
                val_set, batch_size=batch_size, sampler=valid_sampler,
                num_workers=0, pin_memory=True
            )
        else:
            train_loader = torch.utils.data.DataLoader(
                train_set, batch_size=batch_size, sampler=train_sampler,
                num_workers=0, pin_memory=True
            )
            val_loader = torch.utils.data.DataLoader(
                val_set, batch_size=batch_size, sampler=valid_sampler,
                num_workers=0, pin_memory=True
            )
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=False
        )
    else:
        raise ValueError("img_size must be 32")
    return train_loader, val_loader, test_loader

# epoch of train
def train_epoch(model, train_loader, optimizer=None, steps=None, device='cuda'):
    model.to(device)
    model.train()
    losses = np.zeros(0)
    total_loss = 0
    data_t = 0
    train_t = 0 
    criterion = torch.nn.CrossEntropyLoss()
    s = time.time()
    for i, (batch, label) in enumerate(train_loader):
       
        batch, label = batch.to(device), label.to(device)
        data_t += time.time() - s
        s = time.time()     # second

        model.zero_grad()
        output = model(batch)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        total_loss += loss
        losses = np.concatenate([losses, np.array([loss.item()])])

        train_t += time.time() - s
        length = steps if steps and steps < len(train_loader) else len(train_loader)

        if (i % 100 == 0) or (i == length-1):
            print('Training | Batch ({}/{}) | Loss {:.4f} ({:.4f}) | (PerBatchProfile) Data: {:.3f}s, Net: {:.3f}s'.format(i+1, length, total_loss/(i+1), loss, data_t/(i+1), train_t/(i+1)))
        if i == length-1:
            break
        s = time.time()
    return np.mean(losses)


# train
def train(model, train_loader, val_loader, optimizer, scheduler, epochs=400, arch ='resnet_56', steps=None, run_test=True, train_model_Running=False, quant=False, name='', ratio='', device='cuda'):
    model.to(device)
    if optimizer is None:
        optimizer = optim.SGD(model.classifier.parameters(), lr=1e-1, momentum=0.9, weight_decay=5e-4)
    if scheduler is None:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # Use number of steps as unit instead of epochs
    if steps:
        epochs = int(steps / len(train_loader)) + 1
        # print(epochs)
        if epochs > 1:
            steps = steps % len(train_loader)

    best_acc = 0
    for i in range(epochs):
        print('Epoch: {}'.format(i))
        if i == epochs - 1:
            loss = train_epoch(model, train_loader, optimizer, steps=steps, device=device)
        else:
            loss = train_epoch(model, train_loader, optimizer, device=device)
        scheduler.step()

        if run_test:
            acc = test(model, val_loader, device=device)
            print('Testing Accuracy {:.2f}'.format(acc))
            if i and best_acc < acc:
                best_acc = acc
                #  "Pre-trained" model
                if train_model_Running :
                    torch.save(model.state_dict(), os.path.join('./ckpt/', '{}.t7'.format(name)))
                else :
                    if quant:
                        torch.save(model.state_dict(), os.path.join('ckpt_{}'.format(ratio), '{}_quant_best.t7'.format(name)))
                    else :
                        torch.save(model.state_dict(), os.path.join('ckpt_{}'.format(ratio), '{}_best.t7'.format(name)))

    print('best_acc: ' + str(best_acc))


# test
def test(model, data_loader, device='cuda', get_loss=False, n_img=-1):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    if get_loss:
        criterion = nn.CrossEntropyLoss(reduction='none')
        loss = np.zeros(0)
    total_len = len(data_loader)

    if n_img > 0 and total_len > int(np.ceil(float(n_img) / data_loader.batch_size)):
        total_len = int(np.ceil(float(n_img) / data_loader.batch_size))
    for i, (batch, label) in enumerate(data_loader):
        
        if i >= total_len:
            break
        batch, label = batch.to(device), label.to(device)

        output = model(batch)
        if get_loss:
            loss = np.concatenate((loss, criterion(output, label).data.cpu().numpy()))
        pred = output.data.max(1)[1]

        correct += pred.eq(label).sum()
        total += label.size(0)
    
    if get_loss:
        return float(correct)/total*100, loss
    else:
        return float(correct)/total*100

def count_bops(layer, x, sparse_channel, bit):
    cur_bops = 0
    activation_index = 0
    y = layer.old_forward(x)
    if isinstance(layer, nn.Conv2d):
        if activation_index == 0:
            conv_in_channels = 3
            conv_out_channels = sparse_channel[activation_index]
        else:
            conv_in_channels = sparse_channel[activation_index - 1]
            conv_out_channels = sparse_channel[activation_index]
        h = y.shape[2]
        w = y.shape[3]
        cur_bops += h * w * conv_in_channels * conv_out_channels * layer.weight.size(2) * layer.weight.size(3) * bit[activation_index] * bit[activation_index]

    elif isinstance(layer, nn.Linear):
        cur_bops += np.prod(layer.weight.shape)
    cur_bops/= 1E9
    return cur_bops
