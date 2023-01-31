import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from data import *
from backbone import *
from utils.augmentations import SSDAugmentation
from utils.utils import *
from utils import *
from model2.drives import *
import torch.utils.data as data
import tools
from model.resnet_cifar import *
from model.vgg_cifar import *
from model.MobileNetV2 import *
from model.imagenet_resnet import *
# from model2.resnet50 import *

from pruner.filterpruner import FilterPruner
from pruner.fp_resnet import FilterPrunerResNet
from pruner.fp_vgg import FilterPrunerVGG
# from backbone import *
os.environ['CUDA_VISIBLE_DEVICES']='0'
# Hyper-Parameters
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--arch',
        type=str,
        default='mobilenetV2',
        choices=('vgg_16_bn','resnet_56','resnet_110','resnet_50','mobilenetV2','darknet_light'),
        help='architecture to calculate feature maps')
    parser.add_argument(
        '--resume',
        type=str,
        default='./model2/ckpt/MobileNetV2_UCML.t7',
        help='load the model from the specified checkpoint') # default=None
    parser.add_argument(
    '--dataset',
        type=str,
        default='UCML',
        choices=('cifar10','imagenet'),
        help='cifar10 or imagenet or VOC or COCO dataset')
    parser.add_argument(
        '--num_classes', 
        default=21, 
        type=int,
        help='The number of dataset classes')
    parser.add_argument(
        "--pruner",
        type=str,
        default='FilterPrunerMBNetV2',
        choices=('FilterPrunerResNet', 'FilterPrunerVGG', 'FilterPrunerMBNetV2'),
        help='Different network require differnt pruner implementation')
    parser.add_argument(
        "--rank_type",
        type=str,
        default='Rank',
        choices=('l1_bn','l2_bn','l1_weight','l2_weight', 'Rank'),
        help='The ranking criteria for filter pruning')
    parser.add_argument(
        "--global_random_rank",
        action='store_true',
        default=False,
        help='When this is specified, none of the rank_type matters, it will randomly prune the filters')
    parser.add_argument(
        "--safeguard",
        type=float,
        default=0,
        help='A floating point number that represent at least how many percentage of the original number of channel should be preserved. E.g., 0.10 means no matter what ranking, each layer should have at least 10% of the number of original channels.')
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        choices=(16, 32, 64, 128),
        help='Batch size for training.')
    parser.add_argument(
        '--img_size',
        type=int,
        default=256,
        help='32 is the size of CIFAR10')
    parser.add_argument(
        "--no_val",
        action='store_true',
        default=False,
        help='Use full dataset to train (use to compare with prior art in CIFAR-10)')
    parser.add_argument(
        "--gpu",
        type=str,
        default='0',
        help='Select GPU to use.("0" is to use GPU:0))')
    parser.add_argument(
        '--limit',
        type=int,
        default=6,
        help='The number of  batch to get rank.')
    parser.add_argument(
        '--num_workers', 
        default=0, 
        type=int, 
        help='Number of workers used in dataloading')

    args = parser.parse_args()
    return args


# NOTE hook function
criterion = nn.CrossEntropyLoss()
feature_result = torch.tensor(0.)
total = torch.tensor(0)
def get_feature_hook(self, input, output):
    global feature_result
    global total

    a = output.shape[0]
    b = output.shape[1]
    c = torch.tensor([torch.matrix_rank(output[i, j, :, :]).item() for i in range(a) for j in range(b)])

    c = c.view(a, -1).float()
    c = c.sum(0)
    feature_result = feature_result * total + c
    total = total + a
    feature_result = feature_result / total


def get_feature_hook_densenet(self, input, output):
    global feature_result
    global total
    a = output.shape[0]
    b = output.shape[1]
    c = torch.tensor([torch.matrix_rank(output[i, j, :, :]).item() for i in range(a) for j in range(b-12,b)])

    c = c.view(a, -1).float()
    c = c.sum(0)
    feature_result = feature_result * total + c
    total = total + a
    feature_result = feature_result / total


def inReference():
    model.eval()
    temp_loss = 0
    correct = 0
    total = 0

    # with torch.no_grad():
    #     for batch_idx, (images, targets) in enumerate(data_loader):
    #         if batch_idx >= args.limit:  # use the first args.limit+1 batches to estimate the rank
    #             break

    #         targets = [label.tolist() for label in targets]
    #         targets = tools.multi_gt_creator(input_size, yolo_net.stride, targets, version=args.arch)
    #         # to device
    #         images = images.to(device)
    #         targets = torch.tensor(targets).float().to(device)
    #         outputs = model(images)
    #         # loss = criterion(outputs, targets)

    #         # temp_loss += loss.item()
    #         # _, predicted = outputs.max(1)
    #         # total += targets.size(0)
    #         # correct += predicted.eq(targets).sum().item()

    #         progress_bar(batch_idx, args.limit)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if batch_idx >= args.limit:  # use the first args.limit+1 batches to estimate the rank
                break

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            temp_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, args.limit, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (temp_loss/(batch_idx+1), 100.*correct/total, correct, total))


# ----------- start -----------------
if __name__ == "__main__":
    # Parameters
    startTime = time.time()
    args = get_args()
    device = 'cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu'
    print("Environment：\ndevice --- {} \ntorch --- {} \ntorchvision --- {} \n\nParameters:\n{}\n".format(device,torch.__version__, torchvision.__version__, args))

    # Num_classes
    if 'CIFAR100' in args.dataset:
        num_classes = 100
    elif 'CIFAR10' in args.dataset:
        num_classes = 10
    elif 'ImageNet' in args.dataset:
        num_classes = 1000
    elif 'CUB200' in args.dataset:
        num_classes = 200


    # Data
    print('==> Preparing data..')
    train_loader, test_loader = get_ucml_dataloader(args.batch_size)
    # Load pretrained model.
    print('Loading Pretrained Model...')
    # model = darknet_light(pretrained=False, hr=False).cuda()
    # model = vgg_16_bn().cuda()
    model = mobilenetV2(n_class=21).cuda()
    model.load_state_dict(torch.load(args.resume))
    model = model.to(device)
    print(model)

    # NOTE Get ranks of feature map
    if args.arch == 'vgg_16_bn':
        # handle directory
        if 'UCML' in args.dataset:
            if not os.path.isdir('rank_conv/UCML/{}/'.format(args.arch)):
                os.makedirs('rank_conv/UCML/{}/'.format(args.arch))
        elif 'CIFAR100' in args.dataset:
            if not os.path.isdir('rank_conv/CIFAR100/{}/'.format(args.arch)):
                os.makedirs('rank_conv/CIFAR100/{}/'.format(args.arch))
        elif 'CIFAR10' in args.dataset:
            if not os.path.isdir('rank_conv/CIFAR10/{}/'.format(args.arch)):
                    os.makedirs('rank_conv/CIFAR10/{}/'.format(args.arch))
        
        ''' Obtain rank '''
        for i, cov_id in enumerate(model.covcfg):
            cov_layer = model.features[cov_id]
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inReference()
            handler.remove()

            if 'CIFAR100' in args.dataset:
                np.save('rank_conv/CIFAR100/{}/'.format(args.arch) + '/rank_conv%d' % (i + 1) + '.npy', feature_result.numpy())
            elif 'CIFAR10' in args.dataset:
                np.save('rank_conv/CIFAR10/{}/'.format(args.arch) + '/rank_conv%d' % (i + 1) + '.npy', feature_result.numpy())
            elif 'UCML' in args.dataset:
                np.save('rank_conv/UCML/{}/'.format(args.arch) + '/rank_conv%d' % (i + 1) + '.npy', feature_result.numpy())
            

            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)

    elif args.arch == 'resnet_56':
        # handle directory
        if 'UCML' in args.dataset:
            if not os.path.isdir('rank_conv/UCML/{}/'.format(args.arch)):
                os.makedirs('rank_conv/UCML/{}/'.format(args.arch))
        elif 'CIFAR100' in args.dataset:
            if not os.path.isdir('rank_conv/CIFAR100/{}/'.format(args.arch)):
                os.makedirs('rank_conv/CIFAR100/{}/'.format(args.arch))
        elif 'CIFAR10' in args.dataset:
            if not os.path.isdir('rank_conv/CIFAR10/{}/'.format(args.arch)):
                    os.makedirs('rank_conv/CIFAR10/{}/'.format(args.arch))
        
        feature_result = torch.tensor(0.)
        total = torch.tensor(0.)
        
        # First conv layer
        cov_layer = eval('model.conv1')
        handler = cov_layer.register_forward_hook(get_feature_hook)
        inReference()
        handler.remove()
        if 'CIFAR100' in args.dataset:
            np.save('rank_conv/CIFAR100/{}/'.format(args.arch) + '/rank_conv%d' % (1) + '.npy', feature_result.numpy())
        elif 'CIFAR10' in args.dataset:
            np.save('rank_conv/CIFAR10/{}/'.format(args.arch) + '/rank_conv%d' % (1) + '.npy', feature_result.numpy())
        elif 'UCML' in args.dataset:
            np.save('rank_conv/UCML/{}/'.format(args.arch) + '/rank_conv%d' % (1) + '.npy', feature_result.numpy())

        # 
        cnt = 1
        for i in range(3):
            # get block
            block = eval('model.layer%d' % (i + 1))

            for j in range(9):
                # cov_layer = block[j].relu1
                cov_layer = block[j].conv1
                handler = cov_layer.register_forward_hook(get_feature_hook)
                inReference()
                handler.remove()
                if 'CIFAR100' in args.dataset:
                    np.save('rank_conv/CIFAR100/{}/'.format(args.arch) + '/rank_conv%d' % (cnt + 1) + '.npy', feature_result.numpy())
                elif 'CIFAR10' in args.dataset:
                    np.save('rank_conv/CIFAR10/{}/'.format(args.arch) + '/rank_conv%d' % (cnt + 1) + '.npy', feature_result.numpy())
                elif 'UCML' in args.dataset:
                    np.save('rank_conv/UCML/{}/'.format(args.arch) + '/rank_conv%d' % (cnt + 1) + '.npy', feature_result.numpy())

                cnt += 1
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)

                ''' * '''
                cov_layer = block[j].conv2
                # cov_layer = block[j].relu2
                handler = cov_layer.register_forward_hook(get_feature_hook)
                inReference()
                handler.remove()
                if 'CIFAR100' in args.dataset:
                    np.save('rank_conv/CIFAR100/{}/'.format(args.arch) + '/rank_conv%d' % (cnt + 1) + '.npy', feature_result.numpy())
                elif 'CIFAR10' in args.dataset:
                    np.save('rank_conv/CIFAR10/{}/'.format(args.arch) + '/rank_conv%d' % (cnt + 1) + '.npy', feature_result.numpy())
                elif 'UCML' in args.dataset:
                    np.save('rank_conv/UCML/{}/'.format(args.arch) + '/rank_conv%d' % (cnt + 1) + '.npy', feature_result.numpy())
                cnt += 1
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)

    elif args.arch == 'mobilenetV2':
        # handle directory
        if 'UCML' in args.dataset:
            if not os.path.isdir('rank_conv/UCML/{}/'.format(args.arch)):
                os.makedirs('rank_conv/UCML/{}/'.format(args.arch))
        elif 'CIFAR100' in args.dataset:
            if not os.path.isdir('rank_conv/CIFAR100/{}/'.format(args.arch)):
                os.makedirs('rank_conv/CIFAR100/{}/'.format(args.arch))
        elif 'CIFAR10' in args.dataset:
            if not os.path.isdir('rank_conv/CIFAR10/{}/'.format(args.arch)):
                    os.makedirs('rank_conv/CIFAR10/{}/'.format(args.arch))
            
        # first Conv
        cov_layer = eval('model.features[0][2]')
        # print(str(cov_layer) + ': 1')
        handler = cov_layer.register_forward_hook(get_feature_hook)
        inReference()
        handler.remove()

        if 'CIFAR100' in args.dataset:
            np.save('rank_conv/CIFAR100/{}/'.format(args.arch) + '/rank_conv%d' % (1) + '.npy', feature_result.numpy())
        elif 'CIFAR10' in args.dataset:
            np.save('rank_conv/CIFAR10/{}/'.format(args.arch) + '/rank_conv%d' % (1) + '.npy', feature_result.numpy())
        elif 'UCML' in args.dataset:
            np.save('rank_conv/UCML/{}/'.format(args.arch) + '/rank_conv%d' % (1) + '.npy', feature_result.numpy())
        
        feature_result = torch.tensor(0.)
        total = torch.tensor(0.)

        # InvertedResidual block
        cnt = 1
        for i in range(17):
            # First invertedResidual
            if i < 1:
                # First conv layer
                cov_layer = eval('model.features[%d].conv[0]' %(i + 1))
                # print(str(cov_layer) + ': %d' %(cnt + 1))
                handler = cov_layer.register_forward_hook(get_feature_hook)
                inReference()
                handler.remove()

                if 'CIFAR100' in args.dataset:
                    np.save('rank_conv/CIFAR100/{}/'.format(args.arch) + '/rank_conv%d' % (cnt + 1) + '.npy', feature_result.numpy())
                elif 'CIFAR10' in args.dataset:
                    np.save('rank_conv/CIFAR10/{}/'.format(args.arch) + '/rank_conv%d' % (cnt + 1) + '.npy', feature_result.numpy())
                elif 'UCML' in args.dataset:
                    np.save('rank_conv/UCML/{}/'.format(args.arch) + '/rank_conv%d' % (cnt + 1) + '.npy', feature_result.numpy())

                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)
                cnt += 1

                # Second conv layer
                cov_layer = eval('model.features[%d].conv[3]' %(i + 1))
                # print(str(cov_layer) + ': %d' %(cnt + 1))
                handler = cov_layer.register_forward_hook(get_feature_hook)
                inReference()
                handler.remove()

                if 'CIFAR100' in args.dataset:
                    np.save('rank_conv/CIFAR100/{}/'.format(args.arch) + '/rank_conv%d' % (cnt + 1) + '.npy', feature_result.numpy())
                elif 'CIFAR10' in args.dataset:
                    np.save('rank_conv/CIFAR10/{}/'.format(args.arch) + '/rank_conv%d' % (cnt + 1) + '.npy', feature_result.numpy())
                elif 'UCML' in args.dataset:
                    np.save('rank_conv/UCML/{}/'.format(args.arch) + '/rank_conv%d' % (cnt + 1) + '.npy', feature_result.numpy())
                                
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)
                cnt += 1
                
            # 2~17 invertedResidual
            else:
                # First conv layer
                cov_layer = eval('model.features[%d].conv[0]' %(i + 1))
                # print(str(cov_layer) + ': %d' %(cnt + 1))
                handler = cov_layer.register_forward_hook(get_feature_hook)
                inReference()
                handler.remove()

                if 'CIFAR100' in args.dataset:
                    np.save('rank_conv/CIFAR100/{}/'.format(args.arch) + '/rank_conv%d' % (cnt + 1) + '.npy', feature_result.numpy())
                elif 'CIFAR10' in args.dataset:
                    np.save('rank_conv/CIFAR10/{}/'.format(args.arch) + '/rank_conv%d' % (cnt + 1) + '.npy', feature_result.numpy())
                elif 'UCML' in args.dataset:
                    np.save('rank_conv/UCML/{}/'.format(args.arch) + '/rank_conv%d' % (cnt + 1) + '.npy', feature_result.numpy())
                
                
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)
                cnt += 1

                # Second conv layer
                cov_layer = eval('model.features[%d].conv[3]' %(i + 1))
                # print(str(cov_layer) + ': %d' %(cnt + 1))
                handler = cov_layer.register_forward_hook(get_feature_hook)
                inReference()
                handler.remove()

                if 'CIFAR100' in args.dataset:
                    np.save('rank_conv/CIFAR100/{}/'.format(args.arch) + '/rank_conv%d' % (cnt + 1) + '.npy', feature_result.numpy())
                elif 'CIFAR10' in args.dataset:
                    np.save('rank_conv/CIFAR10/{}/'.format(args.arch) + '/rank_conv%d' % (cnt + 1) + '.npy', feature_result.numpy())
                elif 'UCML' in args.dataset:
                    np.save('rank_conv/UCML/{}/'.format(args.arch) + '/rank_conv%d' % (cnt + 1) + '.npy', feature_result.numpy())
                
                
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)
                cnt += 1

                # Third conv layer
                cov_layer = eval('model.features[%d].conv[6]' %(i + 1))
                # print(str(cov_layer) + ': %d' %(cnt + 1))
                handler = cov_layer.register_forward_hook(get_feature_hook)
                inReference()
                handler.remove()

                if 'CIFAR100' in args.dataset:
                    np.save('rank_conv/CIFAR100/{}/'.format(args.arch) + '/rank_conv%d' % (cnt + 1) + '.npy', feature_result.numpy())
                elif 'CIFAR10' in args.dataset:
                    np.save('rank_conv/CIFAR10/{}/'.format(args.arch) + '/rank_conv%d' % (cnt + 1) + '.npy', feature_result.numpy())
                elif 'UCML' in args.dataset:
                    np.save('rank_conv/UCML/{}/'.format(args.arch) + '/rank_conv%d' % (cnt + 1) + '.npy', feature_result.numpy())
        
                
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)
                cnt += 1

        # Last Conv e.g., 52th
        cov_layer = eval('model.features[18][2]')
        # print(str(cov_layer) + ': %d' %(cnt + 1))
        handler = cov_layer.register_forward_hook(get_feature_hook)
        inReference()
        handler.remove()

        if 'CIFAR100' in args.dataset:
            np.save('rank_conv/CIFAR100/{}/'.format(args.arch) + '/rank_conv%d' % (52) + '.npy', feature_result.numpy())
        elif 'CIFAR10' in args.dataset:
            np.save('rank_conv/CIFAR10/{}/'.format(args.arch) + '/rank_conv%d' % (52) + '.npy', feature_result.numpy())
        elif 'UCML' in args.dataset:
            np.save('rank_conv/UCML/{}/'.format(args.arch) + '/rank_conv%d' % (52) + '.npy', feature_result.numpy())
        
        feature_result = torch.tensor(0.)
        total = torch.tensor(0.)

    elif args.arch=='darknet_light':

         # handle directory
        if 'UCML' in args.dataset:
            if not os.path.isdir('rank_conv/UCML/{}/'.format(args.arch)):
                os.makedirs('rank_conv/UCML/{}/'.format(args.arch))
        elif 'CIFAR10' in args.dataset:
            if not os.path.isdir('rank_conv/CIFAR10/{}/'.format(args.arch)):
                    os.makedirs('rank_conv/CIFAR10/{}/'.format(args.arch))
        
        ''' Obtain rank '''
        cov_layer_1 = model.conv_1
        handler = cov_layer_1.register_forward_hook(get_feature_hook)
        inReference()
        handler.remove()
        np.save('rank_conv/UCML/{}/'.format(args.arch) + '/rank_conv%d' % (1) + '.npy', feature_result.numpy())
        feature_result = torch.tensor(0.)
        total = torch.tensor(0.)

        cov_layer_2 = model.conv_2
        handler = cov_layer_2.register_forward_hook(get_feature_hook)
        inReference()
        handler.remove()
        np.save('rank_conv/UCML/{}/'.format(args.arch) + '/rank_conv%d' % (2) + '.npy', feature_result.numpy())
        feature_result = torch.tensor(0.)
        total = torch.tensor(0.)

        cov_layer_3 = model.conv_3
        handler = cov_layer_3.register_forward_hook(get_feature_hook)
        inReference()
        handler.remove()
        np.save('rank_conv/UCML/{}/'.format(args.arch) + '/rank_conv%d' % (3) + '.npy', feature_result.numpy())
        feature_result = torch.tensor(0.)
        total = torch.tensor(0.)

        cov_layer_4 = model.conv_4
        handler = cov_layer_4.register_forward_hook(get_feature_hook)
        inReference()
        handler.remove()
        np.save('rank_conv/UCML/{}/'.format(args.arch) + '/rank_conv%d' % (4) + '.npy', feature_result.numpy())
        feature_result = torch.tensor(0.)
        total = torch.tensor(0.)

        cov_layer_5 = model.conv_5
        handler = cov_layer_5.register_forward_hook(get_feature_hook)
        inReference()
        handler.remove()
        np.save('rank_conv/UCML/{}/'.format(args.arch) + '/rank_conv%d' % (5) + '.npy', feature_result.numpy())
        feature_result = torch.tensor(0.)
        total = torch.tensor(0.)

        cov_layer_6 = model.conv_6
        handler = cov_layer_6.register_forward_hook(get_feature_hook)
        inReference()
        handler.remove()
        np.save('rank_conv/UCML/{}/'.format(args.arch) + '/rank_conv%d' % (6) + '.npy', feature_result.numpy())
        feature_result = torch.tensor(0.)
        total = torch.tensor(0.)

        cov_layer_7 = model.conv_7
        handler = cov_layer_7.register_forward_hook(get_feature_hook)
        inReference()
        handler.remove()
        np.save('rank_conv/UCML/{}/'.format(args.arch) + '/rank_conv%d' % (7) + '.npy', feature_result.numpy())
        feature_result = torch.tensor(0.)
        total = torch.tensor(0.)

    elif args.arch=='resnet_50':

        cov_layer = eval('model.maxpool')
        handler = cov_layer.register_forward_hook(get_feature_hook)
        inReference()
        handler.remove()

        if not os.path.isdir('rank_conv/' + args.arch+'_limit%d'%(args.limit)):
            os.mkdir('rank_conv/' + args.arch+'_limit%d'%(args.limit))
        np.save('rank_conv/' + args.arch+'_limit%d'%(args.limit) + '/rank_conv%d' % (1) + '.npy', feature_result.numpy())
        feature_result = torch.tensor(0.)
        total = torch.tensor(0.)

        # ResNet50 per bottleneck
        cnt=1
        for i in range(4):
            block = eval('model.layer%d' % (i + 1))
            for j in range(model.layers[i]):
                cov_layer = block[j].relu1
                handler = cov_layer.register_forward_hook(get_feature_hook)
                inReference()
                handler.remove()
                np.save('rank_conv/' + args.arch+'_limit%d'%(args.limit) + '/rank_conv%d'%(cnt+1)+'.npy', feature_result.numpy())
                cnt+=1
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)

                cov_layer = block[j].relu2
                handler = cov_layer.register_forward_hook(get_feature_hook)
                inReference()
                handler.remove()
                np.save('rank_conv/' + args.arch + '_limit%d' % (args.limit) + '/rank_conv%d' % (cnt + 1) + '.npy',
                        feature_result.numpy())
                cnt += 1
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)

                cov_layer = block[j].relu3
                handler = cov_layer.register_forward_hook(get_feature_hook)
                inReference()
                handler.remove()
                if j==0:
                    np.save('rank_conv/' + args.arch + '_limit%d' % (args.limit) + '/rank_conv%d' % (cnt + 1) + '.npy',
                            feature_result.numpy())#shortcut conv
                    cnt += 1
                np.save('rank_conv/' + args.arch + '_limit%d' % (args.limit) + '/rank_conv%d' % (cnt + 1) + '.npy',
                        feature_result.numpy())#conv3
                cnt += 1
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)
    print('\n----------- Cost Time: ' + format_time(time.time() - startTime) +" ----------- \n----------- Program Over -----------")
