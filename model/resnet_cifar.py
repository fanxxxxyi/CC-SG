import time
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

from matplotlib import pyplot as plt
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES']='1'


# 3x3 convLayer
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


# down sample
class DownsampleA(nn.Module):
  def __init__(self, nIn, nOut, stride):
    super(DownsampleA, self).__init__() 
    assert stride == 2
    self.out_channels = nOut
    self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)

  def forward(self, x):
    # down sample
    x = self.avg(x)
    if self.out_channels-x.size(1) > 0:
        return torch.cat((x, torch.zeros(x.size(0), self.out_channels-x.size(1), x.size(2), x.size(3), device=x.device)), 1)
    else:
        return x


# ResBasicBlock
class ResBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(ResBasicBlock, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.stride = stride
        self.shortcut = nn.Sequential()

        # Shortcut
        if stride != 1 or inplanes != planes:
            self.shortcut = DownsampleA(inplanes, planes, stride)


    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # Residual
        out += self.shortcut(x)
        out = self.relu2(out)

        return out


# ResNet
class ResNet(nn.Module):
    def __init__(self, block, num_layers, covcfg, num_classes=10):
        super(ResNet, self).__init__()
        assert (num_layers - 2) % 6 == 0, 'depth should be 6n+2'
        # Computer the number of ResBasicBlock needed
        n = (num_layers - 2) // 6
        self.covcfg = covcfg
        self.num_layers = num_layers
        self.inplanes = 16

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16, blocks=n, stride=1)
        self.layer2 = self._make_layer(block, 32, blocks=n, stride=2)
        self.layer3 = self._make_layer(block, 64, blocks=n, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # ResNet-110
        if num_layers == 110:
            self.linear = nn.Linear(64 * block.expansion, num_classes)
        else:
            self.fc = nn.Linear(64 * block.expansion, num_classes)

        # Parameters Intialization
        self.initialize()


    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # m.weight ~ kaiming distribution
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, planes, blocks, stride):
        layers = []
        layers.append(block(self.inplanes, planes, stride))
        self.inplanes = planes * block.expansion
       
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # tsne = TSNE(n_components=2, random_state=0)
        # output1 = x.cpu().detach().numpy()
        # OC, IC, KH, KW = output1.shape
        # output1 = output1.reshape(OC, IC * KH * KW)
        # X_1 = tsne.fit_transform(output1) 
        # y_1 = label.cpu().numpy()
        # target_names = ['0','1','2','3','4','5','6','7','8','9']
        # target_ids = range(len(target_names))
        
        # plt.figure(figsize=(10, 10))
    
        # ax = plt.gca(projection='polar')
        # ax.set_thetagrids(np.arange(0.0, 360.0, 15.0))
        # ax.set_thetamin(0.0)  # 设置极坐标图开始角度为0°
        # ax.set_thetamax(360.0)  # 设置极坐标结束角度为180°
        # ax.grid(True, linestyle="-", color="k", linewidth=0.5, alpha=0.5)
        # ax.set_axisbelow('True')  # 使散点覆盖在坐标系之上
        # colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'pink', 'orange', 'purple'
        # for i, c, label in zip(target_ids, colors, target_names):
        #     plt.scatter(X_1[y_1 == i, 0], X_1[y_1 == i, 1], c=c, label=label)
        # plt.legend()
        # plt.savefig('/home/fanxiaoyi/legr+hrank/plt/conv1.jpg')
        # plt.show() 


        x = self.layer1(x)

        # tsne = TSNE(n_components=2, random_state=0)
        # output2 = x.cpu().detach().numpy()
        # OC, IC, KH, KW = output2.shape
        # output2 = output2.reshape(OC, IC * KH * KW)
        # X_2 = tsne.fit_transform(output2) 
        # y_2 = label.cpu().numpy()
        # target_names = ['0','1','2','3','4','5','6','7','8','9']
        # target_ids = range(len(target_names))
        
        # plt.figure(figsize=(10, 10))
        # ax = plt.gca(projection='polar')
        # ax.set_thetagrids(np.arange(0.0, 360.0, 15.0))
        # ax.set_thetamin(0.0)  # 设置极坐标图开始角度为0°
        # ax.set_thetamax(360.0)  # 设置极坐标结束角度为180°
        # ax.grid(True, linestyle="-", color="k", linewidth=0.5, alpha=0.5)
        # ax.set_axisbelow('True')  # 使散点覆盖在坐标系之上
        # colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'pink', 'orange', 'purple'
        # for i, c, label in zip(target_ids, colors, target_names):
        #     plt.scatter(X_2[y_2 == i, 0], X_2[y_2 == i, 1], c=c, label=label)
        # plt.legend()
        # plt.savefig('/home/fanxiaoyi/legr+hrank/plt/layer1.jpg')
        # plt.show() 

        x = self.layer2(x)

        # tsne = TSNE(n_components=2, random_state=0)
        # output3 = x.cpu().detach().numpy()
        # OC, IC, KH, KW = output3.shape
        # output3 = output3.reshape(OC, IC * KH * KW)
        # X_3 = tsne.fit_transform(output3) 
        # y_3 = label.cpu().numpy()
        # target_names = ['0','1','2','3','4','5','6','7','8','9']
        # target_ids = range(len(target_names))
        
        # plt.figure(figsize=(10, 10))
        # ax = plt.gca(projection='polar')
        # ax.set_thetagrids(np.arange(0.0, 360.0, 15.0))
        # ax.set_thetamin(0.0)  # 设置极坐标图开始角度为0°
        # ax.set_thetamax(360.0)  # 设置极坐标结束角度为180°
        # ax.grid(True, linestyle="-", color="k", linewidth=0.5, alpha=0.5)
        # ax.set_axisbelow('True')  # 使散点覆盖在坐标系之上
        # colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'pink', 'orange', 'purple'
        # for i, c, label in zip(target_ids, colors, target_names):
        #     plt.scatter(X_3[y_3 == i, 0], X_3[y_3 == i, 1], c=c, label=label)
        # plt.legend()
        # plt.savefig('/home/fanxiaoyi/legr+hrank/plt/layer2.jpg')
        # plt.show() 

        x = self.layer3(x)

        # tsne = TSNE(n_components=2, random_state=0)
        # output4 = x.cpu().detach().numpy()
        # OC, IC, KH, KW = output4.shape
        # output4 = output4.reshape(OC, IC * KH * KW)
        # X_4 = tsne.fit_transform(output4) 
        # y_4 = label.cpu().numpy()
        # target_names = ['0','1','2','3','4','5','6','7','8','9']
        # target_ids = range(len(target_names))
        
        # plt.figure(figsize=(10, 10))
        # ax = plt.gca(projection='polar')
        # ax.set_thetagrids(np.arange(0.0, 360.0, 15.0))
        # ax.set_thetamin(0.0)  # 设置极坐标图开始角度为0°
        # ax.set_thetamax(360.0)  # 设置极坐标结束角度为180°
        # ax.grid(True, linestyle="-", color="k", linewidth=0.5, alpha=0.5)
        # ax.set_axisbelow('True')  # 使散点覆盖在坐标系之上
        # colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'pink', 'orange', 'purple'
        # for i, c, label in zip(target_ids, colors, target_names):
        #     plt.scatter(X_4[y_4 == i, 0], X_4[y_4 == i, 1], c=c, label=label)
        # plt.legend()
        # plt.savefig('/home/fanxiaoyi/legr+hrank/plt/layer3.jpg')
        # plt.show() 

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        if self.num_layers == 110:
            x = self.linear(x)
        else:
            x = self.fc(x)

        return x


def resnet_56(num_classes=10):
    cov_cfg = [(3 * i + 2) for i in range(9 * 3 * 2 + 1)]
    return ResNet(ResBasicBlock, 56, cov_cfg, num_classes=num_classes)


def resnet_110(num_classes=10):
    cov_cfg = [(3 * i + 2) for i in range(18 * 3 * 2 + 1)]
    return ResNet(ResBasicBlock, 110, cov_cfg, num_classes=num_classes)

########################## test ##########################
# print('---------------------- model ----------------------' + str(resnet_56()))


if __name__ == "__main__":
    from drives import *
    startTime = time.time()
    print('torch\'s version --- ' + torch.__version__ + '\ntorchvision\'s version --- ' + torchvision.__version__)
    
    # Parameters
    img_size = 32
    dataset = 'torchvision.datasets.CIFAR10'
    datapath = '/home/fanxiaoyi/LED-PQ -all/data'
    batch_size = 128
    no_val = True
    long_ft = 600
    lr = 0.01
    weight_decay = 5e-4
    name = 'resnet_56_CIFAR10'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device --- ' + str(device))

    # Data
    print('==> Preparing data..')
    train_loader, val_loader, test_loader = get_dataloader(img_size, dataset, datapath, batch_size, no_val)
    
    # Model
    print('==> Building model..')
    model = resnet_56(num_classes=10).to(device)
    # model.load_state_dict(torch.load('/home/fanxiaoyi/legr+hrank/model/ckpt/resnet_56_CIFAR10.t7'))
    # model.load_state_dict('/home/fanxiaoyi/legr+hrank/model/ckpt/resnet_56_CIFAR10.t7')
    print(model)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [int(long_ft*0.3), int(long_ft*0.6), int(long_ft*0.8)], gamma=0.2)
    
    # Train
    train(model, train_loader, test_loader, optimizer, epochs=long_ft, scheduler=scheduler,  train_model_Running=True,device=device, name=name)
    # acc = test(model, test_loader, device=device)
    # print('Testing Accuracy {:.2f}'.format(acc))

