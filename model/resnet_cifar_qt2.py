import time
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from model.modules import *
from pruner.filterpruner import FilterPruner
# filters_left = [16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,4,32,4,32,4,32,4,32,12,32,17,32,19,32,18,32,16,32,7,64,7,64,7,64,7,64,7,64,7,64,7,64,7,64,7,64]
# filters_left = [16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,4,32,4,32,4,32,4,32,4,32,4,32,17,32,29,32,27,32,7,64,7,64,7,64,7,64,7,64,7,64,7,64,7,64,7,64]
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

    def __init__(self, inplanes, planes_1, planes_2, stride, ka_1, ka_2, kw_1, kw_2):
        super(ResBasicBlock, self).__init__()
        self.inplanes = inplanes
        self.planes_1 = planes_1
        self.planes_2 = planes_2
        conv2d_q_1 = conv2d_Q_fn(kw_1)
        # self.conv1 = Conv2d_A_W(inplanes, planes_1, ksize = 3, stride= 1, padding= 1, dilation= 1, bias=False)
        self.conv1 = conv2d_q_1(inplanes, planes_1, ksize = 3, stride = stride, padding= 1, dilation= 1, bias=False)
        # self.conv1 = conv3x3(inplanes, planes_1, stride)
        self.bn1 = nn.BatchNorm2d(planes_1)
        # self.relu1 = nn.ReLU(inplace=True)
        self.relu1 = activation_quantize_fn(ka_1)

        conv2d_q_2 = conv2d_Q_fn(kw_2)
        self.conv2 = conv2d_q_2(planes_1, planes_2, ksize = 3, stride = 1, padding= 1, dilation= 1, bias=False)
        # self.conv2 = Conv2d_A_W(planes_1, planes_2, ksize = 3, stride= 1, padding= 1, dilation= 1, bias=False)
        # self.conv2 = conv3x3(planes_1, planes_2)
        self.bn2 = nn.BatchNorm2d(planes_2)
        # self.relu2 = nn.ReLU(inplace=True)
        self.relu2 = activation_quantize_fn(ka_2)
        self.stride = stride
        self.shortcut = nn.Sequential()

        # Shortcut
        # if stride != 1 or inplanes != planes_2:
        if stride != 1:
           self.shortcut = DownsampleA(inplanes, planes_2, stride)


    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # Residual
        out += self.shortcut(x)
        # out = self.relu2(out)

        return out


# ResNet
class QResNet_A_W(nn.Module):
    def __init__(self, block, num_layers, covcfg, num_classes, filters_left, ka, kw):
        super(QResNet_A_W, self).__init__()
        assert (num_layers - 2) % 6 == 0, 'depth should be 6n+2'
        # Computer the number of ResBasicBlock needed
        n = (num_layers - 2) // 6
        self.covcfg = covcfg
        self.num_layers = num_layers
        self.ka = ka
        self.kw = kw
        self.inplanes = filters_left[0]

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        # conv2d_q = conv2d_Q_fn(4)
        # self.conv1 = conv2d_q(3, self.inplanes, ksize = 3, stride= 1, padding= 1, dilation= 1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.relu = activation_quantize_fn(3)

        self.layer1 = self._make_layer(block, 16, blocks=n, stride=1, slim_channel=filters_left[1:2 * n + 1], ka = ka[1:2 * n + 1], kw = kw[1:2 * n + 1])
        self.layer2 = self._make_layer(block, 32, blocks=n, stride=2, slim_channel=filters_left[2 * n + 1:4 * n + 1], ka = ka[2 * n + 1:4 * n + 1], kw = kw[2 * n + 1:4 * n + 1])
        self.layer3 = self._make_layer(block, 64, blocks=n, stride=2, slim_channel=filters_left[4 * n + 1:6 * n + 1], ka = ka[4 * n + 1:6 * n + 1], kw = kw[4 * n + 1:6 * n + 1])
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


    def _make_layer(self, block, planes, blocks, stride, slim_channel, ka, kw):
        layers = []
        layers.append(block(self.inplanes, slim_channel[0], slim_channel[1], stride, ka[0], ka[1], kw[0], kw[1]))
        self.inplanes = planes * block.expansion
    
        for i in range(1, blocks):
            inplanes = slim_channel[2*i-1]
            planes_1 = slim_channel[2*i]
            planes_2 = slim_channel[2*i+1]
            ka_1 = ka[2*i]
            ka_2 = ka[2*i+1]
            kw_1 = kw[2*i]
            kw_2 = kw[2*i+1]
            layers.append(block(inplanes, planes_1, planes_2, 1, ka_1, ka_2, kw_1, kw_2))
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        if self.num_layers == 110:
            x = self.linear(x)
        else:
            x = self.fc(x)

        return x


def qresnet_56_A_W(num_classes, filters_left, bit):
    cov_cfg = [(3 * i + 2) for i in range(9 * 3 * 2 + 1)]
    return QResNet_A_W(ResBasicBlock, 56, cov_cfg, num_classes=num_classes, filters_left=filters_left,ka=bit, kw=bit)


def qresnet_110_A_W(num_classes, filters_left, bit):
    cov_cfg = [(3 * i + 2) for i in range(18 * 3 * 2 + 1)]
    return QResNet_A_W(ResBasicBlock, 110, cov_cfg, num_classes=num_classes, filters_left=filters_left,ka=bit, kw=bit)

########################## test ##########################
# print('---------------------- model ----------------------' + str(resnet_56()))


# if __name__ == "__main__":
#     from drives import *
#     startTime = time.time()
#     print('torch\'s version --- ' + torch.__version__ + '\ntorchvision\'s version --- ' + torchvision.__version__)
    
#     # Parameters
#     img_size = 32
#     dataset = 'torchvision.datasets.CIFAR10'
#     datapath = '/home/fanxiaoyi/legr+hrank/data'
#     batch_size = 500
#     no_val = True
#     long_ft = 10
#     lr = 0.01
#     weight_decay = 5e-4
#     name = 'resnet_56_CIFAR10'
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     print('device --- ' + str(device))

#     # Data
#     print('==> Preparing data..')
#     train_loader, val_loader, test_loader = get_dataloader(img_size, dataset, datapath, batch_size, no_val)
    
#     # Model
#     print('==> Building model..')
#     model = resnet_56(num_classes=10).to(device)
#     # print(model)
#     optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
#     scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [int(long_ft*0.3), int(long_ft*0.6), int(long_ft*0.8)], gamma=0.2)
    
#     # Train
#     train(model, train_loader, test_loader, optimizer, epochs=long_ft, scheduler=scheduler,  train_model_Running=True,device=device, name=name)

