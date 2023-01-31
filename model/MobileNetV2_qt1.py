import math
import time
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model.modules import *


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, mid_channel, oup, stride, expand_ratio, ka_1, ka_2, ka_3):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
                nn.BatchNorm2d(inp * expand_ratio),
                # nn.ReLU6(inplace=True),
                activation_quantize_fn(ka_1),
                # pw-linear
                nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                # nn.ReLU6(inplace=True),
                activation_quantize_fn(ka_2),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, mid_channel, 1, 1, 0, bias=False),
                nn.BatchNorm2d(mid_channel),
                # nn.ReLU6(inplace=True),
                activation_quantize_fn(ka_1),
                # dw
                nn.Conv2d(mid_channel, mid_channel, 3, stride, 1, groups=mid_channel, bias=False),
                nn.BatchNorm2d(mid_channel),
                # nn.ReLU6(inplace=True),
                activation_quantize_fn(ka_2),
                # pw-linear
                nn.Conv2d(mid_channel, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                # nn.ReLU6(inplace=True),
                activation_quantize_fn(ka_3),
            )


    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class QMobileNetV2_CIFAR10_A(nn.Module):
    def __init__(self, n_class, filters_left, bit, input_size=32, width_mult=1.):
        super(QMobileNetV2_CIFAR10_A, self).__init__()
        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s, k
            [1, 16, 1, 1, 0],
            [6, 24, 2, 1, 3], # NOTE: change stride 2->1 for CIFAR10
            [6, 32, 3, 2, 9],
            [6, 64, 4, 1, 18], # NOTE: change stride 2->1 for CIFAR10
            [6, 96, 3, 1, 30],
            [6, 160, 3, 2, 39],
            [6, 320, 1, 1, 48],
        ]

        # building first layer
        # assert input_size % 4 == 0
        # input_channel = int(32 * width_mult)
        # self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        input_channel = filters_left[0]
        self.last_channel = filters_left[51]
        self.features = [conv_bn(3, input_channel, 1)] # NOTE: change stride 2->1 for CIFAR10
        slim_channel = []
        # building inverted residual blocks
        for t, c, n, s, k in self.interverted_residual_setting:
            # output_channel = int(c * width_mult)
            slim_channel = filters_left[k:k + 3 * n]
            ka = bit[k:k + 3 * n]
            
            for i in range(n):
                output_channel = slim_channel[3 * i + 2]
                mid_channel = slim_channel[3 * i + 1]
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, mid_channel, output_channel, s, t, ka[3 * i], ka[3 * i + 1], ka[3 * i + 2]))
                else:               
                    self.features.append(InvertedResidual(input_channel, mid_channel, output_channel, 1, t, ka[3 * i], ka[3 * i + 1], ka[3 * i + 2]))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        self.features.append(nn.AvgPool2d(int(input_size/4)))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()


    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.classifier[1].in_features)
        x = self.classifier(x)
        return x


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels / m.groups
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()



class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(32 * width_mult)
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, s, t))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        self.features.append(nn.AvgPool2d(int(input_size/32)))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()


    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.classifier[1].in_features)
        x = self.classifier(x)
        return x


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels / m.groups
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

def QMobileNetV2_A(n_class, filters_left, bit):
    return QMobileNetV2_CIFAR10_A(n_class = 100, filters_left = filters_left, bit = bit, input_size=32, width_mult=1.)

############################ test ############################
# print(MobileNetV2_CIFAR10(n_class=10, input_size=32, width_mult=1.0))


# if __name__ == "__main__":
#     from drives import *

#     print('torch\'s version --- ' + torch.__version__ + '\ntorchvision\'s version --- ' + torchvision.__version__)
    
#     # Parameters
#     img_size = 32
#     dataset = 'torchvision.datasets.CIFAR100'
#     datapath = './data'
#     batch_size = 128
#     no_val = True
#     long_ft = 500
#     lr = 0.01
#     weight_decay = 5e-4
#     name = 'MobileNetV2_CIFAR100'
#     # device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     if torch.cuda.is_available():
#         device = torch.device("cuda:1")
#     else:
#         device = "cpu"
#     print('device --- ' + str(device))

#     # Data
#     print('==> Preparing data..')
#     train_loader, val_loader, test_loader = get_dataloader(img_size, dataset, datapath, batch_size, no_val)
    
#     # Model
#     print('==> Building model..')
#     model = MobileNetV2_CIFAR10(n_class=100).to(device)
    
#     # print(model)
#     optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
#     scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [int(long_ft*0.3), int(long_ft*0.6), int(long_ft*0.8)], gamma=0.2)
#     # print(test(model, test_loader, device=device, get_loss=False))
#     # Train
#     train(model, train_loader, test_loader, optimizer, epochs=long_ft, scheduler=scheduler,  train_model_Running=True,device=device, name=name)


