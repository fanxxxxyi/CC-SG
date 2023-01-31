import os
import time
import math
import torch
import torchvision
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict


os.environ['CUDA_VISIBLE_DEVICES']='0'
defaultcfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 512]
# relu layer config
relucfg = [2, 6, 9, 13, 16, 19, 23, 26, 29, 33, 36, 39, 42]
# convolution layer config
convcfg = [0, 3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37, 40]


# VGG
class VGG(nn.Module):
    def __init__(self, num_classes=21, init_weights=True, cfg=None):
        super(VGG, self).__init__()
        self.features = nn.Sequential()
        if cfg is None:
            cfg = defaultcfg
        self.relucfg = relucfg
        self.covcfg = convcfg
        self.features = self.make_layers(cfg[:-1], True)
        self.classifier = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(cfg[-2], cfg[-1])),
            ('norm1', nn.BatchNorm1d(cfg[-1])),
            ('relu1', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(cfg[-1], num_classes)),
        ]))

        # Initialize weights
        if init_weights:
            self._initialize_weights()


    def make_layers(self, cfg, batch_norm=True):
        layers = nn.Sequential()
        in_channels = 3
        cnt = 0
        for i, v in enumerate(cfg):
            if v == 'M':
                layers.add_module('pool%d' % i, nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=True)
                cnt += 1
                layers.add_module('conv%d' % i, conv2d)
                layers.add_module('norm%d' % i, nn.BatchNorm2d(v))
                layers.add_module('relu%d' % i, nn.ReLU(inplace=True))
                in_channels = v
        return layers


    def forward(self, x):
        x = self.features(x)

        x = nn.AvgPool2d(15)(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    # Initialize Weights Function
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n)) 
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def vgg_16_bn():
    return VGG(num_classes=21)


#################### Test ####################
# print('--- model ---\n' + str(vgg_16_bn()))



if __name__ == "__main__":
    from drives import *
    startTime = time.time()
    print('torch\'s version --- ' + torch.__version__ + '\ntorchvision\'s version --- ' + torchvision.__version__)
    name = 'vgg16bn_UCML'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device --- ' + str(device))
    # Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=300, help='Epoch')
    parser.add_argument('--batch_size', type=int, default=16, help='Batchsize')
    parser.add_argument('--img_size', type=int, default=256, help='img_size')
    parser.add_argument('--base_lr', type=float, default=0.001, help='Learning_rate') 
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--weight_decay', type=float, nargs='+', default=1e-4, help='Weight_decay')  
    args = parser.parse_args()


    # Data
    print('==> Preparing data..')
    # train_loader, val_loader, test_loader = get_dataloader(img_size, dataset, datapath, batch_size, no_val)
    train_loader, test_loader = get_ucml_dataloader(args.batch_size)
    
    # Model
    print('==> Building model..')
    model = vgg_16_bn().to(device)
    # vggmodel = models.vgg16_bn(pretrained=False)
    # state_dict = torch.load(r"/home/fanxiaoyi/LEDPQ_YOLO/vgg16_bn-6c64b313.pth")
    # vggmodel.load_state_dict(state_dict)
    # new_state_dict = vggmodel.state_dict()

    # model = vgg_16_bn()
    # op = model.state_dict()

    # print(len(new_state_dict.keys()))  # 输出torch官方网络模型字典长度
    # print(len(op.keys()))# 输出自己网络模型字典长度
    # for i in new_state_dict.keys():   # 查看网络结构的名称 并且得出一共有320个key
    #     print(i)
    # for j in op.keys():   # 查看网络结构的名称 并且得出一共有384个key
    #     print(j)

    # 无论名称是否相同都可以使用
    # for new_state_dict_num, new_state_dict_value in enumerate(new_state_dict.values()):
    #     for op_num, op_key in enumerate(op.keys()):
    #         if op_num == new_state_dict_num and op_num <= 90:  # 不需要最后的全连接层的参数
    #             op[op_key] = new_state_dict_value
    # model.load_state_dict(op)  # 更改了state_dict的值记得把它导入网络中



    # print(model)
    # optimizer = get_optimizer(args, model)
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [int(args.epoch*0.5), int(args.epoch*0.75)], gamma=0.1)
    # Train
    train(model, train_loader, test_loader, optimizer, scheduler, epochs=args.epoch,  train_model_Running=True, device=device, name=name)


