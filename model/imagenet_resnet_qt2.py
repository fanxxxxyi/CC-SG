import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from model.modules import *

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
def conv3x3_conv2(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, ka_1, ka_2, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = activation_quantize_fn(ka_1)
        # self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3_conv2(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = activation_quantize_fn(ka_2)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes_1, planes_2, planes_3, ka_1, ka_2, ka_3, kw_1, kw_2, kw_3, stride, downsample):
        super(Bottleneck, self).__init__()
        conv2d_q_1 = conv2d_Q_fn(kw_1)
        self.conv1 = conv2d_q_1(inplanes, planes_1, ksize = 1, stride = 1, padding= 0, dilation= 1, bias=False)
        # self.conv1 = nn.Conv2d(inplanes, planes_1, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes_1)
        self.relu1 = activation_quantize_fn(ka_1)
        conv2d_q_2 = conv2d_Q_fn(kw_2)
        self.conv2 = conv2d_q_2(planes_1, planes_2, ksize = 3, stride = stride, padding= 1, dilation= 1, bias=False)
        # self.conv2 = nn.Conv2d(planes_1, planes_2, kernel_size=3, stride=stride,
        #                        padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes_2)
        self.relu2 = activation_quantize_fn(ka_2)
        conv2d_q_3 = conv2d_Q_fn(kw_3)
        self.conv3 = conv2d_q_3(planes_2, planes_3, ksize = 1, stride = 1, padding= 0, dilation= 1, bias=False)
        # self.conv3 = nn.Conv2d(planes_2, planes_3, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes_3)
        self.relu3 = activation_quantize_fn(ka_3)
        # self.relu = nn.ReLU(inplace=True)
        # self.relu = nn.ReLU(inplace=False)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu3(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, filters_left, ka, kw):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, filters_left[0], kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(filters_left[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], slim_channel=filters_left[0:3 * 3 + 2], ka = ka[1:3 * 3 + 2], kw = kw[1:3 * 3 + 2])
        self.layer2 = self._make_layer(block, 128, layers[1], slim_channel=filters_left[3 * 3 + 1:3 * 7 + 3], ka = ka[3 * 3 + 2:3 * 7 + 3], kw = kw[3 * 3 + 2:3 * 7 + 3], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], slim_channel=filters_left[3 * 7 + 2:3 * 13 + 4], ka = ka[3 * 7 + 3:3 * 13 + 4], kw = kw[3 * 7 + 3:3 * 13 + 4], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], slim_channel=filters_left[3 * 13 + 3:3 * 16 + 5], ka = ka[3 * 13 + 4:3 * 16 + 5], kw = kw[3 * 13 + 4:3 * 16 + 5], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(filters_left[-1], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, slim_channel, ka, kw, stride=1):
        # downsample = None
        # if stride != 1 or self.inplanes != planes * block.expansion:
        #     downsample = nn.Sequential(
        #         nn.Conv2d(slim_channel[0], slim_channel[3],
        #                   kernel_size=1, stride=stride, bias=False),
        #         nn.BatchNorm2d(slim_channel[3]),
        #     )
        conv2d_q = conv2d_Q_fn(kw[3])
        downsample = nn.Sequential(
            conv2d_q(slim_channel[0], slim_channel[4], ksize = 1, stride = stride, padding= 0, dilation= 1, bias=False),
            # nn.Conv2d(slim_channel[0], slim_channel[4],
            #             kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(slim_channel[4]),
        )

        layers = []
        layers.append(block(slim_channel[0], slim_channel[1], slim_channel[2], slim_channel[3], ka[0], ka[1], ka[2], kw[0], kw[1], kw[2], stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
                inplanes = slim_channel[3*i+1]
                planes_1 = slim_channel[3*i+2]
                planes_2 = slim_channel[3*i+3]
                planes_3 = slim_channel[3*i+4]
                ka_1 = ka[3*i+1]
                ka_2 = ka[3*i+2]
                ka_3 = ka[3*i+3]
                kw_1 = kw[3*i+1]
                kw_2 = kw[3*i+2]
                kw_3 = kw[3*i+3]
                layers.append(block(inplanes, planes_1, planes_2, planes_3, ka_1, ka_2, ka_3, kw_1, kw_2, kw_3, stride = 1, downsample = None))
  
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        print('ResNet-18 Use pretrained model for initalization')
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
        print('ResNet-34 Use pretrained model for initalization')
    return model


def qresnet50_A_W(filters_left, bit, pretrained=False,  **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes = 21, filters_left=filters_left, ka=bit, kw=bit, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        print('ResNet-50 Use pretrained model for initalization')
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    #     print('ResNet-101 Use pretrained model for initalization')
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
        print('ResNet-152 Use pretrained model for initalization')
    return model

# if __name__ == "__main__":
#     from drives import *
#     startTime = time.time()
#     print('torch\'s version --- ' + torch.__version__ + '\ntorchvision\'s version --- ' + torchvision.__version__)
    
#     # Parameters
#     img_size = 256
#     batch_size = 16
#     long_ft = 300
#     lr = 0.001
#     weight_decay = 1e-4
#     name = 'resnet50_UCLM'
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     print('device --- ' + str(device))

#     # Data
#     print('==> Preparing data..')
#     # train_loader, val_loader, test_loader = get_dataloader(img_size, dataset, datapath, batch_size, no_val)
#     train_loader, test_loader = get_ucml_dataloader(batch_size)
    
#     # Model
#     print('==> Building model..')
#     model = resnet50(pretrained=False).to(device)
#     print(model)
#     optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
#     scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [int(long_ft*0.5), int(long_ft*0.75)], gamma=0.1)
    
#     # Train
#     train(model, train_loader, test_loader, optimizer, epochs=long_ft, scheduler=scheduler,  train_model_Running=True, device=device, name=name)