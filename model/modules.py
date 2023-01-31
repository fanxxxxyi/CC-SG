import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import numpy as np

def get_device(gpu_ind):
    if torch.cuda.is_available():
        print('Let us use GPU.')
        cudnn.benchmark = True
        if torch.cuda.device_count() == 1:
            device = torch.device('cuda')
        else:
            device = torch.device('cuda:%d' % gpu_ind)
    else:
        print('Come on !! No GPU ?? Who gives you the courage to study Deep Learning ?')
        device = torch.device('cpu')

    return device
## step1:激活函数量化（可作为模块化使用）
class Conv2d_A(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, padding=0, stride=1, dilation=1):
        super(Conv2d_A, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride=stride, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            activation_quantize_fn(3)
        )

    def forward(self, x):
        return self.convs(x)
## step2:权重和激活函数量化（可作为模块化使用）
# class Conv2d_A_W(nn.Module):
#     def __init__(self, in_channels, out_channels, ksize, w_bit, a_bit, padding=0, stride=1, dilation=1, groups=1, bias=False):
#         super(Conv2d_A_W, self).__init__()
#         conv2d_q = conv2d_Q_fn(w_bit)
#         self.convs = nn.Sequential(
#             conv2d_q(in_channels, out_channels, ksize, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False),
#             nn.BatchNorm2d(out_channels),
#             activation_quantize_fn(a_bit)
#         )

#     def forward(self, x):
#         return self.convs(x)

class reorg_layer(nn.Module):
    def __init__(self, stride):
        super(reorg_layer, self).__init__()
        self.stride = stride

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        _height, _width = height // self.stride, width // self.stride
        
        x = x.view(batch_size, channels, _height, self.stride, _width, self.stride).transpose(3, 4).contiguous()
        x = x.view(batch_size, channels, _height * _width, self.stride * self.stride).transpose(2, 3).contiguous()
        x = x.view(batch_size, channels, self.stride * self.stride, _height, _width).transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, _height, _width)

        return x

#####对称量化模块###
def uniform_quantize(k):
    class qfn(torch.autograd.Function):

        @staticmethod
        def forward(ctx, input):
            if k == 32:
                out = input
            elif k == 1:
                out = torch.sign(input)
            else:
                n = float(2 ** k  - 1)
                out = torch.round(input * n) / n
            return out
        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()
            return grad_input

    return qfn().apply

#####权重量化模块###
class weight_quantize_fn(nn.Module):
    def __init__(self, w_bit):
        super(weight_quantize_fn, self).__init__()
        assert w_bit <= 8 or w_bit == 32
        self.w_bit = w_bit
        # 符号位 占一位
        self.uniform_q = uniform_quantize(k=w_bit - 1)

    def forward(self, x):
        if self.w_bit == 32:
            weight_q = x
        elif self.w_bit == 1:
            E = torch.mean(torch.abs(x)).detach()   #torch.mean(input) → float 返回输入张量所有元素的均值。
            weight_q = (self.uniform_q(x / E) + 1) / 2 * E
        else:
            weight = torch.tanh(x)
            weight = weight / torch.max(torch.abs(weight))   ##归一化处理
            weight_q = self.uniform_q(weight)
        return weight_q

#####卷积量化#####
def conv2d_Q_fn(w_bit):
    class Conv2d_Q(nn.Conv2d):
        def __init__(self, in_channels, out_channels, ksize, stride=1, padding=0, dilation=1, groups=1, bias=False):
            super(Conv2d_Q, self).__init__(in_channels, out_channels, ksize, stride, padding, dilation, groups, bias)
            self.w_bit = w_bit
            self.quantize_fn = weight_quantize_fn(w_bit=w_bit)

        def forward(self, input, order=None):
            weight_q = self.quantize_fn(self.weight)
            return F.conv2d(input, weight_q, self.bias, self.stride,self.padding, self.dilation, self.groups)

    return Conv2d_Q

class QConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, ksize, stride, padding, dilation, groups, bias, w_bit):
        super(QConv2d, self).__init__(in_channels, out_channels, ksize, stride, padding, dilation, groups, bias, w_bit)
        self.w_bit = w_bit
        self.uniform_q = uniform_quantize(k=w_bit - 1)
        # self.quantize_fn = weight_quantize_fn(w_bit=w_bit)

    def forward(self, input, order=None):
        if self.w_bit == 32:
            weight_q = self.weight
        elif self.w_bit == 1:
            E = torch.mean(torch.abs(self.weight)).detach()   #torch.mean(input) → float 返回输入张量所有元素的均值。
            weight_q = (self.uniform_q(self.weight / E) + 1) / 2 * E
        else:
            weight = torch.tanh(self.weight)
            weight = weight / torch.max(torch.abs(weight))   ##归一化处理
            weight_q = self.uniform_q(weight)
        # weight_q = self.quantize_fn(self.weight)
        return F.conv2d(input, weight_q, self.bias, self.stride,self.padding, self.dilation, self.groups)
#####激活量化#####
class activation_quantize_fn(nn.Module):
    def __init__(self, a_bit):
        super(activation_quantize_fn, self).__init__()
        assert a_bit <= 8 or a_bit == 32
        self.a_bit = a_bit
        self.uniform_q = uniform_quantize(k=a_bit)  #gai

    def forward(self, x):
        if self.a_bit == 32:
            activation_q = x
        else:
            activation_q = self.uniform_q(torch.clamp(x, 0, 1))   ##将x取值在0-1
            # print(np.unique(activation_q.detach().numpy()))
        return activation_q