import torch
import numpy as np
import torch.nn as nn
from torchvision.models.resnet import Bottleneck
from model.resnet_cifar import *
from model.vgg_cifar import *
from model.MobileNetV2 import *
from pruner.filterpruner import FilterPruner
from model.imagenet_resnet import *
import os
os.environ['CUDA_VISIBLE_DEVICES']='3'

def get_num_gen(gen):
    return sum(1 for x in gen)
def is_leaf(model):
    return get_num_gen(model.children()) == 0

class BOPsCounterResNet(FilterPruner):
    def trace_layer(self, layer, x, sparse_channel, bit):
        y = layer.old_forward(x)
        if isinstance(layer, nn.Conv2d):
            if self.activation_index == 0:
                self.conv_in_channels = 3
                self.conv_out_channels = sparse_channel[self.activation_index]
            else:
                self.conv_in_channels = sparse_channel[self.activation_index - 1]
                self.conv_out_channels = sparse_channel[self.activation_index]
            # x = (torch.rand(conv_out_channels, conv_in_channels, 32, 32))
            # y = m.forward(x)
            h = y.shape[2]
            w = y.shape[3]
            self.cur_bops += h * w * self.conv_in_channels * self.conv_out_channels * layer.weight.size(2) * layer.weight.size(3) * bit[self.activation_index] * bit[self.activation_index]
            self.activation_index += 1

        elif isinstance(layer, nn.Linear):
            self.cur_bops += np.prod(layer.weight.shape) * 32 * 32
        # self.cur_bops/= 1E9

        return y
    def forward(self, x, sparse_channel, bit):
        self.activation_index = 0
        self.cur_bops = 0        
        def modify_forward(model):
            for child in model.children():
                if is_leaf(child):
                    def new_forward(m):
                        def lambda_forward(x):
                            return self.trace_layer(m, x, sparse_channel, bit)
                        return lambda_forward
                    child.old_forward = child.forward
                    child.forward = new_forward(child)
                else:
                    modify_forward(child)

        def restore_forward(model):
            for child in model.children():
                # leaf node
                if is_leaf(child) and hasattr(child, 'old_forward'): 
                    # Update
                    child.forward = child.old_forward
                    child.old_forward = None
                else:
                    restore_forward(child)
        modify_forward(model)
        y = model.forward(x)
        restore_forward(model)
        cur_bops = self.cur_bops
        return y, cur_bops
        
class BOPsCounterVGG(FilterPruner):
    def forward(self, x, sparse_channel, bit):
        if isinstance(self.model, nn.DataParallel):
            model = self.model.module
        else:
            model = self.model
        # Chainning convolutions
        # (use activation index to represent a conv)
        self.linear = []
        self.chains = {}
        self.cur_bops = 0
        self.activation_index = 0
        self.bn_for_conv = {}

        for layer, method in enumerate(model.features.children()):
            h = x.shape[2]
            w = x.shape[3]

            if isinstance(method, nn.Conv2d):
                if self.activation_index == 0:
                    self.conv_in_channels = 3
                    self.conv_out_channels = sparse_channel[self.activation_index]
                else:
                    self.conv_in_channels = sparse_channel[self.activation_index - 1]
                    self.conv_out_channels = sparse_channel[self.activation_index]

                    self.cur_bops +=  h * w * self.conv_in_channels * self.conv_out_channels * method.weight.size(2) * method.weight.size(3) * bit[self.activation_index] * bit[self.activation_index]

                # If this is full group_conv it should be bounded with last conv
                if method.groups == method.out_channels and method.groups == method.in_channels:
                    assert self.activation_index-1 not in self.chains, 'Previous conv has already chained to some other convs!'
                    self.chains[self.activation_index-1] = self.activation_index
                self.activation_index += 1
            elif isinstance(method, nn.BatchNorm2d):
                # activation-1 since we increased the index right after conv
                self.bn_for_conv[self.activation_index-1] = method
            x = method(x)

        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)

        for m in model.classifier.children():
            x = m(x)
            if isinstance(m, nn.Linear):
                self.linear.append(m)
                self.cur_bops += np.prod(m.weight.shape) * 32 * 32
        cur_bops = self.cur_bops
        return x, cur_bops    

class BOPsCounterMBNetV2(FilterPruner):
    def forward(self, x, sparse_channel, bit):
        if isinstance(self.model, nn.DataParallel):
            model = self.model.module
        else:
            model = self.model
        
        self.activations = []
        self.gradients = []
        self.weight_grad = []
        self.grad_index = 0

        self.linear = None
        # activation index to the instance of conv layer
        self.activation_to_conv = {}
        # retrieve next conv using activation index of conv
        self.next_conv = {}
        # retrieve next immediate bn layer using activation index of conv
        self.bn_for_conv = {}
        # Chainning convolutions
        # (use activation index to represent a conv)
        self.chains = {}
        self.cur_bops = 0

        self.activation_index = 0
        prev_blk_last_conv = -1

        for l1, m1 in enumerate(model.features.children()):
            skipped = False
            if isinstance(m1, InvertedResidual):
                if m1.use_res_connect:
                    skipped = True
                # m1 is nn.Sequential now
                m1 = m1.conv 

            # use for residual
            tmp_x = x 

            # In the beginning of InvertedResidual block, get prev_conv for chaining purpose
            if self.activation_index-1 >= 0:
                prev_blk_last_conv = self.activation_index-1

            cnt = 0
            for l2, m2 in enumerate(m1.children()):
                cnt += 1
                x = m2(x)
                h = x.shape[2]
                w = x.shape[3]
                if isinstance(m2, nn.Conv2d):
                    if self.activation_index == 0:
                        self.conv_in_channels = 3
                        self.conv_out_channels = sparse_channel[self.activation_index]
                    else:
                        self.conv_in_channels = sparse_channel[self.activation_index - 1]
                        self.conv_out_channels = sparse_channel[self.activation_index]
                    # self.conv_in_channels[self.activation_index] = m2.weight.size(1)
                    # self.conv_out_channels[self.activation_index] = m2.weight.size(0)
                    # self.cur_flops +=  h * w * m2.weight.size(0) * m2.weight.size(1) * m2.weight.size(2) * m2.weight.size(3)
                    self.cur_bops +=  h * w * self.conv_in_channels * self.conv_out_channels * m2.weight.size(2) * m2.weight.size(3) * bit[self.activation_index] * bit[self.activation_index]

                    # If this is full group_conv it should be bounded with last conv
                    if m2.groups == m2.out_channels and m2.groups == m2.in_channels:
                        assert self.activation_index-1 not in self.chains, 'Previous conv has already chained to some other convs!'
                        self.chains[self.activation_index-1] = self.activation_index

                    self.activation_to_conv[self.activation_index] = m2

                    if self.activation_index > 0:
                        self.next_conv[self.activation_index-1] = [self.activation_index]
                    self.activation_index += 1

                elif isinstance(m2, nn.BatchNorm2d):
                    # activation-1 since we increased the index right after conv
                    self.bn_for_conv[self.activation_index-1] = m2

            if cnt == 0:
                x = m1(x)

            # After we parse through the block, if this block is with residual
            if skipped:
                x = tmp_x + x
                if prev_blk_last_conv >= 0:
                    assert prev_blk_last_conv not in self.chains, 'Previous conv has already chained to some other convs!'
                    # activation-1 is the current convolution since we just increased the pointer
                    self.chains[prev_blk_last_conv] = self.activation_index-1

        for m in model.classifier.modules():
            if isinstance(m, nn.Linear):
                self.linear = m 
                self.cur_bops += np.prod(m.weight.shape) * 32 * 32
                # self.base_flops = np.prod(m.weight.shape)
                # self.cur_flops += self.base_flops
        cur_bops = self.cur_bops
        return model.classifier(x.view(x.size(0), -1)),cur_bops
if __name__ == "__main__":
    device = 'cuda:{}'.format('0') if torch.cuda.is_available() else 'cpu'
    # sparse_channel = [64, 64, 13, 128, 256, 26, 256, 512, 52, 512, 512, 52, 52]#vgg_60
    # bit_w = [32, 8, 6, 8, 6, 4, 6, 4, 2, 4, 4, 2, 2]
    # bit_a = [32, 8, 6, 8, 6, 4, 6, 4, 2, 4, 4, 2, 2]
    sparse_channel = [64, 7, 64, 64, 64, 128, 128, 128, 13, 128, 256, 256, 256, 107, 256, 512, 512, 512, 512, 512]
    bit = [8, 4, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6]
    # bit_w = [32, 4, 6, 6, 6, 6, 6, 6, 4, 6, 4, 4, 4, 3, 4, 4, 4, 4, 4, 4]
    # sparse_channel = [64]*5 + [128]*5 + [256]*5 + [512]*5
    # bit = [32]*20
    # model = vgg_16_bn().to(device)
    # model = MobileNetV2_CIFAR10(n_class=100).to(device)
    model = resnet18(pretrained=False).to(device)
    model.eval()
    # bops_counter = eval('BOPsCounterVGG')(model, 'Rank', num_cls=10, rankPath='./rank_conv/CIFAR10/vgg_16_bn', device=device)
    bops_counter = eval('BOPsCounterResNet')(model, 'Rank', num_cls=1000, rankPath='./rank_conv/resnet18_limit6/', device=device)
    # bops_counter = eval('BOPsCounterMBNetV2')(model, 'Rank', num_cls=10, rankPath='./rank_conv/CIFAR100/mobilenetV2', device=device)
    _, cur_bops = bops_counter.forward(torch.zeros((1,3,224, 224), device=device), sparse_channel, bit)
    print('BOPs: {:.3f}G'.format(cur_bops/1e9))