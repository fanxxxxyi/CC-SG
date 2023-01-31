import torch
import numpy as np
import torch.nn as nn

from pruner.filterpruner import FilterPruner


class FilterPrunerVGG(FilterPruner):
    def parse_dependency(self):
        pass

    def forward(self, x):
        if isinstance(self.model, nn.DataParallel):
            model = self.model.module
        else:
            model = self.model

        self.activations = []
        self.linear = []
        # activation index to the instance of conv layer
        self.activation_to_conv = {}
        # retrieve next conv using activation index of conv
        self.next_conv = {}
        # retrieve next immediate bn layer using activation index of conv
        self.bn_for_conv = {}
        # Chainning convolutions
        # (use activation index to represent a conv)
        self.chains = {}
        
        activation_index = 0

        for layer, method in enumerate(model.features.children()):
            h = x.shape[2]
            w = x.shape[3]

            if isinstance(method, nn.Conv2d):
                self.conv_in_channels[activation_index] = method.weight.size(1)
                self.conv_out_channels[activation_index] = method.weight.size(0)
                self.omap_size[activation_index] = (h, w)
                self.cost_map[activation_index] = h * w * method.weight.size(2) * method.weight.size(3)

                self.in_params[activation_index] = method.weight.size(1) * method.weight.size(2) * method.weight.size(3)
                self.cur_flops +=  h * w * method.weight.size(0) * method.weight.size(1) * method.weight.size(2) * method.weight.size(3)

                # If this is full group_conv it should be bounded with last conv
                if method.groups == method.out_channels and method.groups == method.in_channels:
                    assert activation_index-1 not in self.chains, 'Previous conv has already chained to some other convs!'
                    self.chains[activation_index-1] = activation_index

                if self.rank_type == 'l1_weight':
                    if activation_index not in self.filter_ranks:
                        self.filter_ranks[activation_index] = torch.zeros(method.weight.size(0), device=self.device)
                    values = (torch.abs(method.weight.data)).sum(1).sum(1).sum(1)

                    self.filter_ranks[activation_index] = values
                elif self.rank_type == 'l2_weight': 
                    if activation_index not in self.filter_ranks:
                        self.filter_ranks[activation_index] = torch.zeros(method.weight.size(0), device=self.device)
                    values = (torch.pow(method.weight.data, 2)).sum(1).sum(1).sum(1)

                    self.filter_ranks[activation_index] = values
                elif self.rank_type == 'Rank': 
                    if activation_index not in self.filter_ranks:
                        self.filter_ranks[activation_index] = torch.zeros(method.weight.size(0), device=self.device)

                    values = torch.from_numpy(np.load(self.rankPath + '/rank_conv%d.npy' %(activation_index + 1)))
                    self.filter_ranks[activation_index] = values
                elif self.rank_type == 'ci':
                    if activation_index not in self.filter_ranks:
                        self.filter_ranks[activation_index] = torch.zeros(method.weight.size(0), device=self.device)

                    values = torch.from_numpy(np.load(self.rankPath + '/ci_conv%d.npy' %(activation_index + 1)))
                    self.filter_ranks[activation_index] = values
                elif self.rank_type == 'l2_bn' or self.rank_type == 'l1_bn': 
                    pass
                else:
                    x.register_hook(self.compute_rank)
                    self.activations.append(x)
                self.rates[activation_index] = self.conv_in_channels[activation_index] * self.cost_map[activation_index]
                self.activation_to_conv[activation_index] = method

                if activation_index > 0:
                    self.next_conv[activation_index-1] = [activation_index]
                activation_index += 1
            elif isinstance(method, nn.BatchNorm2d):
                # activation-1 since we increased the index right after conv
                self.bn_for_conv[activation_index-1] = method
                if self.rank_type == 'l2_bn':
                    if activation_index-1 not in self.filter_ranks:
                        self.filter_ranks[activation_index-1] = torch.zeros(method.weight.size(0), device=self.device)
                    values = torch.pow(method.weight.data, 2)
                    self.filter_ranks[activation_index-1] = values

                elif self.rank_type == 'l2_bn_param': 
                    if activation_index-1 not in self.filter_ranks:
                        self.filter_ranks[activation_index-1] = torch.zeros(method.weight.size(0), device=self.device)
                    values = torch.pow(method.weight.data, 2)
                    self.filter_ranks[activation_index-1] = values* self.in_params[activation_index-1]
            x = method(x)

        # x = nn.AvgPool2d(15)(x)
        # x = x.view(x.size(0), -1) #vgg16
        x = nn.AdaptiveAvgPool2d((7, 7))(x)
        x = torch.flatten(x, 1) #vgg19
        

        for m in model.classifier.children():
            x = m(x)
            if isinstance(m, nn.Linear):
                self.linear.append(m)
                self.base_flops += np.prod(m.weight.shape)
                self.cur_flops += np.prod(m.weight.shape)

        self.og_conv_in_channels = self.conv_in_channels.copy()
        self.og_conv_out_channels = self.conv_out_channels.copy()
        self.resource_usage = self.cur_flops
        return x

    def bops_forward(self, x, sparse_channel, bit):
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

        # x = nn.AvgPool2d(15)(x)
        # x = x.view(x.size(0), -1)
        x = nn.AdaptiveAvgPool2d((7, 7))(x)
        x = torch.flatten(x, 1)

        for m in model.classifier.children():
            x = m(x)
            if isinstance(m, nn.Linear):
                self.linear.append(m)
                self.cur_bops += np.prod(m.weight.shape) * 32 * 32
        cur_bops = self.cur_bops
        return x, cur_bops    

    def mask_conv_layer_segment(self, layer_index, filter_range):
        filters_begin = filter_range[0]
        filters_end = filter_range[1]
        pruned_filters = filters_end - filters_begin + 1
        # Retrive conv based on layer_index
        conv = self.activation_to_conv[layer_index]

        #if layer_index in self.pre_padding:
        #    self.pre_padding[layer_index].out_channels -= pruned_filters
        next_bn = self.bn_for_conv[layer_index]
        next_conv_idx = self.next_conv[layer_index] if layer_index in self.next_conv else None

        # Surgery on the conv layer to be pruned
        # dw-conv, reduce groups as well
        conv.weight.data[filters_begin:filters_end+1,:,:,:] = 0
        conv.weight.grad = None

        if not conv.bias is None:
            conv.bias.data[filters_begin:filters_end+1] = 0
            conv.bias.grad = None
            
        next_bn.weight.data[filters_begin:filters_end+1] = 0
        next_bn.weight.grad = None

        next_bn.bias.data[filters_begin:filters_end+1] = 0
        next_bn.bias.grad = None

        next_bn.running_mean.data[filters_begin:filters_end+1] = 0
        next_bn.running_mean.grad = None

        next_bn.running_var.data[filters_begin:filters_end+1] = 0
        next_bn.running_var.grad = None

    def prune_conv_layer_segment(self, layer_index, filter_range):
        filters_begin = filter_range[0]
        filters_end = filter_range[1]
        pruned_filters = int(filters_end - filters_begin + 1)
        # Retrive conv based on layer_index
        conv = self.activation_to_conv[layer_index]
        # print(conv)
        next_bn = self.bn_for_conv[layer_index]
        next_conv_idx = self.next_conv[layer_index] if layer_index in self.next_conv else None

        # Surgery on the conv layer to be pruned
        # dw-conv, reduce groups as well
        if conv.groups == conv.out_channels and conv.groups == conv.in_channels:
            new_conv = \
                torch.nn.Conv2d(in_channels = conv.out_channels - pruned_filters, \
                        out_channels = conv.out_channels - pruned_filters,
                        kernel_size = conv.kernel_size, \
                        stride = conv.stride,
                        padding = conv.padding,
                        dilation = conv.dilation,
                        groups = conv.groups - pruned_filters,
                        # bias = conv.bias
                        bias = True if not conv.bias is None else False
                        )

            conv.in_channels -= pruned_filters
            conv.out_channels -= pruned_filters
            conv.groups -= pruned_filters
        else:
            new_conv = \
                torch.nn.Conv2d(in_channels = conv.in_channels, \
                        out_channels = conv.out_channels - pruned_filters,
                        kernel_size = conv.kernel_size, \
                        stride = conv.stride,
                        padding = conv.padding,
                        dilation = conv.dilation,
                        groups = conv.groups,
                        # bias = conv.bias
                        bias = True if not conv.bias is None else False
                        )

            conv.out_channels -= pruned_filters
        old_weights = conv.weight.data.cpu().numpy()
        new_weights = new_conv.weight.data.cpu().numpy()
        new_weights[: filters_begin, :, :, :] = old_weights[: filters_begin, :, :, :]
        new_weights[filters_begin : , :, :, :] = old_weights[filters_end + 1 :, :, :, :]
        conv.weight.data = torch.from_numpy(new_weights).to(self.device)
        conv.weight.grad = None

        if not conv.bias is None:
            bias_numpy = conv.bias.data.cpu().numpy()

            bias = np.zeros(shape = (bias_numpy.shape[0] - pruned_filters), dtype = np.float32)
            bias[:filters_begin] = bias_numpy[:filters_begin]
            bias[filters_begin : ] = bias_numpy[filters_end + 1 :]
            conv.bias.data = torch.from_numpy(bias).to(self.device)
            conv.bias.grad = None
            
        # Surgery on next batchnorm layer
        next_new_bn = \
            torch.nn.BatchNorm2d(num_features = next_bn.num_features-pruned_filters,\
                    eps =  next_bn.eps, \
                    momentum = next_bn.momentum, \
                    affine = next_bn.affine,
                    track_running_stats = next_bn.track_running_stats)
        next_bn.num_features -= pruned_filters

        old_weights = next_bn.weight.data.cpu().numpy()
        new_weights = next_new_bn.weight.data.cpu().numpy()
        old_bias = next_bn.bias.data.cpu().numpy()
        new_bias = next_new_bn.bias.data.cpu().numpy()
        old_running_mean = next_bn.running_mean.data.cpu().numpy()
        new_running_mean = next_new_bn.running_mean.data.cpu().numpy()
        old_running_var = next_bn.running_var.data.cpu().numpy()
        new_running_var = next_new_bn.running_var.data.cpu().numpy()

        new_weights[: filters_begin] = old_weights[: filters_begin]
        new_weights[filters_begin :] = old_weights[filters_end + 1 :]
        next_bn.weight.data = torch.from_numpy(new_weights).to(self.device)
        next_bn.weight.grad = None

        new_bias[: filters_begin] = old_bias[: filters_begin]
        new_bias[filters_begin :] = old_bias[filters_end + 1 :]
        next_bn.bias.data = torch.from_numpy(new_bias).to(self.device)
        next_bn.bias.grad = None

        new_running_mean[: filters_begin] = old_running_mean[: filters_begin]
        new_running_mean[filters_begin :] = old_running_mean[filters_end + 1 :]
        next_bn.running_mean.data = torch.from_numpy(new_running_mean).to(self.device)
        next_bn.running_mean.grad = None

        new_running_var[: filters_begin] = old_running_var[: filters_begin]
        new_running_var[filters_begin :] = old_running_var[filters_end + 1 :]
        next_bn.running_var.data = torch.from_numpy(new_running_var).to(self.device)
        next_bn.running_var.grad = None
        

        # Found next convolution layer
        if next_conv_idx:
            for next_conv_i in next_conv_idx:
                next_conv = self.activation_to_conv[next_conv_i]
                next_new_conv = \
                    torch.nn.Conv2d(in_channels = next_conv.in_channels - pruned_filters,\
                            out_channels =  next_conv.out_channels, \
                            kernel_size = next_conv.kernel_size, \
                            stride = next_conv.stride,
                            padding = next_conv.padding,
                            dilation = next_conv.dilation,
                            groups = next_conv.groups,
                            # bias = next_conv.bias
                            bias = True if not conv.bias is None else False
                            )
                next_conv.in_channels -= pruned_filters

                old_weights = next_conv.weight.data.cpu().numpy()
                new_weights = next_new_conv.weight.data.cpu().numpy()

                new_weights[:, : filters_begin, :, :] = old_weights[:, : filters_begin, :, :]
                new_weights[:, filters_begin : , :, :] = old_weights[:, filters_end + 1 :, :, :]
                next_conv.weight.data = torch.from_numpy(new_weights).to(self.device)
                next_conv.weight.grad = None
        else:
            # Prunning the last conv layer. This affects the first linear layer of the classifier.
            if self.linear[0] is None:
                raise BaseException("No linear laye found in classifier")
            params_per_input_channel = int(self.linear[0].in_features / (conv.out_channels+pruned_filters))

            new_linear_layer = \
                    torch.nn.Linear(self.linear[0].in_features - pruned_filters*params_per_input_channel, 
                            self.linear[0].out_features)

            self.linear[0].in_features -= pruned_filters*params_per_input_channel
            
            old_weights = self.linear[0].weight.data.cpu().numpy()
            new_weights = new_linear_layer.weight.data.cpu().numpy()	 	

            new_weights[:, : int(filters_begin * params_per_input_channel)] = \
                    old_weights[:, : int(filters_begin * params_per_input_channel)]
            new_weights[:, int(filters_begin * params_per_input_channel) :] = \
                    old_weights[:, int((filters_end + 1) * params_per_input_channel) :]
            
            self.linear[0].weight.data = torch.from_numpy(new_weights).to(self.device)
            self.linear[0].weight.grad = None


