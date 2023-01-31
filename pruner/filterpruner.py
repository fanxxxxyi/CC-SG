import torch
import numpy as np
import torch.nn as nn
from random import shuffle
from model.resnet_cifar import resnet_56
# from model.vgg_cifar import vgg_16_bn
# from count_bops import BOPsCounterResNet, BOPsCounterVGG, BOPsCounterMBNetV2


# FilterPruner
class FilterPruner(object):
    def __init__(self, model, rank_type='Rank', num_cls=100, safeguard=0, random=False, device='cuda', resource='FLOPs', rankPath=None):
        self.model = model
        self.rank_type = rank_type
        # Chainning convolutions
        # (use activation index to represent a conv)
        self.chains = {}
        self.y = None
        self.num_cls = num_cls
        self.safeguard = safeguard
        self.random = random
        self.device = device
        self.resource_type = resource
        self.rankPath = rankPath
        self.reset()


    # num_params
    def num_params(self):
        all_p = 0
        conv_p = 0
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                conv_p += np.prod(m.weight.shape)
                all_p += np.prod(m.weight.shape)
            if isinstance(m, nn.Linear):
                all_p += np.prod(m.weight.shape) 
        return all_p, conv_p
   

    def reset(self):
        self.amc_checked = []
        self.cur_flops = 0
        self.base_flops = 0
        self.cur_size, conv_size = self.num_params()
        self.base_size = self.cur_size - conv_size
        self.quota = None
        self.filter_ranks = {}
        self.rates = {}
        self.cost_map = {}
        self.in_params = {}
        self.omap_size = {}
        self.conv_in_channels = {}
        self.conv_out_channels = {}


    def one_shot_lowest_ranking_filters(self, target):
        # Consolidation of chained channels
        # Use the maximum rank among the chained channels for the criteria for those channels
        # Greedily pick from the lowest rank.
        # 
        # This return list of [layers act_index, filter_index, rank]
        data = []
        chained = []

        # keys of filter_ranks are activation index
        checked = []
        og_filter_size = {}
        new_filter_size = {}
        for i in sorted(self.filter_ranks.keys()):
            og_filter_size[i] = int(self.filter_ranks[i].size(0))
            if i in checked:
                continue
            current_chain = []
            k = i
            while k in self.chains:
               current_chain.append(k) 
               checked.append(k)
               k = self.chains[k]
            current_chain.append(k) 
            checked.append(k)

            sizes = np.array([self.filter_ranks[j].size(0) for j in current_chain])

            max_size = np.max(sizes)
            for k in current_chain:
                new_filter_size[k] = max_size
            ranks = [self.filter_ranks[j].to(self.device) for j in current_chain]
            cnt = torch.zeros(int(max_size), device=self.device)
            for idx in range(len(ranks)):
                # The padding residual
                rank = ranks[idx]
                if rank.size(0) < max_size:
                    cnt += torch.cat((torch.ones(int(rank.size(0)), device=self.device), torch.zeros(int(max_size-rank.size(0)), device=self.device)))
                    ranks[idx] = torch.cat((ranks[idx], torch.zeros(int(max_size-rank.size(0)), device=self.device)))
                else:
                    cnt += torch.ones(int(max_size), device=self.device)

            ranks = torch.stack(ranks, dim=1)
            sum_ranks = ranks.sum(dim=1)
            weight = len(current_chain)
            layers_index = current_chain

            for j in range(sum_ranks.size(0)):
                # layers_index, filter_index, rank, #chain
                rank = sum_ranks[j].cpu().numpy()
                data.append((layers_index, j, rank, weight))

        if self.random:
            s = list(data)
            shuffle(s)
        else:
            # data[layers_index, j, rank, weight]
            s = sorted(data, key=lambda x: x[2])
            import os
            if not os.path.exists('./log_all_rank'):
                os.makedirs('./log_all_rank')
            np.savetxt(os.path.join('./log_all_rank', 'resnet.txt'), np.array(s), fmt = '%s')
        selected = []
        idx = 0

        while idx < len(s):  # [layer, filter, rank, weight]
            # make each layer index an instance to prune
            for lj, l in enumerate(s[idx][0]):
                index = s[idx][1]   # index is the No. of filter
                if self.quota[s[idx][0][lj]] > 0 and index < og_filter_size[l]:
                    selected.append((l, index, s[idx][2]))
                    self.quota[l] -= 1

            # if isinstance(self.model, vgg_16_bn) or isinstance(self.model, resnet_56) or idx % 10 == 0:
            if idx % 10 == 0:
                tmp = sorted(selected, key=lambda x: x[0])
                tmp_in_channels = dict(self.conv_in_channels)
                tmp_out_channels = dict(self.conv_out_channels)
                for f in tmp:
                    tmp_out_channels[f[0]] -= 1
                    next_conv_idx = self.next_conv[f[0]] if f[0] in self.next_conv else None
                    #if not f[0] in self.downsample_conv and next_conv_idx:
                    if next_conv_idx:
                        for i in next_conv_idx:
                            next_conv = self.activation_to_conv[i]
                            if (next_conv.groups != next_conv.out_channels or next_conv.groups != next_conv.in_channels):
                                tmp_in_channels[i] -= 1

                cost = 0
                for key in self.cost_map:
                    cost += self.cost_map[key]*tmp_in_channels[key]*tmp_out_channels[key]
                cost += tmp_out_channels[key]*self.num_cls

                if cost < target:
                    break

            left = 0
            for k in self.quota:
                left += self.quota[k]
            if left <= 0:
                return selected

            idx += 1
        return selected, s


    def pack_pruning_target(self, filters_to_prune_per_layer, get_segment=True, progressive=True):
        if get_segment:
            filters_to_prune = []
            for l in filters_to_prune_per_layer:
                if len(filters_to_prune_per_layer[l]) > 0:
                    filters_to_prune_per_layer[l] = sorted(filters_to_prune_per_layer[l])
                    prev_len = 0
                    first_ptr = 0
                    j = first_ptr+1
                    while j < len(filters_to_prune_per_layer[l]):
                        if filters_to_prune_per_layer[l][j] != filters_to_prune_per_layer[l][j-1]+1:
                            if progressive:
                                begin = filters_to_prune_per_layer[l][first_ptr] - prev_len
                                end = filters_to_prune_per_layer[l][j-1] - prev_len
                            else:
                                begin = filters_to_prune_per_layer[l][first_ptr]
                                end = filters_to_prune_per_layer[l][j-1]
                            filters_to_prune.append((l, (begin, end)))
                            prev_len += (end-begin+1)
                            first_ptr = j
                        j += 1
                    if progressive:
                        begin = filters_to_prune_per_layer[l][first_ptr] - prev_len
                        end = filters_to_prune_per_layer[l][j-1] - prev_len
                    else:
                        begin = filters_to_prune_per_layer[l][first_ptr]
                        end = filters_to_prune_per_layer[l][j-1]
                    filters_to_prune.append((l, (begin, end)))
        else:
            for l in filters_to_prune_per_layer:
                if len(filters_to_prune_per_layer[l]) > 0:
                    filters_to_prune_per_layer[l] = sorted(filters_to_prune_per_layer[l])
                    if progressive:
                        for i in range(len(filters_to_prune_per_layer[l])):
                            # Progressively pruning starts from the lower filters 从下面的滤波器开始逐步修剪
                            filters_to_prune_per_layer[l][i] = filters_to_prune_per_layer[l][i] - i

                    filters_to_prune = []
                    for l in filters_to_prune_per_layer:
                        for i in filters_to_prune_per_layer[l]:
                            filters_to_prune.append((l, i))
        return filters_to_prune
    

    def get_pruning_plan(self, num_filters_to_prune, progressive=True, get_segment=False):
        if not self.quota:
            self.quota = {}
            if self.safeguard == 0:
                for k in self.filter_ranks:
                    # print(self.filter_ranks[k].size(0))
                    self.quota[k] = int(self.filter_ranks[k].size(0)) - 1
            else:
                for k in self.filter_ranks:
                    # print(np.minimum(int(np.floor(self.filter_ranks[k].size(0) * (1-self.safeguard))), int(self.filter_ranks[k].size(0)) - 2))
                    self.quota[k] = np.minimum(int(np.floor(self.filter_ranks[k].size(0) * (1-self.safeguard))), int(self.filter_ranks[k].size(0)) - 2)

        filters_to_prune, s = self.one_shot_lowest_ranking_filters(num_filters_to_prune)
        filters_to_prune_per_layer = {}
        for (l, f, r) in filters_to_prune:      # l=layer , f=filter , r=rank
            if l not in filters_to_prune_per_layer:
                filters_to_prune_per_layer[l] = []
            filters_to_prune_per_layer[l].append(f)

        filters_to_prune = self.pack_pruning_target(filters_to_prune_per_layer, get_segment=get_segment, progressive=progressive)
        return filters_to_prune, s, filters_to_prune_per_layer

    def get_quantization_plan(self, s, pruned_channel, bops_target):
        cur_bops = self.resource_usage * 32 * 32
        # thres = s[200][2]
        thres = 0.1
        while cur_bops > bops_target:
            split_strategy = []
            idx = 0
        
            while idx < len(s):  # [layer, filter, rank, weight]
                for lj, l in enumerate(s[idx][0]):
                    score = s[idx][2]
                    index = s[idx][1]   # index is the No. of filter
                    if score <= thres:
                        split_strategy.append((l, index, 2))
                    elif score > thres and score <= (thres + 100):
                        split_strategy.append((l, index, 4))
                    elif score > (thres + 100):
                        split_strategy.append((l, index, 8))
                idx += 1
            # print(split_strategy)
            flag_for_layers = []
            tmp = sorted(split_strategy, key=lambda x: x[0])
            t = 0
            while t < len(tmp):
                if t == 0:
                    # flag_for_layers.append((tmp[t][0], tmp[t][2]))
                    flag_for_layers.append(32)
                    t += 1
                else:
                    if tmp[t][0] == 0:
                        t += 1
                    elif tmp[t][0] != 0 and tmp[t][0] == tmp[t-1][0] and tmp[t][2] >= tmp[t-1][2]:
                        flag_for_layers.pop()
                        # flag_for_layers.append((tmp[t][0], tmp[t][2]))
                        flag_for_layers.append(tmp[t][2])
                        t += 1
                    else:
                        # flag_for_layers.append((tmp[t][0], tmp[t][2]))
                        flag_for_layers.append(tmp[t][2])
                        t += 1
            _, cur_bops = self.bops_forward(torch.zeros((1,3,256, 256), device=self.device), pruned_channel, flag_for_layers)
            
            if cur_bops > bops_target:
                thres += 5
            else:
                break
        
        return flag_for_layers, cur_bops           

    def pruning_with_transformations(self, original_dist, perturbation, flops_target, bops_target, masking=False):
        flops_target = flops_target * self.resource_usage
        bops_target = bops_target * self.resource_usage * 32 * 32
        print('Targeting resource usage: {:.2f}MFLOPs'.format(flops_target/1e6))
        print('Targeting bit-operations usage: {:.2f}GBOPs'.format(bops_target/1e9))
        
        for k in sorted(self.filter_ranks.keys()):
            self.filter_ranks[k] = original_dist[k] * perturbation[k][0] + perturbation[k][1]

        prune_targets, s, filters_to_prune_per_layer = self.get_pruning_plan(flops_target, progressive=(not masking), get_segment=True)
   
        layers_pruned = {}
        for layer_index, filter_index in prune_targets:
            if layer_index not in layers_pruned:
                layers_pruned[layer_index] = 0
            layers_pruned[layer_index] = layers_pruned[layer_index] + (filter_index[1]-filter_index[0]+1)
        filters_left = {}
        pruned_channel = []
        for k in sorted(self.filter_ranks.keys()):
            if k not in layers_pruned:
                layers_pruned[k] = 0
            filters_left[k] = len(self.filter_ranks[k]) - layers_pruned[k]
            pruned_channel.append(filters_left[k])
        print('Filters left: {}'.format(sorted(filters_left.items())))
        flag_for_layers, cur_bops = self.get_quantization_plan(s, pruned_channel, bops_target)
        print('Bitwidth for each layer: {}'.format(flag_for_layers))
        print('Current BOPs: {:.3f}G'.format(cur_bops/1e9))

        
        print('Prunning filters..')
        for layer_index, filter_index in prune_targets:
            if masking:
                self.mask_conv_layer_segment(layer_index, filter_index)
            else:
                self.prune_conv_layer_segment(layer_index, filter_index)

        # return pruned_channel, flag_for_layers, cur_bops, filters_to_prune_per_layer
        return pruned_channel, flag_for_layers, cur_bops
    
    def get_uniform_ratio(self, target):
        first = 0
        second = 0
        for conv_idx in self.activation_to_conv:
            conv = self.activation_to_conv[conv_idx]
            layer_cost = self.cost_map[conv_idx]*conv.weight.size(0)*conv.weight.size(1)
            if conv_idx == 0:
                first += layer_cost
            else:
                second += layer_cost

        # TODO: this is wrong if there are multiple linear layers
        first += self.base_flops
        ratio = (np.sqrt(first**2+4*second*target)-first) / (2.*second)
        return ratio

        
    def compute_rank(self, grad):
        activation_index = len(self.activations) - self.grad_index - 1
        activation = self.activations[activation_index] 
        if self.rank_type == 'analysis':
            if activation_index not in self.filter_ranks:
                self.filter_ranks[activation_index] = activation*grad
            else:
                self.filter_ranks[activation_index] = torch.cat((self.filter_ranks[activation_index], activation*grad), 0)
        else:
            # This is NVIDIA's approach
            if self.rank_type == 'meanAbsMeanImpact':
                values = torch.abs((activation*grad).sum(2).sum(2).data / (activation.size(2)*activation.size(3)))

            # This is equivalent to NVIDIA's approach when E[mean_impact] = 0 
            elif self.rank_type == 'madMeanImpact':
                mean_impact = (activation*grad).sum(2).sum(2).data / (activation.size(2)*activation.size(3))
                values = torch.abs(mean_impact - mean_impact.mean(dim=0))

            elif self.rank_type == 'varMeanImpact':
                mean_impact = (activation*grad).sum(2).sum(2).data / (activation.size(2)*activation.size(3))
                values = torch.pow(mean_impact - mean_impact.mean(dim=0), 2)

            elif self.rank_type == 'MAIVarMAI':
                mean_impact = torch.abs(activation*grad).sum(2).sum(2).data / (activation.size(2)*activation.size(3))
                values = mean_impact.mean(dim=0) * torch.pow(mean_impact - mean_impact.mean(dim=0), 2)

            elif self.rank_type == 'MSIVarMSI':
                mean_impact = torch.pow(activation*grad, 2).sum(2).sum(2).data / (activation.size(2)*activation.size(3))
                values = mean_impact.mean(dim=0) * torch.pow(mean_impact - mean_impact.mean(dim=0), 2)

            elif self.rank_type == 'meanL1Impact':
                values = torch.abs(activation*grad).sum(2).sum(2).data / (activation.size(2) * activation.size(3))

            elif self.rank_type == 'meanL1ImpactRaw':
                values = torch.abs(activation*grad).sum(2).sum(2).data

            elif self.rank_type == 'meanL1Act':
                values = torch.abs(activation).sum(2).sum(2).data / (activation.size(2) * activation.size(3))

            elif self.rank_type == 'meanL1Grad':
                values = torch.abs(grad).sum(2).sum(2).data / (activation.size(2) * activation.size(3))

            elif self.rank_type == 'meanGrad':
                values = grad.sum(2).sum(2).data / (activation.size(2) * activation.size(3))

            elif self.rank_type == 'meanL2Impact':
                values = torch.pow(activation*grad, 2).sum(2).sum(2).data / (activation.size(2) * activation.size(3))

            elif self.rank_type == 'madL2Impact':
                l2_impact = torch.pow(activation*grad, 2).sum(2).sum(2).data / (activation.size(2) * activation.size(3))
                values = torch.abs(l2_impact - l2_impact.mean(dim=0))

            elif self.rank_type == 'varL2Impact':
                l2_impact = torch.pow(activation*grad, 2).sum(2).sum(2).data / (activation.size(2) * activation.size(3))
                values = torch.pow(l2_impact - l2_impact.mean(dim=0), 2)

            elif self.rank_type == 'varMSImpact':
                ms_impact = torch.pow(activation*grad, 2).sum(2).sum(2).data / (activation.size(2) * activation.size(3))
                values = torch.pow(ms_impact - ms_impact.mean(dim=0), 2)

            elif self.rank_type == 'L2IVarL2I':
                l2_impact = torch.sqrt(torch.pow(activation*grad, 2).sum(2).sum(2).data)
                values = l2_impact.mean(dim=0) * torch.pow(l2_impact - l2_impact.mean(dim=0), 2)

            elif self.rank_type == 'meanSquaredImpact':
                impact = torch.pow(activation * grad, 2)
                values = impact.sum(2).sum(2) / (activation.size(2) * activation.size(3))

            elif self.rank_type == 'meanMadImpact':
                impact = activation * grad
                impact = impact.reshape((impact.size(0), impact.size(1), -1))
                mean = impact.mean(dim=2)
                values = torch.abs(impact - mean.reshape((impact.size(0), impact.size(1), 1))).sum(2) / impact.size(2)

            elif self.rank_type == 'meanVarImpact':
                impact = activation * grad
                impact = impact.reshape((impact.size(0), impact.size(1), -1))
                mean = impact.mean(dim=2)
                values = torch.pow(impact - mean.reshape((impact.size(0), impact.size(1), 1)), 2).sum(2) / impact.size(2)

            elif self.rank_type == 'meanVarAct':
                std = activation.reshape((activation.size(0), activation.size(1), -1))
                values = torch.pow(std - std.mean(dim=2).reshape((std.size(0), std.size(1), 1)), 2).sum(2) / std.size(2)

            elif self.rank_type == 'meanAct':
                values = activation.sum(2).sum(2).data / (activation.size(2)*activation.size(3))

            elif self.rank_type == 'varF2Act':
                f2 = torch.sqrt(torch.pow(activation, 2).sum(2).sum(2).data) / (activation.size(2)*activation.size(3))
                values = torch.pow(f2 - f2.mean(dim=0), 2)

            elif self.rank_type == '2-taylor':
                values1 = (activation*grad).sum(2).sum(2).data
                values2 = (torch.pow(activation*grad, 2)*0.5).sum(2).sum(2).data
                values = torch.abs(values1 + values2) / (activation.size(2)*activation.size(3))

            values = values.sum(0) / activation.size(0)
            
            if activation_index not in self.filter_ranks:
                self.filter_ranks[activation_index] = torch.zeros(activation.size(1), device=self.device)

            self.filter_ranks[activation_index] += values

        self.grad_index += 1


    def get_unit_flops_for_layer(self, layer_id):
        flops = 0
        k = layer_id
        while k in self.chains:
            flops += self.cost_map[k]*self.conv_in_channels[k]*self.conv_out_channels[k]
            next_conv_idx = self.next_conv[k] if k in self.next_conv else None
            if next_conv_idx:
                for next_conv_i in next_conv_idx:
                    next_conv = self.activation_to_conv[next_conv_i]
                    if (next_conv.groups != next_conv.out_channels or next_conv.groups != next_conv.in_channels):
                        flops += self.cost_map[next_conv_i]*self.conv_in_channels[next_conv_i]*self.conv_out_channels[next_conv_i]
            k = self.chains[k]

        flops += self.cost_map[k]*self.conv_in_channels[k]*self.conv_out_channels[k]
        next_conv_idx = self.next_conv[k] if k in self.next_conv else None
        if next_conv_idx:
            for next_conv_i in next_conv_idx:
                next_conv = self.activation_to_conv[next_conv_i]
                if (next_conv.groups != next_conv.out_channels or next_conv.groups != next_conv.in_channels):
                    flops += self.cost_map[next_conv_i]*self.conv_in_channels[next_conv_i]*self.conv_out_channels[next_conv_i]
        return flops

    def get_pruning_plan_from_layer_budget(self, layer_budget):
        filters_to_prune_per_layer = {}
        last_residual = 0
        for layer in sorted(self.filter_ranks.keys()):
            current_chain = []
            k = layer
            while k in self.chains:
               current_chain.append(k) 
               k = self.chains[k]
            current_chain.append(k) 

            sizes = np.array([self.filter_ranks[j].size(0) for j in current_chain])
            max_size = np.max(sizes)
            ranks = [self.filter_ranks[j].to(self.device) for j in current_chain]
            cnt = torch.zeros(int(max_size), device=self.device)
            for idx in range(len(ranks)):
                # The padding residual
                rank = ranks[idx]
                if rank.size(0) < max_size:
                    cnt += torch.cat((torch.ones(int(rank.size(0)), device=self.device), torch.zeros(int(max_size-rank.size(0)), device=self.device)))
                    ranks[idx] = torch.cat((ranks[idx], torch.zeros(int(max_size-rank.size(0)), device=self.device)))
                else:
                    cnt += torch.ones(int(max_size), device=self.device)

            ranks = torch.stack(ranks, dim=1)
            sum_ranks = ranks.sum(dim=1)

            rank = sum_ranks.cpu().numpy()

            tbp = np.argsort(rank)

            for k in current_chain:
                if not k in filters_to_prune_per_layer:
                    cur_layer_size = self.filter_ranks[k].size(0)
                    valid_ind = tbp[tbp < cur_layer_size][:(cur_layer_size-layer_budget[k])]
                    filters_to_prune_per_layer[k] = valid_ind
        return filters_to_prune_per_layer

