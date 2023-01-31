import time
import queue
import torch
import argparse
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

from utils.utils import *
from utils import *
from model2.drives import *
from pruner.filterpruner import FilterPruner

from model.resnet_cifar import *
from pruner.fp_resnet import FilterPrunerResNet
from model.vgg_cifar import *
from pruner.fp_vgg import FilterPrunerVGG

import os
import sys
import shutil
import numpy as np
import time, datetime
import torch
import random
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data.distributed
from tqdm import tqdm

sys.path.append("../../")
from torchvision import datasets, transforms, models 
from torch.autograd import Variable
import torchvision.models as models
# from model.resnet_cifar import *
# from model.resnet_cifar_qt2 import qresnet_56_A_W
# from model.resnet_cifar_qt1 import qresnet_56_A
from model.imagenet_resnet import *
from model.imagenet_resnet_qt1 import qresnet50_A
from model.imagenet_resnet_qt2 import qresnet50_A_W


os.environ['CUDA_VISIBLE_DEVICES']='0'
# Hyper-Parameters
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='resnet_50', choices=('resnet_56'), help='The architecture to prune and the resulting model and logs will use this') ##'vgg_16_bn'
    parser.add_argument('--resume', type=str, default='./model2/ckpt/resnet50_UCML.t7',help='load the model from the specified checkpoint')
    # parser.add_argument("--datapath", type=str, default='./data', help='Path toward the dataset that is used for this experiment')
    parser.add_argument("--dataset", type=str, default='torchvision.datasets.CIFAR10', help='The class name of the dataset that is used, please find available classes under the dataset folder')
    parser.add_argument('--num_classes', default=21, type=int,help='The number of dataset classes')
    parser.add_argument("--pruner", type=str, default='FilterPrunerResNet',choices=('FilterPrunerResNet','FilterPrunerVGG'),help='Different network require differnt pruner implementation')
    # parser.add_argument("--counter", type=str, default='BOPsCounterResNet',choices=('BOPsCounterResNet','BOPsCounterVGG', 'BOPsCounterMBNetV2'),help='Different network require differnt pruner implementation')
    parser.add_argument("--rankPath", type=str, default='./rank_conv/resnet_50_limit6/', help='The path of ranks of convolution layers')
    parser.add_argument("--rank_type", type=str, default='Rank',choices=('l1_bn','l2_bn','l1_weight','l2_weight', 'Rank', 'ci'), help='The ranking criteria for filter pruning')
    parser.add_argument("--lub", type=str, default='./log_0.05/resnet_50_UCML_ea_min.data', help='The affine transformations')
    parser.add_argument("--savename", type=str, default='resnet_50_UCML')
    # parser.add_argument("--ori_channel", type=str, default='[16]*19 + [32]*18 + [64]*18', help='The original_channel of different network')
    parser.add_argument("--global_random_rank", action='store_true', default=False, help='When this is specified, none of the rank_type matters, it will randomly prune the filters')  
    parser.add_argument("--long_ft", type=int, default=300, help='It specifies how many epochs to fine-tune the network once the pruning is done')
    parser.add_argument("--prune_away",type=float, default=50, help='How many percentage of constraints should be pruned away. E.g., 50 means 50% of FLOPs will be pruned away')
    # parser.add_argument("--penalty_factor",type=float, default=6, help=' a constant about quantization bid-width that can be adjusted according to the need')
    # parser.add_argument("--qbw_upper",type=str, default='[32] + [8]*18 + [6]*18 + [4]*18', help='The upper bound of quantization bid-width for different network')
    parser.add_argument("--safeguard", type=float, default=0.1,help='A floating point number that represent at least how many percentage of the original number of channel should be preserved. E.g., 0.10 means no matter what ranking, each layer should have at least 10% of the number of original channels.')
    parser.add_argument("--batch_size", type=int, default=16, choices=(16, 32, 64, 128), help='Batch size for training.')
    parser.add_argument("--tau_hat", type=int, default=200, help='The number of updates before evaluating for fitness (used in EA).   e.g. tau_hat = 200')
    parser.add_argument("--min_lub", action='store_true', default=False, help='Use Evolutionary Algorithm to solve latent variable for minimizing Lipschitz upper bound')
    parser.add_argument("--uniform_pruning", action='store_true', default=False, help='Use Evolutionary Algorithm to solve latent variable for minimizing Lipschitz upper bound')
    parser.add_argument("--no_val", action='store_true', default=False, help='Use full dataset to train (use to compare with prior art in CIFAR-10)')
    parser.add_argument("--gpu", type=str, default='0', help='Select GPU to use.("0" is to use GPU:0))')
    parser.add_argument("--lr", type=float, default=0.01,help='The learning rate for fine-tuning')
    parser.add_argument("--weight_decay", type=float, default=1e-4, help='weight_decay')
    
    args = parser.parse_args()
    return args


class LEDPQ:
    def __init__(self, model, pruner, rank_type='Rank', batch_size=128, lr=1e-3, safeguard=0, global_random_rank=False, lub='', weight_decay=0, device=None):
        self.device = device
        self.sample_for_ranking = 1 if rank_type in ['l1_weight', 'l2_weight', 'l2_bn', 'l1_bn', 'l2_bn_param','Rank'] else 5000
        self.safeguard = safeguard
        self.lub = lub
        self.lr = lr
        self.img_size = 256
        self.batch_size = batch_size
        self.rank_type = rank_type

        self.weight_decay = weight_decay

        self.train_loader, self.test_loader = get_ucml_dataloader(batch_size)

        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.pruner = eval(pruner)(self.model, rank_type, args.num_classes, safeguard, random=global_random_rank, device=device, rankPath=args.rankPath) 

        self.model.train()

    # EA
    def learn_ranking_ea(self, name, model_desc, tau_hat, long_ft, flops_target, bops_target):
        name = name
        start_t = time.time()
        self.pruner.reset()
        self.pruner.model.eval()
        self.pruner.forward(torch.zeros((1,3,self.img_size,self.img_size), device=self.device))
        original_flops = self.pruner.cur_flops
        original_bops = original_flops * 32 * 32
        original_size = self.pruner.cur_size

        print('Before Compressing, FLOPs: {:.3f}M, BOPs: {:.3f}M, Size: {:.3f}M'.format(original_flops/1e6, original_bops/1e6, original_size/1e6))

        mean_loss = []
        num_layers = len(self.pruner.filter_ranks)
        minimum_loss = 10
        beta = 1e-10
        best_perturbation = None
        POPULATIONS = 64
        SAMPLES = 16
        GENERATIONS = 400
        SCALE_SIGMA = 1
        MUTATE_PERCENT = 0.1
        index_queue = queue.Queue(POPULATIONS)
        population_loss = np.zeros(0)
        population_data = []

        original_dist = self.pruner.filter_ranks.copy()
        original_dist_stat = {}
        for k in sorted(original_dist):
            a = original_dist[k].cpu().numpy()
            original_dist_stat[k] = {'mean': np.mean(a), 'std': np.std(a)}

        # Initialize Population
        for i in range(GENERATIONS):
            step_size = 1-(float(i)/(GENERATIONS*1.25))
            # Perturn distribution
            perturbation = []
            if i == POPULATIONS-1:
                for k in sorted(self.pruner.filter_ranks.keys()):
                    perturbation.append((1,0))
            elif i < POPULATIONS-1:
                for k in sorted(self.pruner.filter_ranks.keys()):
                    scale = np.exp(float(np.random.normal(0, SCALE_SIGMA)))
                    shift = float(np.random.normal(0, original_dist_stat[k]['std']))
                    perturbation.append((scale, shift))
            else:
                mean_loss.append(np.mean(population_loss))
                sampled_idx = np.random.choice(POPULATIONS, SAMPLES)
                sampled_loss = population_loss[sampled_idx]
                winner_idx_ = np.argmin(sampled_loss)
                winner_idx = sampled_idx[winner_idx_]
                oldest_index = index_queue.get()

                # Mutate winner
                base = population_data[winner_idx]
                # Perturb distribution
                mnum = int(MUTATE_PERCENT * len(self.pruner.filter_ranks))
                mutate_candidate = np.random.choice(len(self.pruner.filter_ranks), mnum)
                for k in sorted(self.pruner.filter_ranks.keys()):
                    scale = 1
                    shift = 0
                    if k in mutate_candidate:
                        scale = np.exp(float(np.random.normal(0, SCALE_SIGMA*step_size)))
                        shift = float(np.random.normal(0, original_dist_stat[k]['std']))
                    perturbation.append((scale*base[k][0], shift+base[k][1]))

            # Given affine transformations, rank and compress
            sparse_channel, bit, cur_bops = self.pruner.pruning_with_transformations(original_dist, perturbation, flops_target, bops_target)

            # Re-measure the compressed model 
            self.pruner.reset()
            self.pruner.model.eval()
            self.pruner.forward(torch.zeros((1,3,self.img_size,self.img_size), device=self.device))
            cur_flops = self.pruner.cur_flops
            cur_size = self.pruner.cur_size
            self.pruner.model = self.pruner.model.to(self.device)
    
            qmodel = qresnet50_A_W(filters_left=sparse_channel, bit=bit).cuda()
            qmodel.to(device)

            print('Density: {:.3f}% ({:.3f}M/{:.3f}M) | FLOPs: {:.3f}% ({:.3f}M/{:.3f}M)'.format(float(cur_size)/original_size*100, cur_size/1e6, original_size/1e6, float(cur_flops)/original_flops*100, cur_flops/1e6, original_flops/1e6))

            print('Fine tuning to recover from pruning iteration.')
            # optimizer = optim.SGD(self.pruner.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
            # if tau_hat > 0:
            #     train(self.model, self.train_loader, self.val_loader, optimizer, epochs=1, steps=tau_hat, run_test=False, device=self.device)
            # acc, loss = test(self.model, self.val_loader, device=self.device, get_loss=True)
            optimizer = optim.SGD(qmodel.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
            if tau_hat > 0:
                train(qmodel, self.train_loader, self.test_loader, optimizer, scheduler = None, epochs=1, steps=tau_hat, run_test=False, device=self.device)
            acc, loss = test(qmodel, self.test_loader, device=self.device, get_loss=True)
            print(acc)
            compress_loss = np.mean(loss) + beta * np.abs(original_bops * bops_target - cur_bops) 

            # if np.mean(loss) < minimum_loss:
            #     minimum_loss = np.mean(loss)
            #     best_perturbation = perturbation
            if compress_loss < minimum_loss:
                minimum_loss = compress_loss
                best_perturbation = perturbation
            
            if i < POPULATIONS:
                index_queue.put(i)
                population_data.append(perturbation)
                population_loss = np.append(population_loss, [compress_loss])
            else:
                index_queue.put(oldest_index)
                population_data[oldest_index] = perturbation
                population_loss[oldest_index] = compress_loss

            # Restore the model back to origin
            model = resnet50().cuda()
            model.load_state_dict(torch.load(model_desc))
            # model = torch.load(model_desc)

            if isinstance(model, nn.DataParallel):
                model = model.module
            model.eval()
            model = model.to(self.device)
            self.pruner.model = model
            self.model = model
            self.pruner.reset()
            self.pruner.model.eval()
            self.pruner.forward(torch.zeros((1,3,self.img_size,self.img_size), device=self.device))
            print('Generation {}, Step: {:.2f}, Min Loss: {:.3f}'.format(i, step_size, np.min(population_loss)))

        total_t = time.time() - start_t
        print('Finished. Use {:.2f} hours. Minimum Loss: {:.3f}'.format(float(total_t) / 3600, minimum_loss))
        if not os.path.exists('./log_{}'.format(bops_target)):
            os.makedirs('./log_{}'.format(bops_target))
        np.savetxt(os.path.join('./log_{}'.format(bops_target), '{}_ea_loss.txt'.format(args.savename)), np.array(mean_loss))
        np.savetxt(os.path.join('./log_{}'.format(bops_target), '{}_ea_min.data'.format(args.savename)), best_perturbation)

        # Use the best affine transformation to obtain the resulting model
        self.pruner.pruning_with_transformations(original_dist, best_perturbation, flops_target, bops_target)
        if not os.path.exists('./ckpt_{}'.format(bops_target)):
            os.makedirs('./ckpt_{}'.format(bops_target))
        torch.save(self.pruner.model, os.path.join('./ckpt_{}'.format(bops_target), '{}_bestarch_init.pt'.format(args.savename)))

    # model_compress
    def model_compress(self, name, long_ft, flops_target, bops_target, target=-1, arch =''):
        test_acc = []
        b4ft_test_acc = []
        density = []
        flops = []

        acc = test(self.model, self.test_loader, device=self.device)
        test_acc.append(acc)
        b4ft_test_acc.append(acc)

        self.pruner.reset() 
        self.model.eval()
        
        self.pruner.forward(torch.zeros((1,3,self.img_size,self.img_size), device=self.device))
        b4prune_size = self.pruner.cur_size
        b4prune_flops = self.pruner.cur_flops
        density.append(self.pruner.cur_size)
        flops.append(self.pruner.cur_flops)

        print('Before Pruning, Acc: {:.2f}%, FLOPs: {:.3f}M, Size: {:.3f}M'.format(acc, b4prune_flops/1e6, b4prune_size/1e6))

        # If there is learned affine transformation, load it.
        if self.lub != '':
            print('Import the transformations ea min data')
            perturbation = np.loadtxt(self.lub)
        else:
            # print('self.lub = \'\' ')
            perturbation = np.array([[1., 0.] for _ in range(len(self.pruner.filter_ranks))])

        # self.pruner.pruning_with_transformations(self.pruner.filter_ranks, perturbation, target)
        sparse_channel, bit, cur_bops = self.pruner.pruning_with_transformations(self.pruner.filter_ranks, perturbation, flops_target, bops_target)

        self.pruner.reset()
        self.pruner.model.eval()

        self.pruner.forward(torch.zeros((1,3,self.img_size,self.img_size), device=self.device))
        cur_flops = self.pruner.cur_flops
        cur_size = self.pruner.cur_size

        density.append(cur_size)
        flops.append(cur_flops)
        
        print('Density: {:.3f}% ({:.3f}M/{:.3f}M) | FLOPs: {:.3f}% ({:.3f}M/{:.3f}M)'.format(cur_size/b4prune_size*100, cur_size/1e6, b4prune_size/1e6, cur_flops/b4prune_flops*100, cur_flops/1e6, b4prune_flops/1e6))

    
        # if not os.path.exists('./ckpt_{}'.format(bops_target)):
        #     os.makedirs('./ckpt_{}'.format(bops_target))
        # print('Saving untrained model...')
        
        # # save untrained model
        # torch.save(self.pruner.model, os.path.join('ckpt_{}'.format(bops_target), '{}_init.t7'.format(args.savename)))  

        # acc = test(self.model, self.test_loader, device=self.device)
        # # before fune-tuning
        # b4ft_test_acc.append(acc)

        # # handle directory
        # if not os.path.exists('./log_{}'.format(bops_target)):
        #     os.makedirs('./log_{}'.format(bops_target))

        # # Going to fine tune
        # print('Finished. Going to fine tune the model a bit more')
        # if long_ft > 0:
        #     optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay, nesterov=True)

        #     #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, long_ft)
        #     scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [int(long_ft*0.5), int(long_ft*0.75)], gamma=0.1)

        #     if args.no_val:
        #         train(self.model, self.train_loader, self.test_loader, optimizer, epochs=long_ft, arch = arch, scheduler=scheduler, name=args.savename, ratio = bops_target, device=self.device)
        #     else:
        #         train(self.model, self.train_loader, self.test_loader, optimizer, epochs=long_ft, arch = arch, scheduler=scheduler, name=args.savename, ratio = bops_target, device=self.device)
            
        #     acc = test(self.model, self.test_loader, device=self.device)
        #     test_acc.append(acc)
        # else:
        #     acc = test(self.model, self.test_loader, device=self.device)
        #     test_acc.append(acc)

        # log = np.stack([np.array(b4ft_test_acc), np.array(test_acc), np.array(density), np.array(flops)], axis=1)
        # np.savetxt(os.path.join('./log_{}'.format(bops_target), '{}_test_acc.txt'.format(args.savename)), log)

        # print('Summary')
        # print('Before Pruning- Accuracy: {:.3f}, Cost: {:.3f}M'.format(test_acc[0], b4prune_flops/1e6))
        # print('After Pruning- Accuracy: {:.3f}, Cost: {:.3f}M'.format(test_acc[-1], cur_flops/1e6))
        
        epoch_qt = 400
        bit1=[32, 32, 32, 32, 32, 2, 4, 8, 2, 8, 8, 2, 2, 8, 8, 2, 2, 8, 2, 8, 8, 2, 2, 8, 2, 2, 8, 8, 2, 2, 8, 2, 2, 8, 2, 2, 8, 2, 2, 8, 2, 2, 8, 2, 2, 2, 2, 2, 2, 2, 32, 32, 32]

        model_qt1 = qresnet50_A(filters_left=sparse_channel, bit=bit1).cuda()
        model_qt1.load_state_dict(torch.load('./ckpt_{}/'.format(bops_target) + '{}_best.t7'.format(args.savename)))
        model_qt1.to(device)
        optimizer = optim.SGD(model_qt1.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [int(epoch_qt*0.5), int(epoch_qt*0.75)], gamma=0.1)
        train(model_qt1, train_loader, test_loader, optimizer, scheduler, epoch_qt, args.arch, train_model_Running=False, quant=True, name=args.savename, ratio = bops_target, device=self.device)
        

        model_qt2 = qresnet50_A_W(filters_left=sparse_channel, bit=bit).cuda()
        model_qt2.load_state_dict(torch.load('./ckpt_{}/'.format(bops_target) + '{}_quant_best.t7'.format(args.savename)))
        model_qt2.to(device)
        optimizer = optim.SGD(model_qt2.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [int(epoch_qt*0.5), int(epoch_qt*0.75)], gamma=0.1)
        train(model_qt2, train_loader, test_loader, optimizer, scheduler, epoch_qt, args.arch, train_model_Running=False, quant=True, name=args.savename, ratio = bops_target, device=self.device)

if __name__ == "__main__":
    startTime = time.time()
    args = get_args()
    device = 'cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu'
    print('device --- ' + str(device))
    print('torch\'s version --- ' + torch.__version__ + '\ntorchvision\'s version --- ' + torchvision.__version__)
    print(args)

    # Data
    print('==> Preparing data..')
    train_loader, test_loader = get_ucml_dataloader(args.batch_size)

    # model
    basemodel = resnet50(pretrained=False).cuda()
    basemodel.load_state_dict(torch.load(args.resume))
    basemodel.to(device)

    base_acc = test(basemodel, test_loader, device=device)
    print('Baseline Acc: {:.2f}'.format(base_acc))

    filter_pruner = eval('FilterPrunerResNet')(basemodel, 'Rank', num_cls=21, rankPath=args.rankPath, device=device)
    filter_pruner.forward(torch.zeros((1,3,256, 256), device=device))
    # bops_counter = eval('BOPsCounterResNet')(model, 'Rank', num_cls=10, rankPath=args.rankPath, device=device)
    
    # NOTE GRFP
    # ledpq = LEDPQ(basemodel, args.pruner, args.counter, args.rank_type, args.batch_size, args.lr, safeguard=args.safeguard, global_random_rank=args.global_random_rank, lub=args.lub, weight_decay= args.weight_decay, device=device)
    ledpq = LEDPQ(basemodel, args.pruner, args.rank_type, args.batch_size, args.lr, safeguard=args.safeguard, global_random_rank=args.global_random_rank, lub=args.lub, weight_decay= args.weight_decay, device=device)
    if args.prune_away > 0:
        dummy_size =  256
        ledpq.pruner.reset()
        ledpq.model.eval()
    
        # NOTE Have gotten ranks
        ledpq.pruner.forward(torch.zeros((1,3,dummy_size, dummy_size), device=device))
        b4prune_flops = ledpq.pruner.cur_flops
        prune_till = b4prune_flops * (1-(args.prune_away)/100.)
        print('Pruned untill {:.3f}M'.format(prune_till/1000000.))
    
        if args.uniform_pruning:
            ratio = ledpq.pruner.get_uniform_ratio(prune_till)
            ledpq.pruner.safeguard = ratio
            args.prune_away = 99
    
    if args.min_lub:
        ledpq.learn_ranking_ea(args.savename, args.resume, args.tau_hat, args.long_ft, (1-(args.prune_away)/100.), 0.05)
    else:
        ledpq.model_compress(args.savename, args.long_ft, (1-(args.prune_away)/100.), 0.05)
    
    print('\n---------------------- Cost Time: ' + format_time(time.time() - startTime) + ' ----------------------' + '\n---------------------- Program Over ---------------------- \n')



   

        
   


    

