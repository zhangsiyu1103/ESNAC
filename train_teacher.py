import torch
import datasets
import models
from architecture import Architecture
from kernel import Kernel
from record import Record
import acquisition as ac
import graph as gr
import options as opt
import training as tr
import numpy as np
import argparse
from operator import attrgetter
import os
import random
import time
from tensorboardX import SummaryWriter
import torch.nn as nn
from gpu_energy_eval import GPUEnergyEvaluator


def seed_everything(seed=127):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True


def fully_train(model, dataset):
    dataset = getattr(datasets, dataset)()
    model, acc=tr.train_model_student(model, dataset,
                        '%s/sample9_artificial.pth' % (opt.savedir), 0)
    print("accuracy: ",acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Learnable Embedding Space for Efficient Neural Architecture Compression')

    parser.add_argument('--network', type=str, default='resnet34',
                        help='resnet18/resnet34/resnet50/resnet101/vgg19/shufflenet/alexnet')
    parser.add_argument('--dataset', type=str, default='cifar100',
                        help='cifar10/cifar100/imagenet')
    parser.add_argument('--suffix', type=str, default='0', help='0/1/2/3...')
    parser.add_argument('--device', type=str, default='cuda', help='cpu/cuda')
    parser.add_argument('--objective', type=str, default='accuracy', help='maximizing objective')
    parser.add_argument('--constype', type=str, default='size', help='size/energy/latency')
    parser.add_argument('--consval', type=float, default=0.1, help='different value range for different constraint')

    args = parser.parse_args()

    seed_everything()

    #assert args.network in ['resnet18', 'resnet34','resnet50', 'resnet101',
    #    'vgg19', 'shufflenet', 'alexnet', 'sample0', 'groundtruth']
    assert args.dataset in ['cifar10', 'cifar100', 'imagenet', 'artificial']

    #if args.network in ['resnet18', 'resnet34', 'resnet50','resnet101']:
    #    opt.co_graph_gen = 'get_graph_resnet'
    #elif args.network in ['vgg19', 'sample0','groundtruth']:
    #    opt.co_graph_gen = 'get_graph_vgg'
    #elif args.network == 'alexnet':
    #    opt.co_graph_gen = 'get_graph_alex'
    #elif args.network == 'shufflenet':
    #    opt.co_graph_gen = 'get_graph_shufflenet'

    if args.dataset == 'cifar10':
        opt.dataset = 'CIFAR10Val'
    elif args.dataset == 'cifar100':
        opt.dataset = 'CIFAR100Val'
    elif args.dataset == 'imagenet':
        opt.dataset = 'IMAGENETVal'
    elif args.dataset == 'artificial':
        opt.dataset = 'Artificial'


    opt.device = args.device

    opt.model = './models/pretrained/%s_%s.pth' % (args.network, args.dataset)
    opt.savedir = './save/%s_%s_%s' % (args.network, args.dataset,
                                  args.suffix)
    opt.writer = SummaryWriter('./runs/%s_%s_%s' % (args.network, args.dataset,
                                                args.suffix))
    if not os.path.isdir(opt.savedir):
        os.mkdir(opt.savedir)
    print ('Start compression. Please check the TensorBoard log in the folder ./runs/%s_%s_%s.'%
                                                    (args.network, args.dataset, args.suffix))


    model = getattr(models, args.network)()
    #model = torch.load("temp_save/temp.pth")
    #model = torch.load("save/GroundTruth9_artificial_0/sample9_artificial.pth")
    print(model)
    dataset = getattr(datasets, opt.dataset)()
    record = Record()
    fully_train(model, dataset=opt.dataset)
