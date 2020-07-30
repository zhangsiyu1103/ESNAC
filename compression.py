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
#import test_model

def seed_everything(seed=127):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

def new_kernels(teacher, record, kernel_n, cons_type, cons_val, alpha=opt.co_alpha,
                beta=opt.co_beta, gamma=opt.co_gamma):
    start_time = time.time()
    kernels = []
    for i in range(kernel_n):
        if cons_type == 'size':
            kernel = Kernel(teacher.rep, 0.0, 1.0)
        if cons_type == 'energy':
            kernel = Kernel(teacher.rep, 0.0, 1.7)
        indices = []
        for j in range(record.n):
            if random.random() < gamma:
                indices.append(j)
        if len(indices) > 0:
            x = [record.x[i] for i in indices]
            cons_val = [record.cons[i] for i in indices]
            indices = torch.tensor(indices, dtype=torch.long, device=opt.device)
            y = torch.index_select(record.y, 0, indices)
            kernel.add_batch(x, y, cons_val)
        ma = 0.0
        for j in range(100):
            ll = kernel.opt_step()
            opt.writer.add_scalar('step_%d/kernel_%d_loglikelihood' % (opt.i, i),
                                  ll, j)
            ma = (alpha * ll + (1 - alpha) * ma) if j > 0 else ll
            if j > 5 and abs(ma - ll) < beta:
                break
        kernels.append(kernel)
    opt.writer.add_scalar('compression/kernel_time',
                          time.time() - start_time, opt.i)
    return kernels

def next_samples(teacher, kernels, kernel_n):
    start_time = time.time()
    n = kernel_n
    reps_best, acqs_best, archs_best = [], [], []

    if opt.co_graph_gen == 'get_graph_shufflenet':
        for i in range(n):
            arch, rep, acq = ac.random_search_sfn(teacher, kernels[i])
            archs_best.append(arch)
            reps_best.append(rep)
            acqs_best.append(acq)
            opt.writer.add_scalar('compression/acq', acq, opt.i * n + i - n + 1)
        opt.writer.add_scalar('compression/sampling_time',
                            time.time() - start_time, opt.i)
        return archs_best, reps_best

    else:
        for i in range(n):
            action, rep, acq = ac.random_search(teacher, kernels[i])
            reps_best.append(rep)
            acqs_best.append(acq)
            archs_best.append(teacher.comp_arch(action))
            opt.writer.add_scalar('compression/acq', acq, opt.i * n + i - n + 1)
        opt.writer.add_scalar('compression/sampling_time',
                            time.time() - start_time, opt.i)
        return archs_best, reps_best

def reward(teacher, teacher_acc, students, dataset, objective, cons_type, cons_val):
    start_time = time.time()
    n = len(students)
    students_best, students_acc = tr.train_model_search(teacher, students, dataset)
    rs = []
    cs = []
    evaluator = GPUEnergyEvaluator(gpuid=0, watts_offset=False)
    for j in range(n):
        s = 1.0 * students_best[j].param_n() / teacher.param_n()
        a = 1.0 * students_acc[j] / teacher_acc

        #evaluator.start()
        l = tr.test_model_latency(students_best[j], dataset)
        #e = evaluator.end()
        r = 0
        if objective == 'accuracy':
            r += a
        if cons_type == 'size':
            r += 2 * (cons_val -s)
        elif cons_type == 'latency':
            r += 2 * (cons_val -l)
        elif cons_type == 'energy':
            r += 2 * (cons_val - e)
        #r = a
        #r = a + 2*(cons_val-s)
        opt.writer.add_scalar('compression/model_size', s,
                              opt.i * n - n + 1 + j)
        opt.writer.add_scalar('compression/accuracy_score', a,
                              opt.i * n - n + 1 + j)
        opt.writer.add_scalar('compression/reward', r,
                              opt.i * n - n + 1 + j)
        rs.append(r)
        cs.append(s-cons_val)
        #students_best[j].energy = e/10000
        students_best[j].latency = l
        students_best[j].size = s
        students_best[j].accuracy = students_acc[j]
        students_best[j].reward = r
    opt.writer.add_scalar('compression/evaluating_time',
                          time.time() - start_time, opt.i)
    return students_best, rs, cs


def compression(teacher, dataset, record, objective, cons_type, cons_val, step_n=opt.co_step_n,
                kernel_n=opt.co_kernel_n, best_n=opt.co_best_n):

    teacher_acc = tr.test_model(teacher, dataset)
    archs_best = []
    for i in range(1, step_n + 1):
        print ('Search step %d/%d' %(i, step_n))
        start_time = time.time()
        opt.i = i
        kernels = new_kernels(teacher, record, kernel_n, cons_type, cons_val)
        students_best, xi = next_samples(teacher, kernels, kernel_n)
        students_best, yi, cons = reward(teacher, teacher_acc, students_best, dataset, objective, cons_type, cons_val)
        for j in range(kernel_n):
            record.add_cons_sample(xi[j], yi[j], cons[j])
            if yi[j] == record.reward_best:
                opt.writer.add_scalar('compression/reward_best', yi[j], i)
        students_best = [student.to('cpu') for student in students_best]
        archs_best.extend(students_best)
        #filter out unconstraint
        archs_best = list(filter(lambda x:getattr(x, cons_type) <= cons_val, archs_best))

        archs_best.sort(key=attrgetter(objective), reverse=True)
        archs_best = archs_best[:best_n]
        for j, arch in enumerate(archs_best):
            arch.save('%s/arch_%d.pth' % (opt.savedir, j))
        record.save(opt.savedir + '/record.pth')
        opt.writer.add_scalar('compression/step_time',
                              time.time() - start_time, i)

def random_compression(teacher, dataset, objective, cons_type, cons_val, num_model, best_n = opt.co_best_n):
    teacher_acc = tr.test_model(teacher, dataset)
    students_best = []
    for i in range(4):
        archs = []
        for i in range(num_model//4):
            print(i)
            action = teacher.comp_action_rand()
            #print(action)
            archs.append(teacher.comp_arch(action))
        students_best_cur, yi, cons = reward(teacher, teacher_acc, archs, dataset, objective, cons_type, cons_val)
        students_best.extend(students_best_cur)

    students_best = [student.to('cpu') for student in students_best]

    archs_best = list(filter(lambda x:getattr(x, cons_type) <= cons_val, students_best))

    archs_best.sort(key=attrgetter(objective), reverse=True)

    archs_best = archs_best[:best_n]
    for j, arch in enumerate(archs_best):
        arch.save('%s/arch_%d.pth' % (opt.savedir, j))



def fully_train(teacher, dataset, best_n=opt.co_best_n):
    dataset = getattr(datasets, dataset)()
    for i in range(best_n):
        if i < 2:
            continue
        print ('Fully train student architecture %d/%d' %(i+1, best_n))
        model = torch.load('%s/arch_%d.pth' % (opt.savedir, i))
        tr.train_model_student(model, dataset,
                               '%s/fully_kd_%d.pth' % (opt.savedir, i), i)




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

    assert args.network in ['resnet18', 'resnet34','resnet50','resnet101', 'vgg19', 'shufflenet', 'alexnet', 'sample9']
    assert args.dataset in ['cifar10', 'cifar100', 'imagenet',  'artificial']

    if args.network in ['resnet18', 'resnet34', 'resnet50', 'resnet101']:
        opt.co_graph_gen = 'get_graph_resnet'
    #elif args.network in ['resnet50', 'resnet101']:
    #    opt.co_graph_gen = 'get_graph_long_resnet'
    elif args.network in ['vgg19', 'sample9']:
        opt.co_graph_gen = 'get_graph_vgg'
    elif args.network == 'alexnet':
        opt.co_graph_gen = 'get_graph_alex'
    elif args.network == 'shufflenet':
        opt.co_graph_gen = 'get_graph_shufflenet'

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
    opt.savedir = './save/%s_%s_%s' % (args.network, args.dataset, args.suffix)
    opt.writer = SummaryWriter('./runs/%s_%s_%s' % (args.network, args.dataset,
                                                  args.suffix))
    assert not(os.path.exists(opt.savedir)), 'Overwriting existing files!'

    print ('Start compression. Please check the TensorBoard log in the folder ./runs/%s_%s_%s.'%
                                                    (args.network, args.dataset, args.suffix))

    model = torch.load(opt.model).to(opt.device)
    if args.network == 'alexnet':
        model.flatten = models.Flatten()
    elif args.network != 'sample9':
        model.avgpool = nn.AvgPool2d(4, stride=1)
    teacher = Architecture(*(getattr(gr, opt.co_graph_gen)(model)))
    #print(teacher)
    dataset = getattr(datasets, opt.dataset)()
    record = Record()
    if opt.bo:
        compression(teacher, dataset, record, args.objective, args.constype, args.consval)
    else:
        random_compression(teacher, dataset, args.objective, args.constype, args.consval, 100)
    fully_train(teacher, dataset=opt.dataset)
