import torch
import datasets
import models
import training as tr
import options as opt
import torch.nn as nn
from tensorboardX import SummaryWriter
from architecture import Architecture
import graph as gr
from gpu_energy_eval import GPUEnergyEvaluator
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = getattr(datasets,'CIFAR100')()
teacher_model_path = './models/pretrained/resnet34_cifar100.pth'
teacher = torch.load(teacher_model_path).to(device)
teacher.avgpool = nn.AvgPool2d(4, stride=1)
teacher = Architecture(*(getattr(gr, opt.co_graph_gen)(teacher)))
teacher_params = teacher.param_n()
full_acc = tr.test_model(teacher,dataset)
print("teacher model acc: %4.2f" %(full_acc))


model_path = "save/resnet34_cifar100_random/fully_kd_"

opt.writer = SummaryWriter('./runs/resnet34_cifar100_3_kd')

for i in range(4):
    cur_path = model_path+str(i)+".pth"
    model = torch.load(cur_path).to(device)
    model.avgpool = nn.AvgPool2d(4, stride=1)
    #tr.train_model_student_kd(teacher, model, dataset, "save/resnet34_cifar100_1/fully_kd_"+str(i)+".pth",i)
    evaluator = GPUEnergyEvaluator(gpuid=0)
    start_time=time.time()
    evaluator.start()
    acc = tr.test_model(model, dataset)
    energy_used = evaluator.end()
    time_used = time.time() - start_time
    student_params = model.param_n()
    c = 1.0 * student_params/teacher_params
    print("model size: ", c)
    print("Energy used: ", energy_used)
    print("Time_used: ", time_used)
    print("model %d acc: %4.2f" %(i, acc))
