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
import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = getattr(datasets,'Artificial')()
#teacher_model_path = './models/pretrained/sample7_artificial.pth'
teacher_model_path = "save/Sample7_artificial_large_data/sample7_artificial.pth"
#teacher_model_path = './base/groundtruth4.pth'
teacher = torch.load(teacher_model_path)
#teacher = models.GroundTruth4()
#teacher.load_state_dict(torch.load(teacher_model_path))
teacher.to(device)
#print(teacher)
#teacher.avgpool = nn.AvgPool2d(4, stride=1)
#print(teacher)
#teacher.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
opt.co_graph_gen = 'get_graph_vgg'
teacher = Architecture(*(getattr(gr, opt.co_graph_gen)(teacher)))
print(teacher)
teacher_params = teacher.param_n()
full_loss = tr.test_model_regression(teacher,dataset)
print("teacher model loss: %4.3f" %(full_loss))
#print("teacher params number: ", teacher_params)


model_path = "save/sample7_artificial_random/fully_kd_"
opt.writer = SummaryWriter('./runs/sample7_artificial_random')

def in_(x, set1):
    for i in set1:
        if torch.equal(x, i):
            return True
    return False

def diff_(set1, set2):
    ret = set()
    for i in set1:
        if not in_(i, set2):
            ret.add(i)
    return ret

#for i in range(4):
#    cur_path = model_path+str(i)+".pth"
#    model = torch.load(cur_path).to(device)
#    print(model)
#    #model.avgpool = nn.AvgPool2d(4, stride=1)
#    #tr.train_model_student_kd(teacher, model, dataset, "save/resnet34_cifar100_1/fully_kd_"+str(i)+".pth",i)
#    evaluator = GPUEnergyEvaluator(gpuid=0, watts_offset=False)
#    start_time=time.time()
#    evaluator.start()
#    loss = tr.test_model_regression(model, dataset)
#    energy_used = evaluator.end()
#    time_used = time.time() - start_time
#    student_params = model.param_n()
#    c = 1.0 * student_params/teacher_params
#    print("model size: ", c)
#    print("Energy used: ", energy_used/10000)
#    print("Time_used: ", time_used)
#    print("model %d loss: %4.3f" %(i, loss))
