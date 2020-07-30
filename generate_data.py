import torch
import models
import pandas as pd
import torch.nn as nn
import graph as gr
from architecture import Architecture

h = 6
w = 6

model = models.Sample9()
#model.load_state_dict(torch.load('base/groundtruth9.pth'))

model.cuda()

#torch.save(model.state_dict(), 'base/groundtruth9.pth')

model = Architecture(*(getattr(gr, 'get_graph_vgg')(model)))
#model = torch.load('base/groundtruth8.pth')

print(model.param_n())

#for parameter in model.parameters():
#    print(parameter.size())


model.eval()


#data = {"image":[], "category":[]}
#
#for i in range(50000):
#
#    #fake_img = torch.randn([1, 3, h, w], dtype=torch.float32)
#    fake_img = torch.empty(1,3,h,w).normal_(mean=128,std=32)
#    #fake_img.to('cuda')
#    #print(fake_img)
#    output = model(fake_img)
#    _, predicted = torch.max(output, 1)
#    data["image"].append(fake_img[0].detach())
#    #data["category"].append(output[0].detach())
#    data["category"].append(predicted.item())
#
#torch.save(data, "datasets/data/sample/groundtruth9/train.pt")
#
#data = {"image":[], "category":[]}
#
#for i in range(10000):
#
#    #fake_img = torch.randn([1, 3, h, w], dtype=torch.float32)
#    fake_img = torch.empty(1,3,h,w).normal_(mean=128,std=32)
#    #fake_img.to('cuda')
#    output = model(fake_img)
#    _, predicted = torch.max(output, 1)
#    data["image"].append(fake_img[0].detach())
#    #data["category"].append(output[0].detach())
#    data["category"].append(predicted.item())
#
#torch.save(data, "datasets/data/sample/groundtruth9/test.pt")

#torch.save(image, "data/train/image.pt")
#torch.save(category, "data/train/category.pt")
#df = pd.DataFrame(data, columns = ["image", "category"])

#df.to_csv("data/data1.csv", index=False)

