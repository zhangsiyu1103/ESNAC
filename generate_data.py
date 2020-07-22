import torch
import models
import pandas as pd
import torch.nn as nn

h = 6
w = 6

model = models.GroundTruth7()
#model.load_state_dict(torch.load('ground_truth0.pth'))

torch.save(model.state_dict(), 'base/groundtruth7.pth')
#model = torch.load('model/ground_truth0.pt')

#for parameter in model.parameters():
#    print(parameter.size())


model.eval()


data = {"image":[], "category":[]}

for i in range(30000):

    fake_img = torch.randn([1, 3, h, w], dtype=torch.float32)
    #print(fake_img)
    output = model(fake_img)
    #_, predicted = torch.max(output, 1)
    data["image"].append(fake_img[0].detach())
    data["category"].append(output[0].detach())

torch.save(data, "datasets/data/sample/groundtruth7/train.pt")

data = {"image":[], "category":[]}

for i in range(10000):

    fake_img = torch.randn([1, 3, h, w], dtype=torch.float32)
    output = model(fake_img)
    #_, predicted = torch.max(output, 1)
    data["image"].append(fake_img[0].detach())
    data["category"].append(output[0].detach())
    #data["category"].append(predicted.item())

torch.save(data, "datasets/data/sample/groundtruth7/test.pt")

#torch.save(image, "data/train/image.pt")
#torch.save(category, "data/train/category.pt")
#df = pd.DataFrame(data, columns = ["image", "category"])

#df.to_csv("data/data1.csv", index=False)

