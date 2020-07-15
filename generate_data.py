import torch
import models
import pandas as pd
import torch.nn as nn

h = 6
w = 6

model = models.GroundTruth4()
#model.load_state_dict(torch.load('ground_truth0.pth'))

torch.save(model.state_dict(), 'base/groundtruth4.pth')
#model = torch.load('model/ground_truth0.pt')

#for parameter in model.parameters():
#    print(parameter.size())


model.eval()


data = {"image":[], "category":[]}

for i in range(16000):

    fake_img = torch.randn([1, 3, h, w], dtype=torch.float32)
    #print(fake_img)
    output = model(fake_img)
    _, predicted = torch.max(output, 1)
    data["image"].append(fake_img[0].detach())
    data["category"].append(predicted.item())

torch.save(data, "datasets/data/sample/groundtruth4/train.pt")


for i in range(4000):

    fake_img = torch.randn([1, 3, h, w], dtype=torch.float32)
    output = model(fake_img)
    _, predicted = torch.max(output, 1)
    data["image"].append(fake_img[0].detach())
    data["category"].append(predicted.item())

torch.save(data, "datasets/data/sample/groundtruth4/test.pt")

#torch.save(image, "data/train/image.pt")
#torch.save(category, "data/train/category.pt")
#df = pd.DataFrame(data, columns = ["image", "category"])

#df.to_csv("data/data1.csv", index=False)

