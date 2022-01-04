import matplotlib_inline
import torch
import torchvision
from torchvision import datasets, transforms
import os
import time
import matplotlib.pyplot as plt

matplotlib_inline

data_dir = "./data/HANDS"

data_transform = {x:transforms.Compose([transforms.Resize([64,64]),transforms.ToTensor()])
                  for x in ["train","valid"]}

image_datasets = {x:datasets.ImageFolder(root=os.path.join(data_dir,x),transform = data_transform[x])
                  for x in ["train", "valid"]}

dataloader = {x:torch.utils.data.DataLoader(dataset = image_datasets[x],
                                            batch_size = 16,
                                            shuffle = True)
                                            for x in ["train", "valid"]}

a_example, b_example = next(iter(dataloader["train"]))

print(u"a_example 个数：{}".format(len(a_example)))
print(u"b_example 个数：{}".format(len(b_example)))

#热编码
# {'close_lyrics': 0, 'like': 1, 'open_lyrics': 2, 'pause': 3, 'play': 4}

index_classes = image_datasets["train"].class_to_idx
print(index_classes)

#做事留一手
#为了到时候可以还原原本标签进而具有可识别性
#将原来的标签存储起来：

example_classes = image_datasets["train"].classes
print(example_classes)

#使用Matplotlib对一个批次的图片进行绘制。

img1 = torchvision.utils.make_grid(a_example)
img1 = img1.numpy().transpose([1,2,0])

print([example_classes[i] for i in b_example])

plt.imshow(img1)
plt.savefig("./data/figs/data_visualization.jpg")
plt.show()

