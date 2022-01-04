import torch
from torchvision import datasets, models, transforms
import os
from torch.autograd import Variable
import time
import matplotlib.pyplot as plt

data_dir = ".\data\HANDS"

data_transform = {x:transforms.Compose([transforms.Resize([224,224]),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
                  for x in ["train","valid"]}

image_datasets = {x:datasets.ImageFolder(root=os.path.join(data_dir, x),
                                         transform=data_transform[x])
                  for x in ["train", "valid"]}

dataloader = {x:torch.utils.data.DataLoader(dataset=image_datasets[x],
                                            batch_size=16,
                                            shuffle=True)
                  for x in ["train", "valid"]}

X_example, y_example = next(iter(dataloader["train"]))
example_classes = image_datasets["train"].classes
index_classes = image_datasets["train"].class_to_idx

model = models.resnet50(pretrained = True)
for param in model.parameters():
    param.requires_grad = False
model.fc = torch.nn.Linear(2048,5)

Use_gpu = torch.cuda.is_available()

if Use_gpu:
    print("Using GPU for training!!!!! ")
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = model.cuda()

loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.00001)

epoch_n = 500

Loss_list_for_train = []
Loss_list_for_valid = []
Accuracy_list_for_train = []
Accuracy_list_for_valid = []
time_open = time.time()

def train_model(model, loss_func, optimizer, epoch_n):
    for epoch in range(epoch_n):
        print("Epoch {}/{}".format(epoch, epoch_n - 1))
        print("-" * 50)

        for phase in ["train", "valid"]:
            if phase == "train":
                print("Training...")
                model.train(True)
            else:
                print("Validating...")
                model.train(False)

            # 初始化loss和accuracy
            running_loss = 0.0
            running_corrects = 0

            for batch, data in enumerate(dataloader[phase], 1):
                X, y = data
                # if Use_gpu:
                X, y = Variable(X.cuda()), Variable(y.cuda())
                # else:
                #     X, y = Variable(X), Variable(y)
                y_pred = model(X)
                _, pred = torch.max(y_pred.data, 1)
                optimizer.zero_grad()
                loss = loss_func(y_pred, y)
                if phase == "train":
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item()
                running_corrects += torch.sum(pred == y.data)
                if batch % 500 == 0 and phase == "train":
                    print('Batch {},TrainLoss: {:.4f},TrainAcc: {:.4f}'.format(batch, running_loss / batch,
                                                                               100 * running_corrects / (16 * batch)))
            if (phase == "train"):
                Loss_list_for_train.append(running_loss * 16 / len(image_datasets[phase]))
                Accuracy_list_for_train.append(100 * running_corrects / len(image_datasets[phase]))
            if (phase == "valid"):
                Loss_list_for_valid.append(running_loss * 16 / len(image_datasets[phase]))
                Accuracy_list_for_valid.append(100 * running_corrects / len(image_datasets[phase]))
            epoch_loss = running_loss * 16 / len(image_datasets[phase])
            epoch_acc = 100 * running_corrects / len(image_datasets[phase])
            print('{} Loss:{:.4f} {} Acc:{:.4f}%'.format(phase, epoch_loss, phase, epoch_acc))
    torch.save(model.state_dict(), "./data/saved_models/parameter.pkl")
    print("Successfully saved parameter�?")
    torch.save(model, './data/saved_models/whole_model.pkl')
    print("Successfully saved Complete Model �?")
    return model

trained_model = train_model(model, loss_func, optimizer, epoch_n)

x1_epoch = range(0,500)
x2_epoch = range(0,500)

y_train_acc = Accuracy_list_for_train
y_valid_acc = Accuracy_list_for_valid
y_train_loss = Loss_list_for_train
y_valid_loss = Loss_list_for_valid

plt.title("Train_Acc & Valid_Acc")
plt.plot(x1_epoch, y_train_acc,'o-')
plt.plot(x1_epoch, y_valid_acc,'o-')
plt.xlabel("Epoch")
plt.ylabel("Acc %")
plt.savefig("./data/figs/Train_Valid_Accuracy.jpg")
plt.show()

plt.title("Train_Loss & Valid_Loss")
plt.plot(x2_epoch, y_train_loss,'.-')
plt.plot(x2_epoch, y_valid_loss,'.-')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("./data/figs/Train_Valid__Loss.jpg")
plt.show()


time_end = time.time() - time_open
print("Cose time : ", time_end, " seconds")



