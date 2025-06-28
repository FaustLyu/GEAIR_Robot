import numpy as np
import torch
from torch.nn import Sequential, Conv2d, MaxPool2d, ReLU, BatchNorm2d
from torch import nn
from torch.utils import model_zoo
import torch, gc
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


root = "../siamese_data"
device = torch.device('cuda:0')
learning_rate = 0.001


# -----------------ready the dataset--------------------------
def default_loader(path):
    return Image.open(path)


#   return Image.open(path).convert('RGB')
class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], words[1], (float(words[2]), float(words[3]), float(words[4]))))
        # print("imgs=",imgs)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn1, fn2, label = self.imgs[index]
        img1 = self.loader(fn1)
        img2 = self.loader(fn2)

        # Convert grayscale images to pseudo-RGB

        if img1.mode == 'L':
            img1 = img1.convert('RGB')
        if img2.mode == 'L':
            img2 = img2.convert('RGB')

        label = np.array(label, dtype='float32')
        # print(label)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, label

    def __len__(self):
        return len(self.imgs)


train_data = MyDataset(txt=root + r'\train\data.txt', transform=transforms.ToTensor())
test_data = MyDataset(txt=root + r'\test\eval.txt', transform=transforms.ToTensor())
# test_data=train_data
train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=32)


# test_loader=train_loader
# -----------------create the Net and training------------------------

# 预训练的ResNet50提取特征
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # 导入预训练的ResNet50提取特征
      #  self.resnet = resnet50(pretrained=True)
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # 移除最后一层只保留特征提取部分
        modules = list(self.resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

    def forward(self, x):
        return self.resnet(x)


# LSTM model
class SiameseLSTM(nn.Module):
    def __init__(self):
        super(SiameseLSTM, self).__init__()
        self.feature_extractor = FeatureExtractor()

        self.lstm = nn.LSTM(input_size=2048, hidden_size=512, batch_first=True)

        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 3),
        )

    def forward(self, input1, input2):
        output1 = self.feature_extractor(input1).squeeze(-1).squeeze(-1)
        output2 = self.feature_extractor(input2).squeeze(-1).squeeze(-1)

        lstm_out1, _ = self.lstm(output1.unsqueeze(1))
        lstm_out2, _ = self.lstm(output2.unsqueeze(1))

        input = torch.cat([lstm_out1.mean(dim=1), lstm_out2.mean(dim=1)], dim=1)
        output = self.fc2(input)
        return output


def showloss():
    import matplotlib.pyplot as plt
    import numpy as np

    X = Epochs  # X轴坐标数据
    Y = Loss  # Y轴坐标数据
    # plt.plot(X,Y,lable="$sin(X)$",color="red",linewidth=2)

    plt.figure(figsize=(8, 6))  # 定义图的大小
    plt.xlabel("epochs")  # X轴标签
    plt.ylabel("loss")  # Y轴坐标标签
    plt.title("Loss")  # 曲线图的标题

    plt.plot(X, Y)  # 绘制曲线图
    # 在ipython的交互环境中需要这句话才能显示出来
    plt.show()


model = SiameseLSTM().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_func = nn.MSELoss(reduction='sum')
torch.set_printoptions(sci_mode=False)
Loss = []
Epochs = []
epochs = 100
# epochs=1
lowest_val_loss = float('inf')  # Set to positive infinity initially

for epoch in range(epochs):
    print('epoch {}'.format(epoch + 1))
    # training-----------------------------
    if epoch == 400:
        # if epoch == 200:
        learning_rate = 0.0001
    # if epoch == 400:
    #   learning_rate = 0.00001

    train_loss = 0.
    train_acc = np.array([0., 0., 0.])
    for batch_x, batch_y, batch_z in train_loader:
        batch_x, batch_y, batch_z = Variable(batch_x).to(device), Variable(batch_y).to(device), Variable(batch_z).to(
            device)

        out = model(batch_x, batch_y)
        # out=1.6*out-0.8#归一化
        # batch_z=1.6*batch_z-0.8
        loss = loss_func(out, batch_z)
        train_loss += loss.item()
        temp = batch_z.cpu() - out.cpu()
        train_correct = np.array(temp.detach().numpy())
        train_correct = train_correct.sum(0) / len(train_correct)
        train_acc = train_acc + train_correct
        optimizer.zero_grad()
        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()

    print('Train Loss: {:.6f}'.format(train_loss / (len(train_data))))
    print('ACC: ', train_acc)
    Loss.append(train_loss / (len(train_data)))
    Epochs.append(epoch)
# Validation loss
val_loss = 0.
for batch_x, batch_y, batch_z in test_loader:
    batch_x, batch_y, batch_z = batch_x.to(device), batch_y.to(device), batch_z.to(device)
    out = model(batch_x, batch_y)
    loss = loss_func(out, batch_z)
    val_loss += loss.item()

avg_val_loss = val_loss / len(test_data)
print('Validation Loss: {:.6f}'.format(avg_val_loss))

# Save the model if the validationd loss is the lowest
if avg_val_loss < lowest_val_loss:
    print("Saving best model with Validation Loss: {:.6f}".format(avg_val_loss))
    lowest_val_loss = avg_val_loss
    torch.save(model.state_dict(), "best_Siamese_model.pth")

torch.save(model.state_dict(), "Siamese_model.pth")
# model.save('model.h5')


model.eval()
eval_loss = 0.00
eval_acc = 0.
test_acc = np.array([0., 0., 0.])
for batch_x, batch_y, batch_z in test_loader:
    batch_x, batch_y, batch_z = Variable(batch_x).to(device), Variable(batch_y).to(device), Variable(batch_z).to(device)
    print("batch_z:", batch_z)
    out = model(batch_x, batch_y)
    print("out:", out)
    temp = batch_z.cpu() - out.cpu()
    print("temp:", torch.abs(temp))
showloss()
