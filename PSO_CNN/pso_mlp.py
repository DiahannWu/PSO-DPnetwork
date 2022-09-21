import os
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
# import FullyConnect3
import numpy as np
import pandas as pd
import random
import time
import torch.nn.functional as F
# from torchsummary import summary


# 设置超参数
batch_size = 64
learning_rate = 1e-2
whether_tet = True
model_dir = "./PSOinFullyConnect3_First.pth"
loss_value = []



dara_tf = transforms.Compose(
    [transforms.ToTensor(),  # 范围为0-1
     transforms.Normalize([0.5], [0.5])])   # 单通道，均值0.5，方差0.5

# 下载训练集NMIST手写数字训练集
train_dataset = datasets.MNIST(
    root="./data", train=True, transform=dara_tf, download=False)
test_dataset = datasets.MNIST(
    root="./data", train=False, transform=dara_tf)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def tra_model(theWight):
    model.cuda().train()
    # print("现在进行权重的更新")
    # print('weight1:', model.layer1.weight.size())
    # print('weight2:', model.layer2.weight.size())
    # print('weight3:', model.layer3.weight.size())

    model.layer1.weight.data = torch.from_numpy(theWight[0, :, :]).to(torch.float32).cuda()
    model.layer2.weight.data = torch.from_numpy(theWight[1, 0:100, 0:300]).to(torch.float32).cuda()
    model.layer3.weight.data = torch.from_numpy(theWight[2, 0:10, 0:100]).to(torch.float32).cuda()

    # print('权重更新后')
    # print('weight1:', model.layer1.weight.size())
    # print('weight2:', model.layer2.weight.size())
    # print('weight3:', model.layer3.weight.size())

    correct = 0
    total = 0
    for c, data in enumerate(train_loader, 1):
        # print("当前训练批次为： {}次 ".format(c))
        img, label = data
        img, label = img.cuda(), label.cuda()
        img = img.view(img.size(0), -1)
        img = Variable(img)
        label = Variable(label)

        out = model.forward(img)
        _, predicted = torch.max(out, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
        loss = criterion(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_value.append(loss.data)
        # print("当前损失函数为：", loss)
    accuracy = 100*correct/total
    # print('correct: ', correct)
    # print('total: ', total)
    # print('accuracy: ', 100*correct/total)
    # print('loss: ', loss.item())

    return loss, correct, total, accuracy


def train():
    # model.cuda().train()
    # print("现在进行权重的更新")
    # print('weight1:', model.layer1.weight.size())
    # print('weight2:', model.layer2.weight.size())
    # print('weight3:', model.layer3.weight.size())
    correct = 0
    total = 0
    with torch.no_grad():
        for c, data in enumerate(train_loader, 1):
            # print("当前训练批次为： {}次 ".format(c))
            img, label = data
            img, label = img.cuda(), label.cuda()
            img = img.view(img.size(0), -1)
            img = Variable(img)
            label = Variable(label)

            out = model(img)
            _, predicted = torch.max(out, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
            loss = criterion(out, label)
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            loss_value.append(loss.data)
            # print("当前损失函数为：", loss)
    accuracy = 100*correct/total
    # print('correct: ', correct)
    # print('total: ', total)
    # print('accuracy: ', 100*correct/total)
    # print('loss: ', loss.item())

    return loss, correct, total, accuracy

def test():
    print('------ Test Start -----')
    # net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for test_x, test_y in test_loader:
            images, labels = test_x.cuda(), test_y.cuda()
            # images, labels = test_x, test_y
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print('Accuracy of the model is: %.8f %%' % accuracy)
    print('test correct: ', correct)
    return accuracy

# SimpleNet
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.layer1 = nn.Linear(28 * 28, 300)
        self.layer2 = nn.Linear(300, 100)
        self.layer3 = nn.Linear(100, 10)
        # self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        # flatten image input
        x = x.view(-1, 28 * 28)
        # add hidden layer, with relu activation function
        x = F.relu(self.layer1(x))
        # x = self.dropout(x)
        # add hidden layer, with relu activation function
        x = F.relu(self.layer2(x))
        # x = self.dropout(x)
        # add output layer
        x = self.layer3(x)
        return x

model = SimpleNet().cuda()
# FullyConnect3.SimpleNet(in_ch=28*28, n_hidden_1=300, n_hidden_2=100, out_ch=10)
# summary(model, (1,28,28), 64, device='cuda')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 粒子数量
num = 3

# 粒子位置矩阵维数 考虑三层全连接层权重的尺寸依次为[300, 784]\[100, 300]\[10, 100]
numx = 3
numy = 300
numz = 784

# p为粒子位置矩阵，初始为标准正态分布
p = np.random.randn(num, numx, numy, numz)/100

# v为粒子速度矩阵，初始为标准正态分别
v = np.random.randn(num, numx, numy, numz)/100

# 个体最佳位置
perbest_P = np.array(p, copy=True)

# 全局最佳位置
allbest_P = np.zeros((numx, numy, numz))

# 更新适应度函数
new_value = np.zeros(num)

# 粒子个体历史最优值
perbest_V = np.zeros(num)

# 粒子群体历史最优值
allbest_V = 0


# 计算初始粒子群的目标函数值（伴随梯度下降）
for i in range(num):
    perbest_V[i], _, _, _ = tra_model(p[i, :, :, :])


# 确定群体历史最优值
allbest_V = min(perbest_V)

# 确定初始最优位置
allbest_P = p[np.argmin(perbest_V), :, :, :]

# 设置最大迭代次数
max_iter = 300

# 设置速度更新权重值
w_forV = 0.1

# 开始迭代
stat_time = time.time()
for i in range(max_iter):
    print("当前迭代轮数： ", i)

    # 速度更新
    v = w_forV * v + 2.4 * random.random() * (allbest_P - p) + 1.7 * random.random() * (perbest_P - p)

    # 位置更新
    p = p + v

    # 计算每个粒子到达新位置后所得目标的函数值
    for a in range(num):
        new_value[a], _, _, _ = tra_model(p[a, :, :, :])


    # 更新全局最优
    if min(new_value) < allbest_V:
        allbest_V = min(new_value)
        allbest_P = p[np.argmin(perbest_V), :, :, :]

    # 更新个体历史最优
    for b in range(num):
        if new_value[b] < perbest_V[b]:
            perbest_V[b] = new_value[b]
            perbest_P[b, :, :, :] = p[b, :, :, :]

    # print(perbest_P.shape)
    loss, correct, total, accuracy = tra_model(allbest_P)
    print('loss:', loss.item())
    print('correct: ', correct)
    print('total: ', total)
    print('accuracy: %.8f %%' % accuracy)
    test()
end_time = time.time() - stat_time
print(end_time)
state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
torch.save(state, model_dir)
saveLoss = pd.DataFrame(data=loss_value)
saveLoss.to_csv('./lossforMaxNum20.csv')
print("损失函数最小值为： ", allbest_V)
print("对应权重值为： ", allbest_P)
