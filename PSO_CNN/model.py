import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from resnet_cifar import resnet20

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


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.convl = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=0),  # 卷积层
            # 1*28*28
            # 输入通道数，输出通道数 都为整形
            # kernel_size指卷积核的大小；stride指步长，即卷积核或者pooling窗口的滑动位移。
            # padding指对input的图像边界补充一定数量的像素，目的是为了计算位于图像边界的像素点的卷积响应；
            # ( input_size + 2*padding - kernel_size ) / stride+1 = output_size
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0),  # 卷积层
            nn.ReLU(),
            # nn.Dropout(p=0.5),
        )
        self.dense = nn.Sequential(  # 全连接层
            nn.Linear(6*6*64, 10),
        )

    def forward(self, x):  # 前向传播
        x = self.convl(x)  # 输入卷积层
        x = x.view(-1, 6*6*64)  # torch里面，view函数相当于numpy的reshape
        # -1表示一个不确定的数，就是你如果不确定你想要reshape成几行，但是列确定，-1会自动更改为合适的值
        x = self.dense(x)  # 输入全连接层
        return x

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 100)
        self.fc2 = nn.Linear(100, 10)
        # self.fc3 = nn.Linear(100, 10)
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        # flatten image input
        x = x.view(-1, 28 * 28)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        # add hidden layer, with relu activation function
        # x = F.relu(self.fc2(x))
        # x = self.dropout(x)
        # add output layer
        x = self.fc2(x)
        return x


# model = CNN().cuda()
#
# 打印每层参数信息
# summary(model, (1, 28, 28), 64, device='cuda')
#
# # # 查看每层对应名称
# for name in model.state_dict():
#     print(name)
#
# # 根据名称输出相应层权重
# print(model.state_dict()['convl.0.weight'])
# print(model.state_dict()['convl.0.bias'])
#
# # 打印模块名字和参数
# for name, parameters in model.named_parameters():
#     print(name, parameters.size())

# for p in model.modules():
#     # print(type(p))
#     # if type(p) is nn.Linear:
#     #     print(type(p))
#     if (type(p) is nn.Conv2d) | (type(p) is nn.Linear):
#         print(type(p))

