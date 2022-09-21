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
import argparse
from models import resnet20
from utils import get_data_loader, get_sigma, restore_param, checkpoint, adjust_learning_rate, process_grad_batch
from utils import l2_regularization

#package for computing individual gradients
from backpack import backpack, extend
from backpack.extensions import BatchGrad
from model import MLP, CNN, SimpleNet

# 设置超参数
# batch_size = 1000
# learning_rate = 1e-2
whether_tet = True
# model_dir = "./PSOinFullyConnect3_First.pth"
loss_value = []

chanels = 1
size = 28
n_weight1 = 300
n_weight2 = 100
n_weight3 = 10
all_weight = chanels*size*size*n_weight1+n_weight1+n_weight1*n_weight2+n_weight2+n_weight2*n_weight3+n_weight3
n1 = chanels * size * size * n_weight1
n2 = n1 + n_weight1
n3 = n2 + n_weight1*n_weight2
n4 = n3 + n_weight2
n5 = n4 + n_weight2*n_weight3
n6 = n5 + n_weight3

parser = argparse.ArgumentParser(description='Differentially Private learning with DP-SGD')

## general arguments
parser.add_argument('--dataset', default='fashion', type=str, help='dataset name')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--sess', default='resnet20_cifar10', type=str, help='session name')
parser.add_argument('--seed', default=2, type=int, help='random seed')
# parser.add_argument('--weight_decay', default=0., type=float, help='weight decay')
parser.add_argument('--batchsize', default=1000, type=int, help='batch size')
parser.add_argument('--n_epoch', default=300, type=int, help='total number of epochs')
parser.add_argument('--lr', default=0.1, type=float, help='base learning rate (default=0.1)')
# parser.add_argument('--momentum', default=0.9, type=float, help='value of momentum')


## arguments for learning with differential privacy
parser.add_argument('--private', default=True, action='store_true', help='enable differential privacy')
parser.add_argument('--clip', default=1., type=float, help='gradient clipping bound')
parser.add_argument('--eps', default=2., type=float, help='privacy parameter epsilon')
parser.add_argument('--delta', default=1e-5, type=float, help='desired delta')

parser.add_argument('--result_dir', default='fashion_result/mlp', type=str, help='reslult file dir')
parser.add_argument('--file_name', default='fashion_result_clip1_eps2', type=str, help='reslult file name')


args = parser.parse_args()

assert args.dataset in ['mnist', 'fashion', 'cifar10']

best_acc = 0

if(args.seed != -1):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

if (args.dataset == 'mnist') | (args.dataset == 'fashion'):
    transform = transforms.Compose(
        [transforms.ToTensor(),  # 范围为0-1
         transforms.Normalize([0.5], [0.5])])   # 单通道，均值0.5，方差0.5
else:
    transform = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

if args.dataset == 'mnist':
    # 下载训练集NMIST手写数字训练集
    train_dataset = datasets.MNIST(
        root="./data", train=True, transform=transform, download=False)
    test_dataset = datasets.MNIST(
        root="./data", train=False, transform=transform)
elif args.dataset == 'fashion':
    # 下载训练集Fashion-NMIST训练集
    train_dataset = datasets.FashionMNIST(
        root="./data", train=True, transform=transform, download=False)
    test_dataset = datasets.FashionMNIST(
        root="./data", train=False, transform=transform)
else:
    # 下载训练集cifar-10训练集
    train_dataset = datasets.CIFAR10(
        root="./data", train=True, transform=transform, download=False)
    test_dataset = datasets.CIFAR10(
        root="./data", train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=args.batchsize, shuffle=False)

print('# of training examples: ', len(train_dataset), '# of testing examples: ', len(test_dataset))

print('\n==> Computing noise scale for privacy budget (%.1f, %f)-DP'%(args.eps, args.delta))
sampling_prob=args.batchsize/len(train_dataset)
steps = int(args.n_epoch/sampling_prob)
sigma, eps = get_sigma(sampling_prob, steps, args.eps, args.delta, rgp=False)
noise_multiplier = sigma
print('noise scale: ', noise_multiplier, 'privacy guarantee: ', eps)

def tra_model(theWight):
    net.cuda().train()
    # print("现在进行权重的更新")
    # print('weight1:', net.layer1.weight.size())
    # print('weight2:', net.layer2.weight.size())
    # print('weight3:', net.layer3.weight.size())

    net.layer1.weight.data = torch.from_numpy(theWight[0:n1].reshape(n_weight1, chanels*size*size)).to(torch.float32).cuda()
    net.layer1.bias.data = torch.from_numpy(theWight[n1:n2].reshape(n_weight1)).to(torch.float32).cuda()
    net.layer2.weight.data = torch.from_numpy(theWight[n2:n3].reshape(n_weight2, n_weight1)).to(torch.float32).cuda()
    net.layer2.bias.data = torch.from_numpy(theWight[n3:n4].reshape(n_weight2)).to(torch.float32).cuda()
    net.layer3.weight.data = torch.from_numpy(theWight[n4:n5].reshape(n_weight3, n_weight2)).to(torch.float32).cuda()
    net.layer3.bias.data = torch.from_numpy(theWight[n5:n6].reshape(n_weight3)).to(torch.float32).cuda()

    # print('权重更新后')
    # print('weight1:', net.layer1.weight.size())
    # print('weight2:', net.layer2.weight.size())
    # print('weight3:', net.layer3.weight.size())

    train_loss = 0
    correct = 0
    total = 0
    for c, data in enumerate(train_loader, 1):
        # print("当前训练批次为： {}次 ".format(c))
        img, label = data
        img, label = img.cuda(), label.cuda()
        # img = img.view(img.size(0), -1)
        img = Variable(img)
        label = Variable(label)

        if (args.private):
            optimizer.zero_grad()
            outputs = net(img)
            loss = loss_func(outputs, label)
            # add noise to loss
            # noise = torch.normal(0, noise_multiplier * args.clip / args.batchsize, size=[args.batchsize])
            # loss += (torch.sum(noise) * l2_regularization(net))
            # noise = np.random.normal(0, noise_multiplier * args.clip / args.batchsize)
            # loss += (noise * l2_regularization(net))
            with backpack(BatchGrad()):
                loss.backward()
                process_grad_batch(list(net.parameters()), args.clip) # clip gradients and sum clipped gradients
            ## add noise to gradient
            for p in net.parameters():
                shape = p.grad.shape
                numel = p.grad.numel()
                grad_noise = torch.normal(0, noise_multiplier*args.clip/args.batchsize, size=p.grad.shape, device=p.grad.device)
                p.grad.data += grad_noise
        else:
            outputs = net.forward(img)
            loss = loss_func(outputs, label)
            optimizer.zero_grad()
            loss.backward()
            try:
                for p in net.parameters():
                    del p.grad_batch
            except:
                pass
        _, predicted = torch.max(outputs, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
        optimizer.step()
        step_loss = loss.item()
        if (args.private):
            step_loss /= img.shape[0]
        train_loss += step_loss
        loss_value.append(loss.data)
        # print("当前损失函数为：", loss)
    theWight[0: n1] = net.layer1.weight.data.cpu().numpy().flatten()[:]
    theWight[n1: n2] = net.layer1.bias.data.cpu().numpy().flatten()[:]
    theWight[n2: n3] = net.layer2.weight.data.cpu().numpy().flatten()[:]
    theWight[n3: n4] = net.layer2.bias.data.cpu().numpy().flatten()[:]
    theWight[n4: n5] = net.layer3.weight.data.cpu().numpy().flatten()[:]
    theWight[n5: n6] = net.layer3.bias.data.cpu().numpy().flatten()[:]
    accuracy = 100*correct/total
    # print('correct: ', correct)
    # print('total: ', total)
    # print('accuracy: ', 100*correct/total)
    # print('loss: ', loss.item())

    return train_loss/c, correct, total, accuracy, theWight


def tra_model_nograd(theWight):
    net.cuda().train()
    # print("现在进行权重的更新")
    # print('weight1:', net.layer1.weight.size())
    # print('weight2:', net.layer2.weight.size())
    # print('weight3:', net.layer3.weight.size())

    net.layer1.weight.data = torch.from_numpy(theWight[0:n1].reshape(n_weight1, chanels * size * size)).to(torch.float32).cuda()
    net.layer1.bias.data = torch.from_numpy(theWight[n1:n2].reshape(n_weight1)).to(torch.float32).cuda()
    net.layer2.weight.data = torch.from_numpy(theWight[n2:n3].reshape(n_weight2, n_weight1)).to(torch.float32).cuda()
    net.layer2.bias.data = torch.from_numpy(theWight[n3:n4].reshape(n_weight2)).to(torch.float32).cuda()
    net.layer3.weight.data = torch.from_numpy(theWight[n4:n5].reshape(n_weight3, n_weight2)).to(torch.float32).cuda()
    net.layer3.bias.data = torch.from_numpy(theWight[n5:n6].reshape(n_weight3)).to(torch.float32).cuda()

    # print('权重更新后')
    # print('weight1:', net.layer1.weight.size())
    # print('weight2:', net.layer2.weight.size())
    # print('weight3:', net.layer3.weight.size())

    train_loss_nograd = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for c, data in enumerate(train_loader, 1):
            # print("当前训练批次为： {}次 ".format(c))
            img, label = data
            img, label = img.cuda(), label.cuda()
            # img = img.view(img.size(0), -1)
            img = Variable(img)
            label = Variable(label)

            out = net.forward(img)
            _, predicted = torch.max(out, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
            loss = loss_func(out, label)
            step_loss = loss.item()
            if (args.private):
                step_loss /= img.shape[0]
            train_loss_nograd += step_loss
            loss_value.append(loss.data)
            # print("当前损失函数为：", loss)
        accuracy = 100*correct/total
        # print('correct: ', correct)
        # print('total: ', total)
        # print('accuracy: ', 100*correct/total)
        # print('loss: ', loss.item())

    return train_loss_nograd/c, correct, total, accuracy, theWight


'''def test_model(theWight):
    net.cuda().eval()
    # print("现在进行权重的更新")
    # print('weight1:', net.layer1.weight.size())
    # print('weight2:', net.layer2.weight.size())
    # print('weight3:', net.layer3.weight.size())

    net.layer1.weight.data = torch.from_numpy(theWight[0, :, :]).to(torch.float32).cuda()
    net.layer2.weight.data = torch.from_numpy(theWight[1, 0:100, 0:300]).to(torch.float32).cuda()
    net.layer3.weight.data = torch.from_numpy(theWight[2, 0:10, 0:100]).to(torch.float32).cuda()

    # print('权重更新后')
    # print('weight1:', net.layer1.weight.size())
    # print('weight2:', net.layer2.weight.size())
    # print('weight3:', net.layer3.weight.size())

    correct = 0
    total = 0
    for c, data in enumerate(test_loader, 1):
        # print("当前训练批次为： {}次 ".format(c))
        img, label = data
        img, label = img.cuda(), label.cuda()
        img = img.view(img.size(0), -1)
        img = Variable(img)
        label = Variable(label)

        out = net.forward(img)
        _, predicted = torch.max(out, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
        loss = loss_func(out, label)
        # print("当前损失函数为：", loss)
    accuracy = 100*correct/total
    # print('correct: ', correct)
    # print('total: ', total)
    # print('accuracy: ', 100*correct/total)
    # print('loss: ', loss.item())

    return loss, correct, total, accuracy


def train():
    # net.cuda().train()
    # print("现在进行权重的更新")
    # print('weight1:', net.layer1.weight.size())
    # print('weight2:', net.layer2.weight.size())
    # print('weight3:', net.layer3.weight.size())
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

            out = net(img)
            _, predicted = torch.max(out, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
            loss = loss_func(out, label)
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            # print("当前损失函数为：", loss)
    accuracy = 100*correct/total
    # print('correct: ', correct)
    # print('total: ', total)
    # print('accuracy: ', 100*correct/total)
    # print('loss: ', loss.item())

    return loss, correct, total, accuracy'''

# def test():
#     print('------ Test Start -----')
#     # net.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for test_x, test_y in test_loader:
#             images, labels = test_x.cuda(), test_y.cuda()
#             # images, labels = test_x, test_y
#             output = net(images)
#             _, predicted = torch.max(output.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#     accuracy = 100 * correct / total
#     print('Accuracy of the model is: %.8f %%' % accuracy)
#     print('test correct: ', correct)
#     return accuracy

def test():
    print('------ Test Start -----')
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for c, (test_x, test_y) in enumerate(test_loader):
            img, labels = test_x.cuda(), test_y.cuda()
            # images, labels = test_x, test_y
            outputs = net(img)
            loss = loss_func(outputs, labels)
            step_loss = loss.item()
            if (args.private):
                step_loss /= img.shape[0]
            test_loss += step_loss
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    if accuracy > best_acc:
        best_acc = accuracy
    print('test loss: ', test_loss/c)
    print('Accuracy of the net is: %.8f %%' % accuracy)
    print('test correct: ', correct)
    return test_loss/c, accuracy

'''# SimpleNet
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.layer1 = nn.Linear(28 * 28, 300)
        self.layer2 = nn.Linear(300, 100)
        self.layer3 = nn.Linear(100, 10)
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        # flatten image input
        x = x.view(-1, 28 * 28)
        # add hidden layer, with relu activation function
        x = F.relu(self.layer1(x))
        x = self.dropout(x)
        # add hidden layer, with relu activation function
        x = F.relu(self.layer2(x))
        x = self.dropout(x)
        # add output layer
        x = self.layer3(x)
        return x'''

# net = SimpleNet().cuda()
# FullyConnect3.SimpleNet(in_ch=28*28, n_hidden_1=300, n_hidden_2=100, out_ch=10)
# summary(net, (1,28,28), 64, device='cuda')

# criterion = nn.CrossEntropyLoss(reduction='mean')
# optimizer = optim.SGD(net.parameters(), lr=learning_rate)

print('\n==> Creating ResNet20 model instance')
if(args.resume):
    try:
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint_file = './checkpoint/' + args.sess  + '.ckpt'
        checkpoint = torch.load(checkpoint_file)
        net = resnet20()
        net.cuda()
        restore_param(net.state_dict(), checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] + 1
        torch.set_rng_state(checkpoint['rng_state'])
    except:
        print('resume from checkpoint failed')
else:
    net = SimpleNet()
    net.cuda()

net = extend(net)

num_params = 0
for p in net.parameters():
    num_params += p.numel()

print('total number of parameters: ', num_params/(10**6), 'M')

if(args.private):
    loss_func = nn.CrossEntropyLoss(reduction='sum')
else:
    loss_func = nn.CrossEntropyLoss(reduction='mean')

loss_func = extend(loss_func)

optimizer = optim.SGD(net.parameters(), lr=args.lr)

# 粒子数量
num = 3

# 粒子位置矩阵维数 考虑三层全连接层权重的尺寸依次为[300, 784]\[100, 300]\[10, 100]
# numx = 3
# numy = 300
# numz = 784

# p为粒子位置矩阵，初始为标准正态分布
p = np.random.randn(num, all_weight)/100

# v为粒子速度矩阵，初始为标准正态分别
v = np.random.randn(num, all_weight)/100

# 个体最佳位置
perbest_P = np.array(p, copy=True)

# 全局最佳位置
allbest_P = np.zeros((1, all_weight))

# 更新适应度函数
new_value = np.zeros(num)

# 粒子个体历史最优值
perbest_V = np.zeros(num)

# 粒子群体历史最优值
allbest_V = 0

new_weight = np.zeros((num, all_weight))

# 计算初始粒子群的目标函数值（伴随梯度下降）
for i in range(num):
    perbest_V[i], _, _, _, new_weight[i] = tra_model(p[i, :])


# 确定群体历史最优值
allbest_V = min(perbest_V)

# 确定初始最优位置
# allbest_P = p[np.argmin(perbest_V), :, :, :]
allbest_P = new_weight[np.argmin(perbest_V)]


# 设置最大迭代次数
# max_iter = 300

# 设置速度更新权重值
w_forV = 0.1

if not os.path.exists(args.result_dir):
    os.makedirs(args.result_dir, exist_ok=True)
# 创建train_acc.csv和var_acc.csv文件，记录loss和accuracy
df = pd.DataFrame(columns=['epoch', 'train Loss', 'test Loss', 'training accuracy', 'test accuracy'])#列名
df.to_csv(os.path.join(args.result_dir, args.file_name+".csv"), index=False) #路径可以根据需要更改

# 开始迭代
stat_time = time.time()
for i in range(args.n_epoch):
    print("当前迭代轮数： ", i)

    # 速度更新
    # v = w_forV * v + 2.4 * random.random() * (allbest_P - p) + 1.7 * random.random() * (perbest_P - p)
    #
    # # 位置更新
    # p = p + v

    p = perbest_P

    # 计算每个粒子到达新位置后所得目标的函数值
    for a in range(num):
        new_value[a], _, _, _, new_weight[a] = tra_model(p[a, :])


    # 更新全局最优
    if min(new_value) < allbest_V:
        allbest_V = min(new_value)
        # allbest_P = p[np.argmin(perbest_V), :, :, :]
        allbest_P = new_weight[np.argmin(perbest_V)]

    # 更新个体历史最优
    for b in range(num):
        if new_value[b] < perbest_V[b]:
            perbest_V[b] = new_value[b]
            # perbest_P[b, :, :, :] = p[b, :, :, :]
            perbest_P[b, :] = new_weight[b]

    # print(perbest_P.shape)
    loss, correct, total, accuracy, _ = tra_model_nograd(allbest_P)
    print('loss:', loss)
    print('correct: ', correct)
    print('total: ', total)
    print('accuracy: %.8f %%' % accuracy)
    test_loss, test_accuracy = test()

    # 数据保存在一维列表
    result_list = [i, loss, test_loss, accuracy, test_accuracy]
    # 由于DataFrame是Pandas库中的一种数据结构，它类似excel，是一种二维表，所以需要将list以二维列表的形式转化为DataFrame
    data = pd.DataFrame([result_list])
    data.to_csv(os.path.join(args.result_dir, args.file_name+".csv"), mode='a', header=False, index=False)  # mode设为a,就可以向csv文件追加数据了

    # print("---------------test------------------")
    # test_loss, test_correct, test_total, test_accuracy = test_model(allbest_P)
    # print('test_loss:', test_loss.item())
    # print('test_correct: ', test_correct)
    # print('test_total: ', test_total)
    # print('test_accuracy: %.8f %%' % test_accuracy)
end_time = time.time() - stat_time
print(end_time)
print('best accuracy: ', best_acc)
# state = {'model': net.state_dict(), 'optimizer': optimizer.state_dict()}
# torch.save(state, model_dir)
# saveLoss = pd.DataFrame(data=loss_value)
# saveLoss.to_csv('./lossforMaxNum20.csv')
# print("损失函数最小值为： ", allbest_V)
# print("对应权重值为： ", allbest_P)
