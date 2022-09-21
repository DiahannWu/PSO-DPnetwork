import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision

import os
import argparse
import csv
import random
import time
import numpy as np

from models import resnet20
from utils import get_data_loader, get_sigma, restore_param, checkpoint, adjust_learning_rate, process_grad_batch

#package for computing individual gradients
from backpack import backpack, extend
from backpack.extensions import BatchGrad
from model import MLP, CNN, SimpleNet

if __name__ == '__main__':

    chanels = 1
    size = 28
    n_weight1 = 300
    n_weight2 = 100
    n_weight3 = 10
    all_weight = chanels * size * size * n_weight1 + n_weight1 + n_weight1 * n_weight2 + n_weight2 + n_weight2 * n_weight3 + n_weight3
    n1 = chanels * size * size * n_weight1
    n2 = n1 + n_weight1
    n3 = n2 + n_weight1 * n_weight2
    n4 = n3 + n_weight2
    n5 = n4 + n_weight2 * n_weight3
    n6 = n5 + n_weight3

    parser = argparse.ArgumentParser(description='Differentially Private learning with DP-SGD')

    ## general arguments
    parser.add_argument('--dataset', default='mnist', type=str, help='dataset name')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--sess', default='resnet20_cifar10', type=str, help='session name')
    parser.add_argument('--seed', default=2, type=int, help='random seed')
    parser.add_argument('--weight_decay', default=0., type=float, help='weight decay')
    parser.add_argument('--batchsize', default=100, type=int, help='batch size')
    parser.add_argument('--n_epoch', default=100, type=int, help='total number of epochs')
    parser.add_argument('--lr', default=0.01, type=float, help='base learning rate (default=0.1)')
    parser.add_argument('--momentum', default=0.9, type=float, help='value of momentum')


    ## arguments for learning with differential privacy
    parser.add_argument('--private', default=False, action='store_true', help='enable differential privacy')
    parser.add_argument('--clip', default=5., type=float, help='gradient clipping bound')
    parser.add_argument('--eps', default=8., type=float, help='privacy parameter epsilon')
    parser.add_argument('--delta', default=1e-5, type=float, help='desired delta')



    args = parser.parse_args()

    assert args.dataset in ['mnist', 'cifar10', 'svhn']

    use_cuda = True
    best_acc = 0
    start_epoch = 0
    batch_size = args.batchsize

    if(args.seed != -1):
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    print('==> Preparing data..')
    ## preparing data for training && testing
    if(args.dataset == 'mnist'):
        trainloader, testloader, n_training, n_test = get_data_loader('mnist', batchsize=args.batchsize)
        train_samples, train_labels = None, None
    elif(args.dataset == 'svhn'):  ## For SVHN, we concatenate training samples and extra samples to build the training set.
        trainloader, extraloader, testloader, n_training, n_test = get_data_loader('svhn', batchsize = args.batchsize)
        for train_samples, train_labels in trainloader:
            break
        for extra_samples, extra_labels in extraloader:
            break
        train_samples = torch.cat([train_samples, extra_samples], dim=0)
        train_labels = torch.cat([train_labels, extra_labels], dim=0)

    else:
        trainloader, testloader, n_training, n_test = get_data_loader('cifar10', batchsize = args.batchsize)
        train_samples, train_labels = None, None

    print('# of training examples: ', n_training, '# of testing examples: ', n_test)


    print('\n==> Computing noise scale for privacy budget (%.1f, %f)-DP'%(args.eps, args.delta))
    sampling_prob=args.batchsize/n_training
    steps = int(args.n_epoch/sampling_prob)
    sigma, eps = get_sigma(sampling_prob, steps, args.eps, args.delta, rgp=False)
    noise_multiplier = sigma
    print('noise scale: ', noise_multiplier, 'privacy guarantee: ', eps)

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

    # num_params = 0
    # np_list = []
    # for p in net.parameters():
    #     num_params += p.numel()
    #     np_list.append(p.numel())

    # optimizer = optim.Adam(
    #     net.parameters(),
    #     lr=args.lr)

    optimizer = optim.SGD(net.parameters(), lr=args.lr)

    # optimizer = optim.SGD(
    #         net.parameters(),
    #         lr=args.lr,
    #         momentum=args.momentum,
    #         weight_decay=args.weight_decay)

    def train(epoch, theWight):
        print('\nEpoch: %d' % epoch)
        net.train()

        net.layer1.weight.data = torch.from_numpy(theWight[0:n1].reshape(n_weight1, chanels * size * size)).to(torch.float32).cuda()
        net.layer1.bias.data = torch.from_numpy(theWight[n1:n2].reshape(n_weight1)).to(torch.float32).cuda()
        net.layer2.weight.data = torch.from_numpy(theWight[n2:n3].reshape(n_weight2, n_weight1)).to(torch.float32).cuda()
        net.layer2.bias.data = torch.from_numpy(theWight[n3:n4].reshape(n_weight2)).to(torch.float32).cuda()
        net.layer3.weight.data = torch.from_numpy(theWight[n4:n5].reshape(n_weight3, n_weight2)).to(torch.float32).cuda()
        net.layer3.bias.data = torch.from_numpy(theWight[n5:n6].reshape(n_weight3)).to(torch.float32).cuda()

        train_loss = 0
        correct = 0
        total = 0
        t0 = time.time()
        steps = n_training//args.batchsize

        if(train_samples == None): # using pytorch data loader for CIFAR10
            loader = iter(trainloader)
        else: # manually sample minibatchs for SVHN
            sample_idxes = np.arange(n_training)
            np.random.shuffle(sample_idxes)

        for batch_idx in range(steps):
            if(args.dataset=='svhn'):
                current_batch_idxes = sample_idxes[batch_idx*args.batchsize : (batch_idx+1)*args.batchsize]
                inputs, targets = train_samples[current_batch_idxes], train_labels[current_batch_idxes]
            else:
                inputs, targets = next(loader)
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            if(args.private):
                logging = batch_idx % 20 == 0
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = loss_func(outputs, targets)
                # with backpack(BatchGrad()):
                #     loss.backward()
                #     process_grad_batch(list(net.parameters()), args.clip) # clip gradients and sum clipped gradients
                    # ## add noise to gradient
                    # for p in net.parameters():
                    #     shape = p.grad.shape
                    #     numel = p.grad.numel()
                    #     grad_noise = torch.normal(0, noise_multiplier*args.clip/args.batchsize, size=p.grad.shape, device=p.grad.device)
                    #     p.grad.data += grad_noise
            else:
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = loss_func(outputs, targets)
                loss.backward()
                # try:
                #     for p in net.parameters():
                #         del p.grad_batch
                # except:
                #     pass
            optimizer.step()
            step_loss = loss.item()
            if(args.private):
                step_loss /= inputs.shape[0]
            train_loss += step_loss
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).float().cpu().sum()
        acc = 100.*float(correct)/float(total)
        theWight[0: n1] = net.layer1.weight.data.cpu().numpy().flatten()[:]
        theWight[n1: n2] = net.layer1.bias.data.cpu().numpy().flatten()[:]
        theWight[n2: n3] = net.layer2.weight.data.cpu().numpy().flatten()[:]
        theWight[n3: n4] = net.layer2.bias.data.cpu().numpy().flatten()[:]
        theWight[n4: n5] = net.layer3.weight.data.cpu().numpy().flatten()[:]
        theWight[n5: n6] = net.layer3.bias.data.cpu().numpy().flatten()[:]
        t1 = time.time()
        print('Train loss:%.5f'%(train_loss/(batch_idx+1)), 'time: %d s'%(t1-t0), 'train acc:', acc, end=' ')
        return train_loss/batch_idx, acc, theWight


    def train_nograd(epoch,theWight):
        net.train()

        net.layer1.weight.data = torch.from_numpy(theWight[0:n1].reshape(n_weight1, chanels * size * size)).to(torch.float32).cuda()
        net.layer1.bias.data = torch.from_numpy(theWight[n1:n2].reshape(n_weight1)).to(torch.float32).cuda()
        net.layer2.weight.data = torch.from_numpy(theWight[n2:n3].reshape(n_weight2, n_weight1)).to(torch.float32).cuda()
        net.layer2.bias.data = torch.from_numpy(theWight[n3:n4].reshape(n_weight2)).to(torch.float32).cuda()
        net.layer3.weight.data = torch.from_numpy(theWight[n4:n5].reshape(n_weight3, n_weight2)).to(torch.float32).cuda()
        net.layer3.bias.data = torch.from_numpy(theWight[n5:n6].reshape(n_weight3)).to(torch.float32).cuda()

        train_loss = 0
        correct = 0
        total = 0
        t0 = time.time()
        steps = n_training // args.batchsize

        if (train_samples == None):  # using pytorch data loader for CIFAR10
            loader = iter(trainloader)
        else:  # manually sample minibatchs for SVHN
            sample_idxes = np.arange(n_training)
            np.random.shuffle(sample_idxes)
        with torch.no_grad():
            for batch_idx in range(steps):
                if (args.dataset == 'svhn'):
                    current_batch_idxes = sample_idxes[batch_idx * args.batchsize: (batch_idx + 1) * args.batchsize]
                    inputs, targets = train_samples[current_batch_idxes], train_labels[current_batch_idxes]
                else:
                    inputs, targets = next(loader)
                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()

                if (args.private):
                    logging = batch_idx % 20 == 0
                    optimizer.zero_grad()
                    outputs = net(inputs)
                    loss = loss_func(outputs, targets)
                else:
                    # optimizer.zero_grad()
                    outputs = net(inputs)
                    loss = loss_func(outputs, targets)
                    # loss.backward()
                    # try:
                    #     for p in net.parameters():
                    #         del p.grad_batch
                    # except:
                    #     pass
                # optimizer.step()
                step_loss = loss.item()
                if (args.private):
                    step_loss /= inputs.shape[0]
                train_loss += step_loss
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).float().cpu().sum()
            acc = 100. * float(correct) / float(total)
        t1 = time.time()
        print('Train loss:%.5f' % (train_loss / (batch_idx + 1)), 'time: %d s' % (t1 - t0), 'train acc:', acc, end=' ')
        return train_loss / batch_idx, acc


    def test(epoch):
        global best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        all_correct = []
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                outputs = net(inputs)
                loss = loss_func(outputs, targets)
                step_loss = loss.item()
                if(args.private):
                    step_loss /= inputs.shape[0]

                test_loss += step_loss
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct_idx = predicted.eq(targets.data).cpu()
                all_correct += correct_idx.numpy().tolist()
                correct += correct_idx.sum()

            acc = 100.*float(correct)/float(total)
            print('test loss:%.5f'%(test_loss/(batch_idx+1)), 'test acc:', acc)
            ## Save checkpoint.
            if acc > best_acc:
                best_acc = acc
                checkpoint(net, acc, epoch, args.sess)

        return test_loss/batch_idx, acc


    # 粒子数量
    num = 3

    # 粒子位置矩阵维数 考虑三层全连接层权重的尺寸依次为[300, 784]\[100, 300]\[10, 100]
    # numx = 3
    # numy = 300
    # numz = 784

    # p为粒子位置矩阵，初始为标准正态分布
    p = np.random.randn(num, all_weight) / 100

    # v为粒子速度矩阵，初始为标准正态分别
    v = np.random.randn(num, all_weight) / 100

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
        perbest_V[i], acc, new_weight[i] = train(-1, p[i, :])

    # 确定群体历史最优值
    allbest_V = min(perbest_V)

    # 确定初始最优位置
    # allbest_P = p[np.argmin(perbest_V), :, :, :]
    allbest_P = new_weight[np.argmin(perbest_V)]

    train_loss, train_acc = train_nograd(-1, allbest_P)
    print('\n')
    test_loss, test_acc = test(-1)

    # 设置最大迭代次数
    max_iter = 300

    # 设置速度更新权重值
    w_forV = 0.1

    print('\n==> Strat training')

    for epoch in range(start_epoch, args.n_epoch):

        # 速度更新
        v = w_forV * v + 2.4 * random.random() * (allbest_P - p) + 1.7 * random.random() * (perbest_P - p)

        # 位置更新
        p = p + v

        # 计算每个粒子到达新位置后所得目标的函数值
        for a in range(num):
            train_loss, train_acc, new_weight[a] = train(epoch, p[a, :])

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

        train_loss, train_acc = train_nograd(epoch, allbest_P)
        print('\n')
        test_loss, test_acc = test(epoch)

        # # lr = adjust_learning_rate(optimizer, args.lr, epoch, all_epoch=args.n_epoch)
        # train_loss, train_acc = train(epoch)
        # print('\n')
        # test_loss, test_acc = test(epoch)
