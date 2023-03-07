# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 20:20:48 2021

@author: sturm
"""
from scipy.stats import levy_stable
import numpy as np
import torch
import math
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch import optim
import os

PATH_base = './3FCN-MNIST-centralized-new'
#PATH_base = './CNN-CIFAR10-centralized-umut-5'
try:
    os.mkdir(PATH_base)
except OSError as exc:
    pass

# A simple FCN
class simpleNet(nn.Module):

    def __init__(self, input_dim=28*28 , width=128, depth=3, num_classes=10):
        super(simpleNet, self).__init__()
        self.input_dim = input_dim 
        self.width = width
        self.depth = depth
        self.num_classes = num_classes
        
        layers = self.get_layers()

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, self.width, bias=False),
            nn.ReLU(inplace=True),
            *layers,
            nn.Linear(self.width, self.num_classes, bias=False),
        )

    def get_layers(self):
        layers = []
        for i in range(self.depth - 2):
            layers.append(nn.Linear(self.width, self.width, bias=False))
            layers.append(nn.ReLU())
        return layers

    def forward(self, x):
        x = x.view(x.size(0), self.input_dim)
        x = self.fc(x)
        return x
    
    '''
class simpleNet(nn.Module):
    def __init__(self):
        super(simpleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    '''
batch_size=100
data_tf = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])
'''
train_dataset = datasets.CIFAR10(root='./data', train=True, transform=data_tf, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=data_tf)
'''
train_dataset = datasets.MNIST(root='./data', train=True, transform=data_tf, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=data_tf)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

lr_list = [0.0001,
0.001,
0.01,
0.015,
0.02,
0.025,
0.03,
0.04,
0.045,
0.05,
0.06,
0.075,
0.08,
0.09,
0.1]
#lr_list = [0.1]
epoch=200

for i in range(len(lr_list)):
    learning_rate = lr_list[i]
    trainErrorList=[]
    trainAccList=[]
    
    PATH = PATH_base + '/LR' + '{}'.format(i)
    try:
        os.mkdir(PATH)
    except OSError as exc:
        pass
    
                
    model = simpleNet()
    if torch.cuda.is_available():
            model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    iter_count = 0
    iter_name = 1
    for l in range(epoch):
        train_acc=0
        for data in train_loader:
            img, label = data
            # img=img.view(img.size(0),-1)
            if torch.cuda.is_available():
                img = img.cuda()
                label = label.cuda()
            else:
                img = Variable(img)
                label = Variable(label)
            out = model(img)
            loss = criterion(out, label)
            print_loss = loss.data.item()
            _, pred = torch.max(out.data, 1)
            train_acc += pred.eq(label.view_as(pred)).sum().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter_count += 1
            if iter_count > 80000:
                break
            if iter_count > 79000:
                tmp_path = PATH + '/model' + '{}'.format(iter_name) +'.pth'
                torch.save(model, tmp_path)
                iter_name += 1
        if iter_count > 80000:
            break
        trainErrorList.append(loss.data.item())
        trainAccList.append(train_acc/60000)
    print(learning_rate)