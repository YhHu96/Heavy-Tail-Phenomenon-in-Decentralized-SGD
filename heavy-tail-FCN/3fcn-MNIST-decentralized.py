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

PATH_base = './3FCN_MNIST_decentralized_delta005'
try:
    os.mkdir(PATH_base)
except OSError as exc:
    pass

# A simple FCN
'''
class simpleNet(nn.Module):
    def __init__(self,in_dim,n_hidden_2,out_dim):
        super(simpleNet,self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_2)
        self.layer2 = nn.Linear(n_hidden_2, n_hidden_2)
        self.layer4 = nn.Linear(n_hidden_2, out_dim)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer4(x))
        return x

'''
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
# lr_list = np.linspace(0.01,0.2,20)
#lr_list = [0.1]
nodes = 20

def generate_cir(N):
    x=np.zeros([N,N])
    for i in range(1,N-1):
        x[i][i]=1/3
        x[i][i+1]=1/3
        x[i][i-1]=1/3
    x[N-1][N-2]=1/3
    x[N-1][N-1]=1/3
    x[N-1][0]=1/3
    x[0][N-1]=1/3
    x[0][0]=1/3
    x[0][1]=1/3
    return x


def generate_star_new(N, delta):
    x = np.identity(N)
    A = np.zeros([N,N])
    for i in range(N-1):
        j = i + 1
        A[0,j] = 1
        A[j,0] = 1
    degree = [N-1] + [1] *(N-1)
    D = np.diag(degree)
    w = D - A
    x = x - delta * w
    return x


batch_size=5
data_tf = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])

# New added: fix the random seed for dataset split
torch.manual_seed(0)
'''
train_dataset = datasets.CIFAR10(root='./data', train=True, transform=data_tf, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=data_tf)
'''
train_dataset = datasets.MNIST(root='./data', train=True, transform=data_tf, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=data_tf)

size_node = int(1/nodes * len(train_dataset))
nodes_dataloader = []
temp_rest = train_dataset
for n in range(nodes):
    tmp_node, temp_rest = torch.utils.data.random_split(temp_rest, [size_node, len(temp_rest) - size_node])
    # New added: changed the shuffle to False
    # train_loader = DataLoader(tmp_node, batch_size=batch_size, shuffle=False)
    train_loader = DataLoader(tmp_node, batch_size=batch_size, shuffle=True)
    nodes_dataloader.append(train_loader)

def cal_weighted(models, w):
    tmp_models = []
    for i in range(len(models)):
        tmp_mod = simpleNet()
        for param in tmp_mod.parameters():
            param.data *= 0
        for j in range(len(models)):
            for (param1, param2) in zip(tmp_mod.parameters(), models[j].parameters()):
                param1.data += w[i][j] * param2.data
        tmp_models.append(tmp_mod)
    return tmp_models

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
delta_list = [0.05]
iterations = 80000

train_error = []

for j in range(len(delta_list)):
    delta = delta_list[j]
    PATH = PATH_base + '/delta' + '{}'.format(j)
    try:
        os.mkdir(PATH)
    except OSError as exc:
        pass
    w = generate_star_new(nodes, delta)
    for i in range(len(lr_list)):
        learning_rate = lr_list[i]
        
        PATH = PATH + '/LR' + '{}'.format(i)
        try:
            os.mkdir(PATH)
        except OSError as exc:
            pass
        criterion = nn.CrossEntropyLoss()
        
        tmp_train_error_lr = []
        
        ############# initialization############
    #     store the models for every nodes
        models = []
    #     store the optimizers for ever nodes
        optimizers = []
        for n in range(nodes):
            tmp_model = simpleNet()
    #         if torch.cuda.is_available():
    #                 tmp_model = tmp_model.cuda()
            tmp_optimizer = optim.SGD(tmp_model.parameters(), lr=learning_rate)
            models.append(tmp_model)
            optimizers.append(tmp_optimizer)
        
        #############update################
        
        for l in range(iterations):
            
            tmp_train_error_ite = []
            
            if l % 1000 == 0:
                print(l)
    #         calculate the weighted average of parameters
            mod_mean = cal_weighted(models, w)
            for n in range(nodes):
                # New added: changed the way to get data in dataloader, 
                # now we the order will be the same as the original method
                for k, data in enumerate(nodes_dataloader[n]):
                    '''
                    if k < l:
                        continue
                    '''
                    img, label = data
                    #img=img.view(img.size(0),-1)
                    
                    img = Variable(img)
                    label = Variable(label)
                    
    #                 if torch.cuda.is_available():
    #                     img = img.cuda()
    #                     label = label.cuda()
    #                 else:
    #                     img = Variable(img)
    #                     label = Variable(label)
    
                    out = models[n](img)
                    loss = criterion(out, label)
                    print_loss = loss.data.item()
                    _, pred = torch.max(out.data, 1)
    #                 train_acc += pred.eq(label.view_as(pred)).sum().item()
                    optimizers[n].zero_grad()
                    loss.backward()
                    
    #                 update parameters by weighted average
                    
                    for (param, tmp_param) in zip(models[n].parameters(), mod_mean[n].parameters()):
                        param.data *= 0
                        param.data += tmp_param.data
                        
                    optimizers[n].step()
    #                 every time we only iterate once, since we need to calculate the weighted average after one iteration
                    break
                if l >= 79000:
                    if n == 0:
                        
                        node_path = PATH + '/node' + '{}'.format(n)
                        try:
                            os.mkdir(node_path)
                        except OSError as exc:
                            pass
                        tmp_path = node_path + '/model' + '{}'.format(l-79000) + '.pth'
                        torch.save(models[n], tmp_path)
                tmp_train_error_ite.append(loss.data.item())
            
            tmp_train_error_lr.append(np.transpose(tmp_train_error_ite))
        train_error.append(tmp_train_error_lr)
        print(learning_rate)