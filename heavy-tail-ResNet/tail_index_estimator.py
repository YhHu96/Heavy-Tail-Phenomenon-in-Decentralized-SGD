import os
import numpy as np
import torch
import itertools
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from tqdm import tqdm
import time
# import scipy.io as sio

# Corollary 2.4 in Mohammadi 2014 - for 1d
def alpha_estimator_one(m, X):
    N = len(X)
    n = int(N/m) # must be an integer
    
    X = X[0:n*m]
    
    Y = np.sum(X.reshape(n, m),1)
    eps = np.spacing(1)

    Y_log_norm =  np.log(np.abs(Y) + eps).mean()
    X_log_norm =  np.log(np.abs(X) + eps).mean()
    diff = (Y_log_norm - X_log_norm) / math.log(m)
    return 1 / diff

# Corollary 2.4 in Mohammadi 2014 - for multi-d
def alpha_estimator_multi(m, X):
    # X is N by d matrix
    N = X.size()[0]   
    n = int(N/m) # must be an integer
#     print(N,n)
    X = X[0:n*m,:]
#     print(X.size())
    Y = torch.sum(X.view(n, m, -1), 1)
    eps = np.spacing(1)
    Y_log_norm = torch.log(Y.norm(dim=1) + eps).mean()
    X_log_norm = torch.log(X.norm(dim=1) + eps).mean()
    diff = (Y_log_norm - X_log_norm) / math.log(m)
    return 1 / diff.item()

def compute_alphas(etas, PATH, depth, node_path, num_nets):
    alphas_mc    = np.zeros((len(etas), depth))-1
    alphas_multi = np.zeros((len(etas), depth))-1
    alphas_single= np.zeros(len(etas))-1
    alphas_haus    = np.zeros((len(etas), depth))-1
    
    
    
    for ei, eta in tqdm(enumerate(etas)):
        
        tmp_path = PATH + '/'
        print(tmp_path)
        
        weights = []
        weights_unfold = []
        weights_unfold_merge = []
        for i in range(depth):
            weights.append([])
            weights_unfold.append([])

        # record the layers in different arrays
        for i in range(num_nets):
            tmp_path_mod = tmp_path + 'model{}'.format(i) +'.pth'
            tmp_net = torch.load(tmp_path_mod,map_location='cpu')
            tmp_net = tmp_net['model']
            
            ix = 0
            for k, p in tmp_net.items():
                if 'bn' not in k and 'bias' not in k:
                    layer = p.detach().numpy()
                    n = layer.shape[0]
                    layer = layer.reshape(n,-1)
                    
                    if(i == 0):
                        weights_unfold[ix] = layer / (num_nets * 1.0)
                    else:
                        weights_unfold[ix] += layer / (num_nets * 1.0)

                    layer = layer.reshape(-1,1)
                    weights[ix].append(layer)
                    ix += 1

        for i in range(depth):
            weights[i] = np.concatenate(weights[i], axis = 1)

        for i in range(depth):
            tmp_mean    = np.mean(weights_unfold[i], axis=0)
            tmp_mean    = tmp_mean[..., np.newaxis]
            tmp_weights = weights_unfold[i] - tmp_mean.T
            
            print(tmp_weights.shape)
            alphas_multi[ei,i] = np.median([alpha_estimator_multi(mm, torch.from_numpy(tmp_weights)) for mm in (2, 5, 10)])


        for i in range(depth):
            tmp_mean    = np.mean(weights[i], axis=1)
            tmp_mean    = tmp_mean[..., np.newaxis]
            tmp_weights = weights[i] - tmp_mean
            tmp_weights = tmp_weights.reshape(-1,1)     
            tmp_alphas = [alpha_estimator_one(mm, tmp_weights) for mm in (2, 5, 10, 20, 50, 100, 500, 1000)]
            alphas_haus[ei,i] = np.median(tmp_alphas)


        for i in range(depth):
            tmp_weights = np.mean(weights[i], axis=1)
            tmp_weights = tmp_weights.reshape(-1,1)
            tmp_weights = tmp_weights - np.mean(tmp_weights)
            tmp_alphas = [alpha_estimator_one(mm, tmp_weights) for mm in (2, 5, 10, 20, 50, 100, 500, 1000)]
            alphas_mc[ei,i] = np.median(tmp_alphas)

    return alphas_mc, alphas_multi, alphas_haus