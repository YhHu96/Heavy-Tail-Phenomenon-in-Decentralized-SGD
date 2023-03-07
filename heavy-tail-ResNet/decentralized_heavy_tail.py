# Modifications copyright (C) 2020 Bluefog Team. All Rights Reserved.
# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import print_function

from bluefog.common import topology_util
import bluefog.torch as bf
import argparse
import os
import sys
import numpy as np
import math
import warnings
import resnet
import h5py
# import decentlam_opt
# import decentralized_optimizer
warnings.simplefilter('ignore')
import collections
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import datasets, transforms, models
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from noniid_sampler import NonIIDDistSampler
from tail_index_estimator import compute_alphas
import scipy.io as sio
import networkx as nx

import tensorboardX
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))

cwd_folder_loc = os.path.dirname(os.path.abspath(__file__))
# Training settings
parser = argparse.ArgumentParser(
    description="PyTorch CIFAR10 Example",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--log-dir', default='./logs',
                    help='tensorboard log directory')
# parser.add_argument('--checkpoint-format', default='./checkpoints/checkpoint-{counter}-optimizer-{optimizer}-topology-{topology}-time-{time}.pth.tar',
#                     help='checkpoint file format')
parser.add_argument('--checkpoint-format', default='./checkpoints2/model{counter}.pth',
                    help='checkpoint file format')
parser.add_argument('--batches-per-allreduce', type=int, default=1,
                    help='number of batches processed locally before '
                         'executing allreduce across workers; it multiplies '
                         'total batch size.')
parser.add_argument('--model', type=str, default='resnet20',
                    help='model to benchmark')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='The dataset to train with.')
parser.add_argument('--train-dir', default=os.path.expanduser('~/imagenet/train'),
                    help='path to training data')
parser.add_argument('--val-dir', default=os.path.expanduser('~/imagenet/validation'),
                    help='path to validation data')

# Default settings from https://arxiv.org/abs/1706.02677.
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size for training')
parser.add_argument('--val-batch-size', type=int, default=32,
                    help='input batch size for validation')
parser.add_argument('--epochs', type=int, default=90,
                    help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=0.0125,
                    help='learning rate for a single GPU')
parser.add_argument('--warmup-epochs', type=float, default=5,
                    help='number of warmup epochs')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--wd', type=float, default=0.00005,
                    help='weight decay')
parser.add_argument('--delta', type=float, default=0.1,
                    help='small network delta')

parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="disables CUDA training"
)
parser.add_argument(
    "--no-adjust-lr", action="store_true", default=False, help="disables adjust learning rate"
)
parser.add_argument("--seed", type=int, default=42, help="random seed")
parser.add_argument('--dist-optimizer', type=str, default='neighbor_allreduce',
                    help='The type of distributed optimizer. Supporting options are [win_put, ' +
                    'neighbor_allreduce, hierarchical_neighbor_allreduce, allreduce, ' +
                    'gradient_allreduce, horovod]')
parser.add_argument('--topology', type=str, default='ring',
                    help='The topology used in decentralized algorithms to connect all nodes. Supporting options are [ring, ' +
                    'hypercube]')
parser.add_argument('--atc-style', action='store_true', default=False,
                    help='If True, the step of optimizer happened before communication')
parser.add_argument('--dirichlet-beta', type=float, default=-1, metavar='DB',
                    help='Dirichlet distribution beta for non-iid local data generation')
parser.add_argument('--nu', type=float, default=0.01, metavar='NU',
                    help='nu value for decentlam')
parser.add_argument('--h5file', type=str, default='cifar10', metavar='DB',
                    help='File name for std histogram')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
allreduce_batch_size = args.batch_size * args.batches_per_allreduce

if args.dist_optimizer == 'horovod':
    print("importing horovod")
    import horovod.torch as bf

bf.init()

if args.topology == 'ring':
    bf.set_topology(bf.RingGraph(bf.size()))
elif args.topology == 'hypercube':
    bf.set_topology(bf.ExponentialTwoGraph(bf.size()))
elif args.topology == 'grid':
    bf.set_topology(bf.MeshGrid2DGraph(bf.size()))
elif args.topology == 'star':
    bf.set_topology(bf.StarGraph(bf.size()))
elif args.topology == 'fully':
    bf.set_topology(bf.FullyConnectedGraph(bf.size()))
elif args.topology == 'delta':
    assert bf.size() ==8, "Network size has to be 8"
    L = np.zeros((8,8))
    L[0,[0,1,7]] = [2,-1,-1]
    L[1,[0,1,2]] = [-1,2,-1]
    L[2,[1,2,3]] = [-1,2,-1]
    L[3,[2,3,4]] = [-1,2,-1]
    L[4,[3,4,5]] = [-1,2,-1]
    L[5,[4,5,6]] = [-1,2,-1]
    L[6,[5,6,7]] = [-1,2,-1]
    L[7,[0,6,7]] = [-1,-1,2]
    print(bf.rank(), L)
    delta = args.delta
    W = np.eye(8) - delta*L
    print(bf.rank(), W)
    G = nx.from_numpy_array(W, create_using=nx.DiGraph)
    bf.set_topology(G)
    topology = bf.load_topology()
    self_weight, neighbor_weights = topology_util.GetRecvWeights(topology, bf.rank())
    print(bf.rank(), self_weight, neighbor_weights)

torch.manual_seed(args.seed)

if args.cuda:
    # Bluefog: pin GPU to local rank.
    device_id = bf.local_rank() % torch.cuda.device_count()
    torch.cuda.set_device(device_id)
    torch.cuda.manual_seed(args.seed)

cudnn.benchmark = True

# Bluefog: print logs on the first worker.
verbose = 1 if bf.rank() == 0 else 0

# Bluefog: write TensorBoard logs on first worker.
log_writer = tensorboardX.SummaryWriter(
    args.log_dir) if bf.rank() == 0 else None


kwargs = {"num_workers": 4, "pin_memory": True} if args.cuda else {}
if args.dataset == "cifar10":
    train_dataset = datasets.CIFAR10(
        os.path.join(cwd_folder_loc, "..", "data", "data-%d" % bf.rank()),
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Pad(4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ]
        ),
    )
elif args.dataset == "imagenet":
    train_dataset = datasets.ImageFolder(args.train_dir,
                                         transform=transforms.Compose([
                                             transforms.RandomResizedCrop(224),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])
                                         ]))
else:
    raise ValueError("Args dataset should be either cifar10 or imagenet")

# Bluefog: use DistributedSampler to partition data among workers. Manually specify
# `num_replicas=bf.size()` and `rank=bf.rank()`.
# Bluefog: use DistributedSampler to partition the training data.
if args.dirichlet_beta < 0:
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=bf.size(), rank=bf.rank()
    )
else:
    train_sampler = NonIIDDistSampler(
        train_dataset, num_replicas=bf.size(), rank=bf.rank(), beta=args.dirichlet_beta
    )
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=allreduce_batch_size, sampler=train_sampler, **kwargs
)

if args.dataset == "cifar10":
    val_dataset = datasets.CIFAR10(
        os.path.join(cwd_folder_loc, "..", "data", "data-%d" % bf.rank()),
        train=False,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ]
        ),
    )
elif args.dataset == "imagenet":
    val_dataset = datasets.ImageFolder(args.val_dir,
                                       transform=transforms.Compose([
                                           transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225])
                                       ]))

val_sampler = torch.utils.data.distributed.DistributedSampler(
    val_dataset, num_replicas=bf.size(), rank=bf.rank()
)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=args.val_batch_size, sampler=val_sampler, **kwargs
)

if args.dataset == "cifar10":
    model = resnet.__dict__[args.model]()
elif args.dataset == "imagenet":
    model = getattr(models, args.model)(num_classes=1000)

if args.cuda:
    # Move model to GPU.
    model.cuda()

# Bluefog: scale learning rate by the number of GPUs.
# Gradient Accumulation: scale learning rate by batches_per_allreduce
optimizer = optim.SGD(
    model.parameters(),
    lr=(args.base_lr * args.batches_per_allreduce * bf.size()),
    momentum=args.momentum,
    weight_decay=args.wd,
)

# Bluefog: wrap optimizer with DistributedOptimizer.
if args.dist_optimizer != 'horovod':
    base_dist_optimizer = (
        bf.DistributedAdaptThenCombineOptimizer if args.atc_style else
        bf.DistributedAdaptWithCombineOptimizer)
if args.dist_optimizer == 'allreduce':
    optimizer = base_dist_optimizer(
        optimizer, model=model, communication_type=bf.CommunicationType.allreduce)
elif args.dist_optimizer == 'neighbor_allreduce':
    # optimizer = decentralized_optimizer.DSGDOptimizer(
    #     model.parameters(), model, nu=args.nu,
    #     lr=(args.base_lr * args.batches_per_allreduce * bf.size()),
    #     momentum=args.momentum,
    #     weight_decay = args.wd,
    # )
    optimizer = base_dist_optimizer(
        optimizer, model=model, communication_type=bf.CommunicationType.neighbor_allreduce,
        num_steps_per_communication = args.batches_per_allreduce)
# elif args.dist_optimizer == 'exact_diffusion':
#     optimizer = decentralized_optimizer.ExactDiffusionOptimizer(
#         model.parameters(), model, nu=args.nu,
#         lr=(args.base_lr * args.batches_per_allreduce * bf.size()),
#         momentum=args.momentum,
#         weight_decay = args.wd,
#     )
elif args.dist_optimizer == 'empty':
    optimizer = base_dist_optimizer(
        optimizer, model=model,
        communication_type=bf.CommunicationType.empty)
elif args.dist_optimizer == 'gradient_allreduce':
    optimizer = bf.DistributedGradientAllreduceOptimizer(
        optimizer, model=model)
elif args.dist_optimizer == 'horovod':
    optimizer = bf.DistributedOptimizer(
        optimizer, named_parameters=model.named_parameters())
else:
    raise ValueError('Unknown args.dist-optimizer type -- ' + args.dist_optimizer + '\n' +
                     'Please set the argument to be one of ' +
                     '[neighbor_allreduce, gradient_allreduce, allreduce, ' +
                     'hierarchical_neighbor_allreduce, win_put, horovod]')

# Bluefog: broadcast parameters & optimizer state.
bf.broadcast_parameters(model.state_dict(), root_rank=0)
bf.broadcast_optimizer_state(optimizer, root_rank=0)

save_epochs = int(np.ceil(1000/len(train_loader)))
save_models = len(train_loader) * save_epochs

def train(epoch, counter):
    model.train()
    train_sampler.set_epoch(epoch)
    train_loss = Metric("train_loss")
    train_accuracy = Metric("train_accuracy")

    with tqdm(total=len(train_loader), desc="Train Epoch     #{}".format(epoch + 1),
              disable=not verbose,) as t:
        for batch_idx, (data, target) in enumerate(train_loader):
            adjust_learning_rate(epoch, batch_idx, args.no_adjust_lr)

            if args.cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            # Split data into sub-batches of size batch_size
            for i in range(0, len(data), args.batch_size):
                data_batch = data[i: i + args.batch_size]
                target_batch = target[i: i + args.batch_size]
                output = model(data_batch)
                train_accuracy.update(accuracy(output, target_batch))
                loss = F.cross_entropy(output, target_batch)
                train_loss.update(loss)
                # Average gradients among sub-batches
                loss.div_(math.ceil(float(len(data)) / args.batch_size))
                loss.backward()
            # Gradient is applied across all ranks
            optimizer.step()
            t.set_postfix(
                {
                    "loss": train_loss.avg.item(),
                    "accuracy": 100.0 * train_accuracy.avg.item(),
                }
            )
            t.update(1)

            if epoch >= args.epochs - save_epochs:
                save_checkpoint(counter[0])
                counter[0] += 1

    if log_writer:
        log_writer.add_scalar("train/loss", train_loss.avg, epoch)
        log_writer.add_scalar("train/accuracy", train_accuracy.avg, epoch)
    return train_loss.avg, train_accuracy.avg

def validate(epoch, record):
    model.eval()
    val_loss = Metric("val_loss")
    val_accuracy = Metric("val_accuracy")

    with tqdm(total=len(val_loader), desc="Validate Epoch  #{}".format(epoch + 1),
              disable=not verbose) as t:
        with torch.no_grad():
            for data, target in val_loader:
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)

                val_loss.update(F.cross_entropy(output, target))
                val_accuracy.update(accuracy(output, target))
                t.set_postfix(
                    {
                        "loss": val_loss.avg.item(),
                        "accuracy": 100.0 * val_accuracy.avg.item(),
                    }
                )
                t.update(1)

    if log_writer:
        log_writer.add_scalar("val/loss", val_loss.avg, epoch)
        log_writer.add_scalar("val/accuracy", val_accuracy.avg, epoch)
    record.append((val_loss.avg, val_accuracy.avg*100))
    return val_loss.avg, val_accuracy.avg


# Bluefog: using `lr = base_lr * bf.size()` from the very beginning leads to worse final
# accuracy. Scale the learning rate `lr = base_lr` ---> `lr = base_lr * bf.size()` during
# the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
# After the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
def adjust_learning_rate(epoch, batch_idx, no_adjust_lr = False):
    if not no_adjust_lr:
        # adjust learning rate
        # print("adjust learing rate!")
        if epoch < args.warmup_epochs:
            epoch += float(batch_idx + 1) / len(train_loader)
            lr_adj = 1.0 / bf.size() * (epoch * (bf.size() - 1) / args.warmup_epochs + 1)
        elif epoch < 100:
            lr_adj = 1.0
        elif epoch < 150:
            lr_adj = 1e-1
        else:
            lr_adj = 1e-2
    else:
        # do not adjust learning rate
        if epoch < args.warmup_epochs:
            epoch += float(batch_idx + 1) / len(train_loader)
            lr_adj = 1.0 / bf.size() * (epoch * (bf.size() - 1) / args.warmup_epochs + 1)
        else:
            lr_adj = 1.0
        if epoch > 100 and bf.rank() == 0:
            print(epoch, lr_adj)

    for param_group in optimizer.param_groups:
        # if bf.rank() == 0:
        #     print(args.base_lr)
        #     print(bf.size())
        #     print(args.batches_per_allreduce)
        #     print(lr_adj)
        #     print(args.base_lr * bf.size() * args.batches_per_allreduce * lr_adj)
        param_group["lr"] = (
            args.base_lr * bf.size() * args.batches_per_allreduce * lr_adj
        )

def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()

def save_checkpoint(counter):
    if bf.rank() == 0:
        # if args.dist_optimizer == 'neighbor_allreduce':
        #     filepath = args.checkpoint_format.format(counter=counter, optimizer=args.dist_optimizer, topology=args.topology, time = tt)
        # else:
        #     filepath = args.checkpoint_format.format(counter=counter, optimizer=args.dist_optimizer, topology="None", time = tt)
        filepath = args.checkpoint_format.format(counter=counter)
        dirpath = os.path.dirname(filepath)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
        torch.save(state, filepath)

# Bluefog: average metrics from distributed training.
class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.0)  # pylint: disable=not-callable
        self.n = torch.tensor(0.0)  # pylint: disable=not-callable

    def update(self, val):
        self.sum += bf.allreduce(val.detach().cpu(), name=self.name)
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n

def store_status(data_dict, h5file, suffix):
    with h5py.File(f"{h5file}-{suffix}.hdf5", "w") as hf:
        for label, data in data_dict.items():
            hf.create_dataset(label, data=data)

if __name__ == "__main__":
    
    timearray=time.localtime(float(time.time()))
    tt=time.strftime('%Y-%m-%d-%H-%M-%S',timearray)
    counter = [0]

    data_dict = collections.defaultdict(list)
    test_record = []
    for epoch in range(args.epochs):
        train_loss, train_acc = train(epoch, counter)
        val_loss, val_acc = validate(epoch, test_record)

        data_dict['train_loss'].append(train_loss)
        data_dict['train_accuracy'].append(train_acc)
        data_dict['val_loss'].append(val_loss)
        data_dict['val_accuracy'].append(val_acc)
    
    bf.barrier()
    store_status(data_dict, args.h5file, bf.rank())
    if bf.rank() == 0:

        # print basic experimental information
        print()
        print("Seed: ", args.seed)
        print("Nodes: ", bf.size())
        print("Base lr:", args.base_lr)
        print("Optimizer: ", args.dist_optimizer)
        if args.dist_optimizer == "gradient_allreduce":
            print("Topology: None")
        else:
            print("Topology: ", args.topology)
        if args.atc_style:
            print("ATC_style: True")
        else:
            print("ATC_style: False")
        print("Dirichlet Beta: ", args.dirichlet_beta)
        print("Nu: ", args.nu)
        print("Batchsize: ", args.batch_size)
        print("Momentum: ", args.momentum)
        print("Weight Deacy ", args.wd)

        print()
        if args.dirichlet_beta >= 0:
            print('Sample distribution:')
            for i in range(bf.size()):
                print(train_sampler.idx_list[i])
            print('Sample number of each node:')
            print(np.sum(np.array(train_sampler.idx_list), axis=1))
        for epoch, (loss, acc) in  enumerate(test_record):
            train_loss, train_acc = data_dict['train_loss'][epoch], data_dict['train_accuracy'][epoch]
            print(f'[Epoch {epoch+1:2d}] Train Loss: {train_loss}, Train Acc: {train_acc}, Test Loss: {loss}, Test Acc: {acc}%')

# compute heavy-tail index
if bf.rank() == 0:

    model_name = "./checkpoints2/model0.pth"
    tmp_net = torch.load(model_name,map_location='cpu')
    tmp_net = tmp_net['model']

    depth = 0
    for k, v in tmp_net.items():
        if 'bn' not in k and 'bias' not in k:
            depth += 1
            print(k)
            layer = v.detach().numpy()
            print(layer.shape)
            n = layer.shape[0]
            layer = layer.reshape(n,-1)
            print(layer.shape)
    print(depth)

    PATH = './checkpoints2'

    lr_list = [0.075]

    num_nets = save_models
    nets = []
    alphas_mc_nodes = []
    alphas_multi_nodes = []
    alphas_haus_nodes = []

    for i in range(1):
        alphas_mc, alphas_multi,  alphas_haus  = compute_alphas(lr_list, PATH, depth, 'node{}'.format(i), num_nets)
        alphas_mc_nodes.append(alphas_mc)
        alphas_multi_nodes.append(alphas_multi)
        alphas_haus_nodes.append(alphas_haus)

    opt_name = "DSGD" if args.dist_optimizer == "neighbor_allreduce" else "CSGD"
    # store the evaluated tail index
    if args.topology != 'delta':
        mat_name = "Cifar10_" + opt_name + "topo_{}_{}n{}b_lr{}_20220502.mat".format(args.topology, bf.size(), args.batch_size, args.base_lr)
    else:
        mat_name = "Cifar10_" + opt_name + "topo_{}_delta_{}_{}n{}b_lr{}_20220502.mat".format(args.topology, args.delta, bf.size(), args.batch_size, args.base_lr)

    sio.savemat(mat_name, 
        {'etas':args.base_lr, 
        'batch_sizes':args.batch_size,
        'alphas_mc':alphas_mc,
        'alphas_multi':alphas_multi,
        'alphas_haus':alphas_haus})

    print(alphas_mc)

    os.system("rm -r checkpoints2")