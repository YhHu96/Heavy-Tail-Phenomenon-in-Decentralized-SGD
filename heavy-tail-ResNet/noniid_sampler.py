from typing import Optional, Iterator
import warnings
import numpy as np
import torch
import time
from torch.utils.data import Sampler, Dataset
import bluefog.torch as bf


# Modified from https://github.com/Xtra-Computing/NIID-Bench.git using 'noniid-labeldir' partition
# and standard PyTorch DistributedSampler
class NonIIDDistSampler(Sampler):
    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, beta: float = 1.0) -> None:
        if num_replicas is None:
            num_replicas = bf.size()
        if rank is None:
            rank = bf.rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.size = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle
        self.seed = seed
        self.beta = beta # beta is for the Dirichlet distribution

        net_dataidx_map, idx_list = self._compute_dirichlet_dist()
        net_dataidx_map, self.idx_list = self._balance_local_data(net_dataidx_map, idx_list)
        net_dataidx_map = self._shuffle_idx(net_dataidx_map)
        self.num_samples = min([len(idx) for _, idx in net_dataidx_map.items()])
        self.total_size = self.num_samples * self.size
        self.local_idx = torch.LongTensor(net_dataidx_map[self.rank][:self.num_samples])
        # time.sleep(10)

    def _transfer_local_data(self, id_max, class_count_max, id_min, class_count_min):
        avg_local_size = len(self.dataset)//self.size
        max_l = len(id_max)
        min_l = len(id_min)
        move_size = min(max_l-avg_local_size, avg_local_size-min_l)

        move_amounts = []
        propotions = class_count_max/np.sum(class_count_max)
        for i in range(self.n_class):
            move_amounts.append(int(propotions[i]*move_size))
        undistributed_amount = move_size-np.sum(move_amounts)
        loc = self.n_class-1
        while undistributed_amount > 0:
            if move_amounts[loc] < class_count_max[loc]:
                additional_amount = min(undistributed_amount, class_count_max[loc]-move_amounts[loc])
                move_amounts[loc] += additional_amount
                undistributed_amount -= additional_amount
            loc -= 1

        move_amounts[-1] = move_size-np.sum(move_amounts[:-1])

        id_max_n = [None]*(max_l-move_size)
        class_count_max_n = [0]*self.n_class
        id_min_n = [None]*(move_size+min_l)
        class_count_min_n = [0]*self.n_class

        s_max_n, s_max = 0, 0
        s_min_n, s_min = 0, 0
        for i in range(self.n_class):
            class_count_max_n[i] = class_count_max[i]-move_amounts[i]
            id_max_n[s_max_n:s_max_n+class_count_max_n[i]] = id_max[s_max:s_max+class_count_max_n[i]]
            class_count_min_n[i] = class_count_min[i]+move_amounts[i]
            id_min_n[s_min_n:s_min_n+class_count_min[i]] = id_min[s_min:s_min+class_count_min[i]]
            id_min_n[s_min_n+class_count_min[i]:s_min_n+class_count_min[i]+move_amounts[i]] = id_max[s_max+class_count_max_n[i]:s_max+class_count_max_n[i]+move_amounts[i]]

            s_max_n += class_count_max_n[i]
            s_max += class_count_max[i]
            s_min_n += class_count_min_n[i]
            s_min += class_count_min[i]

        return id_max_n, class_count_max_n, id_min_n, class_count_min_n
    
    def _balance_local_data(self, net_dataidx_map, idx_list):
        n_iter = 0
        max_iter = len(idx_list)
        while True:
            local_len = [len(net_dataidx_map[r]) for r in range(self.size)]
            min_k_l, max_k_l = local_len[0], local_len[0]
            min_k, max_k = 0, 0
            for rank in range(1, self.size):
                if min_k_l > local_len[rank]:
                    min_k_l = local_len[rank]
                    min_k = rank
                if max_k_l < local_len[rank]:
                    max_k_l = local_len[rank]
                    max_k = rank
            if min_k_l == max_k_l:
                break # assume the dataset can be evenly divided
            new_idx_1, ns_1, new_idx_2, ns_2 = self._transfer_local_data(
                net_dataidx_map[max_k], idx_list[max_k],
                net_dataidx_map[min_k], idx_list[min_k])
            net_dataidx_map[max_k] = new_idx_1
            idx_list[max_k] = ns_1
            net_dataidx_map[min_k] = new_idx_2
            idx_list[min_k] = ns_2
            n_iter += 1
            if n_iter >= max_iter:
                warnings.warn("Dataset may be not evenly distributed.")
                break
        return net_dataidx_map, idx_list
    
    def _shuffle_idx(self, net_dataidx_map):
        for i in range(self.size):
            np.random.shuffle(net_dataidx_map[i])
        return net_dataidx_map
    
    def _compute_dirichlet_dist(self):
        n_parties = self.size
        min_size = 0
        min_require_size = 10
        targets = torch.LongTensor(self.dataset.targets)
        K = len(torch.unique(torch.LongTensor(self.dataset.targets)))
        self.n_class = K

        N = len(self.dataset)
        np.random.seed(self.seed)
        net_dataidx_map = {}

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            idx_dist = [[] for _ in range(n_parties)]
            for k in range(K):
                idx_k = np.where(targets==k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(self.beta, n_parties))
                # print("[", self.rank, "] proportions1: ", proportions)
                # print("[", self.rank, "] sum pro1:", np.sum(proportions))
                ## Balance
                proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                # print("[", self.rank, "] proportions2: ", proportions)
                proportions = proportions / proportions.sum()
                # print("[", self.rank, "] proportions3: ", proportions)
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                # print("[", self.rank, "] proportions4: ", proportions)
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
                idx_sep = np.zeros(len(proportions)+2)
                idx_sep[1:-1] = proportions
                idx_sep[-1] = len(idx_k)
                idx_dist = [dist+[idx_sep[i+1]-idx_sep[i]] for i, dist in enumerate(idx_dist)]

        for j in range(n_parties):
            net_dataidx_map[j] = idx_batch[j]
        # print("[", self.rank, "] size: ", len(net_dataidx_map[self.rank]))
        # print("[", self.rank, "] non-iid sample: ", net_dataidx_map[self.rank][:10])
        idx_dist = [[int(v) for v in ss] for ss in idx_dist]
        return net_dataidx_map, idx_dist

    def __iter__(self) -> Iterator:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch + self.rank)
            shuffler = torch.randperm(self.num_samples, generator=g)  # type: ignore[arg-type]
            indices = self.local_idx[shuffler]
        else:
            indices = self.local_idx  # type: ignore[arg-type]

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch