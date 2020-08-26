#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
import scipy.sparse as sp 
from torch.utils.data import Dataset
from config import CONFIG


def sparse_ones(indices, size, dtype=torch.float):
    one = torch.ones(indices.shape[1], dtype=dtype)
    return torch.sparse.FloatTensor(indices, one, size=size).to(dtype)

def to_tensor(graph):
    graph = graph.tocoo()
    values = graph.data
    indices = np.vstack((graph.row, graph.col))
    graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), 
                                          torch.Size(graph.shape))
    return graph

def print_statistics(X, string):
    print('>'*10 + string + '>'*10 )
    print('Average interactions', X.sum(1).mean(0).item())
    nonzero_row_indice, nonzero_col_indice = X.nonzero()
    unique_nonzero_row_indice = np.unique(nonzero_row_indice)
    unique_nonzero_col_indice = np.unique(nonzero_col_indice)
    print('Non-zero rows', len(unique_nonzero_row_indice)/X.shape[0])
    print('Non-zero columns', len(unique_nonzero_col_indice)/X.shape[1])
    print('Matrix density', len(nonzero_row_indice)/(X.shape[0]*X.shape[1]))


class BasicDataset(Dataset):
    '''
    generate dataset from raw *.txt
    contains:
        tensors like (`user`, `bundle_p`, `bundle_n1`, `bundle_n2`, ...) for BPR (use `self.user_bundles`)
    Args:
    - `path`: the path of dir that contains dataset dir
    - `name`: the name of dataset (used as the name of dir)
    - `neg_sample`: the number of negative samples for each user-bundle_p pair
    - `seed`: seed of `np.random`
    '''

    def __init__(self, path, name, task, neg_sample):
        self.path = path
        self.name = name
        self.task = task
        self.neg_sample = neg_sample
        self.num_users, self.num_bundles, self.num_items  = self.__load_data_size()

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __load_data_size(self):
        with open(os.path.join(self.path, self.name, '{}_data_size.txt'.format(self.name)), 'r') as f:
            return [int(s) for s in f.readline().split('\t')][:3]
    def load_U_B_interaction(self):
        with open(os.path.join(self.path, self.name, 'user_bundle_{}.txt'.format(self.task)), 'r') as f:
            return list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))
    def load_U_I_interaction(self):
        with open(os.path.join(self.path, self.name, 'user_item.txt'), 'r') as f:
            return list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))
    def load_B_I_affiliation(self):
        with open(os.path.join(self.path, self.name, 'bundle_item.txt'), 'r') as f:
            return list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))


class BundleTrainDataset(BasicDataset):
    def __init__(self, path, name, item_data, assist_data, seed=None):
        super().__init__(path, name, 'train', 1)
        # U-B
        self.U_B_pairs = self.load_U_B_interaction()
        indice = np.array(self.U_B_pairs, dtype=np.int32)
        values = np.ones(len(self.U_B_pairs), dtype=np.float32)
        self.ground_truth_u_b = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_users, self.num_bundles)).tocsr()

        print_statistics(self.ground_truth_u_b, 'U-B statistics in train')

        if CONFIG['sample'] == 'hard': 
            #  1. u_p --> b_n1
            u_b_from_i = item_data.ground_truth_u_i @ assist_data.ground_truth_b_i.T
            u_b_from_i = u_b_from_i.todense()
            bn1_window = [int(i*self.num_bundles) for i in CONFIG['hard_window']]
            self.u_b_for_neg_sample = np.argsort(u_b_from_i, axis=1)[:,bn1_window[0]:bn1_window[1]]

            #  2. b_p --> b_n2
            overlap_graph = assist_data.ground_truth_b_i @ assist_data.ground_truth_b_i.T
            overlap_graph = overlap_graph.todense()
            bn2_window = [int(i*self.num_bundles) for i in CONFIG['hard_window']]
            self.b_b_for_neg_sample = np.argsort(overlap_graph, axis=1)[:,bn2_window[0]:bn2_window[1]]

    def __getitem__(self, index):
        user_b, pos_bundle = self.U_B_pairs[index]
        all_bundles = [pos_bundle]
        if CONFIG['sample'] == 'simple':
            while True:
                i = np.random.randint(self.num_bundles)
                if self.ground_truth_u_b[user_b, i] == 0 and not i in all_bundles:
                    all_bundles.append(i)
                    if len(all_bundles) == self.neg_sample+1:
                        break
        elif CONFIG['sample'] == 'hard':
            hard_probability = round(np.random.uniform(0, 1), 1)
            if  hard_probability <= CONFIG['hard_prob'][0]:
                while True:
                    i = np.random.randint(self.u_b_for_neg_sample.shape[1])
                    b_n1 = self.u_b_for_neg_sample[user_b, i]
                    if self.ground_truth_u_b[user_b, b_n1] == 0 and not b_n1 in all_bundles:
                        all_bundles.append(b_n1)
                        if len(all_bundles) == self.neg_sample+1:
                            break
            elif CONFIG['hard_prob'][0] < hard_probability \
                <= CONFIG['hard_prob'][0] + CONFIG['hard_prob'][1]:
                while True:
                    i = np.random.randint(self.b_b_for_neg_sample.shape[1])
                    b_n2 = self.b_b_for_neg_sample[pos_bundle, i]
                    if self.ground_truth_u_b[user_b, b_n2] == 0 and not b_n2 in all_bundles:
                        all_bundles.append(b_n2)
                        if len(all_bundles) == self.neg_sample+1:
                            break
            else:
                while True:
                    i = np.random.randint(self.num_bundles)
                    if self.ground_truth_u_b[user_b, i] == 0 and not i in all_bundles:
                        all_bundles.append(i)
                        if len(all_bundles) == self.neg_sample+1:
                            break
        else:
            raise ValueError(r"sample's method is wrong")

        return torch.LongTensor([user_b]), torch.LongTensor(all_bundles)

    def __len__(self):
        return len(self.U_B_pairs)  


class BundleTestDataset(BasicDataset):
    def __init__(self, path, name, train_dataset, task='test'):
        super().__init__(path, name, task, None)
        # U-B
        self.U_B_pairs = self.load_U_B_interaction()
        indice = np.array(self.U_B_pairs, dtype=np.int32)
        values = np.ones(len(self.U_B_pairs), dtype=np.float32)
        self.ground_truth_u_b = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_users, self.num_bundles)).tocsr()

        print_statistics(self.ground_truth_u_b, 'U-B statistics in test')

        self.train_mask_u_b = train_dataset.ground_truth_u_b
        self.users = torch.arange(self.num_users, dtype=torch.long).unsqueeze(dim=1)
        self.bundles = torch.arange(self.num_bundles, dtype=torch.long)
        assert self.train_mask_u_b.shape == self.ground_truth_u_b.shape

    def __getitem__(self, index):
        return index, torch.from_numpy(self.ground_truth_u_b[index].toarray()).squeeze(),  \
            torch.from_numpy(self.train_mask_u_b[index].toarray()).squeeze(),  \

    def __len__(self):
        return self.ground_truth_u_b.shape[0]

class ItemDataset(BasicDataset):
    def __init__(self, path, name, assist_data, seed=None):
        super().__init__(path, name, 'train', 1)
        # U-I
        self.U_I_pairs = self.load_U_I_interaction()
        indice = np.array(self.U_I_pairs, dtype=np.int32)
        values = np.ones(len(self.U_I_pairs), dtype=np.float32)
        self.ground_truth_u_i = sp.coo_matrix( 
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_users, self.num_items)).tocsr()

        print_statistics(self.ground_truth_u_i, 'U-I statistics')

    def __getitem__(self, index):
        user_i, pos_item = self.U_I_pairs[index]
        all_items = [pos_item]
        while True:
            j = np.random.randint(self.num_items)
            if self.ground_truth_u_i[user_i, j] == 0 and not j in all_items:
                all_items.append(j)
                if len(all_items) == self.neg_sample+1:
                    break

        return torch.LongTensor([user_i]), torch.LongTensor(all_items)

    def __len__(self):
        return len(self.U_I_pairs)  


class AssistDataset(BasicDataset):
    def __init__(self, path, name):
        super().__init__(path, name, None, None)
        # B-I
        self.B_I_pairs = self.load_B_I_affiliation()
        indice = np.array(self.B_I_pairs, dtype=np.int32)
        values = np.ones(len(self.B_I_pairs), dtype=np.float32)
        self.ground_truth_b_i = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_bundles, self.num_items)).tocsr()

        print_statistics(self.ground_truth_b_i, 'B-I statistics')


def get_dataset(path, name, task='tune', seed=123):
    assist_data = AssistDataset(path, name)
    print('finish loading assist data')
    item_data = ItemDataset(path, name, assist_data, seed=seed)
    print('finish loading item data')

    bundle_train_data = BundleTrainDataset(path, name, item_data, assist_data, seed=seed)
    print('finish loading bundle train data')
    bundle_test_data = BundleTestDataset(path, name, bundle_train_data, task=task)
    print('finish loading bundle test data')

    return bundle_train_data, bundle_test_data, item_data, assist_data

