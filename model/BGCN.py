#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp 
import numpy as np
from .model_base import Info, Model
from config import CONFIG


def graph_generating(raw_graph, row, col):
    if raw_graph.shape == (row, col):
        graph = sp.bmat([[sp.identity(raw_graph.shape[0]), raw_graph],
                             [raw_graph.T, sp.identity(raw_graph.shape[1])]])
    else:
        raise ValueError(r"raw_graph's shape is wrong")
    return graph

def laplace_transform(graph):
    rowsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=1).A.ravel()) + 1e-8))
    colsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=0).A.ravel()) + 1e-8))
    graph = rowsum_sqrt @ graph @ colsum_sqrt
    return graph

def to_tensor(graph):
    graph = graph.tocoo()
    values = graph.data
    indices = np.vstack((graph.row, graph.col))
    graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), 
                                          torch.Size(graph.shape))
    return graph


class BGCN_Info(Info):
    def __init__(self, embedding_size, embed_L2_norm, mess_dropout, node_dropout, num_layers, act=nn.LeakyReLU()):
        super().__init__(embedding_size, embed_L2_norm)
        self.act = act
        assert 1 > mess_dropout >= 0
        self.mess_dropout = mess_dropout
        assert 1 > node_dropout >= 0
        self.node_dropout = node_dropout
        assert isinstance(num_layers, int) and num_layers > 0
        self.num_layers = num_layers


class BGCN(Model):
    def get_infotype(self):
        return BGCN_Info

    def __init__(self, info, dataset, raw_graph, device, pretrain=None):
        super().__init__(info, dataset, create_embeddings=True)
        self.items_feature = nn.Parameter(
            torch.FloatTensor(self.num_items, self.embedding_size))
        nn.init.xavier_normal_(self.items_feature)

        self.epison = 1e-8

        assert isinstance(raw_graph, list)
        ub_graph, ui_graph, bi_graph = raw_graph

        #  deal with weights
        bi_norm = sp.diags(1/(np.sqrt((bi_graph.multiply(bi_graph)).sum(axis=1).A.ravel()) + 1e-8)) @ bi_graph
        bb_graph = bi_norm @ bi_norm.T

        #  pooling graph
        bundle_size = bi_graph.sum(axis=1) + 1e-8
        bi_graph = sp.diags(1/bundle_size.A.ravel()) @ bi_graph

        if ui_graph.shape == (self.num_users, self.num_items):
            # add self-loop
            atom_graph = sp.bmat([[sp.identity(ui_graph.shape[0]), ui_graph],
                                 [ui_graph.T, sp.identity(ui_graph.shape[1])]])
        else:
            raise ValueError(r"raw_graph's shape is wrong")
        self.atom_graph = to_tensor(laplace_transform(atom_graph)).to(device)
        print('finish generating atom graph')
 
        if ub_graph.shape == (self.num_users, self.num_bundles) \
                and bb_graph.shape == (self.num_bundles, self.num_bundles):
            # add self-loop
            non_atom_graph = sp.bmat([[sp.identity(ub_graph.shape[0]), ub_graph],
                                 [ub_graph.T, bb_graph]])
        else:
            raise ValueError(r"raw_graph's shape is wrong")
        self.non_atom_graph = to_tensor(laplace_transform(non_atom_graph)).to(device)
        print('finish generating non-atom graph')

        self.pooling_graph = to_tensor(bi_graph).to(device)
        print('finish generating pooling graph')

        # copy from info
        self.act = self.info.act
        self.num_layers = self.info.num_layers
        self.device = device

        #  Dropouts
        self.mess_dropout = nn.Dropout(self.info.mess_dropout, True)
        self.node_dropout = nn.Dropout(self.info.node_dropout, True)

        # Layers
        self.dnns_atom = nn.ModuleList([nn.Linear(
            self.embedding_size*(l+1), self.embedding_size) for l in range(self.num_layers)])
        self.dnns_non_atom = nn.ModuleList([nn.Linear(
            self.embedding_size*(l+1), self.embedding_size) for l in range(self.num_layers)])

        # pretrain
        if not pretrain is None:
            self.users_feature.data = F.normalize(
                pretrain['users_feature'])  
            self.items_feature.data = F.normalize(
                pretrain['items_feature'])
            self.bundles_feature.data = F.normalize(
                pretrain['bundles_feature'])

    def one_propagate(self, graph, A_feature, B_feature, dnns):
        # node dropout on graph
        indices = graph._indices()
        values = graph._values()
        values = self.node_dropout(values)
        graph = torch.sparse.FloatTensor(
            indices, values, size=graph.shape)

        # propagate
        features = torch.cat((A_feature, B_feature), 0)
        all_features = [features]
        for i in range(self.num_layers):
            features = self.mess_dropout(torch.cat([self.act(
                dnns[i](torch.matmul(graph, features))), features], 1))
            all_features.append(F.normalize(features))

        all_features = torch.cat(all_features, 1)
        A_feature, B_feature = torch.split(
            all_features, (A_feature.shape[0], B_feature.shape[0]), 0)
        return A_feature, B_feature

    def propagate(self):
        #  =============================  item level propagation  =============================
        atom_users_feature, atom_items_feature = self.one_propagate(
            self.atom_graph, self.users_feature, self.items_feature, self.dnns_atom)
        atom_bundles_feature = F.normalize(torch.matmul(self.pooling_graph, atom_items_feature))

        #  ============================= bundle level propagation =============================
        non_atom_users_feature, non_atom_bundles_feature = self.one_propagate(
            self.non_atom_graph, self.users_feature, self.bundles_feature, self.dnns_non_atom)

        users_feature = [atom_users_feature, non_atom_users_feature]
        bundles_feature = [atom_bundles_feature, non_atom_bundles_feature]

        return users_feature, bundles_feature

    def predict(self, users_feature, bundles_feature):
        users_feature_atom, users_feature_non_atom = users_feature # batch_n_f
        bundles_feature_atom, bundles_feature_non_atom = bundles_feature # batch_n_f
        pred = torch.sum(users_feature_atom * bundles_feature_atom, 2) \
            + torch.sum(users_feature_non_atom * bundles_feature_non_atom, 2)
        return pred
    
    def forward(self, users, bundles):
        users_feature, bundles_feature = self.propagate()
        users_embedding = [i[users].expand(- 1, bundles.shape[1], -1) for i in users_feature]  # u_f --> batch_f --> batch_n_f
        bundles_embedding = [i[bundles] for i in bundles_feature] # b_f --> batch_n_f
        pred = self.predict(users_embedding, bundles_embedding)
        loss = self.regularize(users_embedding, bundles_embedding)
        return pred, loss

    def regularize(self, users_feature, bundles_feature):
        users_feature_atom, users_feature_non_atom = users_feature # batch_n_f
        bundles_feature_atom, bundles_feature_non_atom = bundles_feature # batch_n_f
        loss = self.embed_L2_norm * \
            ((users_feature_atom ** 2).sum() + (bundles_feature_atom ** 2).sum() +\
            (users_feature_non_atom ** 2).sum() + (bundles_feature_non_atom ** 2).sum())
        return loss

    def evaluate(self, propagate_result, users):
        '''
        just for testing, compute scores of all bundles for `users` by `propagate_result`
        '''
        users_feature, bundles_feature = propagate_result
        users_feature_atom, users_feature_non_atom = [i[users] for i in users_feature] # batch_f
        bundles_feature_atom, bundles_feature_non_atom = bundles_feature # b_f
        scores = torch.mm(users_feature_atom, bundles_feature_atom.t()) \
            + torch.mm(users_feature_non_atom, bundles_feature_non_atom.t()) # batch_b
        return scores

