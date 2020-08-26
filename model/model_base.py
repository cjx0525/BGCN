#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class Info(object):
    '''
    [FOR `utils.logger`]

    the base class that packing all hyperparameters and infos used in the related model
    '''

    def __init__(self, embedding_size, embed_L2_norm):
        assert isinstance(embedding_size, int) and embedding_size > 0
        self.embedding_size = embedding_size
        assert embed_L2_norm >= 0
        self.embed_L2_norm = embed_L2_norm

    def get_title(self):
        dct = self.__dict__
        if '_info' in dct:
            dct.pop('_info')
        return '\t'.join(map(lambda x: dct[x].get_title() if isinstance(dct[x], Info) else x, dct.keys()))

    def get_csv_title(self):
        return self.get_title().replace('\t', ', ')

    def __getitem__(self, key):
        if hasattr(self, '_info'):
            return self._info[key]
        else:
            return self.__getattribute__(key)

    def __str__(self):
        dct = self.__dict__
        if '_info' in dct:
            dct.pop('_info')
        return '\t'.join(map(str, dct.values()))

    def get_line(self):
        return self.__str__()

    def get_csv_line(self):
        return self.get_line().replace('\t', ', ')


class Model(nn.Module):
    '''
    base class for all MF-based model
    packing embedding initialization, embedding choosing in forward

    NEED IMPLEMENT:
    - `propagate`: all raw embeddings -> processed embeddings(user/bundle)
    - `predict`: processed embeddings of targets(users/bundles inputs) -> scores

    OPTIONAL:
    - `regularize`: processed embeddings of targets(users/bundles inputs) -> extra loss(default: L2)
    - `get_infotype`: the correct type of `info`(default: `object`)
    '''

    def get_infotype(self):
        return object

    def __init__(self, info, dataset, create_embeddings=True):
        super().__init__()
        assert isinstance(info, self.get_infotype())
        self.info = info
        self.embedding_size = info['embedding_size']
        self.embed_L2_norm = info['embed_L2_norm']
        self.num_users = dataset.num_users
        self.num_bundles = dataset.num_bundles
        self.num_items = dataset.num_items
        if create_embeddings:
            # embeddings
            self.users_feature = nn.Parameter(
                torch.FloatTensor(self.num_users, self.embedding_size))
            nn.init.xavier_normal_(self.users_feature)
            self.bundles_feature = nn.Parameter(
                torch.FloatTensor(self.num_bundles, self.embedding_size))
            nn.init.xavier_normal_(self.bundles_feature)
 
    def propagate(self, *args, **kwargs):
        '''
        raw embeddings -> embeddings for predicting
        return (user's,bundle's)
        '''
        raise NotImplementedError

    def predict(self, users_feature, bundles_feature, *args, **kwargs):
        '''
        embeddings of targets for predicting -> scores
        return scores
        '''
        raise NotImplementedError

    def regularize(self, users_feature, bundles_feature, *args, **kwargs):
        '''
        embeddings of targets for predicting -> extra loss(default: L2 loss...)
        '''
        loss = self.embed_L2_norm * \
            ((users_feature ** 2).sum()+(bundles_feature**2).sum())
        return loss

    def forward(self, users, bundles):
        users_feature, bundles_feature = self.propagate()
        bundles_embedding = bundles_feature[bundles]
        users_embedding = users_feature[users].expand(
            -1, bundles.shape[1], -1)  
        pred = self.predict(users_embedding, bundles_embedding)
        loss = self.regularize(users_embedding, bundles_embedding)
        return pred, loss

    def evaluate(self, propagate_result, users):
        '''
        just for testing, compute scores of all bundles for `users` by `propagate_result`
        '''
        raise NotImplementedError

