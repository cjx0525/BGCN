#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import string
import random
from json import dump
from itertools import product
from utils import get_perf
import torch
from config import CONFIG


def hash_model(modelinfo):
    return hex(hash(modelinfo))[-6:]


class Logger(object):
    CHECKPOINT_POLICIES = ['none', 'always', 'best']

    def get_metric_csv_title(self, metrics):
        window = ['best', 'ave']
        metric = list(map(lambda x: x.get_title(), metrics))
        group = product(window, metric)
        group_str = map(lambda x: '-'.join(x), group)
        return ', '.join(group_str)

    def __init__(self, log_path, checkpoint_policy='best', checkpoint_interval=None, checkpoint_target=None):
        '''
        Args:
        - log_path: the dir of every model's log dir
        - checkpoint_policy: when to save model. [ `none` | `always` | `best` (default)]
            - `none`: never save model
            - `always`: save model every `checkpoint_interval` epochs
            - `best`: save the best(`checkpoint_target`) model
        - checkpoint_interval: int, for `always`
        - checkpoint_target: str or list of str, for `best`
        '''
        assert checkpoint_policy in Logger.CHECKPOINT_POLICIES
        if checkpoint_policy == 'always':
            assert checkpoint_interval > 0 and isinstance(
                checkpoint_interval, int)
        if checkpoint_policy == 'best':
            assert isinstance(checkpoint_target, (str, list, tuple))
            if isinstance(checkpoint_target, (list, tuple)):
                for target in checkpoint_target:
                    assert isinstance(target, str)
            else:
                checkpoint_target = [checkpoint_target]

        self.checkpoint_policy = checkpoint_policy
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_epoch = 0
        self.checkpoint_target = checkpoint_target

        self.random = ''.join(random.choice(
            string.ascii_uppercase + string.digits) for _ in range(8))
        self.log_path = log_path
        #  self.time_path = time.strftime(
            #  '%m-%d-%H-%M-%S-', time.localtime(time.time()))+self.random
        self.time_path = time.strftime(
            '%m-%d-%H-%M-%S-', time.localtime(time.time())) + CONFIG['note']

        self.root_path = os.path.join(self.log_path, self.time_path)
        if not os.path.exists(self.root_path):
            os.makedirs(self.root_path, exist_ok=True)
        else:
            raise FileExistsError('{} exists'.format(self.root_path))
        self.csv_log = open(os.path.join(self.root_path, 'model.csv'), 'w')

        self.cnt = 0

    def get_model_Id(self, modelinfo):
        return '_'.join([str(self.cnt), hash_model(modelinfo)])

    def __del__(self):
        try:
            self.csv_log.close()
        finally:
            pass

    def close(self):
        self.__del__()

    def update_modelinfo(self, modelinfo, envinfo, metrics):
        '''
        args:
        - `modelinfo`:  model hyperparameters (`ModelInfo`)
        - `envinfo`:    other hyperparameters like (`lr`) (`Dict`)
        '''
        self.modelinfo = modelinfo
        self.env = envinfo
        if self.cnt is 0:
            self.csv_log.write('hash, {}, {}\n'.format(
                self.modelinfo.get_csv_title(), self.get_metric_csv_title(metrics)))
        self.cnt += 1
        self._metrics_log = None

    def update_log(self, metrics, model):
        # save metrics
        if self._metrics_log is None:
            self._metrics_log = {
                metric.get_title(): [metric.metric] for metric in metrics}
        else:
            for metric in metrics:
                self._metrics_log[metric.get_title()].append(metric.metric)
        with open(os.path.join(
                self.root_path, '{}.json'.format(self.get_model_Id(self.modelinfo))), 'w') as f:
            dump(self._metrics_log, f)
        # save model
        if self.checkpoint_policy == 'always':
            self.checkpoint_epoch += 1
            if self.checkpoint_epoch % self.checkpoint_interval == 0:
                model_path = os.path.join(
                    self.root_path, '{}.pth'.format(self.get_model_Id(self.modelinfo)))
                torch.save(model.state_dict(), model_path)
        elif self.checkpoint_policy == 'best':
            for target in self.checkpoint_target:
                if self.metrics_log[target][-1] == max(self.metrics_log[target]):
                    model_path = os.path.join(self.root_path, '{}_{}.pth'.format(
                        self.get_model_Id(self.modelinfo), target))
                    torch.save(model.state_dict(), model_path)

    def close_log(self, target, window_size=10):
        self.csv_log.write('{}, {}, '.format(
            self.get_model_Id(self.modelinfo), self.modelinfo.get_csv_line()))
        best = get_perf(self.metrics_log, window_size=1,
                        target=target, show=False)
        best_str = ', '.join(map(str, best.values()))
        ave = get_perf(self.metrics_log, window_size=window_size,
                       target=target, show=False)
        ave_str = ', '.join(map(str, ave.values()))
        self.csv_log.write('{}, {}, {}\n'.format(
            best_str, ave_str, str(self.env)))
        self.csv_log.flush()

    @property
    def metrics_log(self):
        return self._metrics_log
