#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from math import ceil


def show(metrics_log):
    x = range(0, len(list(metrics_log.values())[0]))
    i = 1
    columns = 2
    rows = ceil(len(metrics_log)/columns)
    for k, v in metrics_log.items():
        plt.subplot(rows, columns, i)
        plt.plot(x, v, '.-')
        plt.title('{} vs epochs'.format(k))
        i += 1
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    plt.show()


def get_perf(metrics_log, window_size, target, show=True):
    # max
    maxs = {title: 0 for title in metrics_log.keys()}
    assert target in maxs
    length = len(metrics_log[target])
    for v in metrics_log.values():
        assert length == len(v)
    if window_size >= length:
        for k, v in metrics_log.items():
            maxs[k] = np.mean(v)
    else:
        for i in range(length-window_size):
            now = np.mean(metrics_log[target][i:i+window_size])
            if now > maxs[target]:
                for k, v in metrics_log.items():
                    maxs[k] = np.mean(v[i:i+window_size])
    if show:
        for k, v in maxs.items():
            print('{}:{:.5f}'.format(k, v), end=' ')
    return maxs


def check_overfitting(metrics_log, target, threshold=0.02, show=False):
    maxs = get_perf(metrics_log, 1, target, False)
    assert target in maxs
    overfit = (maxs[target]-metrics_log[target][-1]) > threshold
    if overfit and show:
        print('***********overfit*************')
        print('best:', end=' ')
        for k, v in maxs.items():
            print('{}:{:.5f}'.format(k, v), end=' ')
        print('')
        print('now:', end=' ')
        for k, v in metrics_log.items():
            print('{}:{:.5f}'.format(k, v[-1]), end=' ')
        print('')
        print('***********overfit*************')
    return overfit


def early_stop(metric_log, early, threshold=0.01):
    if len(metric_log) >= 2 and metric_log[-1] < metric_log[-2] and metric_log[-1] > threshold:
        return early-1
    else:
        return early
