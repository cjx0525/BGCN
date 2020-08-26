#!/usr/bin/env python3
# -*- coding: utf-8 -*-

CONFIG = {
    'name': '@changjianxin',
    'path': './data',
    'log': './log',
    'visual': './visual',
    'gpu_id': "0",
    'note': 'some_note',
    'model': 'BGCN',
    'dataset_name': 'NetEase',
    'task': 'tune',
    'eval_task': 'test',

    ## search hyperparameters
    #  'lrs': [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3],
    #  'message_dropouts': [0, 0.1, 0.3, 0.5],
    #  'node_dropouts': [0, 0.1, 0.3, 0.5],
    #  'decays': [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],

    ## optimal hyperparameters 
    'lrs': [3e-4],
    'message_dropouts': [0.3],
    'node_dropouts': [0],
    'decays': [1e-7],

    ## hard negative sample and further train
    'sample': 'simple',
    #  'sample': 'hard',
    'hard_window': [0.7, 1.0], # top 30%
    'hard_prob': [0.4, 0.4], # probability 0.8
    'conti_train': 'model_file_from_simple_sample.pth',

    ## other settings
    'epochs': 1000,
    'early': 50,
    'log_interval': 20,
    'test_interval': 1,
    'retry': 1,

    ## test path
    'test':['model_path_from_hard_sample']
}

