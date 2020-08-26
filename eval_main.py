#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import setproctitle
import dataset
from model import BGCN, BGCN_Info
from utils import check_overfitting, early_stop, get_perf, logger 
from train import train
from metric import Recall, NDCG, MRR
from config import CONFIG
from test import test
import loss
import time
import csv

TAG = ''

def main():
    # set env
    setproctitle.setproctitle(f"test{CONFIG['name']}")
    os.environ["CUDA_VISIBLE_DEVICES"] = CONFIG['gpu_id']
    device = torch.device('cuda')

    # load data
    bundle_train_data, bundle_test_data, item_data, assist_data = \
        dataset.get_dataset(CONFIG['path'], CONFIG['dataset_name'], task=CONFIG['eval_task'])
    bundle_test_loader = DataLoader(bundle_test_data, 4096, False,
                             num_workers=16, pin_memory=True)
    test_loader = bundle_test_loader

    #  graph
    ub_graph = bundle_train_data.ground_truth_u_b
    ui_graph = item_data.ground_truth_u_i
    bi_graph = assist_data.ground_truth_b_i

    # metric
    metrics = [Recall(20), NDCG(20), Recall(40), NDCG(40), Recall(80), NDCG(80)]
    TARGET = 'Recall@20'

    # log
    log = logger.Logger(os.path.join(
        CONFIG['log'], CONFIG['dataset_name'], 
        f"{CONFIG['model']}_{CONFIG['eval_task']}", TAG), 'best', checkpoint_target=TARGET)

    for DIR in CONFIG['test']:
        with open(os.path.join(DIR, 'model.csv'), 'r') as f:
            d = csv.DictReader(f)
            d = [line for line in d]
        for i in range(len(d)):
            s = d[i][None][0]
            dd = {'hash': d[i]['hash'],
                  'embed_L2_norm': float(d[i][' embed_L2_norm']),
                  'mess_dropout': float(d[i][' mess_dropout']),
                  'node_dropout': float(d[i][' node_dropout']),
                  'lr': float(s[s.find(':') + 1:])}

            # model
            if CONFIG['model'] == 'BGCN':
                graph = [ub_graph, ui_graph, bi_graph]
                info = BGCN_Info(64, dd['embed_L2_norm'], dd['mess_dropout'], dd['node_dropout'], 2)
                model = BGCN(info, assist_data, graph, device, pretrain=None).to(device)

            assert model.__class__.__name__ == CONFIG['model']

            model.load_state_dict(torch.load(
                os.path.join(DIR, dd['hash']+"_Recall@20.pth")))

            # log
            log.update_modelinfo(info, {'lr': dd['lr']}, metrics)

            # test
            test(model, test_loader, device, CONFIG, metrics)

            # log
            log.update_log(metrics, model) 

            log.close_log(TARGET)
    log.close()


if __name__ == "__main__":
    main()
