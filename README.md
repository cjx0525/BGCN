# Bundle Graph Convolutional Network
This is our Pytorch implementation for the paper:

>Jianxin Chang, Chen Gao, Xiangnan He, Depeng Jin and Yong Li. Bundle Graph Convolutional Network, [Paper in ACM DL](https://dl.acm.org/citation.cfm?doid=3397271.3401198) or [Paper in arXiv](https://arxiv.org/abs/2005.03475). In SIGIR'20, Xi'an, China, July 25-30, 2020.

Author: Jianxin Chang (changjx18@mails.tsinghua.edu.cn)

## Introduction
Bundle Graph Convolutional Network (BGCN) is a bundle recommendation solution based on graph neural network, explicitly re-constructing the two kinds of interaction and an affiliation into the graph. With item nodes as the bridge, graph convolutional propagation between user and bundle nodes makes the learned representations capture the item level semantics.

## Citation 
If you want to use our codes and datasets in your research, please cite:
```
@inproceedings{BGCN20,
  author    = {Jianxin Chang and 
               Chen Gao and 
               Xiangnan He and 
               Depeng Jin and 
               Yong Li},
  title     = {Bundle Recommendation with Graph Convolutional Networks},
  booktitle = {Proceedings of the 43nd International {ACM} {SIGIR} Conference on
               Research and Development in Information Retrieval, {SIGIR} 2020, Xi'an,
               China, July 25-30, 2020.},
  year      = {2020},
}
```

## Requirement
The code has been tested running under Python 3.7.0. The required packages are as follows:
* torch == 1.2.0
* numpy == 1.17.4
* scipy == 1.4.1
* temsorboardX == 2.0

## Usage
The hyperparameter search range and optimal settings have been clearly stated in the codes (see the 'CONFIG' dict in config.py).
* Train

```
python main.py 
```

* Futher Train

Replace 'sample' from 'simple' to 'hard' in CONFIG and add model file path obtained by Train to 'conti_train', then run
```
python main.py 
```

* Test

Add model path obtained by Futher Train to 'test' in CONFIG, then run
```
python eval_main.py 
```

Some important hyperparameters:
* `lrs`
  * It indicates the learning rates. 
  * The learning rate is searched in {1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3}.

* `mess_dropouts`
  * It indicates the message dropout ratio, which randomly drops out the outgoing messages. 
  * We search the message dropout within {0, 0.1, 0.3, 0.5}.

* `node_dropouts`
  * It indicates the node dropout ratio, which randomly blocks a particular node and discard all its outgoing messages. 
  * We search the node dropout within {0, 0.1, 0.3, 0.5}.

* `decays`
  * we adopt L2 regularization and use the decays to control the penalty strength.
  * L2 regularization term is tuned in {1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2}.

* `hard_window`
  * It indicates the difficulty of sampling in the hard-negative sampler.
  * We set it to the top thirty percent.

* `hard_prob`
  * It indicates the probability of using hard-negative samples in the further training stage.
  * We set it to 0.8 (0.4 in the item level and 0.4 in the bundle level), so the probability of simple samples is 0.2.

## Dataset
We provide one processed dataset: Netease.
* `user_bundle_train.txt`
  * Train file.
  * Each line is 'userID\t bundleID\n'.
  * Every observed interaction means user u once interacted bundle b.

* `user_item.txt`
  * Train file.
  * Each line is 'userID\t itemID\n'.
  * Every observed interaction means user u once interacted item i. 

* `bundle_item.txt`
  * Train file.
  * Each line is 'bundleID\t itemID\n'.
  * Every entry means bundle b contains item i.

* `Netease_data_size.txt`
  * Assist file.
  * The only line is 'userNum\t bundleNum\t itemNum\n'.
  * The triplet denotes the number of users, bundles and items, respectively.

* `user_bundle_tune.txt`
  * Tune file.
  * Each line is 'userID\t bundleID\n'.
  * Every observed interaction means user u once interacted bundle b.

* `user_bundle_test.txt`
  * Test file.
  * Each line is 'userID\t bundleID\n'.
  * Every observed interaction means user u once interacted bundle b.
  
