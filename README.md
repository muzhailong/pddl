# BERT

Training BERT on [Wikipedia](https://dumps.wikimedia.org/) and [BookCorpus](http://yknzhu.wixsite.com/mbweb) for Pre-training and [SQuAD](https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/squad_dataset_tools.tgz) for Fine-tuning using [OneFlow](https://github.com/Oneflow-Inc/oneflow#install-with-pip-package)

> BERT 2018 BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
> Paper URL : https://arxiv.org/abs/1810.04805


# oneflow
> https://github.com/Oneflow-Inc/oneflow
> https://github.com/Oneflow-Inc/models

## Introduction
多机多卡混合并行方式训练（数据并行、模型并行、流水并行）

## 使用

### 本地训练
```shell
./run_local.sh
```

### 分布式训练
修改脚本中的node_rank值，然后<h4>分别</h4>在多个节点上执行下述命令
```shell
./run_ppl.sh
```