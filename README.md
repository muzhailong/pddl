# BERT

Training BERT on [Wikipedia](https://dumps.wikimedia.org/) and [BookCorpus](http://yknzhu.wixsite.com/mbweb) for Pre-training and [SQuAD](https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/squad_dataset_tools.tgz) for Fine-tuning using [OneFlow](https://github.com/Oneflow-Inc/oneflow#install-with-pip-package)

> BERT 2018 BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
> Paper URL : https://arxiv.org/abs/1810.04805


# oneflow
> https://github.com/Oneflow-Inc/oneflow
> https://github.com/Oneflow-Inc/models

## Introduction
多机多卡混合并行方式训练（数据并行、模型并行、流水并行）；当前项目采用oneflow计算框架，对bert模型进行按层进行划分，进而实现混合并行训练

## 未来展望
- 支持GPT2、Robart等主流模型
- 更加灵活的并行训练
- 其他模型切分方式

## 使用
### 分布式训练
```shell
./run_ppl.sh ${node_rank}
```
注意：在oneflow中node_rank和local_rank是分开编码的

## 参数
参数名 | 描述 | example
-|-|-|
nproc_per_node | 每个节点的进程数目 | 3
node_rank | 节点的rank | 0
master_addr | master的ip地址 | cn1（可以使用域名）
master_port | master端口 | 7788
num_hidden_layers | encoder layer的层数 | 24
num_attention_heads | attention的头数，默认每一个头是64维 | 16
gpu_num_per_node | 每个节点使用的GPU数量 | 3
grad-acc-steps | 梯度累积 | 3
seq_length | 序列长度 | 128
train-global-batch-size | 全局batch_size | 24
learning_rate | 学习率 | 0.00005
nums_split | 模型切分数目 | 3
ofrecord_path | 训练数据集目录 | ../data/wiki_ofrecord_seq_len_128
strategy | 模型并行策略 | intra_first or inter_first

## 并行策略

根据模型参数量均分划分到多个节点上（以encoder layer为基本划分单位）

<strong>intra_first</strong>：模型并行优先放置的节点内（机器内的多个GPU上，节点间数据并行）

<strong>inter_first</strong>: 模型并行优先放置在节点间 （节点内数据并行）

## 实验
### 实验环境

服务器：Intel(R) Xeon(R) Silver 4216 CPU @ 2.10GHz

GPU：Tesla M40（24G）

oneflow：0.6.0+cu111

### 实验结果
并行策略 | 参数量 | GPU数量 | 单个step执行时间（s） |
-|-|-|-|
intra_first |  24层16头 | 双机双卡（4GPU） | 1.20 |
inter_first | 24层16头 | 双机双卡（4GPU） | 0.80 |
intra_first |  24层32头 | 三机三卡（9GPU） | 5.25 |
inter_first | 24层32头 | 三机三卡（9GPU） | 2.84 |
intra_first |  48层32头 | 四机四卡（16GPU） | 10.96 |
inter_first | 48层32头 | 四机四卡（16GPU） | 5.23 |

![结果](https://github.com/muzhailong/pddl/blob/master/images/result.jpg?raw=true)
