#!/usr/bin/python3
from typing import Dict
import time
####################################################################
import sys
import os
import matplotlib.pyplot as plt
plt.waitforbuttonpress()

from oneflow import sbp
import oneflow
from oneflow.framework.function_util import oneflow_function_config
from oneflow.nn.modules.tensor_ops import cpu
sys.path.append(os.path.abspath("../"))
####################################################################
from config import str2bool
import config
from utils.optimizer import build_optimizer
from utils.ofrecord_data_utils import OfRecordDataLoader
from bert.model import PipelineModule
import numpy as np
import oneflow as flow
from oneflow.nn.parallel import DistributedDataParallel as ddp
from oneflow import nn


BROADCAST = [flow.sbp.broadcast]
# P0 = flow.placement("cuda", {0: [0, 1]})
# P1 = flow.placement("cuda", {1: [0, 1]})
# P0 = flow.placement("cuda", {0: [0],1:[0]})
# P1 = flow.placement("cuda", {0: [1],1:[1]})

def print_0(*params):
    if flow.env.get_rank() == 0:
        print(*params)


def get_config():
    parser = config.get_parser()
    # pretrain bert config
    parser.add_argument(
        "--ofrecord_path",
        type=str,
        default="./data/wiki_ofrecord_seq_len_128",
        help="Path to ofrecord dataset",
    )
    parser.add_argument(
        "--train-dataset-size",
        type=int,
        default=10000000,
        help="dataset size of ofrecord",
    )
    parser.add_argument(
        "--train-data-part", type=int, default=1, help="data part num of ofrecord"
    )
    parser.add_argument(
        "--train-batch-size", type=int, default=8, help="Training batch size"
    )
    parser.add_argument(
        "--val-batch-size", type=int, default=32, help="Validation batch size"
    )
    parser.add_argument(
        "--train-global-batch-size",
        type=int,
        default=None,
        dest="train_global_batch_size",
        help="train batch size",
    )
    parser.add_argument(
        "--val-global-batch-size",
        type=int,
        default=None,
        dest="val_global_batch_size",
        help="val batch size",
    )

    parser.add_argument("-e", "--epochs", type=int,
                        default=1, help="Number of epochs")

    parser.add_argument(
        "--with-cuda",
        type=bool,
        default=True,
        help="Training with CUDA: true, or false",
    )
    parser.add_argument(
        "--cuda_devices", type=int, nargs="+", default=None, help="CUDA device ids"
    )
    parser.add_argument(
        "--optim_name", type=str, default="adamw", help="optimizer name"
    )
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate of adam")
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight_decay of adam"
    )
    parser.add_argument(
        "--loss_print_every_n_iters",
        type=int,
        default=20,
        help="Interval of training loss printing",
    )
    parser.add_argument(
        "--val_print_every_n_iters",
        type=int,
        default=20,
        help="Interval of evaluation printing",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints",
        help="Path to model saving",
    )
    parser.add_argument(
        "--grad-acc-steps", type=int, default=1, help="Steps for gradient accumulation"
    )
    parser.add_argument(
        "--nccl-fusion-threshold-mb",
        type=int,
        default=16,
        dest="nccl_fusion_threshold_mb",
        help="NCCL fusion threshold megabytes, set to 0 to compatible with previous version of OneFlow.",
    )
    parser.add_argument(
        "--nccl-fusion-max-ops",
        type=int,
        default=24,
        dest="nccl_fusion_max_ops",
        help="Maximum number of ops of NCCL fusion, set to 0 to compatible with previous version of OneFlow.",
    )
    parser.add_argument(
        "--use_ddp",
        type=str2bool,
        nargs="?",
        const=True,
        help="Whether to use use fp16",
    )
    parser.add_argument(
        "--use_consistent",
        type=str2bool,
        nargs="?",
        const=True,
        help="Whether to use use consistent",
    )
    parser.add_argument(
        "--metric-local",
        type=str2bool,
        default=False,
        nargs="?",
        const=True,
        dest="metric_local",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--train_steps",
        type=int,
        default=10
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="intra_first"
    )
    parser.add_argument(
        "--nums_split",
        type=int,
        default=2
    )

    args = parser.parse_args()
    return args


args = get_config()
args.hidden_size = 64 * args.num_attention_heads
args.intermediate_size = 4 * args.hidden_size


def sp_model():
    '''
    split,placement
    生成多个GPU的sbp和placement
    '''
    embed_params = args.vocab_size * args.hidden_size + \
        args.max_position_embeddings * args.hidden_size + \
        args.type_vocab_size * args.hidden_size + \
        args.seq_length
    layer_params = args.hidden_size * args.num_attention_heads * 64 * 3 + \
        args.hidden_size * args.hidden_size + args.seq_length * 2 + \
        args.hidden_size * args.intermediate_size * 2
    out_params = args.hidden_size ** 2 + args.hidden_size * 2 + \
        args.hidden_size * args.vocab_size
    num_layer = args.num_hidden_layers
    # 横向均衡切分，保持最小切分单元

    # all_params = embed_params + layer_params * num_layer + out_params
    def split_fn(p1, pk, pn, k, n):
        res = [{0: 0, 1: 0, 2: 0}for _ in range(n)]
        if n == 1:
            res[0][0], res[0][1], res[0][2] = 1, k, 1
        elif n == 2:
            s1, s2 = p1, pn
            res[0][0] = 1
            res[1][2] = 1
            for _ in range(k):
                if s1 < s2:
                    s1 += pk
                    res[0][1] += 1
                else:
                    s2 += pk
                    res[1][1] += 1
        else:
            res[0][0] = 1
            res[n-1][2] = 1
            d = None
            for i in range(k+1):
                for j in range(k+1):
                    if i + j + n-2 > k:
                        break
                    s1 = p1 + i * pk
                    s2 = pn + j * pk
                    tmp = (k-i-j)//(n-2)
                    smi = tmp * pk
                    smx = (tmp if tmp*(n-2) == (k-i-j) else tmp + 1) * pk
                    d_mx = max(smx-smi, abs(s1-smi), abs(s1-smx),
                               abs(s2-smi), abs(s2-smx), abs(s1-s2))
                    if d is None or d_mx < d:
                        d = d_mx
                        res[0][1] = i
                        res[n-1][1] = j
                        for ti in range(n-2):
                            res[ti+1][1] = tmp
                        for ti in range(k-i-j-tmp*(n-2)):
                            res[ti+1][1] += 1
        return res

    def placement_fn(nums_node, gpu_num_per_node, nums_split, strategy):
        if (nums_node * gpu_num_per_node) % nums_split != 0:
            raise Exception(f"{nums_split} is not supported!!!")
        pos = [[]]
        if strategy == 'intra_first':
            for inter_rank in range(nums_node):
                for intra_rank in range(gpu_num_per_node):
                    if len(pos[-1]) < nums_split:
                        pos[-1].append((inter_rank, intra_rank))
                    else:
                        pos.append([(inter_rank, intra_rank)])
        elif strategy == 'inter_first':
            for intra_rank in range(gpu_num_per_node):
                for inter_rank in range(nums_node):
                    if len(pos[-1]) < nums_split:
                        pos[-1].append((inter_rank, intra_rank))
                    else:
                        pos.append([(inter_rank, intra_rank)])
        else:
            raise Exception(f"{strategy} is not supported!!!!")
        # convert sbp
        res = [{} for _ in range(nums_split)]
        for i in range(nums_split):
            for lt in pos:
                inter_rank, intra_rank = lt[i]
                if inter_rank not in res[i]:
                    res[i][inter_rank] = []
                res[i][inter_rank].append(intra_rank)
        return res

    if (flow.env.get_node_size() * args.gpu_num_per_node) % args.nums_split != 0:
        raise Exception(f"nums_split Error!")
    print_0(
        f"嵌入层参数量:{embed_params}, encoder layer参数量:{embed_params}，输出层参数量:{out_params}")
    # print_0(f"总共参数量：{embed_params+embed_params*num_layer+out_params}")
    
    split_res = split_fn(embed_params, layer_params,
                         out_params, num_layer, args.nums_split)
    placement_tmp=placement_fn(flow.env.get_node_size(), args.gpu_num_per_node, args.nums_split, args.strategy)
    placement_gpu = [flow.placement("cuda", p) for p in placement_tmp]

    print_0("split:", split_res)
    print_0("placement:", placement_gpu)

    return split_res, placement_gpu


def get_masked_lm_loss(
    logit,
    masked_lm_positions,
    masked_lm_labels,
    label_weights,
    max_predictions_per_seq,
    mlm_criterion,
):

    # gather valid position indices
    logit = flow.gather(
        logit,
        index=masked_lm_positions.unsqueeze(2).expand(-1, -1, args.vocab_size),
        dim=1,
    )

    logit = flow.reshape(logit, [-1, args.vocab_size])
    label_id = flow.reshape(masked_lm_labels, [-1])

    # The `positions` tensor might be zero-padded (if the sequence is too
    # short to have the maximum number of predictions). The `label_weights`
    # tensor has a value of 1.0 for every real prediction and 0.0 for the
    # padding predictions.
    pre_example_loss = mlm_criterion(logit, label_id)
    pre_example_loss = flow.reshape(
        pre_example_loss, [-1, max_predictions_per_seq])
    numerator = flow.sum(pre_example_loss * label_weights)
    denominator = flow.sum(label_weights) + 1e-5
    loss = numerator / denominator
    return loss


class PipelineGraph(nn.Graph):
    def __init__(self, base_model, optimizer, data_loader, in_placement, out_placement):
        super().__init__()
        self.base_model = base_model
        for i in range(len(self.base_model.m_stage)):
            self.base_model.m_stage[i].config.stage_id = i
        # self.base_model.m_stage0.config.stage_id = 0
        # self.base_model.m_stage1.config.stage_id = 1

        self.ns_criterion = nn.CrossEntropyLoss(reduction="mean")
        self.mlm_criterion = nn.CrossEntropyLoss(reduction="none")
        self.masked_lm_criterion = get_masked_lm_loss

        self._train_data_loader = data_loader
        self.in_placement = in_placement
        self.out_placement = out_placement
        self.config.set_gradient_accumulation_steps(args.grad_acc_steps)
        self.add_optimizer(optimizer)

    def build(self):
        (
            input_ids,
            next_sentence_labels,
            input_mask,
            segment_ids,
            masked_lm_ids,
            masked_lm_positions,
            masked_lm_weights,
        ) = self._train_data_loader()

        input_ids = input_ids.to_consistent(
            self.in_placement, sbp=flow.sbp.split(0))
        input_mask = input_mask.to_consistent(
            self.in_placement, sbp=flow.sbp.split(0))
        segment_ids = segment_ids.to_consistent(
            self.in_placement, sbp=flow.sbp.split(0))

        next_sentence_labels = next_sentence_labels.to_consistent(
            self.out_placement, sbp=flow.sbp.split(0))
        masked_lm_ids = masked_lm_ids.to_consistent(
            self.out_placement, sbp=flow.sbp.split(0))
        masked_lm_positions = masked_lm_positions.to_consistent(
            self.out_placement, sbp=flow.sbp.split(0))
        masked_lm_weights = masked_lm_weights.to_consistent(
            self.out_placement, sbp=flow.sbp.split(0))

        prediction_scores, seq_relationship_scores = self.base_model(
            input_ids, segment_ids, input_mask
        )
        # print(prediction_scores, seq_relationship_scores)
        # 2-1. loss of is_next classification result
        next_sentence_loss = self.ns_criterion(
            seq_relationship_scores.reshape(-1, 2),
            next_sentence_labels.reshape(-1)
        )
    #     get_masked_lm_loss(
    #     logit,
    #     masked_lm_positions,
    #     masked_lm_labels,
    #     label_weights,
    #     max_predictions_per_seq,
    #     mlm_criterion,
    # ):
        masked_lm_loss = self.masked_lm_criterion(
            prediction_scores, masked_lm_positions,
            masked_lm_ids, masked_lm_weights,
            args.max_predictions_per_seq, self.mlm_criterion
        )

        total_loss = next_sentence_loss + masked_lm_loss
        total_loss.backward()
        return (
            total_loss,
            masked_lm_loss,
            next_sentence_loss,
        )
split_res, placement_res = sp_model()
print_0(f"global_batch_size:{args.train_global_batch_size}")
print_0("Creating Dataloader")
train_data_loader = OfRecordDataLoader(
    ofrecord_dir=args.ofrecord_path,
    mode="train",
    dataset_size=args.train_dataset_size,
    batch_size=args.train_global_batch_size,
    data_part_num=args.train_data_part,
    seq_length=args.seq_length,
    max_predictions_per_seq=args.max_predictions_per_seq,
    consistent=args.use_consistent,
)
print_0("Building BERT Model")
base_model = PipelineModule(
    split_res,
    placement_res,
    args.vocab_size,
    args.seq_length,
    args.hidden_size,
    args.num_hidden_layers,
    args.num_attention_heads,
    args.intermediate_size,
    hidden_act=nn.GELU(),
    hidden_dropout_prob=args.hidden_dropout_prob,
    attention_probs_dropout_prob=args.attention_probs_dropout_prob,
    max_position_embeddings=args.max_position_embeddings,
    type_vocab_size=args.type_vocab_size,
)
optimizer = build_optimizer(
    args.optim_name,
    base_model,
    args.lr,
    args.weight_decay,
    weight_decay_excludes=["bias", "LayerNorm", "layer_norm"],
    clip_grad_max_norm=1,
    clip_grad_norm_type=2.0,
)


print_0("Building Graph")
graph_pipeline = PipelineGraph(
    base_model,
    optimizer,
    train_data_loader,
    placement_res[0],
    placement_res[-1]
)
# graph_pipeline.debug(1)
# Train

print_0("Start warmuping!")
base_model.train()
st=time.time()
for step in range(args.warmup_steps):
    loss, mlm_loss, nsp_loss = graph_pipeline()
    # print(f"loss:{loss},mlm_loss:{mlm_loss},nsp_loss:{nsp_loss}")
flow._oneflow_internal.eager.multi_client.Sync()
ed=time.time()
print_0(f"warmup time per step:{(ed-st)/args.warmup_steps}s")

print_0("start training!!!!")
st = time.time()
for step in range(args.train_steps):
    loss, mlm_loss, nsp_loss = graph_pipeline()
    # print(f"loss:{loss.numpy().mean()},mlm_loss:{mlm_loss.numpy().mean()},nsp_loss:{nsp_loss.numpy().mean()}")
flow._oneflow_internal.eager.multi_client.Sync()
ed = time.time()
print_0(f"run time per step:{(ed-st)/args.train_steps}s")