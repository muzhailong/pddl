#!/usr/bin/python3
import time

import numpy as np
import oneflow as flow
from oneflow.nn.parallel import DistributedDataParallel as ddp
from oneflow import nn

from modeling import PipelineModule
from utils.ofrecord_data_utils import OfRecordDataLoader
from utils.lr_scheduler import PolynomialLR
from utils.optimizer import build_optimizer
import config
from config import str2bool
from typing import Dict
import time

BROADCAST = [flow.sbp.broadcast]
P0 = flow.placement("cuda", {0: [0, ], 1: [0]})
P1 = flow.placement("cuda", {0: [1, ], 1: [1]})


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

    args = parser.parse_args()
    return args


args = get_config()
# print_0(args)
# if args.with_cuda:
#     device = flow.device("cuda")
# else:
#     device = flow.device("cpu")
device = flow.device('cpu')


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
    def __init__(self, base_model, optimizer, data_loader):
        super().__init__()
        self.base_model = base_model
        self.base_model.m_stage0.config.stage_id = 0
        self.base_model.m_stage1.config.stage_id = 1

        self.ns_criterion = nn.CrossEntropyLoss(reduction="mean")
        self.mlm_criterion = nn.CrossEntropyLoss(reduction="none")
        self.masked_lm_criterion = get_masked_lm_loss

        self._train_data_loader = data_loader
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

        input_ids = input_ids.to_consistent(P0, sbp=flow.sbp.split(0))
        input_mask = input_mask.to_consistent(P0, sbp=flow.sbp.split(0))
        segment_ids = segment_ids.to_consistent(P0, sbp=flow.sbp.split(0))

        next_sentence_labels = next_sentence_labels.to_consistent(
            P1, sbp=flow.sbp.split(0))
        masked_lm_ids = masked_lm_ids.to_consistent(P1, sbp=flow.sbp.split(0))
        masked_lm_positions = masked_lm_positions.to_consistent(
            P1, sbp=flow.sbp.split(0))
        masked_lm_weights = masked_lm_weights.to_consistent(
            P1, sbp=flow.sbp.split(0))

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
hidden_size = 64 * args.num_attention_heads
intermediate_size = 4 * hidden_size
base_model = PipelineModule(
    P0,
    P1,
    args.vocab_size,
    args.seq_length,
    hidden_size,
    args.num_hidden_layers//2,
    args.num_attention_heads,
    intermediate_size,
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
graph_pipeline = PipelineGraph(base_model, optimizer, train_data_loader)
# graph_pipeline.debug(1)
# Train

print_0("Start training!")
base_model.train()
if flow.env.get_rank() == 0:
    t = []
for step in range(args.warmup_steps):
    if flow.env.get_rank() == 0:
        st = time.time()
    loss, mlm_loss, nsp_loss = graph_pipeline()
    if flow.env.get_rank() == 0:
        ed = time.time()
        t.append((ed-st)*1000)
    # print(f"loss:{loss},mlm_loss:{mlm_loss},nsp_loss:{nsp_loss}")
if flow.env.get_rank() == 0:
    print_0(f"warmup time per step:{sum(t)/len(t)}ms")
    t.clear()

print_0("start training!!!!")
for step in range(args.train_steps):
    if flow.env.get_rank() == 0:
        st = time.time()
    loss, mlm_loss, nsp_loss = graph_pipeline()
    if flow.env.get_rank() == 0:
        ed = time.time()
        t.append((ed-st)*1000)
    # print(f"loss:{loss.numpy().mean()},mlm_loss:{mlm_loss.numpy().mean()},nsp_loss:{nsp_loss.numpy().mean()}")
if flow.env.get_rank() == 0:
    print_0(f"train time per step:{sum(t)/len(t)}ms")
