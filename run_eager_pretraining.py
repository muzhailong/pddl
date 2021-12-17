#!/usr/bin/python3
import time
from functools import partial

import numpy as np
import oneflow as flow
from oneflow.cuda import device_count
from oneflow.nn.parallel import DistributedDataParallel as ddp
from oneflow import nn

from modeling import BertForPreTraining, PipelineModule
from utils.ofrecord_data_utils import OfRecordDataLoader
from utils.lr_scheduler import PolynomialLR
from utils.optimizer import build_optimizer
from utils.metric import Metric
from utils.comm import ttol, tton
from utils.checkpoint import save_model
import config
from config import str2bool
from typing import Dict

BROADCAST = [flow.sbp.broadcast]
P0 = flow.placement("cuda", {0: [0, 1]})
P1 = flow.placement("cuda", {1: [0, 1]})

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
        "--train-data-part", type=int, default=64, help="data part num of ofrecord"
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

    args = parser.parse_args()
    return args


def pretrain(graph: nn.Graph, metric_local: bool) -> Dict:
    # NOTE(xyliao): when using gradient accmulation, graph call 1 step for 1 mini-batch(n micro-batch)
    next_sent_output, next_sent_labels, loss, mlm_loss, nsp_loss = graph()
    # to local
    next_sent_output = ttol(next_sent_output, metric_local)
    next_sent_labels = ttol(next_sent_labels, metric_local)
    # next sentence prediction accuracy
    correct = (
        next_sent_output.argmax(dim=1)
        .to(dtype=next_sent_labels.dtype)
        .eq(next_sent_labels.squeeze(1))
        .to(dtype=flow.float32)
        .sum()
        .numpy()
        .item()
    )
    pred_acc = np.array(correct / next_sent_labels.nelement())
    return {
        "total_loss": tton(loss.mean(), metric_local),
        "mlm_loss": tton(mlm_loss.mean(), metric_local),
        "nsp_loss": tton(nsp_loss.mean(), metric_local),
        "pred_acc": pred_acc,
    }


args = get_config()
print(args)
# if args.with_cuda:
#     device = flow.device("cuda")
# else:
#     device = flow.device("cpu")
device=flow.device('cpu')


class PipelineGraph(nn.Graph):
    def __init__(self, base_model, optimizer, data_loader):
        super().__init__()
        self.base_model = base_model
        self.base_model.m_stage0.config.stage_id = 0
        self.base_model.m_stage1.config.stage_id = 1

        self.ns_criterion = nn.CrossEntropyLoss(reduction="mean")
        self.mlm_criterion = nn.CrossEntropyLoss(reduction="none")
        self._train_data_loader = data_loader
        self.config.set_gradient_accumulation_steps(2)
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

        next_sentence_labels = next_sentence_labels.to_consistent(P1, sbp=flow.sbp.split(0))
        masked_lm_ids = masked_lm_ids.to_consistent(P1, sbp=flow.sbp.split(0))
        masked_lm_positions = masked_lm_positions.to_consistent(P1, sbp=flow.sbp.split(0))
        masked_lm_weights = masked_lm_weights.to_consistent(P1, sbp=flow.sbp.split(0))
        prediction_scores, seq_relationship_scores = self.base_model(
            input_ids, segment_ids, input_mask
        )
        # 2-1. loss of is_next classification result
        next_sentence_loss = self.ns_criterion(
            seq_relationship_scores.reshape(-1,
                                            2), next_sentence_labels.reshape(-1)
        )

        masked_lm_loss = self.masked_lm_criterion(
            prediction_scores, masked_lm_positions, masked_lm_ids, masked_lm_weights
        )
        total_loss = next_sentence_loss + masked_lm_loss
        total_loss.backward()
        return (
            seq_relationship_scores,
            next_sentence_labels,
            total_loss,
            masked_lm_loss,
            next_sentence_loss,
        )

print("Creating Dataloader")
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

print("Building BERT Model")
hidden_size = 64 * args.num_attention_heads
intermediate_size = 4 * hidden_size
base_model = PipelineModule(
    args.vocab_size,
    args.seq_length,
    hidden_size,
    args.num_hidden_layers,
    args.num_attention_heads,
    intermediate_size,
    nn.GELU(),
    args.hidden_dropout_prob,
    args.attention_probs_dropout_prob,
    args.max_position_embeddings,
    args.type_vocab_size,
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
print("Building Graph")
graph_pipeline = PipelineGraph(base_model, optimizer, train_data_loader)
graph_pipeline.debug(1)
# Train

print("Start training!")
base_model.train()
metric = Metric(
    desc="bert pretrain",
    print_steps=args.loss_print_every_n_iters,
    batch_size=args.train_global_batch_size * args.grad_acc_steps,
    keys=["total_loss", "mlm_loss", "nsp_loss", "pred_acc"],
)
train_total_losses = []
for step in range(len(train_data_loader)):
    bert_outputs = pretrain(graph_pipeline, args.metric_local)
    if flow.env.get_rank() == 0:
        metric.metric_cb(step, epoch=0)(bert_outputs)
        train_total_losses.append(bert_outputs["total_loss"])
