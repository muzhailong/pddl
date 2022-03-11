#!/bin/bash
export PYTHONUNBUFFERED=1
# export NCCL_SHM_DISABLE=0
export NCCL_IB_DISABLE=0
# export NCCL_SOCKET_IFNAME=enp1s0f0
export NCCL_DEBUG=INFO


python3 ./local_train.py \
    --use_consistent False \
    --num_hidden_layers 3 \
    --num_attention_heads 50 \
    --max_position_embeddings 512 \
    --gpu_num_per_node 1 \
    --grad-acc-steps 1 \
    --seq_length 128 \
    --vocab_size 30522 \
    --type_vocab_size 2 \
    --attention_probs_dropout_prob 0.1 \
    --train-data-part 1 \
    --hidden_dropout_prob 0.1 \
    --max_predictions_per_seq 20 \
    --train-global-batch-size 8 \
    --learning_rate 0.00005 \
    --ofrecord_path ../data/wiki_ofrecord_seq_len_128 \
    --nums_split 1 \
    --strategy intra_first \
    2>&1 | tee bert_eager_pretrain.log
