#!/bin/bash
export PYTHONUNBUFFERED=1
python3 -m oneflow.distributed.launch \
    --nproc_per_node 2 \
    --nnodes 2 \
    --node_rank $1 \
    --master_addr cn1 \
    --master_port 7788 \
    train.py \
    --use_consistent True \
    --num_hidden_layers 24 \
    --num_attention_heads 32 \
    --max_position_embeddings 512 \
    --gpu_num_per_node 2 \
    --grad-acc-steps 2 \
    --seq_length 128 \
    --vocab_size 30522 \
    --type_vocab_size 2 \
    --attention_probs_dropout_prob 0.1 \
    --train-data-part 1 \
    --hidden_dropout_prob 0.1 \
    --max_predictions_per_seq 20 \
    --train-global-batch-size 32 \
    --learning_rate 0.00005 \
    --ofrecord_path ../data/wiki_ofrecord_seq_len_128 \
    2>&1 | tee bert_eager_pretrain.log