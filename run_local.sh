#!/bin/bash
export PYTHONUNBUFFERED=1
python3 run_eager_pretraining.py \
    --train-batch-size 32 \
    --train-data-part 1 \
    --learning_rate 0.001 \
    --num_hidden_layers 24 \
    --num_attention_heads 16 \
    --max_position_embeddings 512 \
    --seq_length 128 \
    --vocab_size 30522 \
    --type_vocab_size 2 \
    --attention_probs_dropout_prob 0.1 \
    --hidden_dropout_prob 0.1 \
    --max_predictions_per_seq 20 \
    --ofrecord_path ./data/wiki_ofrecord_seq_len_128 \
    2>&1 | tee bert_eager_pretrain.log

    # parser.add_argument("--seq_length", type=int, default=512)
    # parser.add_argument("--max_predictions_per_seq", type=int, default=80)
    # parser.add_argument("--num_hidden_layers", type=int, default=24)
    # parser.add_argument("--num_attention_heads", type=int, default=16)
    # parser.add_argument("--max_position_embeddings", type=int, default=512)
    # parser.add_argument("--type_vocab_size", type=int, default=2)
    # parser.add_argument("--vocab_size", type=int, default=30522)
    # parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.1)
    # parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)
    # parser.add_argument("--hidden_size_per_head", type=int, default=64)