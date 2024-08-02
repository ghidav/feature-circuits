#!/bin/bash

python circuit_tl.py \
    --model google/gemma-2-2b \
    --num_examples 1024 \
    --batch_size 32 \
    --dataset simple_train \
	--node_threshold 0.1 \
	--edge_threshold 0.01 \
	--aggregation none \
    --dict_id id \