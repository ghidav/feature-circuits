#!/bin/bash

python circuit.py \
    --model google/gemma-2-2b \
    --num_examples 100 \
    --batch_size 10 \
    --dataset rc_train \
	--node_threshold 0.1 \
	--edge_threshold 0.01 \
	--aggregation none \
    --example_length 6 \
    --dict_id id #10
    