#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
python run_ranker.py --version 2 --batch_size 4 --num_epochs 20 --gradient_accumulation_steps 512 --lr 4e-5 --top_k 12 --context --wandb --mode add_s

