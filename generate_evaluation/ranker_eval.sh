#!/bin/bash

# 修改
python eval_ranker.py \
--model t5_v2_4e_ranker_context_extract_all_truncate150_top12_add_b_4_accu_512_10_epoch_7.pth \
--mode add_b \
--file data/t5_v2_ranker_context_extract_all_truncate150_top12_test.txt \
--batch_size 64 \
--top_k 12

