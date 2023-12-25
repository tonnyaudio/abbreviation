#!/bin/bash


python predict.py \
--ranker_model glm_v2_4e_ranker_context_extract_all_top12_add_s_4_accu_512_20_contrastive_context_best.pth \
--batch_size 32 \
--num_beams 32 \
--top_k 12 \
--file data/final_dataset_gen_test_context.txt \
--file2  eval/can_g_topk12_2prompt_context_test_bs32t05p1_2.txt \
--context




