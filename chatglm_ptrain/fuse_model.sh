#!/bin/bash
# 融合预训练模型
python fuse_model.py --model model/chatglm2-6b --peft_model output/adgen-chatglm2-6b-lora_ptrain/checkpoint-10000  --output_model output/chatglm-v2_ptrain_model

# 融合微调模型
# python fuse_model.py --model output/chatglm-v2_ptrain_model --peft_model output/adgen-chatglm2-6b-lora_ptrain_fintune/checkpoint-10000  --output_model output/chatglm-v2_ptrain_fintune

# 融合微调模型 context
# python fuse_model.py --model output/chatglm-v2_ptrain_model --peft_model output/adgen-chatglm2-6b-lora_ptrain_fintune_context/checkpoint-10000  --output_model output/chatglm-v2_ptrain_fintune_context