export CUDA_VISIBLE_DEVICES=0

python main.py \
    --do_train \
    --train_file data/glm_train_context.json \
    --validation_file data/glm_val_context.json \
    --preprocessing_num_workers 10 \
    --prompt_column context \
    --response_column target \
    --overwrite_cache \
    --model_name_or_path output/chatglm-v2_ptrain_model \
    --output_dir output/adgen-chatglm2-6b-lora_ptrain_fintune_context \
    --overwrite_output_dir \
    --max_source_length 200 \
    --max_target_length 32 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --predict_with_generate \
    --max_steps 100 \
    --logging_steps 50 \
    --save_steps 10000 \
    --learning_rate 2e-5 \
    --lora_r 64 \
    --model_parallel_mode True
    
