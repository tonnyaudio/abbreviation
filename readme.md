
##  直接推理见 文件夹 inference 

 

## 环境说明

几个主要的依赖：

- python=3.10.12 
- pytorch=2.0.1
- transformers=4.27.1
- datasets
- jieba
- sklearn
- seqeval
- wandb


## 总体框架
    一 ： 对chatglm进行 关于缩略词的预训练和微调
    二 ： 训练缩略词类型分类器
    三 ： 生成缩略词候选，构造评估阶段数据，训练评估模型


一 ： 对chatglm进行 关于缩略词的预训练和微调


    文件夹 chatglm_ptrain

        data：
            预训练数据(不包含 上下文)：
                ptrain.json ： "context":prompt模板 ：全称的简称是：   "target": 简称 
                pval.json
                
            微调数据：
                包含上下文
                glm_train_context.json  "context"： 全称+上下文+类型的 prompt模板  "target":简称 
                glm_val_context.json

        model：
            chatglm2-6b https://huggingface.co/THUDM/chatglm2-6b

        1 使用预训练数据集 对chatglm预训练  运行 ptrain.sh
            --do_train \   
            --train_file data/ptrain.json \  训练集
            --validation_file data/pval.json \  验证集
            --preprocessing_num_workers 10 \
            --prompt_column context \      prompt名称
            --response_column target \     标签名称
            --overwrite_cache \
            --model_name_or_path model/chatglm2-6b \   模型路径
            --output_dir output/adgen-chatglm2-6b-lora_ptrain \  输出路径
            --overwrite_output_dir \
            --max_source_length 200 \
            --max_target_length 32 \
            --per_device_train_batch_size 64 \
            --per_device_eval_batch_size 64 \
            --gradient_accumulation_steps 1 \
            --predict_with_generate \
            --max_steps 10000 \       总步数
            --logging_steps 50 \
            --save_steps 100 \
            --learning_rate 2e-5 \  
            --lora_r 64 \
            --model_parallel_mode True   \ 可多卡训练
            
        2 将更新的参数与原模型融合       运行 fuse_model.sh
            将chatglm2-6b 与 进行预训练lora更新的参数进行融合
            --model model/chatglm2-6b    加载模型
            --peft_model output/adgen-chatglm2-6b-lora_ptrain/checkpoint-10000    微调参数
            --output_model output/chatglm-v2_ptrain_model    输出模型
        
        3 使用微调数据集 对预训练的chatglm微调 运行 train.sh
            使用上一步输出的融合参数的模型 进行微调 
            
            
        4 将更新的参数与模型融合，        更改为融合微调模型那一行，运行 fuse_model.sh
             将模型输出到output_model output/chatglm-v2_ptrain_fintune



二 ： 对全称 按照类型 进行预测分类


    文件夹 type_classify

        data：
        训练数据
        train_context_type_fenci.json
            "full": 经过分词的全称，以空格连接   "label": 全称缩略词类型, "context": 全称上下文
        val_context_type_fenci.json
        test_context_type_fenci.json

    
        model：
            mengzi-t5-base https://huggingface.co/Langboat/mengzi-t5-base
    

        1 使用训练数据集 对mengzi-t5微调，得到微调模型  运行 t5_huggingface.sh
        
            python t5_huggingface.py 
            --model model/mengzi-t5-base  使用模型
            --batch_size 32  
            --num_epochs 10  训练轮数
            --lr 5e-5  学习率
            --context  
        
            模型输出到 hugging_result/T5best_context_fenci  /T5best_fenci
        2 运行 t5_huggingface_inference.sh  生成 测试集预测结果 
        
            测试集结果输出到 ../generate_evaluation/data/classify_data/test_predict.txt

    

三 ： 对全称按照分类构建prompt，生成缩略词候选，并且构建数据集进行训练评估模型
    
    文件夹 generate_evaluation


        data：
            生成阶段数据
                final_dataset_gen_test_context.txt
                    数据包含：src, target, context, context_with_placeholder
                final_dataset_gen_train_context.txt
                final_dataset_gen_val_context.txt
                classify_data
                    test_predict.txt
                        全称  0/1   表示全称的缩略词类型 0 为截取型 1 为缩合型
                    test.txt
                    train.txt
                    val.txt

            评估阶段数据
            rank_data
                t5_v2_ranker_context_extract_all_truncate150_top12_test_score_withtrue_last_rule_glmv2_ptrain2_fintune_bs32t05p1_2.txt
                    
                t5_v2_ranker_context_extract_all_truncate150_top12_train_score_withtrue_last_rule_glmv2_ptrain2_fintune_bs32t05p1_2.txt
                t5_v2_ranker_context_extract_all_truncate150_top12_val_score_withtrue_last_rule_glmv2_ptrain2_fintune_bs32t05p1_2.txt
        
        config.py 评估模型的配置项
        model.py 评估模型代码
        preprocess.py 数据处理代码
        train_eval.py 训练和评估的工具函数
        eval/ 存放模型评估产生的模型预测，目录下给了当前数据集中的一些例子，主要是生成模型生成的top_k 个 candidates

        model：
            chinese-macbert-base https://huggingface.co/hfl/chinese-macbert-base
            

        1 使用预训练模型生成数据集的缩略词k个候选   运行 gen_eval_constrain.sh
            --model_name ../chatglm_ptrain/output/chatglm-v2_ptrain2_fintune    选择生成模型
            --top_k 12 
            --num_beams 32 
            --batch_size 16
            --file data/final_dataset_gen_test_context.txt         输入数据
            --constrain   
            --data_set test_predict                          缩略词类型数据集设置
            
            输出位置 eval/can_g_topk12_2prompt_context_test_bs32t05p1_2.txt
            
        2 将生成的缩略词候选构造 评估数据集       
            运行
                get_rank_score_withtrue_data.py
                输入： 
                    data/final_dataset_gen_{}.txt
                    eval/can_g_topk12_2prompt_{}_bs32t05p1_2.txt
                输出：
                    data/rank_data/glm_v2_ranker_top12_{}_bs32t05p1_2.txt
            
        3 使用评估数据集训练评估模型             运行 train_ranker.sh
            python run_ranker.py 
                --version 2 
                --batch_size 4 
                --num_epochs 20 
                --gradient_accumulation_steps 512 
                --lr 4e-5 
                --top_k 12        /生成候选数
                --context        
                --wandb 
                --mode add_s      /add_s 为加上下文  none 为不加上下文
                
                输入： 上一步输出的数据集
                    data/rank_data/glm_v2_ranker_top12_train_bs32t05p1_2.txt
                    data/rank_data/glm_v2_ranker_top12_test_bs32t05p1_2.txt
                    data/rank_data/glm_v2_ranker_top12_val_bs32t05p1_2.txt
                    
                输出： 
                    训练好的模型
                    glm_v2_4e_ranker_extract_all_top12_none_4_accu_512_20_contrastive_best.pth
                    glm_v2_4e_ranker_context_extract_all_top12_add_s_4_accu_512_20_contrastive_context_best.pth
                    
                    
          4 使用评估模型进行推理  运行 predict.sh
          
              # # 带context
                python predict.py \
                --ranker_model glm_v2_4e_ranker_context_extract_all_top12_add_s_4_accu_512_20_contrastive_context_best.pth \
                推理模型
                --batch_size 32 \
                --num_beams 32 \
                --top_k 12 \
                --file data/final_dataset_gen_test_context.txt \   输入 全称及上下文
                --file2  eval/can_g_topk12_2prompt_context_test_bs32t05p1_2.txt \    输入生成候选
                --context

