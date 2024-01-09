

## Environment Description

Several major dependencies:

- python=3.10.12 
- pytorch=2.0.1
- transformers=4.27.1
- datasets
- jieba
- sklearn
- seqeval
- wandb


## Overall Framework

Pretraining and Fine-tuning of chatglm for acronyms

Training Acronym Type Classifier

Generating Acronym Candidates, Constructing Evaluation Data, and Training Evaluation Model

Pretraining and Fine-tuning of chatglm for acronyms

Folder: chatglm_ptrain

Data:
Pretraining Data (without context):
ptrain.json: "context": prompt template for full name abbreviation: "target": abbreviation
pval.json
Fine-tuning Data:
Includes context
glm_train_context.json: "context": prompt template with full name, context, and type: "target": abbreviation
glm_val_context.json

        model：
            chatglm2-6b https://huggingface.co/THUDM/chatglm2-6b

        1 To pretrain chatglm using the pretraining dataset, run the ptrain.sh script with the following command:


            --do_train \   
            --train_file data/ptrain.json \  
            --validation_file data/pval.json \ 
            --preprocessing_num_workers 10 \
            --prompt_column context \     
            --response_column target \     
            --overwrite_cache \
            --model_name_or_path model/chatglm2-6b \   
            --output_dir output/adgen-chatglm2-6b-lora_ptrain \  
            --overwrite_output_dir \
            --max_source_length 200 \
            --max_target_length 32 \
            --per_device_train_batch_size 64 \
            --per_device_eval_batch_size 64 \
            --gradient_accumulation_steps 1 \
            --predict_with_generate \
            --max_steps 10000 \       
            --logging_steps 50 \
            --save_steps 100 \
            --learning_rate 2e-5 \  
            --lora_r 64 \
            --model_parallel_mode True   
            
        2 Integrate the updated parameters with the original model       run fuse_model.sh
            Integrate Chatglm2-6B with the parameters of pre-training LoRA updates
            --model model/chatglm2-6b    
            --peft_model output/adgen-chatglm2-6b-lora_ptrain/checkpoint-10000    
            --output_model output/chatglm-v2_ptrain_model    
        
        3 Use fine -tuning data set to fine -tune the pre -trained Chatglm run train.sh
            Use the model of the fusion parameter output in the previous step to fine -tune
            
            
        4 Form the updated parameters with the model and change to the line of fusion and fine -tuning model，run fuse_model.sh
             Output the model to output_model output/chatglm-v2_ptrain_fintune



二 ： Predict the full name according to the type

    folder type_classify

        data：
        Training data
        train_context_type_fenci.json
            "full": After the full name of the word segmentation, connect "label" with space: full name shrinking word type, "context": full name context
        val_context_type_fenci.json
        test_context_type_fenci.json

    
        model：
            mengzi-t5-base https://huggingface.co/Langboat/mengzi-t5-base
    

        1 Use the training data set to fine-tune Mengzi-T5 to get a fine-tuning model run t5_huggingface.sh
        
            python t5_huggingface.py 
            --model model/mengzi-t5-base  
            --batch_size 32  
            --num_epochs 10  
            --lr 5e-5  
            --context  
        
            Model output to hugging_result/T5best_context_fenci  /T5best_fenci
        2 run t5_huggingface_inference.sh  Generate test set prediction results
        
            The test set result is output to ../generate_evaluation/data/classify_data/test_predict.txt

    

三 ： Construct a Prompt in the full name according to the classification, generate tap components, and build a data set for training and evaluation model
    
    folder generate_evaluation


        data：
            Generate data
                final_dataset_gen_test_context.txt
                    Data inclusion：src, target, context, context_with_placeholder
                final_dataset_gen_train_context.txt
                final_dataset_gen_val_context.txt
                classify_data
                    test_predict.txt
                        Full name 0/1 indicates that the full name zoom type type 0 is intercepting type 1 as a contraction type
                    test.txt
                    train.txt
                    val.txt

            Evaluation phase data
            rank_data
                t5_v2_ranker_context_extract_all_truncate150_top12_test_score_withtrue_last_rule_glmv2_ptrain2_fintune_bs32t05p1_2.txt
                    
                t5_v2_ranker_context_extract_all_truncate150_top12_train_score_withtrue_last_rule_glmv2_ptrain2_fintune_bs32t05p1_2.txt
                t5_v2_ranker_context_extract_all_truncate150_top12_val_score_withtrue_last_rule_glmv2_ptrain2_fintune_bs32t05p1_2.txt
        
        config.py The configuration item of the evaluation model
        model.py Evaluation model code
        preprocess.py Data processing code
        train_eval.py The tool function of training and evaluation
        eval/ The model prediction generated by the storage model assessment, some examples of the current data set are given in the directory, mainly to generate the TOP_K CANDIDATES generated by the model
        model：
            chinese-macbert-base https://huggingface.co/hfl/chinese-macbert-base
            

        1 Use the pre -training model to generate a thumbnail of the dataset k.   run gen_eval_constrain.sh
            --model_name ../chatglm_ptrain/output/chatglm-v2_ptrain2_fintune    
            --top_k 12 
            --num_beams 32 
            --batch_size 16
            --file data/final_dataset_gen_test_context.txt        
            --constrain   
            --data_set test_predict                         
            
            Output position eval/can_g_topk12_2prompt_context_test_bs32t05p1_2.txt
            
        2 The data set will be generated by the selection candidates     
            run
                get_rank_score_withtrue_data.py
                input： 
                    data/final_dataset_gen_{}.txt
                    eval/can_g_topk12_2prompt_{}_bs32t05p1_2.txt
                output：
                    data/rank_data/glm_v2_ranker_top12_{}_bs32t05p1_2.txt
            
        3 Use evaluation data set training evaluation model             run train_ranker.sh
            python run_ranker.py 
                --version 2 
                --batch_size 4 
                --num_epochs 20 
                --gradient_accumulation_steps 512 
                --lr 4e-5 
                --top_k 12        
                --context        
                --wandb 
                --mode add_s      
                
                input： The data set output in the previous step
                    data/rank_data/glm_v2_ranker_top12_train_bs32t05p1_2.txt
                    data/rank_data/glm_v2_ranker_top12_test_bs32t05p1_2.txt
                    data/rank_data/glm_v2_ranker_top12_val_bs32t05p1_2.txt
                    
                output： 
                    Trained model
                    glm_v2_4e_ranker_extract_all_top12_none_4_accu_512_20_contrastive_best.pth
                    glm_v2_4e_ranker_context_extract_all_top12_add_s_4_accu_512_20_contrastive_context_best.pth
                    
                    
          4 Use the evaluation model for reasoning  run predict.sh
          
              # # 
                python predict.py \
                --ranker_model glm_v2_4e_ranker_context_extract_all_top12_add_s_4_accu_512_20_contrastive_context_best.pth \
            
                --batch_size 32 \
                --num_beams 32 \
                --top_k 12 \
                --file data/final_dataset_gen_test_context.txt \  
                --file2  eval/can_g_topk12_2prompt_context_test_bs32t05p1_2.txt \    
                --context

