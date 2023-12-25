import argparse
import pandas as pd
import numpy as np
from datasets import load_dataset, load_metric
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AutoModelForSeq2SeqLM, T5Tokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os
import evaluate

# 设置 gpu ，加载评估
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


print("loading seqeval\n")
metric = evaluate.load("seqeval")
# metric =evaluate.load("accuracy")




def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='model/mengzi-t5-base',
                        help='how ')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--context', action='store_true', help='whether the contexts are used in classify stage')
    return parser.parse_args()

argss = set_args()



context="_context" if argss.context  else ""
train_data_path=f'data/train{context}_type_fenci.json'
val_data_path=f'data/val{context}_type_fenci.json'
hugging_result_path=f"hugging_result/T5_hugging{context}_fc"
save_model_path=f"hugging_result/T5best{context}_fc"


# 加载预训练模型

print("loading model\n")

model_name = argss.model
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)





# 设置超参数
max_input_length = 200
max_target_length = 10
batch_size = argss.batch_size
epoch=argss.num_epochs
learning_rate=argss.lr


# 预处理数据
prefix = "请问下面文本属于 缩合型 、 截取型 二者中的哪一类？\n"
suffix = "\n选项： 缩合型, 截取型 \n答案："




def preprocess_function(examples):
    
    if argss.context:
        inputs = [prefix + full + "\n 上下文是：" + context + "\n" + suffix for full, context in
                  zip(examples["full"], examples["context"])]
    else:
        inputs = [prefix + full + "\n" + suffix for full in examples["full"]]
        
    
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['label'], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs



# 加载数据 ，encode
print("loading\n")
raw_datasets = load_dataset('json',data_files={'train': train_data_path,
                                        'val': val_data_path}).shuffle( )
print("finish")
tokenized_datasets = raw_datasets.map(preprocess_function, batched=True,
                                      remove_columns=raw_datasets['train'].column_names)

# 设置评估函数
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
#     logger.info(predictions, labels)
    decoded_preds = [tokenizer.batch_decode(predictions, skip_special_tokens=True)]
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = [tokenizer.batch_decode(labels, skip_special_tokens=True)]
    

    return metric.compute(predictions=decoded_preds, references=decoded_labels)


# 设置参数
# metric_name = "acc"
metric_name = "overall_accuracy"
args = Seq2SeqTrainingArguments(hugging_result_path,
                                evaluation_strategy="epoch",
                                learning_rate=learning_rate,
                                per_device_train_batch_size=batch_size,
                                per_device_eval_batch_size=batch_size * 10,
                                weight_decay=0.01,
                                save_total_limit=10,
                                num_train_epochs=epoch,
                                metric_for_best_model=metric_name,
                                predict_with_generate=True,
                                logging_dir="t5log.txt",
                                fp16=True)


data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
trainer = Seq2SeqTrainer(model, args,
                         train_dataset=tokenized_datasets["train"],
                         eval_dataset=tokenized_datasets["val"],
                         data_collator=data_collator,
                         tokenizer=tokenizer,
                         compute_metrics=compute_metrics)


trainer.train()
print("test")
print(trainer.evaluate())
trainer.save_model(save_model_path)
