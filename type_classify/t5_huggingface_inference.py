from transformers import AutoModelForSeq2SeqLM, T5Tokenizer
import pandas as pd
import torch
from datasets import load_dataset, load_metric
import argparse


# 设置 gpu 
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='hugging_result/T5best_context_fc ',
                        help='how ')
    parser.add_argument('--context', action='store_true', help='whether the contexts are used in classify stage')
    return parser.parse_args()

argss = set_args()

context="_context" if argss.context  else ""
test_data_path=f'data/test{context}_type_fenci.json'
save_predict_path=f"../generate_evaluation/data/classify_data/test_predict.txt"



# 加载模型
model_name =argss.model
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)


# 设置超参数
max_input_length = 200
max_target_length = 20
batch_size = 32


# 加载数据 ，encode
raw_datasets = load_dataset('json',data_files={ 'test': test_data_path})
data=raw_datasets["test"]


if torch.cuda.is_available():
    device = "cuda:0"
    model.half()
else:
    device = "cpu"
model = model.to(device)



# 预处理数据
prefix = "请问下面文本属于 缩合型 、 截取型 二者中的哪一类？\n"
suffix = "\n选项： 缩合型, 截取型 \n答案："


# 进行推理
results=[]
with torch.no_grad():
    for i in range(0, len(data["label"]), batch_size):
#         使用上下文
        if argss.context:
            batch_full_names, batch_abbrs, batch_contexts = data['full'][i: i + batch_size], data['label'][i: i + batch_size], data['context'][i: i + batch_size]
            batch_inputs=[prefix + full + "\n 上下文是：" + context + "\n" + suffix for full, context in zip(batch_full_names, batch_contexts)]
        else:
            batch_full_names, batch_abbrs = data['full'][i: i + batch_size], data['label'][i: i + batch_size]
            batch_inputs=[prefix + full+ "\n" + suffix for full in batch_full_names]
            
#         print(batch_inputs)
            
        encoded = tokenizer(batch_inputs, return_tensors='pt', padding=True,truncation=True)
        encoded = {k: v.to(device) for k, v in encoded.items()}
        outputs = model.generate(**encoded, max_new_tokens=5, return_dict_in_generate=True)
        batch_generates = tokenizer.batch_decode( outputs[0], skip_special_tokens=True)
        for j, abbr in enumerate(batch_abbrs):
            if argss.context:
                results.append([batch_full_names[j], abbr, batch_generates[j],batch_contexts[j]])
            else:
                results.append([batch_full_names[j], abbr, batch_generates[j]])

# 结果写入文件
cut=0
total=len(results)
with open(save_predict_path ,'w',encoding='UTF-8')as f:
    for cc in results:
        if(cc[2]==cc[1]):
            cut+=1
            
        if(cc[2]=="截取型"):
            f.write(cc[0].replace(" ","")+"\t0"+"\n")
        else :
            f.write(cc[0].replace(" ","")+"\t1"+"\n")
print(cut,total,cut/total)