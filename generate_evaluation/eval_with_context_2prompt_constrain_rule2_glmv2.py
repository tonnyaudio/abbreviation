from transformers import BertTokenizer, BartForConditionalGeneration, AutoModel, AutoTokenizer, AutoModel
import os
import torch
from tqdm import tqdm
from preprocess import load_dataset_with_context, generate_candidates_by_rule_classifyByAbb, is_subsequence, \
    generate_candidates_by_rule
from utils import seed_everything
import thwpy
import argparse
from pprint import pprint
import os
import numpy as np
from peft import get_peft_model, LoraConfig, TaskType
from typing import Optional

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from tqdm import *


def softmax(x):
    x_exp = np.exp(x)
    # 如果是列向量，则axis=0
    x_sum = np.sum(x_exp, axis=0, keepdims=True)
    s = x_exp / x_sum
    return s



def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='chatglm-6b')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--top_k', type=int, default=12)
    parser.add_argument('--num_beams', type=int, default=32)
    parser.add_argument('--file', type=str, default='data/final_dataset_gen_test_context.txt')
    parser.add_argument('--constrain', action='store_true')
    parser.add_argument('--score', action='store_true')
    parser.add_argument('--data_set', type=str, default='test_predict')
    return parser.parse_args()


# 设置参数
seed_everything()
args = set_args()
pprint(args)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = args.model_name
batch_size = args.batch_size
top_k = args.top_k
num_beams = args.num_beams
use_prompt = False
constrain = args.constrain
score = args.score


# 加载模型
if 'glm' in model_name:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, device_map='auto',
                                      torch_dtype=torch.bfloat16)  # .half().cuda()
model = model.to(device)
model.eval()

# 加载数据 和 全称类型数据
split = thwpy.check_split(args.file)
file_prefix = args.file.split('/')[-1].split('.')[0]
mod = '_mod' if 'mod' in args.file else ''

full_names, abbrs, contexts = load_dataset_with_context(args.file, to_dataset=False)
file2 = 'data/classify_data/{}.txt'.format(args.data_set)

classfs = []
with open(file2, "r", encoding="utf-8") as ff:
    for line in ff:
        if (line.split("\t")[1].replace("\n", '') == '0'):
            classfs.append("截取型")
        else:
            classfs.append("缩合型")

correct = 0
total = 0
results = []
# 输出分数
score = True

with torch.no_grad():
    if constrain:

        for i in tqdm(range(0, len(abbrs), batch_size)):
            batch_full_names, batch_abbrs, batch_contexts = full_names[i: i + batch_size], abbrs[
                                                                                           i: i + batch_size], contexts[
                                                                                                               i: i + batch_size]
            bfull_names = batch_full_names
            batch_full_prompts = classfs[i: i + batch_size]
            batch_full_context = [f'{f}的上下文是：{cc}\n' for cc, f in zip(batch_contexts, batch_full_names)]
            batch_full_abbs = [f'{name}的简称是：' for name in batch_full_names]

            batch_input = [b + f'类型是：{a}\n' + c for a, b, c in
                           zip(batch_full_prompts, batch_full_context, batch_full_abbs)]

            print(batch_input)
            encoded = tokenizer(batch_input, return_tensors='pt', padding=True, truncation=True)

            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = model.generate(**encoded, max_new_tokens=32, num_beams=num_beams, num_return_sequences=num_beams,
                                     temperature=0.5, repetition_penalty=1.2, output_scores=score,
                                     return_dict_in_generate=score)
            batch_generates = tokenizer.batch_decode(outputs.sequences if score else outputs, skip_special_tokens=True)
            if score:
                batch_scores = outputs.sequences_scores.cpu().tolist()
            for j, abbr in enumerate(batch_abbrs):
                generate = batch_generates[j * num_beams: (j + 1) * num_beams]
                generate = [w.strip().replace(' ', '') for w in generate]
                generate = [
                    a.split("：")[-1].replace(" ", "").replace("。", "").replace("）", "").replace(")", "").replace("\n",
                                                                                                                 "") for
                    a in generate]

                print(generate)
                gen_scores = []
                gen_constrains = []
                for ss, cscore in zip(generate, batch_scores):
                    if (batch_full_prompts[j] == "截取型"):
                        if (is_subsequence(ss, bfull_names[j]) and (ss != bfull_names[j]) and (ss != "") and (
                                ss in bfull_names[j]) and (ss not in gen_constrains)):
                            gen_constrains.append(ss)
                            gen_scores.append(cscore)
                    else:
                        if (is_subsequence(ss, bfull_names[j]) and (ss != bfull_names[j]) and (ss != "") and (
                                ss not in bfull_names[j]) and (ss not in gen_constrains)):
                            gen_constrains.append(ss)
                            gen_scores.append(cscore)
                    if len(gen_constrains) == top_k:
                        break
                for ss, cscore in zip(generate, batch_scores):
                    if len(gen_constrains) == top_k:
                        break
                    if (is_subsequence(ss, bfull_names[j]) and (ss != bfull_names[j]) and (ss != "") and (
                            ss not in gen_constrains)):
                        gen_constrains.append(ss)
                        gen_scores.append(cscore)

                # 当生成候选不满足top_k，采用启发规则生成
                while len(gen_constrains) < top_k:
                    gen_constrains.append("a")
                    gen_scores.append(0.1)

                #       加入规则
                if (batch_full_prompts[j] == "截取型"):
                    gen_constrains = generate_candidates_by_rule_classifyByAbb(bfull_names[j], gen_constrains, "0")
                else:
                    gen_constrains = generate_candidates_by_rule_classifyByAbb(bfull_names[j], gen_constrains, "1")


                generate = gen_constrains


                if score:
                    if len(gen_scores)!=top_k:
                        print("!!!!!!!!!!!!!!!!!!!")
                    scores = softmax(gen_scores)
                    print(f"Generate: {gen_constrains} Abbr: {abbr} Score: {scores}  Full: {bfull_names[j]} \n")
                    results.append([batch_full_names[j], abbr, ';'.join(generate), ';'.join([str(s) for s in scores])])
                else:
                    results.append([batch_full_names[j], abbr, ';'.join(generate)])
                if abbr in generate:
                    correct += 1
                total += 1

with open("eval/can_g_topk12_2prompt_context_{}_bs32t05p1_2.txt".format(
        args.data_set), 'w', encoding='UTF-8') as f:
    for cc in results:
        f.write(cc[2] + "\t" + cc[3] + "\n")



print(f"Model({split}){'(constrain)' if constrain else ''}: {model_name}")
print(f"Hit@{top_k}(beam={num_beams}): {correct} / {total} = ", correct / total)
