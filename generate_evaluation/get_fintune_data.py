
from preprocess import load_dataset_with_context, generate_candidates_by_rule_classifyByAbb, is_subsequence
import os
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_data(a):

    full_names, abbrs, contexts = load_dataset_with_context("data/final_dataset_gen_{}_context.txt".format(a), to_dataset=False)
    file2 = 'data/classify_data/{}.txt'.format(a)
    classfs = []
    with open(file2, "r", encoding="utf-8") as ff:
        for line in ff:
            if (line.split("\t")[1].replace("\n", '') == '0'):
                classfs.append("截取型")
            else:
                classfs.append("缩合型")

    glmdata=[]

    for i in range(len(abbrs)):
        full_name, abbr, context = full_names[i], abbrs[i], contexts[i]
        use_2prompt = True
        if use_2prompt:
            prompt2 = classfs[i]

            glmdata.append({"context":f'{full_name}的上下文是：{context}\n'+f'类型是：{prompt2}\n'+f'{full_name}的简称是：',"target":abbr})

    file4="../chatglm_ptrain/data/glm_{}.json".format(a)

    with open(file4, "w", encoding="utf-8") as f:
        for p in glmdata[:]:
            json.dump(p, f, ensure_ascii=False)
            f.write("\n")
            
data_set=['test','val','train']
for a in data_set:
    get_data(a)
        
print("完成")