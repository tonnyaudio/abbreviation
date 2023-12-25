
import pandas as pd
def get_rankdata(a):
    
    file1="data/final_dataset_gen_{}_context.txt".format(a)
    file2 = "eval/can_g_topk12_2prompt_context_{}_bs32t05p1_2.txt".format(a)
    file3 = "data/rank_data/glm_v2_ranker_top12_context_{}_bs32t05p1_2.txt".format(a)
    data = pd.read_csv(file1, sep='\t',
                       names=['src', 'target', 'context', 'context_with_placeholder'])
    
    
    candidates = []
    scores = []
    with open(file2, "r", encoding="utf-8") as f:
        for line in f:
            s = line.replace("\n", "").split("\t")
            candidates.append(s[0].split(";"))
            scores.append([float(t) for t in s[1].split(";")])
#     print(candidates[:2])
#     print(scores[:2])

    srcs = data["src"].tolist()
    targets = data["target"].tolist()
    contexts = data["context"].tolist()
    context_with_placeholders = data["context_with_placeholder"].tolist()
    candidatess = ["1"]*len(srcs)
    labels = [1]*len(srcs)

    clean_data = []
    for i, candidate in enumerate(candidates):

        if (targets[i] not in candidate):

            candidate[11] = targets[i]
            labels[i] = 11
        else:
            labels[i] = candidate.index(targets[i])
        candidatess[i] = ";".join(candidate)
        if labels[i] == 11:
            scores[i] = 1
        else:
            scores[i] = scores[i][labels[i]]
    a = {}
    a["src"] = srcs
    a["target"] = targets
    a["context"] = contexts
    a["context_with_placeholder"] = context_with_placeholders
    a["candidates"] = candidatess
    a["label"] = labels
    a["scores"] = scores

    print(a["scores"][0])

    s = pd.DataFrame.from_dict(a)
    s.to_csv(file3, sep='\t', index=False, header=False)


data_set = ["test", "val", "train"]
for a in data_set[:]:
    get_rankdata(a)


