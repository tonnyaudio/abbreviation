import torch
from torch.utils.data import DataLoader
from config import RankerConfig
from preprocess import read_ranker_data, PLH, PREFIX, generate_candidates_by_rule, \
    generate_candidates_by_rule_classifyByAbb
from model import Ranker
from dataset import RankerDataset, RankerOrderDataset
from train_eval import  save_ranker_predicts
from utils import seed_everything
import argparse
from pprint import pprint
import pandas as pd

import re
import jieba
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_beams', type=int, default=32)
    parser.add_argument('--ranker_model', type=str,
                        default='t5_v2_4e_ranker_context_extract_all_truncate150_top12_add_b_4_accu_512_10_epoch_7.pth',
                        help='the evaluation model')
    parser.add_argument('--top_k', type=int, default=12, help='number of candidates')
    parser.add_argument('--file', type=str, default='data/predict.txt', help='file for prediction')
    parser.add_argument('--file2', type=str, default='data/predict.txt', help='file for generate')
    parser.add_argument('--context', action="store_true")
    return parser.parse_args()




def prepare_ranker_context(fulls, contexts, candidates):
    #   加入 e1
    reference = [context.replace(PLH, f"<e1>{full}</e1>") for full, context in zip(fulls, contexts)]
    candidate_contexts = [[context.replace(PLH, f"{full}，简称“<e1>{candidate}</e1>”，") for candidate in candidate_list]
                          for full, context, candidate_list in zip(fulls, contexts, candidates)]

    labels = [0] * len(contexts)
    return reference, candidate_contexts, labels


if __name__ == '__main__':
    seed_everything()
    args = set_args()
    args_dict = vars(args)

    ranker_model_name = args_dict.pop('ranker_model')
    num_beams = args_dict.pop('num_beams')
    file = args_dict.pop('file')
    file2=args_dict.pop('file2')
    use_context = args_dict.pop('context')
    pprint(args)
    print(f'Use context: {use_context}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    config = RankerConfig(**args_dict)
    ranker = Ranker(config).to(device)
    ranker.load_state_dict(torch.load(ranker_model_name))
    ranker.eval()
    save = True

    data = pd.read_csv(file, sep='\t', names=['full','label', 'context', 'context_with_placeholder'] if use_context else ['full','label'])

    fulls = data['full'].tolist()

    contexts = [PLH] * len(data)
    if use_context:
        contexts = data['context_with_placeholder'].tolist()
        gen_reference = [context.replace(PLH, full) for full, context in zip(fulls, contexts)]

    gen_results = []

    # 读取结果
    candidates = []


    with open(file2, "r", encoding="utf-8") as f:
        for line in f:
            s = line.replace("\n", "").split("\t")
            candidates.append(s[0].split(";"))

    references, contexts, labels = prepare_ranker_context(fulls, contexts, candidates)

    test_set = RankerDataset(references, contexts, labels, config)

    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, collate_fn=test_set.collate_fn)

    _, _, predicts = save_ranker_predicts(test_loader, ranker, device, topk=config.num_candidates)

    predict_candidates = []
    for full, candidate_list, predict in zip(fulls, candidates, predicts):
        predict_candidates.append(';'.join([candidate_list[j] for j in predict]))

    data['predict_candidates'] = predict_candidates
    file_name = file.split('/')[-1]

    save_path = f"eval/{file_name.split('.')[0]}_output_constrain_bs32t05p1_2.txt"


    print(f"Save in {save_path}")

    if save:
        data.to_csv(save_path, sep='\t', index=False, header=True)
