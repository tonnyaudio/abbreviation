from datasets import Dataset
import pandas as pd
import thwpy
import random
from utils import seed_everything
import numpy as np
import re
from pprint import pprint
from itertools import combinations
import jieba
import json
from sklearn.model_selection import KFold
from seqeval.metrics import accuracy_score
from collections import Counter
from rouge import Rouge
import  torch

seed_everything()
PLH = '[PLH]'
PREFIX = f'以下是{PLH}的介绍：'
MID_EDIT_DISTANCE = 4
UNK = '[UNK]'
GEN_DATA_NAMES = ['src', 'target']
CANDIDATE_DATA_NAMES = ['src', 'target', 'candidates']
RANKER_DATA_NAMES = ['src', 'target', 'context', 'context_with_placeholder', 'candidates', 'label']
RANKER_NO_CONTEXT_DATA_NAMES = ['src', 'target', 'candidates', 'label']
RAW_CONTEXT_DATA_NAMES = ['src', 'target', 'context']
CONTEXT_DATA_NAMES = ['src', 'target', 'context', 'context_with_placeholder']
# ABLATIONS = ['noheu', 'nofull', 'noword', 'notruth', 'nochar', 'all_v2', 'all']
ABLATIONS = ['noheu', 'noword', 'notruth', 'nochar', 'all_v2', 'all']
header_name_dict = {
    'gen': GEN_DATA_NAMES,
    'ranker': RANKER_DATA_NAMES,
    'context': CONTEXT_DATA_NAMES,
    'raw_context': RAW_CONTEXT_DATA_NAMES
}


def read_text_pairs(file):
    data = pd.read_csv(file, sep='\t', names=['src', 'target'])
    return data['src'].tolist(), data['target'].tolist()


def read_company_names(file):
    data = pd.read_csv(file, sep='\t')
    # data = data.fillna(axis=1, method='bfill')
    return data['full_name'].tolist(), data['short_name'].tolist()


def read_ranker_data(file, mode='add_b'):
    # add_b: fill candidates into context
    # none: no context
    assert mode in ['add_b', 'add_s', 'sub', 'none', 'nofuse']

    data = pd.read_csv(file, sep='\t', names=['src', 'target', 'context',
                                              'context_with_placeholder', 'candidates', 'label'])
    if data['label'].isna().any():
        data = data.drop(columns=['candidates', 'label'])
        data = data.rename(columns={'context': 'candidates', 'context_with_placeholder': 'label'})

    assert all(not data[key].isna().any() for key in data.keys())

    data['candidates'] = data['candidates'].map(lambda x: x.split(';'))
    if mode != 'none':
        references = [context.replace(PLH, full) for context, full in
                      zip(data['context_with_placeholder'], data['src'])]
    else:
        references = data['src'].tolist()
    if mode == 'add_b':
        contexts = [[context.replace(PLH, f"{full}（简称“{candidate}”）") for candidate in candidates] for
                    candidates, full, abbr, context in
                    zip(data['candidates'], data['src'], data['target'], data['context_with_placeholder'])]
    elif mode == 'add_s':
        contexts = [[context.replace(PLH, f"{full}，简称“{candidate}”，") for candidate in candidates] for
                    candidates, full, abbr, context in
                    zip(data['candidates'], data['src'], data['target'], data['context_with_placeholder'])]
    elif mode == 'sub':
        contexts = [[context.replace(PLH, candidate) for candidate in candidates] for
                    candidates, full, abbr, context in
                    zip(data['candidates'], data['src'], data['target'], data['context_with_placeholder'])]
    else:
        contexts = [[f"{full}（简称“{candidate}”）" for candidate in candidates]
                    for full, candidates in zip(data['src'], data['candidates'])]

    assert len(references) == len(contexts) and len(contexts) == len(data['label'])

    return references, contexts, data['label'].tolist()

def read_ranker_data_order(file, mode='add_b'):
    # add_b: fill candidates into context
    # none: no context
    assert mode in ['add_b', 'add_s', 'sub', 'none', 'nofuse']
    
    if mode=='add_s':
        data = pd.read_csv(file, sep='\t',
                           names=['src', 'target', 'context', 'context_with_placeholder', 'candidates', 'label', 'order'])
    if mode=='none':
        data = pd.read_csv(file, sep='\t',
                           names=['src', 'target', 'candidates', 'label', 'order'])
        
    
    data['candidates'] = data['candidates'].map(lambda x: x.split(';'))


    if mode != 'none':
        if mode == 'add_b':
        
            references = [context.replace(PLH, full) for context, full in
                          zip(data['context_with_placeholder'], data['src'])]
        if mode == 'add_s': 
# 添加 e1 
            references = [context.replace(PLH, f"<e1>{full}</e1>") for context, full in
                          zip(data['context_with_placeholder'], data['src'])]

    else:
        # nocontext
        references = [f"<e1>{full}</e1>" for  full in data['src']]

    if mode == 'add_b':
        contexts = [[context.replace(PLH, f"{full}（简称“{candidate}”）") for candidate in candidates] for
                    candidates, full, abbr, context in
                    zip(data['candidates'], data['src'], data['target'], data['context_with_placeholder'])]
    elif mode == 'add_s':
# 加e1
        contexts = [[context.replace(PLH, f"{full}，简称“<e1>{candidate}</e1>”，") for candidate in candidates] for
                    candidates, full, abbr, context in
                    zip(data['candidates'], data['src'], data['target'], data['context_with_placeholder'])]

    elif mode == 'none':
        #     nocontext
        contexts = [[f"{full}（简称“<e1>{candidate}</e1>”）" for candidate in candidates] for full, candidates in
                    zip(data['src'], data['candidates'])]
        
    elif mode == 'sub':
        
        contexts = [[context.replace(PLH, candidate) for candidate in candidates] for candidates, full, abbr, context in
                    zip(data['candidates'], data['src'], data['target'], data['context_with_placeholder'])]
        print(contexts[0][0])


    else:
        contexts = [[f"{full}（简称“{candidate}”）" for candidate in candidates] for full, candidates in
                    zip(data['src'], data['candidates'])]

    assert len(references) == len(contexts) and len(contexts) == len(data['label'])

    return references, contexts, data['label'].tolist(), data['order'].tolist()

def rfind(s, find_chars):
    """从s的尾部开始，找到第一个满足 在find_chars中的字符 的位置"""
    pos = len(s) - 1
    while pos >= 0 and s[pos] not in find_chars:
        pos -= 1
    return pos


def truncate_context_len(file, max_len, truncate_len, save=False):
    """
    对带有PLH的context进行truncate
    :param save:
    :param file: 带有PLH的context文件
    :param max_len: 对于大于max_len的context需要进行truncate，truncate的方式就是直接截断
    :param truncate_len: 截断的长度，即选取context[:truncate_len]
    :return:
    """
    file_split = thwpy.check_split(file)
    data = pd.read_csv(file, sep='\t', names=['src', 'target', 'context', 'context_with_placeholder'])
    print(len(data))
    context_type = '_short' if 'short' in file else ''
    print(len(data))
    print(file_split)
    find_chars = ['。', '；', '，', '、', ';', ',', '？', ' ']
    cnt = 0
    truncate_context_with_placeholder = []
    for i, row in data.iterrows():
        context = row['context_with_placeholder']
        if len(context) > max_len:
            idx = -1
            cnt += 1
            context = row['context_with_placeholder'][:truncate_len]
            for ch in find_chars:
                idx = context.rfind(ch)
                if idx > 0:
                    break
            if idx == -1:
                idx = truncate_len
            context = context[:idx] + '。' if PLH in context[
                                                    :idx] else f"以下是{PLH}的介绍：{context[:idx] + '。' if not context.endswith('）') else context[:idx + 1]}"
            print(f"{context} | {len(context)} | {idx}")
        if PLH not in context:
            print(f"{i} | {context}")
        assert len(context) <= truncate_len + len(PREFIX) + 1
        truncate_context_with_placeholder.append(context)
    data['context_with_placeholder'] = truncate_context_with_placeholder
    assert all(len(c) <= truncate_len + len(PREFIX) + 1 and PLH in c for c in data['context_with_placeholder'])
    print(f"Truncated: {cnt}")
    save_path = f'data/final_ranker_truncate{truncate_len}{context_type}_context_{file_split}.txt'
    print(data)
    print(f"Saved in: {save_path}")
    if save:
        data.to_csv(save_path, sep='\t', header=False, index=False)


def extract_context_sentence(file, min_len=75, truncate_len=150, save=False):
    """
    抽取context中PLH所在的关键句子作为最终的context，为中心扩展的思想，即:
    1. 通过标点符号将context转为一个句子的序列
    2. 定位PLH所在的句子
    3. 从该句子向两边扩展，当扩展到一定的长度区间后 [min_len, truncate_len] 后停止
    :param file: 带有PLH的context文件, 总共4列，[src(全称), target(缩略词), context, context_with_placeholder]
    :param min_len: 中心扩展的最小长度
    :param truncate_len: 中心扩展的最大长度
    :param save: 是否保存
    :return: 保存处理的数据，格式和file一样，返回保存位置
    """
    file_split = thwpy.check_split(file)
    data = pd.read_csv(file, sep='\t', names=['src', 'target', 'context', 'context_with_placeholder'])
    print(len(data))
    print(file_split)
    cnt = 0
    find_chars = ''.join(['。', '；', '，', '、', ';', ',', '？', '?', '！', '!', ' '])
    pattern = re.compile(rf'[{find_chars}]')
    truncate_context_with_placeholder = []
    for i, row in data.iterrows():
        context = row['context_with_placeholder']
        context_sentences = re.split(pattern, context)
        context_sentences = [s for s in context_sentences if len(s) > 0]
        # print(context_sentences)
        pos = 0
        if PREFIX in context:
            len_sum = 0
            for idx, s in enumerate(context_sentences):
                len_sum += len(s)
                if len_sum > truncate_len:
                    pos = idx
                    break
            context = '，'.join(context_sentences[:pos] if pos > 0 else context_sentences)
        else:
            for idx, s in enumerate(context_sentences):
                if PLH in s:
                    context = s
                    offset = 1
                    while len(context) < min_len and (idx + offset <= len(context_sentences) or idx - offset >= 0):
                        context = '，'.join(
                            context_sentences[max(0, idx - offset): min(len(context_sentences), idx + offset)])
                        offset += 1
                    break
        assert PLH in context
        if PREFIX == context:
            cnt += 1
        print(f"{context} | {i} | {len(context)}")
        truncate_context_with_placeholder.append(context)
    assert len(truncate_context_with_placeholder) == len(data)
    print(
        f"Average Length: {sum(len(c) for c in truncate_context_with_placeholder) / len(truncate_context_with_placeholder)}")
    print(f"Max Length: {max(len(c) for c in truncate_context_with_placeholder)}")
    print(f"context = PREFIX: {cnt}")
    data['context_with_placeholder'] = truncate_context_with_placeholder
    save_path = f'data/final_ranker_extract_truncate{truncate_len}_context_{file_split}.txt'
    print(data)
    print(f"Saved in {save_path}")
    if save:
        data.to_csv(save_path, sep='\t', header=False, index=False)
    return save_path


def read_candidates(file, top_k=16):
    """读取candidates文件file，并确保每个样本都有top_k个candidates"""
    data = pd.read_csv(file, sep='\t', names=['src', 'target', 'candidates'])
    data['candidates'] = data['candidates'].map(lambda x: x.split(';'))
    print(data['candidates'])
    for c in data['candidates']:
        assert len(c) == top_k


def remove_spans(context, abbr):
    """移除context中带有abbr的span"""
    if re.search(rf'（.*?{abbr}.*?）', context) is not None:
        context = re.sub(rf'（.*?{abbr}.*?）', '', context)
    while abbr in context:
        idx = context.find(abbr)
        start = idx - 1
        end = idx + len(abbr)
        punctuation = '，。；：！？,.;:!?'
        while start >= 0:
            if context[start] in punctuation:
                break
            start -= 1
        while end < len(context):
            if context[end] in punctuation:
                break
            end += 1
        if start < 0:
            context = context[end + 1:]
        elif end >= len(context):
            context = context[:start + 1]
        else:
            context = context[:start + 1] + context[end + 1:]
    return context


def prepare_context_with_placeholder(file, save=False):
    """
    给context文件加入一列，为带有PLH的context，方便填入全称和candidates
    :param file: context文件，包含3列 [src(全称), target(缩略词), context]
    :param save: 是否保存文件
    :return: 保存添加了 context_with_placeholder 列的数据，返回保存路径
    """
    data = pd.read_csv(file, sep='\t', names=['src', 'target', 'context'])
    file_split = thwpy.check_split(file)
    cnt = 0
    abbr_in_contexts = []
    context_with_placeholder = []
    for idx, row in data.iterrows():
        context = row['context']
        if row['src'] not in row['context']:
            cnt += 1
            context = f"以下是{PLH}的介绍：{context}"
        else:
            context = row['context'].replace(row['src'], PLH)
        if row['target'] in context:
            context = remove_spans(context, row['target'])
            abbr_in_contexts.append([row['target'], row['context'], context])
        if PLH not in context:
            context = f"以下是{PLH}的介绍：{context}"
        assert row['target'] not in context and PLH in context
        context_with_placeholder.append(context)
    print(f"Not in the context: {cnt}")
    data['context_with_placeholder'] = context_with_placeholder
    # thwpy.save_2d_list(abbr_in_contexts, f'data/{file_split}_abbr_in_short_context.txt')
    print(data)
    print(data.keys())
    save_path = f'data/final_ranker_context_{file_split}.txt'
    print(f"Saved in {save_path}")
    if save:
        data.to_csv(save_path, sep='\t', header=False, index=False)
    return save_path


def check_ablation(file_name):
    for ablation in ABLATIONS:
        if ablation in file_name:
            return ablation
    return ABLATIONS[0]


def prepare_label(file, top_k=8, has_duplicate=False, version=6, save=False):
    """
    找到candidates中的正样本，构建label
    对于训练数据，如果candidates中不存在ground truth，那么将ground truth替换candidates中的最后一个
    对于测试数据，如果candidates中不存在ground truth，那么将label设为-1
    :param save: 是否保存
    :param file: 带有candidates的文件
    :param top_k: 最终共有几个candidates
    :param has_duplicate: candidate中是否有重复
    :return: 保存添加了 label 列的数据
    """
    data = pd.read_csv(file, sep='\t', names=['src', 'target', 'candidates'])
    context = '_context' if 'context' in file else ''
    ratio = thwpy.regex_match(rf'_ratio\d*', file)
    gen_context_truncate = thwpy.regex_match(rf'_truncate\d*', file)
    file_split = thwpy.check_split(file)
    ablation = check_ablation(file)
    print(file_split)
    cnt = 0
    labels = []
    print(has_duplicate)
    for idx, row in data.iterrows():
        candidate_list = row['candidates'].split(';')
        if has_duplicate:
            tmp_list = list(set(candidate_list))
            tmp_list.sort(key=candidate_list.index)
            candidate_list = tmp_list[:top_k]
        if row['target'] in candidate_list:
            label = candidate_list.index(row['target'])
        else:
            cnt += 1
            label = top_k - 1 if file_split == 'train' else -1
            if file_split == 'train':
                candidate_list[-1] = row['target']

        assert len(candidate_list) == top_k, "candidate个数必须正确"
        assert candidate_list[label] == row['target'] or (row['target'] not in candidate_list and label == -1), \
            "标签对应，且测试集中不存在ground truth的样本label为-1"

        row['candidates'] = ';'.join(candidate_list)
        labels.append(label)
    data['label'] = labels
    print(data)
    print(data.keys())
    print(f"Hit@{top_k}: {len(data) - cnt} / {len(data)} = {(len(data) - cnt) / len(data)}")
    # save_path = f'data/t5_v{version}_candidate_context_{ablation}_top{top_k}_w_label_{file_split}.txt'
    save_path = f'data/t5_v{version}_candidate{ratio}{context}{gen_context_truncate}_{ablation}_top{top_k}_w_label_{file_split}.txt'
    print(f"Saved in {save_path}")
    if save:
        data.to_csv(save_path, sep='\t', header=False, index=False)
    return save_path, (len(data) - cnt) / len(data)


def merge_data(context_file, candidate_file, top_k=8, truncate_len=200, version=6, save=False):
    """
    将context文件和candidate文件合并，构造最终模型训练和测试读取的数据文件
    :param save: 是否保存
    :param context_file: 格式为['full', 'abbr', 'context', 'context_with_placeholder']
    :param candidate_file: 格式为['full', 'abbr', 'candidates', 'label']
    :param top_k: 有几个candidates
    :param truncate_len: 超过truncate_len进行truncate
    :return: 保存merge后的数据
    """
    file_split = thwpy.check_split(context_file)
    candidate_file_split = thwpy.check_split(candidate_file)
    post_edit = f'_{check_ablation(candidate_file)}'
    # uni = '_uni' if 'uni' in candidate_file else ''
    extract = '_extract' if 'extract' in context_file else ''
    short = '_short' if 'short' in context_file else ''
    context = '_context' if 'context' in candidate_file else ''
    ratio = thwpy.regex_match(rf'_ratio\d*', candidate_file)
    gen_context_truncate = thwpy.regex_match(rf'_truncate\d*', candidate_file)

    assert file_split == candidate_file_split, 'merge同一个split'

    context_data = pd.read_csv(context_file, sep='\t', names=['full', 'abbr', 'context', 'context_with_placeholder'])
    max_len = max([len(c) for c in context_data['context_with_placeholder']])
    print(f"max context len: {max_len}")
    candidate_data = pd.read_csv(candidate_file, sep='\t', names=['full', 'abbr', 'candidates', 'label'])

    assert len(context_data) == len(candidate_data)

    data = pd.merge(context_data, candidate_data, how='right', on=['full', 'abbr'])
    print(data)
    print(data.keys())
    save_path = f'data/t5_v{version}_ranker{ratio}{context}{gen_context_truncate}{extract}{post_edit}_truncate{truncate_len}{short}_top{top_k}_{file_split}.txt'
    print(f"Saved in {save_path}")
    if save:
        data.to_csv(save_path, sep='\t', header=False, index=False)
    return save_path


def get_subsequence_by_label(word, label):
    """
    根据label选取word中的元素组成一个word的子序列
    :param word:
    :param label: 0/1序列，str/list
    :return: 获取到的子序列，str类型
    """
    return ''.join([ch for ch, l in zip(word, label) if l == 1])


def sample_by_edit_distance(word, label, d=1):
    """
    根据和label的编辑距离d从word中选取子序列
    :param word:
    :param label: 0/1序列，str/list
    :param d: 最终的子序列和label的距离
    :return: 获取的子序列列表
    """
    label = [int(l) for l in label]
    # print(label)
    assert len(label) == len(word)

    idx_list = list(range(len(label)))
    indices = list(combinations(idx_list, d))
    random.shuffle(indices)
    candidates = []
    for idx in indices:
        current_label = label.copy()
        for i in idx:
            current_label[i] = 1 - current_label[i]
        candidate = get_subsequence_by_label(word, current_label)
        if len(candidate) > 0:
            candidates.append(candidate)
    return candidates


def sample_by_word_segmentation(seg_word, d=1):
    """
    选取和分词后的结果seg_word的编辑距离为d的子序列
    :param seg_word: 分词后的结果 list
    :param d: 编辑距离
    :return: 子序列的列表
    """
    labels = [1] * len(seg_word)
    idx_list = list(range(len(labels)))
    indices = list(combinations(idx_list, d))
    random.shuffle(indices)
    candidates = []
    for idx in indices:
        current_label = labels.copy()
        for i in idx:
            current_label[i] = 1 - current_label[i]
        candidate = get_subsequence_by_label(seg_word, current_label)
        if len(candidate) > 0:
            candidates.append(candidate)
    return candidates


def prepare_candidates(label_seq_file, candidate_file, top_k=8, version=6, save=False, ratio=0.0,
                       no_full=False, no_word=False, no_truth=False, no_char=False):
    """
    对candidate_file中模型生成的candidates通过启发式的规则进行后处理
    首先，对于训练集，首先全称是一个负样本
    1st. 根据全称的分词结果，选取和全称词编辑距离为d的子序列
    2nd. 只是对于训练集而言，选取和ground truth编辑距离为d的子序列
    3th. 选取和全称字符编辑距离为d的子序列，其中 0 < d < word_len
    :param label_seq_file: 带有0/1序列的文件
    :param candidate_file: 带有candidates的文件
    :param top_k: 最终保证top_k个candidates
    :param ratio:
    :param version:
    :param no_char: 是否移除 char-based rule
    :param no_truth: 是否移除 ground-truth-based rule
    :param no_word: 是否移除 word-based rule
    :param no_full: candidates是否移除全称，调用时默认设为True
    :param save: 是否保存
    :return: 保存最终的candidate数据的路径，以及Hit@K指标
    """
    file_split = thwpy.check_split(candidate_file)
    print(file_split)
    print(thwpy.check_split(label_seq_file))
    assert file_split == thwpy.check_split(label_seq_file) or file_split == 'infer'
    context = '_context' if 'context' in candidate_file else ''
    ratio_str = f'_ratio{int(ratio * 100)}' if ratio > 0 else ''
    gen_context_truncate = thwpy.regex_match(rf'_truncate\d*', candidate_file)
    train = file_split == 'train'
    if ratio > 0 and train:
        prefix, suffix = label_seq_file.split('.')
        label_seq_file = prefix + f'_ratio{int(ratio * 100)}.' + suffix
    # label_data = pd.read_csv(label_seq_file, sep='\t', names=['full', 'label'])
    label_data = thwpy.load_csv(label_seq_file, sep='\t')
    label_data = [row[1] for row in label_data]
    candidate_data = pd.read_csv(candidate_file, sep='\t', names=['full', 'abbr', 'candidate'])
    # candidate_data['label'] = label_data['label']

    candidate_data['label'] = label_data

    assert not train or all(len(row['full']) == len(row['label']) for i, row in candidate_data.iterrows())

    statistics = {'word': 0, 'ground-truth': 0, 'char': 0, 'pad': 0}
    candidates = []
    diff = 0
    edit_distances = []
    cnt = 0
    for idx, row in candidate_data.iterrows():
        diff += len(row['full']) - len(row['abbr'])
        edit_distances.append(len(row['full']) - len(row['abbr']))
        candidate_list = row['candidate'].split(';')

        assert len(candidate_list) == top_k

        tmp_list = list(set(candidate_list))
        tmp_list.sort(key=candidate_list.index)
        tmp_list = [c for c in tmp_list if is_subsequence(c, row['full']) and c != '']
        # tmp_list = [c for c in tmp_list if is_subsequence(c, row['full'])]
        if not no_full:
            if train and len(tmp_list) < top_k and row['full'] not in tmp_list:
                tmp_list.append(row['full'])
        else:
            tmp_list = [c for c in tmp_list if c != row['full']]
        word_len = len(row['full'])
        seg_word = jieba.lcut(row['full'])
        seg_word_len = len(seg_word)

        # 1st
        if not no_word:
            d = 1
            while len(tmp_list) < top_k and d <= seg_word_len:
                examples = sample_by_word_segmentation(seg_word, d)
                for example in examples:
                    if example not in tmp_list:
                        tmp_list.append(example)
                        statistics['word'] += 1
                        if len(tmp_list) == top_k:
                            break
                d += 1
        # 2nd
        if (not no_truth) and train:
            d = 1
            while len(tmp_list) < top_k and d <= word_len:
                examples = sample_by_edit_distance(row['full'], row['label'], d)
                for example in examples:
                    if example not in tmp_list:
                        tmp_list.append(example)
                        statistics['ground-truth'] += 1
                        if len(tmp_list) == top_k:
                            break
                d += 1
        # 3rd
        if not no_char:
            d = 1
            while len(tmp_list) < top_k and d < word_len:
                examples = sample_by_word_segmentation(row['full'], d)
                for example in examples:
                    if example not in tmp_list:
                        tmp_list.append(example)
                        statistics['char'] += 1
                        if len(tmp_list) == top_k:
                            break
                d += 1
        # len(word) <= 3，枚举肯定会有空的，需要补全

        # v1
        if no_full:
            tmp_list = [c for c in tmp_list if c != row['full'] and c != '']
            # tmp_list = [c for c in tmp_list if c != row['full']]
            d = len(tmp_list) - 1
            while d >= 0 and tmp_list[d] == row['abbr']:
                d -= 1
            while d >= 0 and len(tmp_list) < top_k:
                tmp_list.append(tmp_list[d])
                statistics['pad'] += 1
        else:
            while len(tmp_list) < top_k:
                tmp_list.append(row['full'])

        if no_full:
            assert row['full'] not in tmp_list and '' not in tmp_list
            # assert row['full'] not in tmp_list
        # padding
        while len(tmp_list) < top_k:
            tmp_list.append(row['full'])
            statistics['pad'] += 1
        assert len(tmp_list) == top_k
        if row['abbr'] in tmp_list:
            cnt += 1
        candidates.append(';'.join(tmp_list))
        print(f"{row['full']} | {row['abbr']} | {';'.join(tmp_list)}")
    print(f"Split: {file_split}")
    print(f"Average Edit Distance: {diff / len(candidate_data)}")
    edit_distances.sort()
    print(f"Middle Edit Distance: {edit_distances[len(candidate_data) // 2]}")
    print(f"Hit@{top_k}: {cnt / len(candidate_data)}")
    print(f"Statistics: {statistics}")
    candidate_data['candidate'] = candidates

    candidate_data = candidate_data.drop(columns=['label'])

    print(candidate_data)
    print(candidate_data.keys())

    if no_word:
        post_edit = 'noword'
    elif no_char:
        post_edit = 'nochar'
    elif no_truth:
        post_edit = 'notruth'
    else:
        post_edit = 'all'
    # save_path = f'data/final_v{version}_candidate_context_{post_edit}_top{top_k}_{file_split}.txt'
    save_path = f'data/t5_v{version}_candidate{ratio_str}{context}{gen_context_truncate}_{post_edit}_top{top_k}_{file_split}.txt'
    # save_path = f'data/t5_v{version}_candidate_{post_edit}_top{top_k}_{file_split}.txt'

    print(f"Saved in {save_path}")
    if save:
        candidate_data.to_csv(save_path, sep='\t', header=False, index=False)
    return save_path, cnt / len(candidate_data)


def generate_candidates_by_rule(full, candidate_list, no_word=False, no_char=False):
    top_k = len(candidate_list)
    tmp_list = list(set(candidate_list))
    tmp_list.sort(key=candidate_list.index)
    tmp_list = [c for c in tmp_list if is_subsequence(c, full) and c != '']
    # tmp_list = [c for c in tmp_list if is_subsequence(c, row['full'])]
    tmp_list = [c for c in tmp_list if c != full]
    word_len = len(full)
    seg_word = jieba.lcut(full)
    seg_word_len = len(seg_word)

    # 1st
    if not no_word:
        d = 1
        # if (len(full > 11)):
        #     d= int((len(full)-8)/2)
        while len(tmp_list) < top_k and d <= seg_word_len:
            examples = sample_by_word_segmentation(seg_word, d)
            for example in examples:
                if example not in tmp_list:
                    tmp_list.append(example)
                    if len(tmp_list) == top_k:
                        break
            d += 1
    if not no_char:
        d = 1
        while len(tmp_list) < top_k and d < word_len:
            examples = sample_by_word_segmentation(full, d)
            for example in examples:
                if example not in tmp_list:
                    tmp_list.append(example)
                    if len(tmp_list) == top_k:
                        break
            d += 1
    # len(word) <= 3，枚举肯定会有空的，需要补全

    # v1
    tmp_list = [c for c in tmp_list if c != full and c != '']
    # tmp_list = [c for c in tmp_list if c != row['full']]
    d = len(tmp_list) - 1
    while d >= 0 and len(tmp_list) < top_k:
        tmp_list.append(tmp_list[d])

    while len(tmp_list) < top_k:
        tmp_list.append(full)
    assert len(tmp_list) == top_k
    return tmp_list

def generate_candidates_by_rule_classifyByAbb(full, candidate_list,classify,no_word=False, no_char=False):
    top_k = len(candidate_list)
    tmp_list = list(set(candidate_list))
    tmp_list.sort(key=candidate_list.index)
    tmp_list = [c for c in tmp_list if is_subsequence(c, full) and c != '']
    # tmp_list = [c for c in tmp_list if is_subsequence(c, row['full'])]
    tmp_list = [c for c in tmp_list if c != full]
    word_len = len(full)
    seg_word = jieba.lcut(full)
    seg_word_len = len(seg_word)

    if(classify=="0"):
        if not no_word:
            d = 1
#             if (len(full > 11)):
#                 d= int((len(full)-8)/2)
            while len(tmp_list) < top_k and d <= seg_word_len:
                examples = sample_by_word_segmentation(seg_word, d)
                for example in examples:
                    if example not in tmp_list and example in full:
                        tmp_list.append(example)
                        if len(tmp_list) == top_k:
                            break
                d += 1
        if not no_char:
            if word_len<=6:
                d = 1
            else :
                d=int(word_len *0.3)
            while len(tmp_list) < top_k and d < word_len:
                examples = sample_by_word_segmentation(full, d)
                for example in examples:
                    if example not in tmp_list and example in full:
                        tmp_list.append(example)
                        if len(tmp_list) == top_k:
                            break
                d += 1
    else :
        if not no_char:
            if word_len<=6:
                d = 1
            else :
                d=int(word_len *0.3)
            while len(tmp_list) < top_k and d < word_len:
                examples = sample_by_word_segmentation(full, d)
                for example in examples:
                    if example not in tmp_list and example not in full:
                        tmp_list.append(example)
                        if len(tmp_list) == top_k:
                            break
                d += 1
        if not no_word:
            if word_len <= 10:
                d = 1
            else:
                d = int(word_len * 0.2)
            while len(tmp_list) < top_k and d <= seg_word_len:
                examples = sample_by_word_segmentation(seg_word, d)
                for example in examples:
                    if example not in tmp_list and example not in full:
                        tmp_list.append(example)
                        if len(tmp_list) == top_k:
                            break
                d += 1

    # v1
    tmp_list = [c for c in tmp_list if c != full and c != '']
    # tmp_list = [c for c in tmp_list if c != row['full']]
    d = len(tmp_list) - 1
    while d >= 0 and len(tmp_list) < top_k:
        tmp_list.append(tmp_list[d])

    while len(tmp_list) < top_k:
        tmp_list.append(full)
    assert len(tmp_list) == top_k
    return tmp_list

def prepare_candidates_fuse(label_seq_file, candidate_file, top_k=8, version=6, save=False,
                            no_full=False, no_word=False, no_truth=False, no_char=False):
    file_split = thwpy.check_split(label_seq_file)
    assert file_split == thwpy.check_split(candidate_file)
    train = file_split == 'train'
    # label_data = pd.read_csv(label_seq_file, sep='\t', names=['full', 'label'])
    label_data = thwpy.load_csv(label_seq_file, sep='\t')
    label_data = [row[1] for row in label_data]
    candidate_data = pd.read_csv(candidate_file, sep='\t', names=['full', 'abbr', 'candidate'])
    # candidate_data['label'] = label_data['label']
    candidate_data['label'] = label_data

    assert all(len(row['full']) == len(row['label']) for i, row in candidate_data.iterrows())

    statistics = {'word': 0, 'ground-truth': 0, 'char': 0, 'pad': 0, 'fuse': 0}
    candidates = []
    diff = 0
    edit_distances = []
    cnt = 0
    for idx, row in candidate_data.iterrows():
        diff += len(row['full']) - len(row['abbr'])
        edit_distances.append(len(row['full']) - len(row['abbr']))
        candidate_list = row['candidate'].split(';')

        assert len(candidate_list) == top_k

        tmp_list = list(set(candidate_list))
        tmp_list.sort(key=candidate_list.index)
        tmp_list = [c for c in tmp_list if is_subsequence(c, row['full']) and c != '']
        # tmp_list = [c for c in tmp_list if is_subsequence(c, row['full'])]
        if not no_full:
            if train and len(tmp_list) < top_k and row['full'] not in tmp_list:
                tmp_list.append(row['full'])
        else:
            tmp_list = [c for c in tmp_list if c != row['full']]
        word_len = len(row['full'])
        seg_word = jieba.lcut(row['full'])
        seg_word_len = len(seg_word)

        # 1st
        if not no_word:
            d = 1
            while len(tmp_list) < top_k and d <= seg_word_len:
                examples = sample_by_word_segmentation(seg_word, d)
                for example in examples:
                    if example not in tmp_list:
                        tmp_list.append(example)
                        statistics['word'] += 1
                        if len(tmp_list) == top_k:
                            break
                d += 1
        # 2nd
        if (not no_truth) and (not no_char) and train:
            d = 1
            while len(tmp_list) < top_k and d < word_len:
                truth_examples = sample_by_edit_distance(row['full'], row['label'], d)
                char_examples = sample_by_word_segmentation(row['full'], d)
                # examples = [e for e in thwpy.cyclic_merge(truth_examples, char_examples)]
                examples = [e for e in thwpy.cyclic_merge(char_examples, truth_examples)]
                for example in examples:
                    if example not in tmp_list:
                        tmp_list.append(example)
                        statistics['fuse'] += 1
                        if len(tmp_list) == top_k:
                            break
                d += 1
        else:
            if (not no_truth) and train:
                d = 1
                while len(tmp_list) < top_k and d <= word_len:
                    examples = sample_by_edit_distance(row['full'], row['label'], d)
                    for example in examples:
                        if example not in tmp_list:
                            tmp_list.append(example)
                            statistics['ground-truth'] += 1
                            if len(tmp_list) == top_k:
                                break
                    d += 1
            # 3rd
            if not no_char:
                d = 1
                while len(tmp_list) < top_k and d < word_len:
                    examples = sample_by_word_segmentation(row['full'], d)
                    for example in examples:
                        if example not in tmp_list:
                            tmp_list.append(example)
                            statistics['char'] += 1
                            if len(tmp_list) == top_k:
                                break
                    d += 1
        # len(word) <= 3，枚举肯定会有空的，需要补全

        # v1
        if no_full:
            tmp_list = [c for c in tmp_list if c != row['full'] and c != '']
            # tmp_list = [c for c in tmp_list if c != row['full']]
            d = len(tmp_list) - 1
            while d >= 0 and tmp_list[d] == row['abbr']:
                d -= 1
            while d >= 0 and len(tmp_list) < top_k:
                tmp_list.append(tmp_list[d])
                statistics['pad'] += 1
        else:
            while len(tmp_list) < top_k:
                tmp_list.append(row['full'])

        if no_full:
            assert row['full'] not in tmp_list and '' not in tmp_list
            # assert row['full'] not in tmp_list
        # padding
        while len(tmp_list) < top_k:
            tmp_list.append(row['full'])
            statistics['pad'] += 1
        assert len(tmp_list) == top_k
        if row['abbr'] in tmp_list:
            cnt += 1
        candidates.append(';'.join(tmp_list))
        print(f"{row['full']} | {row['abbr']} | {';'.join(tmp_list)}")
    print(f"Split: {file_split}")
    print(f"Average Edit Distance: {diff / len(candidate_data)}")
    edit_distances.sort()
    print(f"Middle Edit Distance: {edit_distances[len(candidate_data) // 2]}")
    print(f"Hit@{top_k}: {cnt / len(candidate_data)}")
    print(f"Statistics: {statistics}")
    candidate_data['candidate'] = candidates
    candidate_data = candidate_data.drop(columns=['label'])
    print(candidate_data)
    print(candidate_data.keys())

    if no_word:
        post_edit = 'noword'
    elif no_char:
        post_edit = 'nochar'
    elif no_truth:
        post_edit = 'notruth'
    else:
        post_edit = 'all'
    save_path = f'data/t5_v{version}_candidate_context_{post_edit}_top{top_k}_{file_split}.txt'

    print(f"Saved in {save_path}")
    if save:
        candidate_data.to_csv(save_path, sep='\t', header=False, index=False)


def load_dataset(file, to_dataset=True):
    fulls, abbrs = read_text_pairs(file)
    fulls = [str(s) for s in fulls]
    abbrs = [str(s) for s in abbrs]
    if to_dataset:
        return Dataset.from_dict({'full': fulls, 'abbr': abbrs})
    return fulls, abbrs


def load_dataset_with_context(file, to_dataset=True):
    data = pd.read_csv(file, sep='\t', names=CONTEXT_DATA_NAMES)
    fulls = [str(s) for s in data[CONTEXT_DATA_NAMES[0]]]
    abbrs = [str(s) for s in data[CONTEXT_DATA_NAMES[1]]]
    context_with_placeholders = [str(s) for s in data[CONTEXT_DATA_NAMES[-1]]]
#     暂时注释
#     for abbr, context_with_placeholder in zip(abbrs, context_with_placeholders):
#         assert abbr not in context_with_placeholder
    contexts = [context.replace(PLH, full) for context, full in zip(context_with_placeholders, fulls)]
    if to_dataset:
        return Dataset.from_dict({'full': fulls, 'abbr': abbrs, 'context': contexts})
    return fulls, abbrs, contexts


def save_text_pairs(src, target, file):
    pd.DataFrame({'src': src, 'target': target}).to_csv(file, sep='\t', header=False, index=False)


def split_pretrain_data(file, ratio=0.01):
    long, short = read_text_pairs(file)
    idx = list(range(len(long)))
    random.shuffle(idx)
    val_len = int(len(long) * ratio)
    print(val_len)
    print(len(long) - val_len)
    val_idx = idx[:val_len]
    train_idx = idx[val_len:]
    long, short = np.array(long), np.array(short)
    save_text_pairs(long[train_idx], short[train_idx], f"{file.split('.')[0]}_train.txt")
    save_text_pairs(long[val_idx], short[val_idx], f"{file.split('.')[0]}_val.txt")


def compare(file1, file2):
    l1 = []
    with open(file1, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            l1.append(line)
            if line == '':
                print(i + 1)
    print(len(l1))
    print(l1)
    _, abbrs = read_text_pairs(file2)
    hit = 0
    print(len(abbrs))
    for i, j in zip(l1, abbrs):
        if i != j:
            print(f"{i}, {j}")
        else:
            hit += 1
    print(hit)
    print(len(abbrs))
    print((hit + 2) / (len(abbrs) - 2))


def prepare_m2e_pairs(m2e_file):
    mentions, entities = read_text_pairs(m2e_file)
    cnt = 0
    for i, (m, e) in enumerate(zip(mentions, entities)):
        m = str(m)
        e = str(e)
        mentions[i] = m.strip().replace(' ', '')
        entities[i] = e.strip().replace(' ', '')
        if len(mentions[i]) < len(entities[i]):
            tmp = mentions[i]
            mentions[i] = entities[i]
            entities[i] = tmp
            cnt += 1
            print(f"Line {i + 1}: {mentions[i]}\t{entities[i]}")
    pd.DataFrame({'long': mentions, 'short': entities}).to_csv('data/pretrain.txt', sep='\t', header=False, index=False)


def is_subsequence(s, t):
    """判断s是否为t的子序列"""
    i = 0
    for ch in s:
        while i < len(t) and t[i] != ch:
            i += 1
        if i >= len(t):
            return False
        i += 1
    return True


def is_available(mention, entity):
    return re.search(r'\(.*\)', mention) is None and re.search(r'（.*）', mention) is None \
        # and is_subsequence(entity, mention)


def find_bracket(s):
    match = re.search(r'（.*）', s)
    if match is None:
        match = re.search(r'\(.*\)', s)
    return match


def filter_available_data(m2e_file):
    mentions, entities = read_text_pairs(m2e_file)
    new_mentions, new_entities = [], []
    total = len(mentions)
    hit = 0
    for m, e in zip(mentions, entities):
        m, e = str(m), str(e)
        if not is_available(m, e):
            match = find_bracket(m)
            start, end = match.start(), match.end()
            print(f"{m} | {e}")
            m = m[start + 1: end - 1] + m[:start]
            match = find_bracket(e)
            if match is not None:
                start, end = match.start(), match.end()
                e = e[start + 1: end - 1] + e[:start]
            print(f"{m} | {e}")
            print((start, end))
            hit += 1
        new_mentions.append(m)
        new_entities.append(e)
    print("Total: ", total)
    print("Hit: ", hit)
    assert len(new_mentions) == len(new_entities)
    print("After Filtering, Total: ", len(new_mentions))
    prefix = m2e_file.split('.')[0]
    pd.DataFrame({'long': new_mentions, 'short': new_entities}) \
        .to_csv(f'{prefix}_no_paren.txt', sep='\t', header=False, index=False)


def check_format(file):
    data = pd.read_csv(file, sep='\t', names=['src', 'target', 'context', 'context_with_placeholder'])
    print(data)


def check_logits():
    hit_1 = thwpy.load_csv('hit@1.txt')
    hit_3 = thwpy.load_csv('hit@3.txt')
    cnt = 0
    for i, (hit_1_item, hit_3_item) in enumerate(zip(hit_1, hit_3)):
        if hit_1_item[0] != hit_3_item[0]:
            cnt += 1
            print(f"Line: {i + 1}")
            print(hit_1_item)
            print(hit_3_item)
    print("Total: ", cnt)


def check_duplicate_candidates(file):
    data = pd.read_csv(file, sep='\t', names=['src', 'target', 'candidates', 'label'])
    data['candidates'] = data['candidates'].map(lambda x: x.split(';'))
    cnt = 0
    for idx, row in data.iterrows():
        abbr = row['target']
        candidates = row['candidates']
        if abbr in candidates:
            if candidates.count(abbr) > 1:
                cnt += 1
                print(f"Line: {idx + 1}")
                print(f"{row['src']} | {abbr} | {candidates}")
    print(cnt)


def check_subsequence_candidates(file):
    data = pd.read_csv(file, sep='\t', names=['full', 'abbr', 'candidate', 'label'])
    fulls, abbrs, candidates = data['full'].tolist(), data['abbr'].tolist(), [x.split(';') for x in data['candidate']]
    cnt = 0
    for full, abbr, candidate in zip(fulls, abbrs, candidates):
        if len(full) <= 5:
            cnt += 1
            print(f"{full}\t{abbr}\t{candidate}")
    print(cnt)


def truncate_pretrain_data(pretrain_file, filtering_file, match_target_file):
    pretrain_src, pretrain_target = read_text_pairs(pretrain_file)
    pretrain_src = [str(s) for s in pretrain_src]
    pretrain_target = [str(s) for s in pretrain_target]
    for i, (src, target) in enumerate(zip(pretrain_src, pretrain_target)):
        if len(src) < len(target):
            pretrain_src[i] = target
            pretrain_target[i] = src
    filtering_src, filtering_target = read_text_pairs(filtering_file)
    filtering_src = [str(s) for s in filtering_src]
    filtering_target = [str(s) for s in filtering_target]
    match_target_src, _ = read_text_pairs(match_target_file)
    match_target_src = [str(s) for s in match_target_src]
    nums = len(match_target_src) + len(filtering_src)
    print(f"Pretrain Total: {len(pretrain_src)}")
    print(f"Filtered Total: {len(filtering_src)}")
    print(f"Target Total: {len(match_target_src)}")
    print(f"Truncate: {nums}")
    final_src, final_target = [], []
    for src, target in zip(pretrain_src, pretrain_target):
        if src not in filtering_src:
            final_src.append(src)
            final_target.append(target)
    total = len(final_src)
    print(total)
    indices = list(range(total))
    random.shuffle(indices)
    final_src = [final_src[idx] for idx in indices[:nums]]
    final_target = [final_target[idx] for idx in indices[:nums]]
    pd.DataFrame({'src': final_src, 'target': final_target}).to_csv('data/pretrain_truncate.txt',
                                                                    sep='\t', index=False, header=False)


def sample_human_evaluate(file, header_type='gen', num=100, save=False):
    data = pd.read_csv(file, sep='\t', names=header_name_dict[header_type])
    indices = list(range(len(data)))
    random.shuffle(indices)
    indices = indices[:num]
    # indices = random.sample(indices, num)
    data = data.iloc[indices]
    print(data)
    save_path = f'eval/t5_{header_type}_human_evaluate_{num}.txt'
    print(f"Saved in {save_path}")
    if save:
        data.to_csv(save_path, sep='\t', header=False)


def sample_ranker_human_evaluate(candidate_file, predict_file, num=100, save=False):
    data = pd.read_csv(candidate_file, sep='\t', names=CANDIDATE_DATA_NAMES)
    predict_data = thwpy.load_list(predict_file)
    indices = list(range(len(data)))
    random.shuffle(indices)
    # indices = indices[:num]
    indices = indices[-num:]
    data = data.iloc[indices].reset_index()
    predict_data = [predict_data[i] for i in indices]
    format_data = []

    assert len(data) == len(predict_data)

    for i, row in data.iterrows():
        candidates = row[CANDIDATE_DATA_NAMES[-1]].split(';')
        predicts = predict_data[i].split(';')
        ranked = [candidates[int(idx)] for idx in predicts]
        format_data.append({
            'idx': row['index'],
            'full': row[CANDIDATE_DATA_NAMES[0]],
            'abbreviation': row[CANDIDATE_DATA_NAMES[1]],
            'candidate': ';'.join(ranked),
            'label': -1
        })
    print(format_data)
    save_path = f'eval/ranker_human_evaluate_{num}_format_v2.json'
    print(f"Saved in {save_path}")
    if save:
        json.dump(format_data, open(save_path, 'w+', encoding='utf-8'), ensure_ascii=False, indent=4)


def human_evaluation_hits(file):
    data = pd.read_csv(file, sep='\t', names=['full', 'src', 'candidate'], index_col=0)
    labels = [int(candidate[-1]) for candidate in data['candidate']]
    print(sum(labels) / len(labels))


def human_evaluation_prec(file):
    with open(file, 'r') as f:
        data = json.load(f)
    top_k = len(data[0]['candidate'].split(';'))
    print(sum(item['label'] for item in data) / len(data))


def check_difference(pretrained_eval_file, no_pretrained_eval_file, save=False):
    pretrained_result = pd.read_csv(pretrained_eval_file, sep='\t', names=['full', 'abbr', 'candidate'])
    no_pretrained_result = pd.read_csv(no_pretrained_eval_file, sep='\t', names=['full', 'abbr', 'candidate'])
    pretrained_wrong = []
    no_pretrained_wrong = []
    pretrained_cnt = 0
    no_pretrained_cnt = 0
    for i, (full, abbr, pretrained_candidate, no_pretrained_candidate) in \
            enumerate(zip(pretrained_result['full'], pretrained_result['abbr'], pretrained_result['candidate'],
                          no_pretrained_result['candidate'])):
        pretrained_candidate = str(pretrained_candidate).split(';')
        no_pretrained_candidate = str(no_pretrained_candidate).split(';')
        if abbr in pretrained_candidate:
            pretrained_cnt += 1
        if abbr in no_pretrained_candidate:
            no_pretrained_cnt += 1
        if abbr not in no_pretrained_candidate and abbr in pretrained_candidate:
            no_pretrained_wrong.append(
                [str(i), full, abbr, ';'.join(pretrained_candidate), ';'.join(no_pretrained_candidate)])
            # print('\t'.join([str(i), full, abbr, ';'.join(pretrained_candidate), ';'.join(no_pretrained_candidate)]))
        if abbr not in pretrained_candidate and abbr in no_pretrained_candidate:
            pretrained_wrong.append(
                [str(i), full, abbr, ';'.join(no_pretrained_candidate), ';'.join(pretrained_candidate)])
    print(len(no_pretrained_wrong))
    print(len(pretrained_wrong))
    print(pretrained_cnt)
    print(no_pretrained_cnt)
    if save:
        thwpy.save_csv(no_pretrained_wrong, 'eval/no_pretrained_wrong.txt', sep='\t')
        thwpy.save_csv(pretrained_wrong, 'eval/pretrained_wrong.txt', sep='\t')


def format_human_evaluation(file, save=False):
    data = pd.read_csv(file, sep='\t', names=['idx', 'full', 'abbr', 'candidate'])
    format_data = []
    for idx, row in data.iterrows():
        format_data.append({
            'idx': row['idx'],
            'full': row['full'],
            'abbreviation': row['abbr'],
            'candidate': row['candidate'],
            'label': -1
        })
    save_path = f"{file.split('.')[0]}_format_v1.json"
    print(f"Saved in {save_path}")
    if save:
        json.dump(format_data, open(save_path, 'w+', encoding='utf-8'), ensure_ascii=False,
                  indent=4)


def merge_multi_human_evaluation(gen_file, rule_file, constrain_file, save=False):
    gen_data = json.load(open(gen_file, 'r', encoding='utf-8'))
    rule_data = json.load(open(rule_file, 'r', encoding='utf-8'))
    constrain_data = json.load(open(constrain_file, 'r', encoding='utf-8'))
    merge = []
    for idx, item in enumerate(gen_data):
        merge.append({
            'idx': item['idx'],
            'full': item['full'],
            'abbreviation': item['abbreviation'],
            'gen': item['candidate'],
            'rule': rule_data[idx]['candidate'],
            'constrain': constrain_data[idx]['candidate'],
            'label': -1
        })
    save_path = f"eval/t5_format_v1.json"
    print(f"Saved in {save_path}")
    if save:
        json.dump(merge, open(save_path, 'w+', encoding='utf-8'), ensure_ascii=False, indent=4)


def common_prefix(s, t):
    for i, (s_ch, t_ch) in enumerate(zip(s, t)):
        if s_ch != t_ch:
            return s[:i].rstrip('_')
    return s.split('.')[0]


def append_file(*files, file_type='gen', save=False):
    assert file_type in header_name_dict

    prefix = common_prefix(files[0], files[1])
    names = header_name_dict[file_type]
    total_data = pd.read_csv(files[0], sep='\t', names=names)
    for file in files[1:]:
        data = pd.read_csv(file, sep='\t', names=names)
        total_data = total_data.append(data)
    total_data = total_data.reset_index(drop=True)
    print(total_data)
    print(total_data.keys())
    save_path = f"data/final_{files[0].split('/')[-1]}"
    print(f"Saved in {save_path}")
    if save:
        total_data.to_csv(save_path, sep='\t', header=False, index=False)


def split_to_half(file, file_type='gen', save=False):
    assert file_type in header_name_dict

    names = header_name_dict[file_type]
    total_data = pd.read_csv(file, sep='\t', names=names)
    val_len = len(total_data) // 2 + 1
    val_data = total_data.iloc[:val_len]
    test_data = total_data.iloc[val_len:].reset_index(drop=True)
    val_save_path = file.replace('test', 'val')
    val_save_path = f"data/final_{val_save_path.split('/')[-1]}"
    test_save_path = f"data/final_{file.split('/')[-1]}"
    print(val_data)
    print(test_data)
    print(f"Val set saved in {val_save_path}")
    print(f"Test set saved in {test_save_path}")
    if save:
        val_data.to_csv(val_save_path, sep='\t', header=False, index=False)
        test_data.to_csv(test_save_path, sep='\t', header=False, index=False)


def restore_like_original_dataset(current_file, file_type='context', original_file='data/dataset_gen.txt'):
    names = header_name_dict[file_type]
    current_data = pd.read_csv(current_file, sep='\t', names=names)
    original_data = pd.read_csv(original_file, sep='\t', names=GEN_DATA_NAMES)
    current_data_dict = {}
    for i, row in current_data.iterrows():
        value = {}
        key = f"{row[names[0]]}-{row[names[1]]}"
        assert key not in current_data_dict
        for name in names[2:]:
            value[name] = row[name]
        current_data_dict[key] = value
    pprint(len(current_data_dict))

    contexts, contexts_with_placeholder = [], []
    for i, row in original_data.iterrows():
        key = f"{row[GEN_DATA_NAMES[0]]}-{row[GEN_DATA_NAMES[1]]}"
        assert key in current_data_dict
        contexts.append(current_data_dict[key][names[2]])
        contexts_with_placeholder.append(current_data_dict[key][names[3]])

    assert len(contexts) == len(current_data_dict) and len(contexts_with_placeholder) == len(current_data_dict)

    original_data[names[2]] = contexts
    original_data[names[3]] = contexts_with_placeholder
    print(original_data)
    print(original_data.keys())
    original_data.to_csv(current_file, sep='\t', index=False, header=False)


def k_fold_split(file, k=5):
    data = pd.read_csv(file, sep='\t', names=CONTEXT_DATA_NAMES)
    k_fold = KFold(n_splits=k, shuffle=False)
    prefix = file.split('.')[0]
    for i, (train, test) in enumerate(k_fold.split(data[CONTEXT_DATA_NAMES[0]])):
        fold_train, fold_test = data.iloc[train], data.iloc[test]
        print(f"fold-{i + 1} test: {fold_test}")
        print(f"fold-{i + 1} test num: {len(fold_test)}")
        print(f"fold-{i + 1} train: {fold_train}")
        print(f"fold-{i + 1} train num: {len(fold_train)}")
        fold_train.to_csv(f'{prefix}_fold_{i + 1}_train.txt', sep='\t', index=False, header=False)
        fold_test.to_csv(f'{prefix}_fold_{i + 1}_test.txt', sep='\t', index=False, header=False)


def get_statistics(*files):
    context_len_dict = dict()
    for file in files:
        total_full_len = 0
        total_abbr_len = 0
        total_context_len = 0
        total = 0
        max_len = 0
        data = pd.read_csv(file, sep='\t', names=CONTEXT_DATA_NAMES)
        total += len(data)
        context_lens = []
        for idx, row in data.iterrows():
            total_full_len += len(row[CONTEXT_DATA_NAMES[0]])
            total_abbr_len += len(row[CONTEXT_DATA_NAMES[1]])
            total_context_len += len(row[CONTEXT_DATA_NAMES[2]])
            context_lens.append(len(row[CONTEXT_DATA_NAMES[2]]))
            max_len = max(max_len, len(row[CONTEXT_DATA_NAMES[2]]))
            context_len_dict[len(row[CONTEXT_DATA_NAMES[2]])] = \
                context_len_dict.get(len(row[CONTEXT_DATA_NAMES[2]]), 0) + 1
        context_lens.sort()
        print("Total: ", total)
        print("Avg. length of full forms: ", total_full_len / total)
        print("Avg. length of abbreviations: ", total_abbr_len / total)
        print("Avg. length of context: ", total_context_len / total)
        print("Median. length of context: ", context_lens[total // 2])
        print("Maximum length of context: ", max_len)
    print(context_len_dict)


def get_context_distribution(*files):
    intervals = ['<100', '100~200', '200~300', '300~400', '400~500', '500~600', '>600']
    context_dict = {k: 0 for k in intervals}
    for file in files:
        data = pd.read_csv(file, sep='\t', names=CONTEXT_DATA_NAMES)
        for context in data['context']:
            if len(context) < 100:
                context_dict[intervals[0]] += 1
            elif len(context) < 200:
                context_dict[intervals[1]] += 1
            elif len(context) < 300:
                context_dict[intervals[2]] += 1
            elif len(context) < 400:
                context_dict[intervals[3]] += 1
            elif len(context) < 500:
                context_dict[intervals[4]] += 1
            elif len(context) < 600:
                context_dict[intervals[5]] += 1
            else:
                context_dict[intervals[6]] += 1
    print(context_dict)


def truncate_training_data_by_ratio(file, k=4, context=True, save=False):
    assert thwpy.check_split(file) == 'train'

    data = pd.read_csv(file, sep='\t', names=RANKER_DATA_NAMES)
    total = len(data)
    prefix = file.split('.')[0]
    for i in range(k - 1):
        ratio = (i + 1) / k
        truncate_len = int(total * ratio)
        truncate_train_data = data.iloc[:truncate_len]
        truncate_train_data = truncate_train_data[CONTEXT_DATA_NAMES] if context else truncate_train_data[
            GEN_DATA_NAMES]
        ratio = int(1000 * ratio)
        save_path = f"data/final_dataset_gen_train{'_context' if context else ''}_{ratio}.txt"
        print(f"Saved in {save_path}")
        if save:
            truncate_train_data.to_csv(save_path, sep='\t', index=False, header=False)


def construct_few_shot_dataset(file, shots=1, context=True, save=False):
    assert thwpy.check_split(file) == 'train'

    data = pd.read_csv(file, sep='\t', names=RANKER_DATA_NAMES)
    few_shot_data = data.iloc[:shots]
    few_shot_data = few_shot_data[CONTEXT_DATA_NAMES] if context else few_shot_data[GEN_DATA_NAMES]
    print(few_shot_data)
    save_path = f"data/final_dataset_gen_train{'_context' if context else ''}_shot{shots}.txt"
    print(f"Saved in {save_path}")
    if save:
        few_shot_data.to_csv(save_path, sep='\t', index=False, header=False)


def construct_ratio_of_dataset(file, ratio=0.01, context=True, save=False):
    assert thwpy.check_split(file) == 'train'
    data = pd.read_csv(file, sep='\t', names=RANKER_DATA_NAMES)
    total = len(data)
    samples = int(ratio * total)
    indices = list(range(total))
    random.shuffle(indices)
    indices = indices[:samples]
    few_shot_data = data.iloc[indices]
    few_shot_data = few_shot_data[CONTEXT_DATA_NAMES] if context else few_shot_data[GEN_DATA_NAMES]
    save_path = f"data/final_dataset_gen_train{'_context' if context else ''}_ratio{int(100 * ratio)}.txt"
    print(few_shot_data)
    print(f"Saved in {save_path}")
    if save:
        few_shot_data.to_csv(save_path, sep='\t', index=False, header=False)


def construct_ratio_of_context(file, ratio=0.01, save=False):
    assert thwpy.check_split(file) == 'train'
    assert thwpy.check_split(file) == 'train'
    data = pd.read_csv(file, sep='\t', names=CONTEXT_DATA_NAMES)
    total = len(data)
    samples = int(ratio * total)
    indices = list(range(total))
    random.shuffle(indices)
    indices = indices[:samples]
    few_shot_data = data.iloc[indices]
    prefix, suffix = file.split('.')
    save_path = prefix + f'_ratio{int(ratio * 100)}.' + suffix
    print(few_shot_data)
    print(f"Saved in {save_path}")
    if save:
        few_shot_data.to_csv(save_path, sep='\t', index=False, header=False)


def construct_ratio_of_origin_dataset(ratio=0.01, save=False):
    data = thwpy.load_csv('data/final_dataset_train.txt')
    total = len(data)
    samples = int(ratio * total)
    indices = list(range(total))
    random.shuffle(indices)
    indices = indices[:samples]
    few_shot_origin_data = [data[i] for i in indices]
    save_path = f"data/final_dataset_train_ratio{int(100 * ratio)}.txt"

    assert all(len(item[0]) == len(item[1]) for item in few_shot_origin_data)
    pprint(few_shot_origin_data)
    print(len(few_shot_origin_data))
    print(f"Saved in {save_path}")
    if save:
        thwpy.save_csv(few_shot_origin_data, save_path, sep='\t')


def construct_gen_context_dataset(file, truncate=0, save=False):
    assert not (truncate > 0 and 'extract' in file)
    split = thwpy.check_split(file)
    data = pd.read_csv(file, sep='\t', names=RANKER_DATA_NAMES)
    data = data[CONTEXT_DATA_NAMES]
    save_path = f"data/final_dataset_gen_{split}_context{f'_truncate{truncate}' if truncate > 0 else ''}.txt"
    print(data)
    print(f"Saved in {save_path}")
    if save:
        data.to_csv(save_path, sep='\t', index=False, header=False)


def filter_testing_set(test_file, filter_file, filter_num=30, save=False):
    test_data = pd.read_csv(test_file, sep='\t', names=GEN_DATA_NAMES)
    filter_data = pd.read_csv(filter_file, sep='\t', names=['id', 'full', 'abbr', 'right', 'wrong'])
    print(filter_data)
    filter_idx = filter_data['id'].tolist()[:filter_num]
    test_data = test_data.drop(filter_idx).reset_index(drop=True)
    print(test_data)
    prefix = test_file.split('.')[0]
    save_path = f'{prefix}_mod.txt'
    if save:
        test_data.to_csv(save_path, sep='\t', index=False, header=False)


def avoid_leakage(file):
    data = pd.read_csv(file, sep='\t', names=CONTEXT_DATA_NAMES)
    for i, row in data.iterrows():
        assert row['target'] not in row['context_with_placeholder']
    print("NO LEAKAGE")


def check_prefix_data(file):
    data = pd.read_csv(file, sep='\t', names=['full', 'abbr', 'predict'])
    prefix_total, other_total = 0, 0
    prefix_cnt, other_cnt = 0, 0
    for i, row in data.iterrows():
        predict = row['predict'].split(';')
        if row['full'].startswith(row['abbr']):
            prefix_total += 1
            if row['abbr'] in predict:
                prefix_cnt += 1
        else:
            other_total += 1
            if row['abbr'] in predict:
                other_cnt += 1
    print(file)
    print(f"Prefix: {prefix_cnt} / {prefix_total} = {prefix_cnt / prefix_total}")
    print(f"Other: {other_cnt} / {other_total} = {other_cnt / other_total}")


def check_pretrain_leakage(pretrain_file, abbr_file):
    pretrain_data = pd.read_csv(pretrain_file, sep='\t', names=GEN_DATA_NAMES)
    data = pd.read_csv(abbr_file, sep='\t', names=GEN_DATA_NAMES)
    data_dict = {k: v for k, v in zip(data[GEN_DATA_NAMES[1]], data[GEN_DATA_NAMES[0]])}
    cnt = 0
    remove_idx = []
    for i, row in pretrain_data.iterrows():
        if row[GEN_DATA_NAMES[1]] in data_dict and row[GEN_DATA_NAMES[0]] == data_dict[row[GEN_DATA_NAMES[1]]]:
            cnt += 1
            remove_idx.append(i)
    print(len(pretrain_data))
    print(cnt)
    print("After Filtering: ")
    pretrain_data = pretrain_data.drop(remove_idx).reset_index(drop=True)
    print(len(pretrain_data))
    print(pretrain_data)
    pretrain_data.to_csv('data/pretrain_noleak_train.txt', sep='\t', header=False, index=False)


def search_ranker_data(file, query):
    data = pd.read_csv(file, sep='\t', names=RANKER_DATA_NAMES)
    for idx, row in data.iterrows():
        if row[RANKER_DATA_NAMES[0]] == query:
            pprint(f"context w [PLH]: {row[RANKER_DATA_NAMES[-3]]}")
            pprint(f"candidates: {row[RANKER_DATA_NAMES[-2]]}")
            break


def search_candidate_data(file, query):
    data = pd.read_csv(file, sep='\t', names=CANDIDATE_DATA_NAMES)
    for idx, row in data.iterrows():
        if row[CANDIDATE_DATA_NAMES[0]] == query:
            pprint(f"candidates: {row[CANDIDATE_DATA_NAMES[-1]]}")
            break


def get_label_seq(full, abbr):
    label = [0] * len(full)
    idx = 0
    if len(abbr) == 0:
        return label
    for i, ch in enumerate(full):
        if ch == abbr[idx]:
            label[i] = 1
            idx += 1
            if idx == len(abbr):
                break
    return label


def char_acc(eval_labels, golden_labels, fulls, abbrs, predicts):
    correct = 0
    total = 0
    ratio_sum = 0
    for i, (eval_label, golden_label) in enumerate(zip(eval_labels, golden_labels)):
        print(eval_label)
        print(golden_label)
        assert len(eval_label) == len(golden_label)

        total += len(golden_label)
        correct += sum(e == g for e, g in zip(eval_label, golden_label))
        ratio = sum(e == g for e, g in zip(eval_label, golden_label)) / len(golden_label)
        ratio_sum += ratio
        # if ratio < 0.6:
        #     print(golden_label)
        #     print(eval_label)
        #     print(fulls[i])
        #     print(abbrs[i])
        #     print(predicts[i])
    print(correct / total)
    print(ratio_sum / len(eval_labels))
    # return ratio / len(eval_labels)


def generation_char_acc(eval_file, golden_file, top_k=1):
    eval_data = pd.read_csv(eval_file, sep='\t', names=CANDIDATE_DATA_NAMES)
    golden_file = thwpy.load_csv(golden_file, sep='\t')
    print("Hit@1: ", sum(
        c.split(';')[0] == t for c, t in zip(eval_data[CANDIDATE_DATA_NAMES[-1]], eval_data[CANDIDATE_DATA_NAMES[-2]]))
          / len(eval_data))
    eval_labels = []
    golden_labels = [[int(l) for l in label[1]] for label in golden_file]
    for i, row in eval_data.iterrows():
        candidates = row[CANDIDATE_DATA_NAMES[2]].split(';')[:top_k]
        max_ratio = 0
        best_label = []
        for candidate in candidates:
            eval_label = get_label_seq(row[CANDIDATE_DATA_NAMES[0]], candidate)
            ratio = sum(e == g for e, g in zip(eval_label, golden_labels[i]))
            if ratio >= max_ratio:
                max_ratio = ratio
                best_label = eval_label
        eval_labels.append(best_label)
    char_acc(eval_labels, golden_labels, eval_data[CANDIDATE_DATA_NAMES[0]].tolist(),
             eval_data[CANDIDATE_DATA_NAMES[1]].tolist(), eval_data[CANDIDATE_DATA_NAMES[2]].tolist())
    print(accuracy_score(golden_labels, eval_labels))


def ranker_char_acc(predict_file, golden_file, top_k=1):
    predicts = pd.read_csv(predict_file, sep='\t', names=['predict'], header=None)
    goldens = pd.read_csv(golden_file, sep='\t', names=CANDIDATE_DATA_NAMES)
    goldens['predict'] = predicts['predict']
    correct = 0
    golden_labels, eval_labels = [], []
    predict_candidates = []
    for i, row in goldens.iterrows():
        predict = row['predict'].split(';')
        predict_idx = [int(p) for p in predict[:top_k]]
        candidate = row[CANDIDATE_DATA_NAMES[-1]].split(';')
        predict_candidate = [candidate[i] for i in predict_idx]
        if predict_candidate[0] == row[CANDIDATE_DATA_NAMES[1]]:
            correct += 1
        max_ratio = 0
        best_label = []
        golden_label = get_label_seq(row[CANDIDATE_DATA_NAMES[0]], row[CANDIDATE_DATA_NAMES[1]])
        golden_labels.append(golden_label)
        for c in predict_candidate:
            assert is_subsequence(c, row[CANDIDATE_DATA_NAMES[0]])
            eval_label = get_label_seq(row[CANDIDATE_DATA_NAMES[0]], c)
            ratio = sum(e == g for e, g in zip(eval_label, golden_label))
            if ratio >= max_ratio:
                max_ratio = ratio
                best_label = eval_label
        eval_labels.append(best_label)
        predict_candidates.append(predict_candidate)
    print(correct / len(goldens))
    char_acc(eval_labels, golden_labels, goldens[CANDIDATE_DATA_NAMES[0]].tolist(),
             goldens[CANDIDATE_DATA_NAMES[1]].tolist(), predict_candidates)
    print(accuracy_score(golden_labels, eval_labels))


def check_label_seq(file, gen_file):
    data = thwpy.load_csv(file, sep='\t')
    gen_data = pd.read_csv(gen_file, sep='\t', names=GEN_DATA_NAMES)
    cnt = 0
    goldens = []
    assert all(len(row[0]) == len(row[1]) for row in data)
    for i, row in enumerate(data):
        golden = []
        if len(row[0]) != len(row[1]):
            cnt += 1
            print(i)
        for ch, label in zip(row[0], row[1]):
            if label == '1':
                golden.append(ch)
        goldens.append(''.join(golden))
    # abbrs = gen_data[GEN_DATA_NAMES[1]].tolist()
    # print(data[:5])
    # assert all(abbr == golden for abbr, golden in zip(abbrs, goldens))
    print(cnt)


def append_gen_file(*files, save=False):
    total_data = thwpy.load_csv(files[0], sep='\t')
    for file in files[1:]:
        data = thwpy.load_csv(file, sep='\t')
        total_data.extend(data)
    pprint(total_data)
    save_path = f"data/final_{files[0].split('/')[-1]}"

    assert all(len(row[0]) == len(row[1]) for row in total_data)

    pprint(len(total_data))
    pprint(f"Saved in {save_path}")
    if save:
        thwpy.save_csv(total_data, save_path, sep='\t')


def split_gen_test_to_half(file, save=False):
    total_data = thwpy.load_csv(file, sep='\t')
    val_len = len(total_data) // 2 + 1
    val_data = total_data[:val_len]
    test_data = total_data[val_len:]
    val_save_path = file.replace('test', 'val')
    val_save_path = f"data/final_{val_save_path.split('/')[-1]}"
    test_save_path = f"data/final_{file.split('/')[-1]}"
    pprint(len(val_data))
    pprint(len(test_data))
    print(f"Val set saved in {val_save_path}")
    print(f"Test set saved in {test_save_path}")
    pprint(val_data[:5])
    pprint(test_data[:5])
    if save:
        thwpy.save_csv(val_data, val_save_path, sep='\t')
        thwpy.save_csv(test_data, test_save_path, sep='\t')


def check_identical(file1, file2):
    data1 = pd.read_csv(file1, sep='\t', names=CANDIDATE_DATA_NAMES)
    data2 = pd.read_csv(file2, sep='\t', names=CANDIDATE_DATA_NAMES)
    for i, (d1, d2) in enumerate(zip(data1[CANDIDATE_DATA_NAMES[-1]], data2[CANDIDATE_DATA_NAMES[-1]])):
        if set(d1) != set(d2):
            print(i)
            print(d1)
            print(d2)


def label_human_evaluation_file_acc(file, save=False):
    data = json.load(open(file, 'r', encoding='utf-8'))
    cnt = 0
    for item in data:
        candidates = item['candidate'].split(';')
        if candidates[0] == item['abbreviation']:
            cnt += 1
            item['label'] = 'a'
    if save:
        json.dump(data, open(file, 'w+', encoding='utf-8'), ensure_ascii=False, indent=4)
    print(f"ACC: {cnt} / {len(data)} = {cnt / len(data)}")


def label_human_evaluation_file_hit(file, save=False):
    data = json.load(open(file, 'r', encoding='utf-8'))
    cnt = 0
    bad_cases = []
    for idx, item in enumerate(data):
        candidates = item['candidate'].split(';')
        if item['abbreviation'] in candidates:
            cnt += 1
            item['label'] = 'h'
        else:
            bad_cases.append(item)
    if save:
        json.dump(data, open(file, 'w+', encoding='utf-8'), ensure_ascii=False, indent=4)
        save_path = f"{file.split('.')[0]}_bad.json"
        json.dump(bad_cases, open(save_path, 'w+', encoding='utf-8'), ensure_ascii=False, indent=4)
    print(f"Hit: {cnt} / {len(data)} = {cnt / len(data)}")


def count_human_evaluation_file(file):
    data = json.load(open(file, 'r', encoding='utf-8'))
    cnt = 0
    counter = Counter([item['label'] for item in data])
    print(counter)


def compare_human_evaluation(generate_file, ranker_file):
    generates = json.load(open(generate_file, 'r+', encoding='utf-8'))
    rankers = json.load(open(ranker_file, 'r+', encoding='utf-8'))
    assert len(generates) == len(rankers)
    for g, r in zip(generates, rankers):
        if g['label'] == 'o' or g['label'] == 'n':
            if r['label'] == 'r' or r['label'] == 'a':
                print(g)
                print(r)


def transform_human_evaluation_file(file, save=False):
    data = json.load(open(file, 'r', encoding='utf-8'))
    cnt = 0
    for item in data:
        item.pop('label')
        candidates = item['candidate'].split(';')
        if item['abbreviation'] in candidates:
            item['hit'] = 1
            cnt += 1
        else:
            item['hit'] = -1
        item['precision'] = -1
    prefix = file.split('.')[0]
    print(f"Hit@12: {cnt} / {len(data)} = {cnt / len(data)}")
    save_path = f'{prefix}_hp.json'
    print(f"Saved in {save_path}")
    print(data)
    if save:
        json.dump(data, open(save_path, 'w+', encoding='utf-8'), ensure_ascii=False, indent=4)


def save_server_eval_bad_case(file, save=False):
    data = pd.read_csv(file, sep='\t')
    bad_cases = []
    for i, row in data.iterrows():
        candidates = row['candidates'].split(';')
        if row['target'] not in candidates:
            bad_cases.append({
                'full': row['src'],
                'abbreviation': row['target'],
                'candidate': row['candidates'],
                'label': -1,
                'id': i
            })
    save_path = f"{file.split('.')[0]}_bad.json"
    print(bad_cases)
    print(f"Saved in {save_path}")
    if save:
        json.dump(bad_cases, open(save_path, 'w+', encoding='utf-8'), ensure_ascii=False, indent=4)


def count_no_subsequence(file):
    data = pd.read_csv(file, sep='\t', names=CANDIDATE_DATA_NAMES)
    cnt = 0
    total = 0
    for i, row in data.iterrows():
        candidates = row[CANDIDATE_DATA_NAMES[-1]].split(';')
        total += len(candidates)
        for c in candidates:
            if not is_subsequence(c, row[CANDIDATE_DATA_NAMES[0]]):
                cnt += 1
    print(f"{cnt} / {total} = {cnt / total}")


def sample_predictions_with_scores(predict_file, context_predict_file, num=300, save=False):
    names = CANDIDATE_DATA_NAMES + ['score']
    predicts = pd.read_csv(predict_file, names=names, sep='\t')
    context_predicts = pd.read_csv(context_predict_file, names=names, sep='\t')

    assert len(predicts) == len(context_predicts)

    indices = list(range(len(predicts)))
    random.shuffle(indices)
    indices = indices[:num]
    predicts = predicts.iloc[indices]
    context_predicts = context_predicts.iloc[indices]
    print(predicts)
    merge = []
    for (i, row), (j, context_row) in zip(predicts.iterrows(), context_predicts.iterrows()):
        merge.append({
            'idx': i,
            'full': row[CANDIDATE_DATA_NAMES[0]],
            'abbreviation': row[CANDIDATE_DATA_NAMES[1]],
            'candidate': row[CANDIDATE_DATA_NAMES[2]],
            'context_candidate': context_row[CANDIDATE_DATA_NAMES[2]],
            'score': row[names[-1]],
            'context_score': context_row[names[-1]],
            'label': -1
        })
    pprint(merge)
    save_path = f'eval/sample_{num}_compare_score.json'
    print(f"Saved in {save_path}")
    if save:
        json.dump(merge, open(save_path, 'w+', encoding='utf-8'), ensure_ascii=False, indent=4)


def construct_ranker_context(context_file, save=False):
    """
    构建用于ranker的context数据
    :param context_file: context文件，包含3列 [src(全称), target(缩略词), context]
    :param save: 是否保存
    :return: 保存的context数据路径
    """
    context_with_placeholder_save_path = prepare_context_with_placeholder(file=context_file, save=save)
    extracted_context_save_path = extract_context_sentence(context_with_placeholder_save_path, truncate_len=150, save=save)
    return extracted_context_save_path


def construct_ranker_dataset(label_seq_file, candidate_file, context_file, top_k=12, version=1, save=False,
                             no_word=False, no_truth=False, no_char=False):
    """
    构建ranker的数据集
    :param label_seq_file: 带有0/1序列的文件，2列[src(全称), label(对应缩略词的0/1序列)]
    :param candidate_file: 带有candidates的文件，3列[src(全称), target(缩略词), candidates]
    :param context_file: 需要合并的context文件，4列[src(全称), target(缩略词), context, context_with_placeholder]
    :param top_k: candidates个数，默认12
    :param version:
    :param save: 是否保存
    :param no_char: 是否移除 char-based rule
    :param no_truth: 是否移除 ground-truth-based rule
    :param no_word: 是否移除 word-based rule
    :return: 最终保存的数据集的路径
    """
    split = thwpy.check_split(label_seq_file)
    assert split == thwpy.check_split(candidate_file) and split == thwpy.check_split(context_file), '数据集划分不同'
    candidate_save_path, candidate_hit_k = prepare_candidates(label_seq_file=label_seq_file,
                                                              candidate_file=candidate_file,
                                                              version=version,
                                                              top_k=top_k,
                                                              no_full=True,
                                                              no_word=no_word, no_truth=no_truth, no_char=no_char,
                                                              save=save)
    candidate_w_label_save_path, label_hit_k = prepare_label(candidate_save_path, top_k=top_k, version=version,
                                                             save=save)

    assert candidate_hit_k == label_hit_k

    dataset_save_path = merge_data(context_file,
                                   candidate_w_label_save_path,
                                   top_k=top_k,
                                   truncate_len=150,
                                   version=version,
                                   save=save)
    return dataset_save_path


if __name__ == '__main__':
    # val, train, test
    # prepare context
    extracted_context_path = construct_ranker_context(context_file='data/final_dataset_gen_test_with_long_context_merged.txt', save=False)
    construct_ranker_dataset(label_seq_file='data/final_dataset_test.txt',
                             candidate_file='eval/final_t5_pretrain_noleak_64_accum_1_10_checkpoint-51843_context_64_10_best_test_beam_32_candidates_12.txt',
                             context_file=extracted_context_path,
                             top_k=12,
                             version=2,
                             save=False)

    # data = pd.read_csv('data/final_ranker_extract_truncate150_context_test.txt', sep='\t', names=CONTEXT_DATA_NAMES)
    # data = data[[CONTEXT_DATA_NAMES[0], CONTEXT_DATA_NAMES[2]]]
    # data.to_csv('data/predict_w_context.txt', sep='\t', index=False, header=False)

    # construct_gen_context_dataset('data/t5_v2_ranker_context_all_truncate150_top12_train.txt', truncate=150, save=False)

    # prepare_candidates(label_seq_file='data/final_dataset_test.txt',
    #                    candidate_file='eval/final_t5_pretrain_noleak_64_accum_1_10_checkpoint-51843_context_truncate50_64_10_best_test_beam_32_candidates_12.txt',
    #                    top_k=12,
    #                    save=False,
    #                    version=2,
    #                    no_full=True)

    # prepare_label('data/t5_v2_candidate_context_truncate50_all_top12_test.txt', top_k=12, version=2, save=False)

    # merge_data('data/final_ranker_truncate50_context_train.txt',
    #            'data/t5_v2_candidate_context_truncate50_all_top12_w_label_train.txt',
    #            top_k=12, truncate_len=50, version=2, save=False)


    # truncate_context_len('data/final_ranker_context_test.txt', max_len=50, truncate_len=50, save=False)

    # extract_context_sentence('data/ranker_context_train.txt', truncate_len=150)

    # print(sample_by_edit_distance('北京大学', '1010', d=1))

    # split_pretrain_data('data/pretrain_truncate.txt')

    # human_evaluation_hits('eval/human_evaluate_100_labeled.txt')

    # check_difference('../bart_abbr/eval/cpt_pretrain_128_10_checkpoint-69320_8_10_best_final_test_test_beam_5_candidates_1.txt',
    #                  'eval/final_cpt-base_8_10_best_test_candidates_1.txt', save=True)

    # human_evaluation_prec('eval/human_evaluate_100_format_prec.json')

    # append_file('data/dataset_gen_train_with_long_context_merged.txt',
    #             'data/dataset_gen_val_with_long_context_merged.txt', file_type='raw_context', save=False)

    # split_to_half('data/dataset_gen_test_with_long_context_merged.txt', file_type='raw_context', save=False)

    # restore_like_original_dataset('data/ranker_extract_truncate150_context.txt',
    #                               file_type='context', original_file='data/dataset_gen.txt')

    # get_statistics('data/final_ranker_context_train.txt',
    #                'data/final_ranker_context_val.txt',
    #                'data/final_ranker_context_test.txt')

    # get_context_distribution('data/final_ranker_context_train.txt',
    #                         'data/final_ranker_context_val.txt',
    #                         'data/final_ranker_context_test.txt')

    # avoid_leakage('data/final_ranker_extract_truncate150_context_test.txt')

    # check_difference('eval/final_cpt-base_128_10_best_test_beam_5_candidates_1.txt',
    #                  'eval/final_cpt_pretrain_128_10_checkpoint-69320_256_10_best_test_beam_5_candidates_1.txt', save=False)

    # search_ranker_data('data/final_v3_ranker_all_truncate150_top12_train.txt', '切尔西足球俱乐部')
    # search_candidate_data('eval/final_cpt_pretrain_noleak_128_accum_1_10_checkpoint-60487_128_10_best_train_beam_32_candidates_12.txt', '复旦大学附属中学')

    # generation_char_acc('eval/final_cpt_pretrain_noleak_128_accum_1_10_checkpoint-60487_128_10_best_test_beam_5_candidates_3.txt', 'data/final_dataset_test.txt', top_k=3)

    # ranker_char_acc('eval/final_v4_4e_ranker_extract_all_truncate150_top12_add_b_4_accu_512_10_epoch_9.txt',
    #                 'data/final_v4_candidate_all_top12_test.txt', top_k=3)

    # count_human_evaluation_file('eval/human_evaluate_300_format_v1.json')
    # count_human_evaluation_file('eval/ranker_human_evaluate_300_format_v1.json')

    # compare_human_evaluation('eval/human_evaluate_300_format_v1.json', 'eval/ranker_human_evaluate_300_format_v1.json')

    # sample_human_evaluate('eval/final_t5_pretrain_noleak_64_accum_1_10_checkpoint-51843_context_64_10_best_test_beam_32_candidates_12_constrain.txt', num=300, save=False)
    
    # sample_human_evaluate('data/final_dataset_gen_test.txt', header_type='gen', num=100, save=False)

    # sample_ranker_human_evaluate('data/final_v4_candidate_all_top12_test.txt',
    #                              'eval/final_v4_4e_ranker_extract_all_truncate150_top12_add_b_4_accu_512_10_epoch_9.txt',
    #                              num=300, save=True)

    # merge_multi_human_evaluation(gen_file='eval/t5_human_evaluate_300_format_v1.json',
    #                              rule_file='eval/t5_rule_human_evaluate_300_format_v1.json',
    #                              constrain_file='eval/t5_constrain_human_evaluate_300_format_v1.json', save=False)

    # format_human_evaluation('eval/t5_constrain_human_evaluate_300.txt', save=True)

    # label_human_evaluation_file_hit('eval/t5_rule_human_evaluate_300_format_v1.json', save=True)

    # transform_human_evaluation_file('eval/human_evaluate_300_format_v1.json', save=False)

    # sample_predictions_with_scores('eval/final_t5_pretrain_noleak_64_accum_1_10_checkpoint-51843_128_10_best_test_beam_32_candidates_12_score.txt',
    #                                'eval/final_t5_pretrain_noleak_64_accum_1_10_checkpoint-51843_context_64_10_best_test_beam_32_candidates_12_score.txt',
    #                                num=300,
    #                                save=False)
