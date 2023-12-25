import torch
import numpy as np
import random
from collections import defaultdict
import unicodedata
import re


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_head_data(*lists, num=10000):
    return [data[:num] for data in lists]


# 对英文做归一化 "Crème Brulée" -> "Creme Brulee"
def token_normalize(token):
    return re.sub('[\u0300-\u036F]', '', unicodedata.normalize('NFKD', token))


def gen_token_list_inv_pointer(sent, token_list):
    """
    :param sent: 文本字符串
    :param token_list: sent被tokenize后的list
    :return: 和token_list等长的每个token所在的sent的位置
    """
    sent = sent.lower()
    otiis = []
    iis = 0
    for it, token in enumerate(token_list):
        otoken = token.lstrip('#').lower()
        if token[0] == '[' and token[-1] == ']': otoken = ''
        niis = iis
        while niis <= len(sent):
            if sent[niis:].startswith(otoken): break
            if niis >= len(sent): break
            if otoken in '-"' and sent[niis][0] in '—“”': break
            niis += 1
        if niis >= len(sent):
            niis = iis
        otiis.append(niis)
        iis = niis + max(1, len(otoken))
    return otiis


# restore [UNK] tokens to the original tokens
def restore_token_list(sent, token_list):
    # 去除特殊字符
    if token_list[0] == '[CLS]':
        token_list = token_list[1:-1]
    invp = gen_token_list_inv_pointer(sent, token_list)
    print(invp)
    invp.append(len(sent))
    otokens = [sent[u:v] for u, v in zip(invp, invp[1:])]
    processed = -1
    for ii, tk in enumerate(token_list):
        if tk != '[UNK]':
            continue
        if ii < processed:
            continue
        for jj in range(ii + 1, len(token_list)):
            if token_list[jj] != '[UNK]':
                break
        else:
            jj = len(token_list)
        allseg = sent[invp[ii]:invp[jj]]
        print(allseg)

        if ii + 1 == jj: continue
        seppts = [0] + [i for i, x in enumerate(allseg) if
                        i > 0 and i + 1 < len(allseg) and x == ' ' and allseg[i - 1] != ' ']
        if allseg[seppts[-1]:].replace(' ', '') == '': seppts = seppts[:-1]
        seppts.append(len(allseg))
        print(f"seppts: {seppts}")
        if len(seppts) == jj - ii + 1:
            for k, (u, v) in enumerate(zip(seppts, seppts[1:])):
                otokens[ii + k] = allseg[u:v]
        processed = jj + 1
        print(f"processed: {processed}")
    if invp[0] > 0: otokens[0] = sent[:invp[0]] + otokens[0]
    if ''.join(otokens) != sent:
        raise Exception('restore tokens failed, text and restored:\n%s\n%s' % (sent, ''.join(otokens)))
    return otokens


def find_val_pos(sent, value):
    ret = []
    value = value.replace(' ', '').lower()
    if value == '': return ret
    ss = [x.replace(' ', '').lower() for x in sent]
    for k, v in enumerate(ss):
        if not value.startswith(v): continue
        vi = 0
        for j in range(k, len(ss)):
            if value[vi:].startswith(ss[j]):
                vi += len(ss[j])
                if vi == len(value):
                    ret.append((k, j + 1))
            else:
                break
    return ret


def get_top_spans(tokens, rr, K=40):
    cands = defaultdict(float)
    start_indexes = sorted(enumerate(rr[:, 0]), key=lambda x: -x[1])[:K]
    end_indexes = sorted(enumerate(rr[:, 1]), key=lambda x: -x[1])[:K]
    for start_index, start_score in start_indexes:
        if start_score < 0.1: continue
        if start_index >= len(tokens): continue
        for end_index, end_score in end_indexes:
            if end_score < 0.1: continue
            if end_index >= len(tokens): continue
            if end_index < start_index: continue
            length = end_index - start_index + 1
            if length > 40: continue
            ans = ''.join(tokens[start_index:end_index + 1]).strip()
            if '》' in ans: continue
            if '、' in ans and len(ans.split('、')) > 2 and '，' not in ans and ',' not in ans:
                aas = ans.split('、')
                for aa in aas: cands[aa.strip()] += start_score * end_score / len(aas)
                continue
            cands[ans] += start_score * end_score

    cand_list = sorted(cands.items(), key=lambda x: len(x[0]))
    removes = set()
    contains = {}
    for i, (x, y) in enumerate(cand_list):
        for j, (xx, yy) in enumerate(cand_list[:i]):
            if xx in x and len(xx) < len(x):
                contains.setdefault(x, []).append(xx)

    for i, (x, y) in enumerate(cand_list):
        sump = sum(cands[z] for z in contains.get(x, []) if z not in removes)
        suml = sum(len(z) for z in contains.get(x, []) if z not in removes)
        if suml > 0: sump = sump * min(1, len(x) / suml)
        if sump > y:
            removes.add(x)
        else:
            for z in contains.get(x, []): removes.add(z)

    ret = [x for x in cand_list if x[0] not in removes]
    ret.sort(key=lambda x: -x[1])
    return ret[:K]


if __name__ == '__main__':
    a = [0] * 100000
    b = [1] * 100000
    a, b = get_head_data(a, b)
    print(a)
    print(b)
    print(len(a))
    print(len(b))