import jieba
import thwpy
import pandas as pd


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
    assert len(label) == len(word)
    label = [int(l) for l in label]
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


def prepare_candidates_dump(label_seq_file, candidate_file, top_k=8, save=False,
                            no_full=False, no_word=False, no_truth=False, no_char=False):
    """
    对candidate_file中模型生成的candidates通过启发式的规则进行后处理
    首先，对于训练集，首先全称是一个负样本
    1st. 根据全称的分词结果，选取和全称词编辑距离为d的子序列
    2nd. 只是对于训练集而言，选取和ground truth编辑距离为d的子序列
    3th. 选取和全称字符编辑距离为d的子序列，其中 0 < d < word_len
    :param no_char:
    :param no_truth:
    :param no_word:
    :param no_full:
    :param save:
    :param label_seq_file: 带有0/1序列的文件
    :param candidate_file: 带有candidates的文件
    :param top_k: 最终保证top_k个candidates
    :return: 保存最终的candidate数据
    """
    file_split = thwpy.check_split(label_seq_file)
    assert file_split == thwpy.check_split(candidate_file)
    train = file_split == 'train'
    label_data = pd.read_csv(label_seq_file, sep='\t', names=['full', 'label'])
    candidate_data = pd.read_csv(candidate_file, sep='\t', names=['full', 'abbr', 'candidate'])
    candidate_data['label'] = label_data['label']
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
        tmp_list = [c for c in tmp_list if is_subsequence(c, row['full'])]
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
                        if len(tmp_list) == top_k:
                            break
                d += 1
        # len(word) <= 3，枚举肯定会有空的，需要补全

        # v1
        if no_full:
            tmp_list = [c for c in tmp_list if c != row['full']]
            d = len(tmp_list) - 1
            while tmp_list[d] == row['abbr']:
                d -= 1
            while len(tmp_list) < top_k:
                tmp_list.append(tmp_list[d])
        else:
            while len(tmp_list) < top_k:
                tmp_list.append(row['full'])

        # v2
        # if no_full:
        #     tmp_list = [c for c in tmp_list if c != row['full']]
        # d = len(tmp_list) - 1
        # while len(tmp_list) < top_k:
        #     while tmp_list[d] == row['abbr']:
        #         d -= 1
        #     tmp_list.append(tmp_list[d])

        if no_full:
            assert row['full'] not in tmp_list
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
    candidate_data['candidate'] = candidates
    candidate_data = candidate_data.drop(columns=['label'])
    print(candidate_data)
    print(candidate_data.keys())

    if no_full:
        post_edit = 'nofull'
    elif no_word:
        post_edit = 'noword'
    elif no_truth:
        post_edit = 'notruth'
    elif no_char:
        post_edit = 'nochar'
    else:
        post_edit = 'all'
    save_path = f'data/final_candidate_{post_edit}_top{top_k}_{file_split}.txt'

    print(f"Saved in {save_path}")
    if save:
        candidate_data.to_csv(save_path, sep='\t', header=False, index=False)
    # return cnt / len(candidate_data)
