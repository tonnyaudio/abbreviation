from tqdm import tqdm
import torch
import torch.nn as nn
import thwpy
import numpy as np
import torch.nn.functional


def train(data_loader, model, loss, optimizer, device, config, scheduler=None):
    model.train()
    final_loss = 0.0
    steps = 0

    bar = tqdm(data_loader, desc='train')
    for data in bar:

        for k, v in data.items():
            data[k] = v.to(device)

        label = data.pop('labels')
        if ("orders" in data):
            orders = data.pop("orders")
            outputs,word_outputs = model(**data)

            lss1 = loss(outputs, label)
            lss2 = loss(word_outputs, label)
            ls = lss1.sum() / torch.count_nonzero(lss1)+1.0*torch.mean(lss2)

        else:
            outputs = model(**data)
            ls = loss(outputs, label)

        ls = ls / config.gradient_accumulation_steps
        ls.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.clip_norm)

        if ((steps + 1) % config.gradient_accumulation_steps) == 0:
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        final_loss += ls.item()
        steps += 1
        bar.set_postfix({'avg_loss': final_loss / steps})
    return final_loss / steps


def duplicated_accuracy(data_loader, model, device, topk=3):
    model.eval()
    predicts = []
    labels = []
    logits = []
    bar = tqdm(data_loader, desc='accuracy')
    with torch.no_grad():
        for data in bar:

            for k, v in data.items():
                data[k] = v.to(device)

            label = data.pop('labels')
            if ("orders" in data):
                orders = data.pop("orders")
                outputs,_ = model(**data)
            else:
                outputs = model(**data)

            if topk == 1:
                predicts.extend(outputs.argmax(dim=-1).tolist())
            else:
                logit, preds = torch.topk(outputs, topk, dim=-1)
                predicts.extend(preds.tolist())
                logits.extend(logit.tolist())
            labels.extend(label.tolist())

    if topk == 1:
        results = [[str(i), str(j)] for i, j in zip(predicts, labels)]
        thwpy.save_csv(results, 'hit@1.txt')
        return sum(i == j for i, j in zip(predicts, labels)) / len(labels)
    else:
        results = [[str(i[0]), str(j), ';'.join([str(idx) for idx in i]), ';'.join([str(log) for log in logit])] for
                   i, j, logit in zip(predicts, labels, logits)]
        thwpy.save_csv(results, 'hit@3.txt')
        return sum(j == i[0] for i, j in zip(predicts, labels)) / len(labels), sum(
            j in i for i, j in zip(predicts, labels)) / len(labels)


def accuracy(data_loader, model, device, topk=3):
    model.eval()
    predicts = []
    hit3_predicts = []
    labels = []
    bar = tqdm(data_loader, desc='accuracy')
    with torch.no_grad():
        for data in bar:

            for k, v in data.items():
                data[k] = v.to(device)

            label = data.pop('labels')
            if ("orders" in data):
                orders = data.pop("orders")
                outputs,word_outputs = model(**data)
            else:
                outputs,word_outputs = model(**data)

            logit, preds = torch.topk(outputs+1.0*word_outputs, topk, dim=-1)
    
            hit3_predicts.extend(preds.tolist())

            labels.extend(label.tolist())

    # acc = sum(i == j for i, j in zip(predicts, labels)) / len(labels)
    hit_1 = sum(j == i[0] for i, j in zip(hit3_predicts, labels)) / len(labels)
    hit_3 = sum(j in i for i, j in zip(hit3_predicts, labels)) / len(labels)

    return hit_1, hit_3



def save_ranker_predicts(data_loader, model, device, topk=3):
    model.eval()
    predicts = []
    hit3_predicts = []
    labels = []
    bar = tqdm(data_loader, desc='accuracy')
    with torch.no_grad():
        for data in bar:

            for k, v in data.items():
                data[k] = v.to(device)

            label = data.pop('labels')

            if ("orders" in data):
                orders = data.pop("orders")
                #                 print(orders)
                outputs,word_outputs = model(**data)
            #                 outputs = torch.mul(outputs, orders)
            else:
                outputs,word_outputs = model(**data)


            logit, preds = torch.topk(outputs+1.0*word_outputs, topk, dim=-1)
    
            hit3_predicts.extend(preds.tolist())

            labels.extend(label.tolist())

    hit_1 = sum(j == i[0] for i, j in zip(hit3_predicts, labels)) / len(labels)
    hit_3 = sum(j in i for i, j in zip(hit3_predicts, labels)) / len(labels)

    return hit_1, hit_3, hit3_predicts