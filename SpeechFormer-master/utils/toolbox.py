
import torch
import re
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
import numpy as np

def _majority_target_Pitt(source_tag: list):
    return [re.match('.*-.*-', mark).group() for mark in source_tag]

def _majority_target_DAIC_WOZ(source_tag: list):
    return [mark.split('_')[0] for mark in source_tag]

def majority_vote(source_tag: list, source_value: torch.Tensor, source_label: torch.Tensor, modify_tag, task='classification'):
    '''
    Args:
    source_tag: Guideline for voting, e.g. sample same.
    source_value: value before voting.
    source_label: label before voting.
    task: classification / regression

    Return:
    target: voting object.
    vote_value: value after voting.
    vote_label: label after voting.
    '''
    source_tag = modify_tag(source_tag)
    target = set(source_tag)
    vote_value_dict = {t:[] for t in target}
    vote_label_dict = {t:[] for t in target}

    if task == 'regression':
        logit_vote = True
    else:
        if source_value.dim() != 1:
            logit_vote = True
        else:
            logit_vote = False

    for i, (mark) in enumerate(source_tag):
        value = source_value[i]
        label = source_label[i]
        vote_value_dict[mark].append(value)
        vote_label_dict[mark].append(label)
    for key, value in vote_value_dict.items():
        if logit_vote:
            logit = torch.mean(torch.stack(value, dim=0), dim=0)
            if task == 'regression':
                vote_value_dict[key] = logit
            else:
                vote_value_dict[key] = torch.argmax(logit)
        else:
            vote_value_dict[key] = max(value, key=value.count)

    vote_value, vote_label = [], []
    for t in target:
        vote_value.append(vote_value_dict[t])
        vote_label.append(vote_label_dict[t][0])

    vote_value = torch.tensor(vote_value)
    vote_label = torch.tensor(vote_label)
    
    return target, vote_value, vote_label

def calculate_score_classification(preds, labels, average_f1='weighted', probs=None):  # weighted, macro
    """Return extended classification metrics.
    Args:
        preds: 1D tensor/list of predicted label indices
        labels: 1D tensor/list of true label indices
        average_f1: 'weighted' or 'macro'
        probs: (N, C) ndarray / tensor or (N,) positive-class probabilities for AUC
    Returns (order kept compatible then extended):
        accuracy, ua(recall-macro), f1, precision-macro, confusion_matrix,
        auc, sensitivity, specificity
    """
    if torch.is_tensor(preds):
        preds_np = preds.cpu().numpy()
    else:
        preds_np = np.asarray(preds)
    if torch.is_tensor(labels):
        labels_np = labels.cpu().numpy()
    else:
        labels_np = np.asarray(labels)

    accuracy = accuracy_score(labels_np, preds_np)
    f1 = f1_score(labels_np, preds_np, average=average_f1, zero_division=0)
    precision = precision_score(labels_np, preds_np, average='macro', zero_division=0)
    ua = recall_score(labels_np, preds_np, average='macro', zero_division=0)
    confuse_matrix = confusion_matrix(labels_np, preds_np)

    # Sensitivity & Specificity (binary). For multi-class compute macro-average.
    if confuse_matrix.shape == (2,2):
        TN, FP, FN, TP = confuse_matrix[0,0], confuse_matrix[0,1], confuse_matrix[1,0], confuse_matrix[1,1]
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0  # recall of positive class
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    else:
        # multi-class sensitivity is UA; specificity macro computed per class
        sensitivity = ua
        spec_list = []
        cm = confuse_matrix.astype(float)
        for i in range(cm.shape[0]):
            TP_i = cm[i,i]
            FN_i = cm[i,:].sum() - TP_i
            FP_i = cm[:,i].sum() - TP_i
            TN_i = cm.sum() - (TP_i + FN_i + FP_i)
            spec_list.append(TN_i / (TN_i + FP_i) if (TN_i + FP_i) > 0 else 0.0)
        specificity = float(np.mean(spec_list)) if spec_list else 0.0

    # AUC
    auc = 0.0
    try:
        if probs is not None:
            if torch.is_tensor(probs):
                probs_np = probs.detach().cpu().numpy()
            else:
                probs_np = np.asarray(probs)
            if probs_np.ndim == 1 or probs_np.shape[1] == 2:  # binary
                if probs_np.ndim > 1:
                    # assume column 1 is positive class
                    pos_prob = probs_np[:, -1]
                else:
                    pos_prob = probs_np
                # Need both classes present
                if len(np.unique(labels_np)) == 2:
                    auc = roc_auc_score(labels_np, pos_prob)
            else:
                # multi-class: macro one-vs-rest
                auc = roc_auc_score(labels_np, probs_np, multi_class='ovr', average='macro')
    except Exception:
        pass

    return accuracy, ua, f1, precision, confuse_matrix, auc, sensitivity, specificity

def calculate_basic_score(preds, labels):
    return accuracy_score(labels, preds)

def tidy_csvfile(csvfile, colname, ascending=True):
    '''
    tidy csv file base on a particular column.
    '''
    print(f'tidy file: {csvfile}, base on column: {colname}')
    df = pd.read_csv(csvfile)
    df = df.sort_values(by=[colname], ascending=ascending, na_position='last')
    df = df.round(3)
    df.to_csv(csvfile, index=False, sep=',')

    
