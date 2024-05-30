#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/5/16 14:35
# @Author : LT
# @FileName: __init__.py.py
# @Software: PyCharm
import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
import os
import numpy as np
import  random
def get_random_seed(seed):
    # https://blog.csdn.net/weixin_48249563/article/details/115031039
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


get_random_seed(20230226)
torch.backends.cudnn.deterministic = True

amino_acids = 'XACDEFGHIKLMNPQRSTVWY'


def getSequenceData(direction: str):
    data, label = [], []
    max_length = 0
    min_length = 8000

    with open(direction) as f:
        for each in f:
            each = each.strip()
            each = each.upper()
            if each[0] == '>':
                label.append(np.array(list(each[1:]), dtype=int))  # Converting string labels to numeric vectors
            else:
                if len(each) > max_length:
                    max_length = len(each)
                elif len(each) < min_length:
                    min_length = len(each)
                data.append(each)

    return np.array(data), np.array(label), max_length, min_length
def PadEncode(data, label, max_len: int = 50):
    data_e, label_e, seq_length, temp = [], [], [], []
    sign, b = 0, 0
    for i in range(len(data)):
        length = len(data[i])
        if len(data[i]) > max_len:
            continue
        element, st = [], data[i].strip()
        for j in st:
            if j not in amino_acids:
                sign = 1
                break
            index = amino_acids.index(j)
            element.append(index)
            sign = 0

        if length <= max_len and sign == 0:
            temp.append(element)
            seq_length.append(len(temp[b]))
            b += 1
            element += [0] * (max_len - length)
            data_e.append(element)
            label_e.append(label[i])

    return torch.LongTensor(np.array(data_e)), torch.LongTensor(np.array(label_e))
def LabelEmbeddingData(x_train, y_train):
    label_input = np.ones((y_train.shape[0], 21))
    return x_train,y_train,torch.LongTensor(np.array(label_input))




def data_load(train_direction=None, test_direction=None, batch=None, subtest: bool = True, CV: bool = False):
    dataset_train, dataset_test = [], []
    dataset_subtest = None
    weight = None
    train_seq_data, train_seq_label, max_len_train, min_len_train = getSequenceData(train_direction)
    test_seq_data, test_seq_label, max_len_test, min_len_test = getSequenceData(test_direction)
    print(f"max_length_train:{max_len_train}")
    print(f"min_length_train:{min_len_train}")
    print(f"max_length_test:{max_len_test}")
    print(f"min_length_test:{min_len_test}")

    x_train, y_train= PadEncode(train_seq_data, train_seq_label, max_len_train)
    x_train,y_train,label_input=LabelEmbeddingData(x_train, y_train)
    x_test, y_test= PadEncode(test_seq_data, test_seq_label, max_len_test)
    x_test, y_test, testlabel_input= LabelEmbeddingData(x_test, y_test)

    if CV is False:
        # Create datasets
        train_data = TensorDataset(x_train,  y_train,label_input)
        test_data = TensorDataset(x_test, y_test,testlabel_input)
        dataset_train.append(DataLoader(train_data, batch_size=batch, shuffle=True))
        dataset_test.append(DataLoader(test_data, batch_size=batch, shuffle=True))

        if subtest:
            dataset_subtest = []
            for i in range(5):
                sub_size = int(0.8 * len(test_data))
                _ = len(test_data) - sub_size
                subtest, _ = torch.utils.data.random_split(test_data, [sub_size, _])
                sub_test = DataLoader(subtest, batch_size=batch, shuffle=True)
                dataset_subtest.append(sub_test)
    else:
        cv = KFold(n_splits=5, random_state=42, shuffle=True)
        # 构造五折交叉验证的训练集和验证集
        for split_index, (train_index, test_index) in enumerate(cv.split(x_train)):
            sequence_train, label_train, length_train = x_train[train_index], y_train[train_index], \
                                                        train_length[train_index]
            sequence_test, label_test, length_test = x_train[test_index], y_train[test_index], train_length[
                test_index]
            train_data = TensorDataset(sequence_train, length_train, label_train)
            test_data = TensorDataset(sequence_test, length_test, label_test)
            dataset_train.append(DataLoader(train_data, batch_size=batch, shuffle=True))
            dataset_test.append(DataLoader(test_data, batch_size=batch, shuffle=True))

    return dataset_train, dataset_test, dataset_subtest, weight

