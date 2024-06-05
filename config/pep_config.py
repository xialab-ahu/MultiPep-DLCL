#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/5/16 14:35
# @Author : lt,fhh
# @FileName: __init__.py.py
# @Software: PyCharm
import argparse
import torch
from loss_functions import FocalDiceLoss, ContrastiveLoss


def get_config():
    parse = argparse.ArgumentParser(description='peptide default config')
    parse.add_argument('-task', type=str, default='model_select')
    parse.add_argument('-train_test', type=bool, default=0)
    parse.add_argument('-subtest', type=bool, default=False)
    parse.add_argument('-vocab_size', type=int, default=21,
                       help='The size of the vocabulary')
    parse.add_argument('-output_size', type=int, default=21,
                       help='Number of peptide functions')

    parse.add_argument('-batch_size', type=int, default=256,
                       help='Batch size')
    parse.add_argument('-epochs', type=int, default=200)  # 50
    parse.add_argument('-learning_rate', type=float, default=0.0018)
    parse.add_argument('-threshold', type=float, default=0.6)
    parse.add_argument('-criterion', type=list, default=[FocalDiceLoss(), torch.nn.BCEWithLogitsLoss(),ContrastiveLoss()])
    parse.add_argument('-alpha', type=float, default=0.3)
    parse.add_argument('-beta', type=float, default=0.5)



    parse.add_argument('-embedding_size', type=int, default=64*4,
                       help='Dimension of the embedding')
    parse.add_argument('-dropout', type=float, default=0.6)
    parse.add_argument('-filter_num', type=int, default=64*2,
                       help='Number of the filter')
    parse.add_argument('-filter_size', type=list, default=[3, 4, 5, 6],
                       help='Size of the filter')


    parse.add_argument('-model_path', type=str, default='tea_model.pth',
                       help='Path of the training data')
    parse.add_argument('-train_direction', type=str, default='dataset/train.txt',
                       help='Path of the training data')
    parse.add_argument('-test_direction', type=str, default='dataset/test.txt',
                       help='Path of the test data')

    config = parse.parse_args()
    return config
