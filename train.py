#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/5/16 14:35
# @Author : LT
# @FileName: __init__.py.py
# @Software: PyCharm
import time
import math
from torch.optim import lr_scheduler

class DataTrain:

    def __init__(self, model, optimizer, criterion, scheduler=None, device="cuda"):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_scheduler = scheduler
        self.device = device
    def train_step(self, train_iter, epochs=None, model_name='', alpha=0.3, beta=0.2,device="cuda"):
        global loss
        steps = 1
        for epoch in range(1,epochs + 1):
            start_time = time.time()
            train_loss = []
            for train_data, train_label,label_input in train_iter:
                #x_train,y_train,label_input,train_length
                self.model.train()

                train_data,train_label ,label_input= train_data.to(self.device), train_label.to(self.device),label_input.to(self.device)

                result, result_label, result_cl, result_label_cl= self.model(train_data, label_input)
                loss1 = self.criterion[0](result, train_label.float())
                loss2 = self.criterion[1](result, train_label.float())
                loss3 = self.criterion[1](result_label, train_label.float())
                loss4 = self.criterion[2](result_cl, result_label_cl, train_label.float())

                loss =0.3*loss2+0.7*loss1+0.3*loss3+0.2*loss4

                self.optimizer.zero_grad()
                loss.backward()

                self.optimizer.step()
                train_loss.append(loss.item())

                if self.lr_scheduler:
                    if self.lr_scheduler.__module__ == lr_scheduler.__name__:
                        # Using PyTorch In-Built scheduler
                        self.lr_scheduler.step()
                    else:
                        # Using custom defined scheduler
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.lr_scheduler(steps)
            steps += 1

            end_time = time.time()
            epoch_time = end_time - start_time

            print(f'{model_name}|Epoch:{epoch:003}/{epochs}|Run time:{epoch_time:.2f}s')
            print(f'Train loss:{np.mean(train_loss)}')




class CosineScheduler:

    def __init__(self, max_update, base_lr=0.01, final_lr=0, warmup_steps=0, warmup_begin_lr=0):
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps

    def get_warmup_lr(self, epoch):
        increase = (self.base_lr_orig - self.warmup_begin_lr) * float(epoch - 1) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase

    def __call__(self, epoch):
        if epoch < self.warmup_steps:
            return self.get_warmup_lr(epoch)
        if epoch <= self.max_update:
            self.base_lr = self.final_lr + (self.base_lr_orig - self.final_lr) * \
                           (1 + math.cos(math.pi * (epoch - 1 - self.warmup_steps) / self.max_steps)) / 2
        return self.base_lr
def predict(model, data, device="cuda"):

    model.to(device)
    model.eval()
    predictions = []
    labels = []

    with torch.no_grad():
        for train_data, train_label,label_input in data:
            x = train_data.to(device)
            label_input = label_input.to(device)
            train_label = train_label.to(device)
            x, _, _, _= model(x, label_input)
            label = torch.sigmoid(x)
            predictions.extend(label.tolist())
            labels.extend(train_label.tolist())

    return np.array(predictions), np.array(labels)

import numpy as np
import pandas as pd
import torch

print("NumPy 版本:", np.__version__)
print("Pandas 版本:", pd.__version__)
print("PyTorch 版本:", torch.__version__)