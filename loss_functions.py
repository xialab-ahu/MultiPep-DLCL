#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/5/16 14:35
# @Author : lt,fhh
# @FileName: __init__.py.py
# @Software: PyCharm

import torch
import torch.nn as nn

class FocalDiceLoss(nn.Module):
    """multi-label focal-dice loss"""

    def __init__(self, p_pos=2, p_neg=2, clip_pos=0.7, clip_neg=0.5, pos_weight=0.3, reduction='mean'):
        super(FocalDiceLoss, self).__init__()
        self.p_pos = p_pos
        self.p_neg = p_neg
        self.reduction = reduction
        self.clip_pos = clip_pos
        self.clip_neg = clip_neg
        self.pos_weight = pos_weight

    def forward(self, input, target):
        assert input.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = nn.Sigmoid()(input)
        # predict = input
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        xs_pos = predict
        p_pos = predict
        if self.clip_pos is not None and self.clip_pos >= 0:
            m_pos = (xs_pos + self.clip_pos).clamp(max=1)
            p_pos = torch.mul(m_pos, xs_pos)
        num_pos = torch.sum(torch.mul(p_pos, target), dim=1)
        den_pos = torch.sum(p_pos.pow(self.p_pos) + target.pow(self.p_pos), dim=1)

        xs_neg = 1 - predict
        p_neg = 1 - predict
        if self.clip_neg is not None and self.clip_neg >= 0:
            m_neg = (xs_neg + self.clip_neg).clamp(max=1)
            p_neg = torch.mul(m_neg, xs_neg)
        num_neg = torch.sum(torch.mul(p_neg, (1 - target)), dim=1)
        den_neg = torch.sum(p_neg.pow(self.p_neg) + (1 - target).pow(self.p_neg), dim=1)

        loss_pos = 1 - (2 * num_pos) / den_pos
        loss_neg = 1 - (2 * num_neg) / den_neg
        loss = loss_pos * self.pos_weight + loss_neg * (1 - self.pos_weight)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))





class ContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
#L [21,256]
    def forward(self, V, L, Y):
        sim = torch.matmul(V, L.T) / self.temperature
        sim_max, _ = torch.max(sim, dim=1,
                               keepdim=True)
        sim = sim - sim_max.detach()
        positive_samples_mask = Y == 1
        negative_samples_mask = ~positive_samples_mask
        exp_sim = torch.exp(sim)
        exp_sim_neg = exp_sim * negative_samples_mask.float()
        exp_sim_neg_sum = exp_sim_neg.sum(dim=1, keepdim=True)
        loss = -torch.log((exp_sim * positive_samples_mask.float()).sum(dim=1) / (
                    exp_sim_neg_sum + exp_sim * positive_samples_mask.float()).sum(dim=1)).mean()
        return loss

