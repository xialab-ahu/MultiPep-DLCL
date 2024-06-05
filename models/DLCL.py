#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/5/16 14:35
# @Author : lt,fhh
# @FileName: __init__.py.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class DLCL(nn.Module):
    def __init__(self, vocab_size=21,dropout=0.6,output_size=21,layer=1):
        super(DLCL, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = 256
        self.output_size = output_size
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.filter_sizes = [3, 4,5, 6]
        self.filter_num = 128
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=self.embedding_size,
                                              out_channels=self.filter_num,
                                              padding='same',
                                              kernel_size=fs)
                                    for fs in self.filter_sizes])
        self.Mish = nn.Mish()
        self.gru = nn.GRU(256,128,num_layers=2,dropout=0.6)
        self.attn = MultiHeadSelfAttention(128, 8, 0.6)
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(256)
        self.norm1=nn.LayerNorm(128)
        self.block1 = nn.Sequential(
            nn.Linear(5120, 2560),
            nn.Mish(),
        )
        self.block2= nn.Sequential(
            nn.Linear(6400, 2560),
            nn.Mish(),
        )
        self.block4 = nn.Sequential(
            nn.Linear(6400, 5376),
            nn.Mish(),
        )
        self.block3 = nn.Sequential(
            nn.Linear(5120, len(self.filter_sizes) * self.filter_num),
            nn.Mish(),
            nn.Dropout(),
            nn.Linear(len(self.filter_sizes) * self.filter_num, len(self.filter_sizes) * self.filter_num // 2),
            nn.Mish(),
        )
        self.classification = nn.Sequential(
            nn.Linear(len(self.filter_sizes) * self.filter_num // 2, self.output_size),
        )
        "-------label"
        self.label_lt = torch.nn.Embedding(21, 256, padding_idx=None)
        self.encoder_layers = nn.ModuleList([TransformerEncoderLayer(d_model=256, nhead=4, dropout=0.1) for _ in range(layer)])
        self.Decoder =nn.ModuleList(
            [TransformerDecoderLayer(d_model=256, nhead=8, batch_first=True, dim_feedforward=2048,
                                     dropout=0.1) for _ in range(layer)]
        )

        self.output_linear1 = torch.nn.Linear(256, 21)
        self.head = nn.Sequential(nn.Linear(5376, 2688),
                                  nn.Mish(),
                                  nn.Dropout(),
                                  nn.Linear(2688, 1344),
                                  nn.Mish(),
                                  nn.Dropout(),
                                  nn.Linear(1344, 672),
                                  nn.Mish(),
                                  nn.Dropout(),
                                  nn.Linear(672, 256))
        self.fc = NormedLinear(256, 21)
        self.head_fc = nn.Sequential(nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
                                 nn.Linear(256, 256))

    def forward(self, x,label):

        x = self.embedding(x)

        conved = [self.Mish(conv(x.permute(0, 2, 1))) for conv in self.convs]
        pooled = [F.max_pool1d(conv, math.ceil(conv.shape[-1] //10))for conv in
                  conved]
        flatten = [pool.view(pool.size(0), -1) for pool in pooled]
        cat = self.dropout(torch.cat(flatten, dim=1))
        cat1=self.block1(cat)
        output, hn = self.gru(x )
        output= self.attn(output)
        output = output.reshape(output.shape[0], -1)
        output1=self.block2(output)

        cat_label = cat.reshape(cat.shape[0], -1, 256)
        output_label = self.block4(output)
        output_label=output_label.reshape(output_label.shape[0],21,-1)


        result = self.dropout(torch.cat((output1, cat1), dim=1))
        result_cl = self.block3(result)
        result = self.classification(result_cl )

        init_label_embeddings = self.label_lt(label)
        init_label_embeddings = init_label_embeddings + output_label
        embeddings = torch.cat((cat_label , init_label_embeddings), 1)
        embeddings = self.LayerNorm(embeddings)
        for layer in self.encoder_layers:
            embeddings,attn_encoder = layer(embeddings)
        cat_label_embeddings = embeddings[:, 0:cat_label .size(1), :]
        label_embeddings = embeddings[:, -init_label_embeddings.size(1):, :]
        label_embeddings = label_embeddings + output_label
        for layer in self.Decoder:
            label_embeddings = layer(label_embeddings,  cat_label_embeddings)
        label_embeddings1=label_embeddings.reshape(label_embeddings.shape[0],-1)

        result_label = self.head(label_embeddings1)
        result_label = self.fc(result_label)
        result_label_cl = F.normalize(self.head_fc(self.fc.weight.T), dim=1)
        return result,result_label,result_cl,result_label_cl

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead=8, dropout=0.1, dim_feedforward=2048, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):

        src2, attn_encoder = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src,attn_encoder
class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.s = 30

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return self.s * out
class TransformerDecoderLayer(nn.Module):
    """
    Transformer decoder
    """
    __constants__ = ['batch_first']

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, layer_norm_eps=1e-5, batch_first=False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                 **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, tgt, memory ,tgt_mask=None, memory_mask= None,
                tgt_key_padding_mask= None, memory_key_padding_mask= None):

        tgt2, _= self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, _ = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)#[256, 21, 256]
        return tgt
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size=128, num_heads=8, dropout=0.6):
        super(MultiHeadSelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads

        assert hidden_size % num_heads == 0

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, inputs, attention_mask=None):
        batch_size, seq_len, hidden_size = inputs.size()


        query = self.query(inputs).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2).to(inputs.device)
        key = self.key(inputs).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2).to(inputs.device)
        value = self.value(inputs).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2).to(inputs.device)
        scores = torch.matmul(query, key.transpose(-1, -2)) / torch.sqrt(torch.FloatTensor([self.head_size])).to(inputs.device)
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask.unsqueeze(1).unsqueeze(1) == 0, -1e9)

        attn_weights = nn.Softmax(dim=-1)(scores)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, value)
        # (batch_size, num_heads, seq_len, head_size)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len,
                                                            self.hidden_size)  # (batch_size, seq_len, hidden_size)
        output = self.fc(context)

        return output



