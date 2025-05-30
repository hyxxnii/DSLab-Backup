# -*- coding: utf-8 -*-
# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/24
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionBatch(nn.Module):
    def __init__(self, dim, da, alpha=0.2, dropout=0.5):
        super(SelfAttentionBatch, self).__init__()
        self.dim = dim
        self.da = da
        self.alpha = alpha
        self.dropout = dropout
        self.a = nn.Parameter(torch.zeros(size=(self.dim, self.da)), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(size=(self.da, 1)), requires_grad=True)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        nn.init.xavier_uniform_(self.b.data, gain=1.414)

    def forward(self, h):
        # h: (N, dim) => user가 상호작용한 item/entity 개수만큼의 행렬
        e = torch.matmul(torch.tanh(torch.matmul(h, self.a)), self.b).squeeze(dim=1)  # (N)
        attention = F.softmax(e, dim=0)  # (N)
        return torch.matmul(attention, h)  # (dim)
    
    
class ApplySentSelfAttentionBatch(nn.Module):
    def __init__(self, dim, da, alpha=0.2, dropout=0.5): #, ent_sent=None):
        super(ApplySentSelfAttentionBatch, self).__init__()
        self.dim = dim
        self.da = da
        self.alpha = alpha
        self.dropout = dropout
        
        # self.ent_sent = ent_sent
        self.a = nn.Parameter(torch.zeros(size=(self.dim, self.da)), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(size=(self.da, 1)), requires_grad=True)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        nn.init.xavier_uniform_(self.b.data, gain=1.414)

    def forward(self, h, entity_list, conv_id, ent_sent):
        # h: (N, dim) => user가 상호작용한 item/entity 개수만큼의 행렬
        e = torch.matmul(torch.tanh(torch.matmul(h, self.a)), self.b).squeeze(dim=1)  # (N)
        attention = F.softmax(e, dim=0)  # (N)
        
        adjusted_attention = attention.clone() 
        
        ent_sent_dict = ent_sent[conv_id]
        for i, entity_id in enumerate(entity_list.tolist()):
            if entity_id in ent_sent_dict:
                sent_score = ent_sent_dict[entity_id]
                adjusted_attention[i] *= sent_score #.detach().cpu()
        
        # 조정된 attention 가중치를 정규화
        norm_attention = adjusted_attention / adjusted_attention.sum()
        
        #print(norm_attention)
        
        return torch.matmul(norm_attention, h)  # (dim)