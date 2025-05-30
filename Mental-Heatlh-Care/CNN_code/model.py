import numpy as np
import pandas as pd
from tqdm import tqdm
import json
from collections import Counter
import collections
import pickle
from termcolor import colored

import os
import time
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data_utils

import torchtext
from torchtext.vocab import Vocab, Vectors


from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class WORDEBD(nn.Module):
    '''
        An embedding layer that maps the token id into its corresponding word
        embeddings. The word embeddings are kept as fixed once initialized.
    '''
    def __init__(self, vocab, finetune_ebd):#, specific_vocab_size=None):
        super(WORDEBD, self).__init__()

        self.vocab_size, self.embedding_dim = vocab.vectors.size()
        # if specific_vocab_size != None: self.vocab_size = specific_vocab_size
        self.embedding_layer = nn.Embedding(
                self.vocab_size, self.embedding_dim)
        self.embedding_layer.weight.data = vocab.vectors

        self.finetune_ebd = finetune_ebd

        if self.finetune_ebd:
            self.embedding_layer.weight.requires_grad = True
        else:
            self.embedding_layer.weight.requires_grad = False

    def forward(self, data, weights=None):
        '''
            @param text: batch_size * max_text_len
            @return output: batch_size * max_text_len * embedding_dim
        '''
        if (weights is None): #or (self.finetune_ebd == False):
            return self.embedding_layer(data['text'])

        else:
            return F.embedding(data['text'],
                               weights['ebd.embedding_layer.weight'])

class CNNseq(nn.Module):
    '''
        An aggregation method that encodes every document through different
        convolution filters (followed by max-over-time pooling).
    '''
    def __init__(self, ebd, args):
        super(CNNseq, self).__init__()
        self.args = args

        self.ebd = ebd # pre-trained FastText로 initialization된 token representation => WORDEBD => nn.Embedding layer로 매핑된 것

        self.input_dim = self.ebd.embedding_dim

        # Convolution
        self.convs = nn.ModuleList([nn.Conv1d(
                    in_channels=self.input_dim,
                    out_channels=args.cnn_num_filters,
                    kernel_size=K) for K in args.cnn_filter_sizes])
        
        self.relu = nn.ReLU()

        self.ebd_dim = args.cnn_num_filters * len(args.cnn_filter_sizes)

        self.mlp = self._init_mlp(self.ebd_dim, self.args.mlp_hidden, self.args.dropout)
        self.out = nn.Linear(args.mlp_hidden[-1], self.args.n_classes)
        self.dropout = nn.Dropout(self.args.dropout)

    def _init_mlp(self, in_d, hidden_ds, drop_rate):
        modules = []

        for d in hidden_ds[:-1]:
            modules.extend([
                nn.Dropout(drop_rate),
                nn.Linear(in_d, d),
                nn.ReLU()])
            in_d = d

        modules.extend([
            nn.Dropout(drop_rate),
            nn.Linear(in_d, hidden_ds[-1])])

        return nn.Sequential(*modules)

    def _conv_max_pool(self, x, conv_filter=None, weights=None):
        '''
        Compute sentence level convolution
        Input:
            x:      batch_size, max_doc_len, embedding_dim
        Output:     batch_size, num_filters_total
        '''
        assert(len(x.size()) == 3) # [batch_size==max_sentences, max_tokens, embedding_dim]

        x = x.permute(0, 2, 1)  # batch_size, embedding_dim, doc_len
        x = x.contiguous()

        # Apply the 1d conv. Resulting dimension is
        # [batch_size, num_filters, doc_len-filter_size+1] * len(filter_size)
        assert(not ((conv_filter is None) and (weights is None)))
        if conv_filter is not None:
            x = [conv(x) for conv in conv_filter]

        
        ## nn.MaxPool1d로 다시 생성
        max_pooled_outputs = []
        for sub_x in x:
            pool_size = sub_x.size(2)
            max_pool = nn.MaxPool1d(pool_size) # output shape: [batch_size, num_filters, 1]
            pooled = max_pool(sub_x).squeeze(2) # output shape: [batch_size, num_filters]
            max_pooled_outputs.append(pooled)
        
        # concatenate along all filters. Resulting dimension is
        # output: [batch_size, num_filters_total]
        x = torch.cat(max_pooled_outputs, 1) # output shape: [batch_size, num_filters*3]
        #x = torch.cat(x, 1)
        x = self.relu(x) #F.relu(x)
        return x

    def forward(self, data, weights=None):
        '''
            @param data dictionary
                @key text: batch_size * max_text_len
            @param weights placeholder used for maml

            @return output: batch_size * embedding_dim
        '''

        device = data['text'].device
        
        # Apply the word embedding, result:  batch_size, doc_len, embedding_dim
        
        ebd = self.ebd(data, weights) # ouptut: [batch_size, max_sentences, max_tokens, embedding_dim]

        # apply 1d conv + max pool, result:  batch_size, num_filters_total        
        ref = tuple(data['text'].size())
        shape = (ref[0], ref[1], ( len(self.args.cnn_filter_sizes) * self.args.cnn_num_filters))
        output = torch.randn(shape).to(device)
        
        if weights is None:
            for i in range(ebd.size(0)): # 각 배치에 대해
                out = self._conv_max_pool(ebd[i], conv_filter=self.convs) # 각 문장에 대해 처리 => (35, 300)
                output[i] = out
        
        else:
            for i in range(ebd.size(0)):
                out = self._conv_max_pool(ebd[i], weights=weights)
                output[i] = out

        output = self.mlp(output)
        output = self.out(output) # output: [batch, max_sentence, n_class]
        
        return output
    
def get_embedding(vocab, args):
    ebd = WORDEBD(vocab, args.finetune_ebd)
    model = CNNseq(ebd, args)
    model.to(device)
    return model

def get_classifier(emb_dim, args):
    model = MLPseq(emb_dim, args)
    model.to(device)
    return model


"""
class MLPseq(BASE):
    def __init__(self, ebd_dim, args, top_layer=None):
        super(MLPseq, self).__init__(args)

        self.args = args
        self.ebd_dim = ebd_dim

        self.mlp = self._init_mlp(ebd_dim, self.args.mlp_hidden, self.args.dropout)
        self.out = nn.Linear(args.mlp_hidden[-1], self.args.n_classes)
        self.dropout = nn.Dropout(self.args.dropout)

        
    def _init_mlp(self, in_d, hidden_ds, drop_rate):
        modules = []

        for d in hidden_ds[:-1]:
            modules.extend([
                nn.Dropout(drop_rate),
                nn.Linear(in_d, d),
                nn.ReLU()])
            in_d = d

        modules.extend([
            nn.Dropout(drop_rate),
            nn.Linear(in_d, hidden_ds[-1])])

        return nn.Sequential(*modules)
    
    def forward(self, XS):
        '''
            if y is specified, return loss and accuracy
            otherwise, return the transformed x

            @param: XS: batch_size * input_dim
            @param: YS: batch_size (optional)

            @return: XS: batch_size * output_dim
        '''


        XS = self.mlp(XS)
        XS = self.out(XS) # output: [batch, max_sentence, n_class]
        
        return XS
"""