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

def _get_dailydialog_seq_classes():
    '''
        seq consider no emotion label
        @return list of classes associated with each split
    '''
    label_dict = { 
        'no emotion': 0, 
        'anger': 1, 
        'disgust': 2,
        'fear': 3,
        'happiness': 4,
        'sadness': 5,
        'surprise': 6
    }

    train_classes = [0, 1, 2, 3, 4, 5, 6]
    val_classes = [0, 1, 2, 3, 4, 5, 6]
    test_classes = [0, 1, 2, 3, 4, 5, 6]

    return train_classes, val_classes, test_classes

def _load_json_seq(path, args):
    '''
        load data file
        @param path: str, path to the data file
        @return data: list of examples
    '''
    label = {}
    text_len = []
    with open(path, 'r', errors='ignore') as f:
        data = []
        for i, line in enumerate(f):
            row = json.loads(line)

            # count the number of examples per label
            for l in row['labels']:
                if int(l) not in label: label[int(l)] = 1
                else: label[int(l)] += 1

            item = {
                'id': i+1,
                'label': [int(r) for r in row['labels'] ],
                # 'text': [ r[:args['maxtokens']] for r in row['texts'] ]  # 30 # 50 # 80 truncate the text to 500 tokens
                'text': [ r[-args['maxtokens']:] for r in row['texts'] ]  # 30 # 50 # 80 truncate the text to the last tokens
            }

            if args.authors:
                item.update({'authors': [ int(a) for a in row['authors'] ]})

            if 'split' in row: item['split'] = row['split']

            text_len.append(len(row['texts']))

            data.append(item)

        #tprint('Class balance (load_json_seq):')

        print(label)

        print('Avg len: {}'.format(sum(text_len) / (len(text_len))))
        print('Max len: {}'.format(max(text_len)))

        return data
    
def _read_words(data, convmode=None):
    '''
        Count the occurrences of all words
        @param convmode: str, None for non conversational scope, 'naive' for classic or naive approach, 'conv' for conversation depth into account (one additional dim and nested values)
        @param data: list of examples
        @return words: list of words (with duplicates)
    '''
    words = []
    if convmode is None:
        for example in data:
            words += example['text']
    else:
        for example in data:
            for m in example['text']: 
                words += m     
    
    return words

def _meta_split_by_field(all_data, train_classes, val_classes, test_classes, seqmode=False):
    '''
        Split the dataset according to the specified train_classes, val_classes
        and test_classes
        Consider a 'split' field for the different train test val sets

        seqmode is a special mode to ensure sequences of labels to be taken into account

        @param all_data: list of examples (dictionaries)
        @param train_classes: list of int
        @param val_classes: list of int
        @param test_classes: list of int
        @param seqmode: bool 

        @return train_data: list of examples
        @return val_data: list of examples
        @return test_data: list of examples
    '''
    train_data, val_data, test_data = [], [], []

    if seqmode:
        for example in all_data:
            if example['split'] == 'train' and len(set(example['label']) & set(train_classes)) > 0: train_data.append(example)
            if example['split'] == 'val' and len(set(example['label']) & set(val_classes)) > 0: val_data.append(example)
            if example['split'] == 'test' and len(set(example['label']) & set(test_classes)) > 0: test_data.append(example)
    else: 
        for example in all_data:
            if example['split'] == 'train' and example['label'] in train_classes: train_data.append(example)
            if example['split'] == 'val' and example['label'] in val_classes: val_data.append(example)
            if example['split'] == 'test' and example['label'] in test_classes: test_data.append(example)

    return train_data, val_data, test_data

def _del_by_idx(array_list, idx, axis):
    '''
        Delete the specified index for each array in the array_lists

        @params: array_list: list of np arrays
        @params: idx: list of int
        @params: axis: int

        @return: res: tuple of pruned np arrays
    '''
    if type(array_list) is not list:
        array_list = [array_list]

    # modified to perform operations in place
    for i, array in enumerate(array_list):
        array_list[i] = np.delete(array, idx, axis)

    if len(array_list) == 1:
        return array_list[0]
    else:
        return array_list
    
    
def _data_to_nparray(data, vocab, args):
    '''
        Convert the data into a dictionary of np arrays for speed.
    '''
    doc_label = np.array([x['label'] for x in data], dtype=np.int64)

    raw = np.array([e['text'] for e in data], dtype=object)

    # compute the max text length
    text_len = np.array([len(m) for e in data for m in e['text']])
    max_text_len = max(text_len)
    seq_len = np.array(  [len(e['text']) for e in data]  )
    max_seq_len =  max(seq_len)
    ids = np.array([e['id'] for e in data])

    # initialize the big numpy array by <pad>
    text = vocab.stoi['<pad>'] * np.ones([len(data), max_seq_len, max_text_len], dtype=np.int64)

    del_idx = []
    # convert each token to its corresponding id
    for i in tqdm(range(len(data)), desc='converting tokens to ids'): # 모든 대화를 돌면서 한 대화 뭉텅이씩 처리
        for idx_x, x in enumerate(data[i]['text']): # 대화 내 모든 문장을 돌면서 한 문장씩 처리
                for idx_message, message in enumerate(x): # 각 문장 내 모든 토큰을 돌면서 각 토큰별 vocab을 이용하여 id로 변환
                        if message in vocab.stoi:
                                text[i, idx_x, idx_message] = vocab.stoi[message]
                        else:
                                text[i, idx_x, idx_message] = vocab.stoi['<unk>']
            # try:
            #     for idx_message, message in enumerate(x):
            #         text[i, idx_x, :len(message)] = [
            #                             vocab.stoi[token] if token in vocab.stoi else vocab.stoi['<unk>'] 
            #                             for token in message
            #                             ]
            # except Exception as e:
            #     print(e)
            #     print(x, idx_x)
            #     exit()

        # filter out document with only unk and pad
        if np.max(text[i]) < 2:
            del_idx.append(i)

    vocab_size = vocab.vectors.size()[0]
    
    print("del_idx: ", del_idx) # 빈 리스트 반환됨
    
    ## Curation for padding (string instead of list of list)
    raw = [ ["<pad>" if m == ["<pad>", "<pad>", "<pad>", "<pad>", "<pad>"] else m for m in c ] for c in raw ]

    if args.authors:
        # trim and pad authors (should have been done in dtaa creation but left here for comparison purposes)
        authors = list()
        for x in data:
            a = len(x['authors'])
            if a < args.context_size: 
                authors.append(x['authors'] + [0 for i in range(18-a)])
            elif a > args.context_size:
                authors.append( x['authors'][int(-args.context_size):] )
            else:
                authors.append(x['authors'])
        authors = np.array(authors, dtype=np.int64)

        ids, text_len, text, doc_label, raw, authors = _del_by_idx(
                [ids, text_len, text, doc_label, raw, authors], del_idx, 0)
        new_data = {
            'ids': ids,
            'text': text,
            'text_len': text_len,
            'label': doc_label,
            'raw': raw,
            'authors': authors,
            'vocab_size': vocab_size,
        }
        
    else: ## authors = False
        #ids, text_len, text, doc_label, raw = _del_by_idx( [ids, text_len, text, doc_label, raw], del_idx, 0)
        new_data = {
            'ids': ids,
            'text': text,
            'text_len': text_len,
            'label': doc_label,
            'raw': raw,
            'vocab_size': vocab_size,
        }
    return new_data
    


class dailydialog_DataLoader:
    def __init__(self, args):
        self.args = args
        
    def load_dataset(self):
        train_classes, val_classes, test_classes = _get_dailydialog_seq_classes()
        assert(len(train_classes) == self.args.n_train_class)
        assert(len(val_classes) == self.args.n_val_class)
        assert(len(test_classes) == self.args.n_test_class)
        
        all_data = _load_json_seq(self.args.data_path, self.args)
        
        # Loading word vector
        path = os.path.join(self.args.wv_path, self.args.word_vector)
        if not os.path.exists(path):
            # Download the word vector and save it locally:
            print('Downloading word vectors')
            import urllib.request
            urllib.request.urlretrieve(
                'https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec',
                path)
        
        vectors = Vectors(self.args.word_vector, cache=self.args.wv_path)
        min_freq = 2
        vocab = Vocab(collections.Counter(_read_words(all_data, convmode=self.args.convmode)), vectors=vectors,
                  specials=['<pad>', '<unk>'], min_freq=min_freq)
        
         # print word embedding statistics
        wv_size = vocab.vectors.size()
        print('Total num. of words: {}, word vector dimension: {}'.format(
            wv_size[0],
            wv_size[1]))
        
        num_oov = wv_size[0] - torch.nonzero(
            torch.sum(torch.abs(vocab.vectors), dim=1)).size()[0]
        print(('Num. of out-of-vocabulary words'
              '(they are initialized to zeros): {}').format( num_oov))
        
        # Split into meta-train, meta-val, meta-test data (or just splits)
        train_data, val_data, test_data = _meta_split_by_field(all_data, train_classes ,val_classes, test_classes, seqmode=True)
        trainset = Counter([l for d in train_data for l in d['label']])
        valset = Counter([l for d in val_data for l in d['label']])
        testset = Counter([l for d in test_data for l in d['label']])
        print(colored('check sets splits', 'yellow'), trainset, len(list(trainset.keys())), valset, len(list(valset.keys())),  testset, len(list(testset.keys())))
        
        # Convert everything into np array for fast data loading
        train_data = _data_to_nparray(train_data, vocab, self.args)
        val_data = _data_to_nparray(val_data, vocab, self.args)
        test_data = _data_to_nparray(test_data, vocab, self.args)

        train_data['is_train'] = True
        
        #return converted_train_data, converted_val_data, converted_test_data, train_data, val_data, test_data, vocab
        return train_data, val_data, test_data, vocab

class SupervisedDataset(Dataset):
    def __init__(self, data, args):
        '''
            data : dict_keys(['ids', 'text', 'text_len', 'label', 'raw', 'vocab_size', 'is_train']) 'authors'
        '''
        # self.berttokenizer = berttokenizer
        # if self.berttokenizer:
        #     self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.pretrained_bert))
        self.args = args
        self.ids = data['ids']
        self.text = data['text']
        self.text_len = data['text_len']
        self.label = data['label']
        self.raw = data['raw']
        #self.authors = data['authors']
        self.vocab_size = data['vocab_size']
        self.train = False
        if 'is_train' in data:
            self.is_train = data['is_train']
            self.train = True

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        
        item = {
            'ids': self.ids[idx], 
            'text': self.text[idx], 
            'text_len': self.text_len[idx], 
            # 'label': np.expand_dims(self.label[idx],0),  # .expand_dims(x, axis=0) unsqueeze(0) for seq labelling (bert)
            'label': self.label[idx],
            # 'raw': self.raw[idx].tolist(), 
            'vocab_size': self.vocab_size,
            #'authors': self.authors[idx]
        }

       
        if self.train: item.update({'is_train': self.is_train})
        
        return item


class BASE(nn.Module):
    '''
        BASE model
    '''
    def __init__(self, args):
        super(BASE, self).__init__()
        self.args = args

        # cached tensor for speed
        # self.I_way = nn.Parameter(torch.eye(self.args.way, dtype=torch.float),
        #                           requires_grad=False)
    
    @staticmethod
    def compute_acc(pred, true, dim=1, nomax=False):
        '''
            Compute the accuracy.
            @param pred: batch_size * num_classes
            @param true: batch_size
        '''
        if nomax: return torch.mean((pred == true).float()).item()
        else: return torch.mean((torch.argmax(pred, dim=dim) == true).float()).item()
        
    @staticmethod
    def compute_f1(y_pred, true, dim=1, nomax=False,  labels=None, average='weighted'):
        '''
            Compute the weighted f1 score.
            @param pred: batch_size * num_classes
            @param true: batch_size
        '''
        if not nomax: _, y_pred = torch.max(y_pred, dim)

        f1 = f1_score(true.cpu().detach().numpy(), y_pred.cpu().detach().numpy(), average=average, labels=labels )

        return f1

    @staticmethod
    def compute_mcc(y_pred, true, dim=1, nomax=False):
        '''
            Compute the matthews correlation coeficient.
            @param pred: batch_size * num_classes
            @param true: batch_size
        '''
        if not nomax: _, y_pred = torch.max(y_pred, dim)

        mcc = matthews_corrcoef(true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())

        return mcc

    import itertools

def grad_param(model, keys):
    '''
        Return a generator that generates learnable parameters in
        model[key] for key in keys.
    '''
    if len(keys) == 1:
        return filter(lambda p: p.requires_grad,
                model[keys[0]].parameters())
    else:
        return filter(lambda p: p.requires_grad,
                itertools.chain.from_iterable(
                    model[key].parameters() for key in keys))

def get_norm(model):
    '''
        Compute norm of the gradients
    '''
    total_norm = 0

    for p in model.parameters():
        if p.grad is not None:
            p_norm = p.grad.data.norm()
            total_norm += p_norm.item() ** 2

    total_norm = total_norm ** 0.5

    return total_norm