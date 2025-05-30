    
###############################################################################
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
###############################################################################

import numpy as np
import pandas as pd
from tqdm import tqdm
import json
from collections import Counter
import collections
import pickle
from termcolor import colored
import time
import datetime
import easydict
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data_utils

import torchtext
from torchtext.vocab import Vocab, Vectors


from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score
from sklearn.metrics import precision_recall_fscore_support

from utils import *
from model import *




def set_seed(seed):
    """
        Setting random seeds
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)



def _CalACC(model, dataloader, mode, epoch =0, directory=''):
    model.eval()
    correct = 0
    label_list = []
    pred_list = []
    p1num, p2num, p3num = 0, 0, 0    
    # label arragne
    with torch.no_grad():
        for i_batch, data in enumerate(tqdm(dataloader, desc="Eval: ", position=1, leave=False, ncols=150)):            
            
            YS = data['label']
            
            """Prediction"""
            data['text'] = data['text'].to(device)
            data['label'] = data['label'].to(device)
            pred_logits = model(data) # (1, clsNum)#, batch_labels

            """Calculation"""    
            pred_logits_sort = pred_logits.sort(descending=True)
            indices = pred_logits_sort.indices.tolist()
            true_label = YS.view(-1)#batch_labels.item()


            for conv in indices:
                for uttr in conv:
                    pred_list.append(uttr[0])
            
            for true in true_label:
                label_list.append(true)

            
        
    return [0,0,0], pred_list, label_list #[p1, p2, p3]

def _SaveModel(model, path):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), os.path.join(path, 'model.bin'))


def main():

    save_path = os.path.join('dailydialog_models')
    log_path = os.path.join(save_path, 'test.log')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fileHandler = logging.FileHandler(log_path)
    
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)    
    logger.setLevel(level=logging.DEBUG)     


    set_seed(args.seed)
    loader = dailydialog_DataLoader(args)
    train_data, val_data, test_data, vocab = loader.load_dataset()
    test_loader = data_utils.DataLoader(SupervisedDataset(test_data, args), batch_size=args.batch_size, num_workers=2, shuffle=False)

    model = get_embedding(vocab, args)
    save_path = os.path.join('dailydialog_models')
    modelfile = os.path.join(save_path, 'model.bin')
    model.load_state_dict(torch.load(modelfile))
    model = model.cuda()
    
    model.eval()

    test_prek, test_pred_list, test_label_list = _CalACC(model, test_loader,'test', 0, save_path)
    test_pre_macro, test_rec_macro, test_fbeta_macro, _ = precision_recall_fscore_support(test_label_list, test_pred_list, labels=[0,1,2,3,4,5,6], average='macro', zero_division = 0)
    test_pre_micro, test_rec_micro, test_fbeta_micro, _ = precision_recall_fscore_support(test_label_list, test_pred_list, labels=[1,2,3,4,5,6], average='micro', zero_division = 0) # neutral x                
    
    
        

    with open('dumps/TRUE_text.csv', 'w') as f:
            # np.savetxt(f, true.cpu().detach().numpy())
            np.savetxt(f, np.array(test_label_list), fmt='%d', delimiter = ',')
        
    with open('dumps/PRED_text.csv', 'w') as f:
            # np.savetxt(f, pred.cpu().detach().numpy())
            np.savetxt(f, np.array(test_pred_list),fmt='%d', delimiter = ',')

    correct_predictions = np.sum(np.array(test_label_list) == np.array(test_pred_list)) 
    acc = correct_predictions / len(test_pred_list)
    print('Final Fscore ## test-accuracy: {},test-precision: {}, test-macro: {}, test-micro: {}'.format(acc, test_prek, test_fbeta_macro, test_fbeta_micro)) 


if __name__ == "__main__":

    print(">>>>>>>>>>> CUDA available :", torch.cuda.is_available())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args = easydict.EasyDict({
    'data_path' : "data/dailydialog_conv35seq_splits.json", # 데이터 경로
    'result_path': "/home/hyuns6100/Mental-Heatlh-Care/Result/",
    'result_text_path': "/home/hyuns6100/Mental-Heatlh-Care/Result/result_text/",
    'best_result_path': "/home/hyuns6100/Mental-Heatlh-Care/Result/best_results.pkl",
    'wv_path': "/home/hyuns6100/Mental-Heatlh-Care/data/", 
    'word_vector':"wiki-news-300d-1M.vec",
    
    'lr': 1e-3,
    'batch_size':32,
    'train_epochs':100,
    'n_classes': 7,
    'n_train_class': 7, 
	'n_val_class': 7, 
	'n_test_class': 7, 
    'labels': [1, 2, 3, 4, 5, 6],
 
    'cnn_num_filters': 100,
    'cnn_filter_sizes': [3,4,5],
    'context_size': 35,
    'maxtokens': 30,
    'mlp_hidden': [300,300],
    'dropout': 0.1,
    
    'seed':330,
    'patience_metric': 'f1',
    'finetune_ebd': False,
    'patience': 100,
    'save': True,
    'authors':False,
    'convmode': 'seq',
    'embedding': 'cnn',
    'classifier': 'mlp'
    })


    logger = logging.getLogger(__name__)
    streamHandler = logging.StreamHandler()

    main()