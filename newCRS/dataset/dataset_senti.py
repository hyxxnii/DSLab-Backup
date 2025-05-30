import json
import os
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from utils import padded_tensor


class CRSSentiDataset(Dataset): 
    def __init__(
        self, dpath, split, tokenizer, context_max_length=None
    ):
        super(CRSSentiDataset, self).__init__()
        #self.debug = debug
        self.tokenizer = tokenizer
        # self.prompt_tokenizer = prompt_tokenizer
        # self.use_resp = use_resp

        self.context_max_length = context_max_length
        if self.context_max_length is None:
            self.context_max_length = self.tokenizer.model_max_length

        # data_file = os.path.join(dpath, f'reformat_senti_{split}_data_processed.jsonl')
        #data_file = os.path.join(dpath, f'remove_role_reformat_senti_{split}_data_processed.jsonl')
        #data_file = os.path.join(dpath, f'original_movieid_reformat_senti_{split}_data_processed.jsonl')
        data_file = os.path.join(dpath, f'only_user_uni_item_reformat_senti_{split}_data_processed.jsonl')
        
        self.data = []
        self.prepare_data(data_file)
        
    def prepare_data(self,data_file):
    
        with open(data_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # if self.debug:
            #     lines = lines[:1024]
            
            for line in tqdm(lines):
                dialog = json.loads(line)
                
                context = ''
                for i, utt in enumerate(dialog['contexts']):
                    context += utt
                    context += self.tokenizer.eos_token # </s>
                
                context_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(context))
                context_ids = context_ids[-self.context_max_length:]
                
                data = {
                    "conv_id": dialog["conv_id"],
                    #"uni_conv_id": dialog["unique_conv_id"],
                    "context": context_ids,
                    "entity": dialog["entities"],
                    #"senti": "unk"
                }
                self.data.append(data)


    def __getitem__(self, ind):
        return self.data[ind]

    def __len__(self):
        return len(self.data)


class CRSSentiDataCollator:
    def __init__(
        self, tokenizer, context_max_length, device
    ):
        self.tokenizer = tokenizer
        self.padding = True
        self.context_max_length = context_max_length
        self.device = device
        
    def __call__(self, data_batch):
        conv_id_batch = []
        #uni_conv_id_batch = []
        context_batch = defaultdict(list)
        entity_batch = []
        #senti_batch = []
        
        for data in data_batch:
            conv_id_batch.append(data['conv_id'])
            #uni_conv_id_batch.append(data['uni_conv_id'])
            input_ids = data['context']       
            context_batch['input_ids'].append(input_ids)
            entity_batch.append(data['entity'])
            #senti_batch.append(data['senti'])

        input_batch = {}

        context_batch = self.tokenizer.pad(
            context_batch, padding=self.padding, max_length=self.context_max_length)
        
        # context_batch['rec_labels'] = label_batch
        
        for k, v in context_batch.items():
            if not isinstance(v, torch.Tensor):
                context_batch[k] = torch.as_tensor(v, device=self.device)
        
        # context_batch['senti_labels'] = senti_batch
        
        # 모델링 시
        # 1. **batch["context"] 로 model 인자 주고 
        # 2. 모델 forward 에서 sentiment logits 값 구하면 
        # 3. if senti_labels is not None: 코드에서 해당 logits 값 저장 (필요시 3개의 label [pos/neg/neu] 에 대한 각각의 값을 딕셔너리로 저장해야할수도~
        
        input_batch['conv_id'] = conv_id_batch
        #input_batch['uni_conv_id'] = uni_conv_id_batch
        input_batch['context'] = context_batch
        input_batch['entity'] = entity_batch

        return input_batch