import json
import os
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm.auto import tqdm

from utils import padded_tensor


class CRSRecDataset(Dataset): 
    def __init__(
        self, dpath, split, tokenizer, context_max_length=None, entity_max_length=None
    ):
        super(CRSRecDataset, self).__init__()
        #self.debug = debug
        self.tokenizer = tokenizer
        # self.prompt_tokenizer = prompt_tokenizer
        # self.use_resp = use_resp

        self.context_max_length = context_max_length
        if self.context_max_length is None:
            self.context_max_length = self.tokenizer.model_max_length
            
        self.entity_max_length = entity_max_length
        if self.entity_max_length is None:
            self.entity_max_length = self.tokenizer.model_max_length

        data_file = os.path.join(dpath, f'{split}_data_processed.jsonl')
        self.data = []
        self.prepare_data(data_file)
        
    def prepare_data(self, data_file):
        with open(data_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # if self.debug:
            #     lines = lines[:1024]
            
            for line in tqdm(lines):
                dialog = json.loads(line)
                if len(dialog['rec']) == 0:
                    continue
                
                context = ''
                
                for i, utt in enumerate(dialog['contexts']):
                    context += utt
                    context += self.tokenizer.eos_token # </s>
                
                context_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(context))
                context_ids = context_ids[-self.context_max_length:] # max_length에 맞춰 뒷부분을 유지
                
                for item in dialog['rec']:
                    data = {
                        "conv_id": dialog["conv_id"],
                        "context": context_ids,
                        "entity": dialog['entities'][-self.entity_max_length:],
		                "rec": item
                    }
                    self.data.append(data)

    def __getitem__(self, ind):
        return self.data[ind]

    def __len__(self):
        return len(self.data)


class CRSRecDataCollator:
    def __init__(
        self, tokenizer, context_max_length, pad_entity_id, device
    ):
        self.tokenizer = tokenizer
        self.padding = True #batch 내에서 가장 긴 sequence의 길이에 맞춰 padding (이미 위에서 max_length만큼 잘라짐)
        self.context_max_length = context_max_length
        self.pad_entity_id = pad_entity_id
        self.device = device
        
    def __call__(self, data_batch):
        conv_id_batch = []
        context_batch = defaultdict(list)
        entity_batch = []
        label_batch = []
        
        for data in data_batch:
            conv_id_batch.append(data['conv_id'])
            input_ids = data['context']       
            context_batch['input_ids'].append(input_ids)
            entity_batch.append(data['entity'])
            label_batch.append(data['rec'])

        input_batch = {}

        context_batch = self.tokenizer.pad(
            context_batch, padding=self.padding, max_length=self.context_max_length)
        
        context_batch['rec_labels'] = label_batch
        for k, v in context_batch.items():
            if not isinstance(v, torch.Tensor):
                context_batch[k] = torch.as_tensor(v, device=self.device)
        
        input_batch['context'] = context_batch
        
        entity_batch = padded_tensor(entity_batch, pad_idx=self.pad_entity_id, pad_tail=True, device=self.device)
        
        input_batch['conv_id'] = conv_id_batch
        input_batch['context'] = context_batch
        input_batch['entity'] = entity_batch

        return input_batch