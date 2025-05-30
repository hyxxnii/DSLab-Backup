import json
import os
from collections import defaultdict
from tqdm.auto import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class CRSAscDataset(Dataset): 
    def __init__(
        self, dpath, split, tokenizer, context_max_length=None, version=None
    ):
        super(CRSAscDataset, self).__init__()
        self.tokenizer = tokenizer
        self.context_max_length = context_max_length

        data_file = os.path.join(dpath, f'{version}_asc_{split}_data_processed.jsonl')
        self.data = []
        self.prepare_data(data_file)
        
    def prepare_data(self, data_file):
        with open(data_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    
            for line in tqdm(lines):
                dialog = json.loads(line)
                #contexts = self.tokenizer.cls_token
                contexts = ''
                
                for i, utt in enumerate(dialog["contexts"]):
                    contexts += utt
                    contexts += self.tokenizer.eos_token # 각 role의 발화를 구분하기 위함 #self.tokenizer.eos_token # </s>
                
                contexts += self.tokenizer.sep_token
                contexts_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(contexts))
                # sentences_ids = sentences_ids[-self.context_max_length:]
                
                context_segments = [0] * len(contexts_ids)
                
                for i, mv_set in enumerate(dialog["aspects"].items()):                    
                    mv_entityid = mv_set[0]
                    mv_aspect = mv_set[1]

                    mv_aspect += self.tokenizer.sep_token
                    mv_aspect_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(mv_aspect))
                    
                    asp_segments = [1] * len(mv_aspect_ids)
                    segments = context_segments + asp_segments
                    
                    sentences_ids = contexts_ids + mv_aspect_ids
                    sentences_ids = sentences_ids[-(self.context_max_length - 1):] # max_length에 맞춰 뒷부분을 유지
                    segments = segments[-(self.context_max_length - 1):]
                    
                    sentences_ids = [self.tokenizer.cls_token_id] + sentences_ids
                    segments = [0] + segments # for [CLS] token
                    
                    assert len(sentences_ids) == len(segments)
                    
                    # mv_set[0]: mv_id
                    # mv_set[1]: mv_aspect
                    #  self.tokenizer.encode_plus(text=sentences, text_pair=mv_aspect, add_special_tokens=True,
                    #         padding='max_length', max_length=self.context_max_length, pad_to_max_length=True,
                    #         return_token_type_ids=True, return_tensors='pt')

                    data = {
                        "conv_id": dialog["conv_id"],
                        "sentence": sentences_ids,
                        "segment": segments,
                        "polarity": dialog["polarity"][mv_entityid]
                    }
                    
                    self.data.append(data)
                    
        
    def __getitem__(self, ind):
        return self.data[ind]

    def __len__(self):
        return len(self.data)


class CRSAscDataCollator:
    def __init__(
        self, tokenizer, context_max_length, device
    ):
        self.tokenizer = tokenizer
        self.context_max_length = context_max_length
        self.device = device
        
    # def pad_contexts(self, contexts):     
    #     inputs = [torch.tensor(i, dtype=torch.long) for i in contexts["input_ids"]]
    #     segments = [torch.tensor(s, dtype=torch.long) for s in contexts["token_type_ids"]]
        
    #     # pad_sequence 적용
    #     inputs = pad_sequence(inputs, batch_first=True)
    #     segments = pad_sequence(segments, batch_first=True)
        
    #     # attention masks 생성
    #     attn_masks = torch.zeros(inputs.shape, dtype=torch.long)
    #     attn_masks = attn_masks.masked_fill(inputs != 0, 1)
    
    #     return inputs, segments, attn_masks
        
    def __call__(self, data_batch):
        conv_id_batch = []
        #uni_conv_id_batch = []
        context_batch = defaultdict(list)
        label_batch = []
        
        for data in data_batch: # "conv_id", "sentence", "segment", "polarity"
            conv_id_batch.append(data["conv_id"])
            #uni_conv_id_batch.append(data['uni_conv_id'])
            input_ids = data["sentence"]
            token_type_ids = data["segment"]  
            context_batch["input_ids"].append(input_ids)
            context_batch["token_type_ids"].append(token_type_ids)
            label_batch.append(data["polarity"])
            
        input_batch = {}
     
        #context_batch["input_ids"], context_batch["token_type_ids"], context_batch["attention_mask"] = self.pad_contexts(context_batch)
        context_batch = self.tokenizer.pad(
            context_batch, padding=True, max_length=self.context_max_length)
        
        context_batch['labels'] = label_batch
        
        for k, v in context_batch.items():
            context_batch[k] = torch.as_tensor(v, device=self.device)
           
        # 모델링 시
        # 1. **batch["context"] 로 model 인자 주고 
        # 2. 모델 forward 에서 sentiment logits 값 구하면 
        # 3. if senti_labels is not None: 코드에서 해당 logits 값 저장 (필요시 3개의 label [pos/neg/neu] 에 대한 각각의 값을 딕셔너리로 저장해야할수도~
        
        input_batch["conv_id"] = conv_id_batch
        input_batch["context"] = context_batch

        return input_batch