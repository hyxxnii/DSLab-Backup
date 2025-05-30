from collections import defaultdict
import torch.nn.functional as F
import torch.nn as nn

class EntitySentModel(nn.Module):
    def __init__(self, model):
        super(EntitySentModel, self).__init__()
        
        self.model = model
        self.ent_sent = defaultdict(dict)

    def forward(self, batch):
        input_ids, attention_mask = batch["context"]['input_ids'], batch["context"]['attention_mask']
        
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
            
        logits = logits.detach() ## 추론 시 기울기 흐르는 것 방지 (오류 발생했었음) => 결론적으로 해결 안됐음;;
        
        probabilities = F.softmax(logits, dim=1)
        
        # Extracting probabilities for negative and positive sentiments
        neg, neu, pos = probabilities.unbind(dim=-1)
        sent_score = pos / (pos + neg)
        
        for i, score in enumerate(sent_score):
            if len(batch['entity'][i]) != 0:
                for entity in batch['entity'][i]:
                    #Store the sentiment score in train_ent_sent
                    self.ent_sent[batch['conv_id'][i]][entity] = score
