import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import RGCNConv

from models.utils.attention import SelfAttentionBatch, ApplySentSelfAttentionBatch

class RecModel(nn.Module):
    def __init__(self, kg_emb_dim, n_entity, num_relations,
                 num_bases, edge_index, edge_type, device, pad_entity_id, 
                 train_ent_sent, val_ent_sent, test_ent_sent
    ):
        
        super(RecModel, self).__init__()
        
        self.device = device
        self.train_ent_sent = train_ent_sent
        self.val_ent_sent = val_ent_sent
        self.test_ent_sent = test_ent_sent
        # self.mode = None # decision for train/val/test entity-sentiment
        
        self.kg_emb_dim = kg_emb_dim
        self.kg_encoder = RGCNConv(n_entity, kg_emb_dim, num_relations, num_bases=num_bases)
        # self.kg_encoder = RGCNConv(kg_emb_dim, kg_emb_dim, num_relations=num_relations,
        #                            num_bases=num_bases)
        
        self.edge_index = nn.Parameter(edge_index, requires_grad=False) # edge_index가 entity 정보 및 연결이 담긴 인덱스들 => grad=False로 변하지 말아라 (관계성이니)
        self.edge_type = nn.Parameter(edge_type, requires_grad=False)
        self.pad_entity_id = pad_entity_id
    
        #self.kg_attn = SelfAttentionBatch(self.kg_emb_dim, self.kg_emb_dim)
        self.kg_sent_attn = ApplySentSelfAttentionBatch(self.kg_emb_dim, self.kg_emb_dim)
        
        self.rec_bias = nn.Linear(self.kg_emb_dim, n_entity)
        self.rec_loss = nn.CrossEntropyLoss()
    
    def get_user_rep(self, batch, ):
        user_rep, item_rep = self.get_kg_user_rep(batch)

        return user_rep, item_rep

    def get_kg_user_rep(self, batch):
        user_rep, kg_embedding = self.get_kg_rep(batch) # (bs, user_dim), (n_entities, user_dim)

        return user_rep, kg_embedding # (bs, user_dim), (n_entities, user_dim)
    
    def get_kg_rep(self, batch):
        context_entities = batch['entity']
        conv_ids = batch['conv_id']

        kg_user_rep, kg_embedding = self._get_kg_user_rep(context_entities, conv_ids)  # (bs, dim), (n_entity, dim)

        return kg_user_rep, kg_embedding
    
    def _get_kg_user_rep(self, context_entities, conv_ids):
        # ipdb.set_trace()
        kg_embedding = self.kg_encoder(None, self.edge_index, self.edge_type)
        # print(self.config.edge_idx.shape)
        # ipdb.set_trace()
        
        #print(kg_embedding.shape) # torch.Size([30890, 128])
        
        user_rep = self._encode_user(context_entities, kg_embedding, conv_ids)  # (bs, dim)

        return user_rep, kg_embedding
    
    def _encode_user(self, entity_lists, kg_embedding, conv_ids):
        user_repr_list = []
            
        # if self.mode == 'train':
        #     ent_sent = self.train_ent_sent
        # elif self.mode == 'val':
        #     ent_sent = self.val_ent_sent
        # elif self.mode == 'test':
        #     ent_sent = self.test_ent_sent
        
        for i, entity_list in enumerate(entity_lists): # entity_lists = batch['entity'] => i번째 batch 데이터를 반복하면서
            # pad_entity_id를 제외한 실제 entity ID만 필터링
            user_entity_list = [entity for entity in entity_list if entity != self.pad_entity_id]
            
            if not user_entity_list:
                # 실제 엔티티가 없는 경우
                user_repr_list.append(torch.zeros(self.kg_emb_dim, device=self.device))
                continue
            
            user_entity_tensor_list = torch.tensor(user_entity_list, dtype=torch.long, device=self.device)
            user_repr = kg_embedding[user_entity_tensor_list] # (n_e, dim)
            #user_repr = self.kg_attn(user_repr) # (dim)
            user_repr = self.kg_sent_attn(user_repr, user_entity_tensor_list, conv_ids[i], self.ent_sent)#ent_sent)
            user_repr_list.append(user_repr)

        return torch.stack(user_repr_list, dim=0)  # (bs, dim)
    
    def forward(self, batch, ent_sent):#mode):
        
        self.ent_sent = ent_sent
        
        print(list(self.ent_sent.items())[-50])
        
        #if mode=='test':
            # print("      test_ent_sent 길이 출력: ", len(self.test_ent_sent))
        
        labels = batch['context']['rec_labels']
        user_rep, item_rep = self.get_user_rep(batch) # (bs, user_dim), (n_entities, user_dim)
        rec_scores = F.linear(user_rep, item_rep, self.rec_bias.bias)  # (bs, n_entity)
        rec_loss = self.rec_loss(rec_scores, labels)
        
        #self.mode = None
        
        return rec_scores, rec_loss
    
    def save_model(self, model_path):
        torch.save(self.state_dict(), f'{model_path}.bin')

    def load_model(self, model_path):
        self.load_state_dict(torch.load(f'{model_path}.bin'))