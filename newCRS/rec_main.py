
import os
import random
import datetime

import numpy as np
from loguru import logger
from tqdm.auto import tqdm

import torch
import torch.optim as optim


from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig

from dataset.dataset_dbpedia import DBpedia
from dataset.load_data import SentDataLoader, RecDataLoader

from models.toy_sent_model import EntitySentModel
from models.recommender_module import RecModel
from evaluate_rec import RecEvaluator
# from model_gpt2 import PromptGPT2forCRS
# from config import gpt2_special_tokens_dict, prompt_special_tokens_dict
# from model_prompt import KGPrompt
from config import parse_args

###############################################################################
## Environment Setting
###############################################################################

## without sentiment-aware user embedding (only recommendation module)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

print(">>>>>>>>>>> CUDA available :", torch.cuda.is_available())

###############################################################################
## Recommendation task
###############################################################################

class TrainLoop():
    def __init__(self, args, model, train_dataloader, val_dataloader, train_ent_set, val_ent_sent, save_name,
        ):
        self.args = args
        self.device = args.device
        self.save_name = save_name # 'base_rec', 'ent_sent_rec'
        
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        #self.test_dataloader = test_dataloader
        
        self.train_ent_set = train_ent_set
        self.val_ent_sent = val_ent_sent
        
        self.rec_evaluator = RecEvaluator()

        self.best_metric_dir = os.path.join(args.output_dir, 'best', 'models')
        os.makedirs(self.best_metric_dir, exist_ok=True)

        self.best_metric = -np.inf
    
        self.rec_model = model
        self.rec_model.to(self.device)
        
        self.optimizer = optim.AdamW(self.rec_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    def train(self):
        logger.info('>>> Training and Valid Evaluation')
        
        # epoch = 1
        # save_path = os.path.join(self.best_metric_dir, f'{self.args.save_name}_{epoch}')
        # with open(save_path, 'wb') as f:
        #     torch.save(self.rec_model, f)
        # print("save 성공")
        
        for epoch in tqdm(range(self.args.num_epochs)):
            train_loss = []
            self.rec_model.train()
            
            for step, batch in enumerate(self.train_dataloader):#(tqdm(self.train_dataloader)):
                rec_score, rec_loss = self.rec_model(batch, self.train_ent_set) #mode='train')
                train_loss.append(rec_loss.detach().cpu())
                
                self.backward(rec_loss)
                self.update_params() # gradient clipping & optimizer update
                # lr_scheduler.step()
                self.zero_grad()
                
                # if step % 100 == 0:
                #     # print('rec loss is %f'%(sum([l[0] for l in train_loss])/len(train_loss)))
                #     logger.info(f"  Rec loss is {sum([l for l in train_loss])/len(train_loss)}")
                #     train_loss = []
            
            train_loss = np.mean(train_loss)
            logger.info(f'epoch {epoch} : train loss {train_loss}') #sum([l for l in train_loss])/len(train_loss)
            del train_loss
            
            # evaluation
            self.val(epoch)
            
            intermed = datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S')
            logger.info(f'end of epoch {epoch}')
            logger.info(f'{intermed}')
            
    def val(self, epoch):
        valid_loss = []
        self.rec_model.eval()
        
        for step, batch in enumerate(self.val_dataloader):#(tqdm(self.val_dataloader)):
            with torch.no_grad():
                rec_score, rec_loss = self.rec_model(batch, self.val_ent_sent) #mode='val')
                valid_loss.append(rec_loss.detach().cpu())
                
                score = rec_score[:, kg['item_ids']]
                ranks = torch.topk(score, k=50, dim=-1).indices.tolist()
                ranks = [[kg['item_ids'][rank] for rank in batch_rank] for batch_rank in ranks] # 각 배치에 대한 상위 50개 엔티티의 인덱스 리스트
                labels = batch['context']['rec_labels']
                self.rec_evaluator.evaluate(ranks, labels)  #self.metrics_cal_rec(rec_loss, scores, labels)
            
        # metric
        metric_report = self.rec_evaluator.report()
        for k, v in metric_report.items():
            metric_report[k] = v.sum().item()
            
        valid_report = {}
        for k, v in metric_report.items():
            if k != 'count':
                valid_report[f'valid/{k}'] = v / metric_report['count']
        
        valid_report['valid/loss'] = np.mean(valid_loss)
        valid_report['epoch'] = epoch
        
        logger.info(f'{valid_report}')
        
        self.rec_evaluator.reset_metric()
        
        now_best_metric = valid_report['valid/recall@1'] + valid_report['valid/recall@50']
        
        if now_best_metric > self.best_metric:
            save_path = os.path.join(self.best_metric_dir, f'{self.args.save_name}_{epoch}')
            self.rec_model.save_model(save_path)
            
            # with open(save_path, 'wb') as f:
            #     torch.save(self.rec_model, f)
                
            self.best_metric = now_best_metric
            logger.info(f'>>> Save new best model with recall@1 + recall@50 at epoch {epoch}')
        
        logger.info('end of validation')
        print()
    
    def backward(self, loss):
        """
        Perform a backward pass. It is recommended you use this instead of
        loss.backward(), for integration with distributed training and FP16
        training.
        """
        loss.backward()

    def update_params(self):
        """
        Perform step of optimization, clipping gradients and adjusting LR
        schedule if needed. Gradient accumulation is also performed if agent
        is called with --update-freq.

        It is recommended (but not forced) that you call this in train_step.
        """
        update_freq = 1
        if update_freq > 1:
            # we're doing gradient accumulation, so we don't only want to step
            # every N updates instead
            
            ##  매 업데이트마다 바로 그래디언트를 적용하는 것이 아니라, 지정된 횟수(update_freq)만큼 그래디언트를 누적한 후에 
            ## 한 번에 업데이트를 수행 => 이는 메모리 제한으로 인해 큰 배치 사이즈를 사용할 수 없을 때 유용
            
            self._number_grad_accum = (self._number_grad_accum + 1) % update_freq
            if self._number_grad_accum != 0:
                return

        if self.args.global_norm_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.rec_model.parameters(), self.args.global_norm_clip
            )

        self.optimizer.step()
        
    def zero_grad(self):
            """
            Zero out optimizer.

            It is recommended you call this in train_step. It automatically handles
            gradient accumulation if agent is called with --update-freq.
            """
            self.optimizer.zero_grad()
            
            
class TestLoop():
    def __init__(self, args, model, test_dataloader, test_ent_sent, save_name):
        self.args = args
        self.device = args.device
        self.save_name = save_name
        
        self.test_dataloader = test_dataloader
        self.test_ent_sent = test_ent_sent
        
        self.rec_model = model
        self.rec_model.to(self.device)
        
        self.rec_evaluator = RecEvaluator()
        self.best_metric_dir = os.path.join(args.output_dir, 'best', 'models')
        
    # def load_model(self):
    #     path = os.path.join(self.best_metric_dir, 'models', self.save_name) # self.args.save_name_{epoch}'
    #     with open(path, 'rb') as f:
    #         self.rec_model = torch.load(f)
        
    def test(self):        
        logger.info('>>> Test Evaluation')
        
        save_path = os.path.join(self.best_metric_dir, self.save_name)
        self.rec_model.load_model(save_path)
        
        #self.rec_model.test_ent_sent = self.test_ent_sent
        self.rec_model.eval()
        
        test_loss = []
        for step, batch in enumerate(tqdm(self.test_dataloader)):
            with torch.no_grad():
                rec_score, rec_loss = self.rec_model(batch, self.test_ent_sent)#mode='test')
                test_loss.append(rec_loss.detach().cpu())
                
                score = rec_score[:, kg['item_ids']]
                ranks = torch.topk(score, k=50, dim=-1).indices.tolist()
                ranks = [[kg['item_ids'][rank] for rank in batch_rank] for batch_rank in ranks] # 각 배치에 대한 상위 50개 엔티티의 인덱스 리스트
                labels = batch['context']['rec_labels']
                self.rec_evaluator.evaluate(ranks, labels)
                
        # metric
        metric_report = self.rec_evaluator.report()
        for k, v in metric_report.items():
            metric_report[k] = v.sum().item()
            
        test_report = {}
        for k, v in metric_report.items():
            if k != 'count':
                test_report[f'test/{k}'] = v / metric_report['count']
        
        test_report['test/loss'] = np.mean(test_loss)
        # test_report['epoch'] = epoch
        
        logger.info(f'{test_report}')
        logger.info('end of test')
        
        self.rec_evaluator.reset_metric()
 
    
    
    
    
    
    
if __name__ == '__main__':
    args = parse_args()
    print(args)

    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        
    set_seed(args.seed)

    ###############################################################################
    ## Load Data
    ###############################################################################

    logger.info('***** Load Dataset *****')
    
    kg = DBpedia(dataset=args.dataset).get_entity_kg_info()

    task='sentiment'
    MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

    sent_model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    sent_model = sent_model.to(args.device)

    # Load the tokenizer
    senti_tokenizer = AutoTokenizer.from_pretrained(MODEL)

    sent_train_dataloader, sent_val_dataloader, sent_test_dataloader = SentDataLoader(args, senti_tokenizer)
    train_dataloader, val_dataloader, test_dataloader = RecDataLoader(args, senti_tokenizer, pad_entity_id=kg['pad_entity_id'])

    ###############################################################################
    ## Entity-Sentiment Prediction (Toy experiments)
    ###############################################################################
    
    logger.info("***** Load Data-Entity Sentiment Dictionary *****")
    
    # train_ent_sent_model = EntitySentModel(sent_model).to(args.device)
    # val_ent_sent_model = EntitySentModel(sent_model).to(args.device)
    test_ent_sent_model = EntitySentModel(sent_model).to(args.device)

    # for batch in tqdm(sent_train_dataloader):
    #     with torch.no_grad():
    #         train_ent_sent_model(batch)
        
    # for batch in tqdm(sent_val_dataloader):
    #     with torch.no_grad():
    #         val_ent_sent_model(batch)
        
    for batch in tqdm(sent_test_dataloader):
        with torch.no_grad():
            test_ent_sent_model(batch)

    # train_ent_sent = train_ent_sent_model.ent_sent
    # val_ent_sent = val_ent_sent_model.ent_sent
    test_ent_sent = test_ent_sent_model.ent_sent # 잘 됨
    
    logger.info("***** Load Model *****")
    
    model = RecModel(args.kg_emb_dim, n_entity=kg['num_entities'], num_relations=kg['num_relations'], 
                    num_bases=args.num_bases, edge_index=kg['edge_index'], edge_type=kg['edge_type'],
                    device=args.device, pad_entity_id=kg['pad_entity_id'],
                    train_ent_sent=None, val_ent_sent=None, test_ent_sent=test_ent_sent)#test_ent_sent)
   
    # training info
    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Instantaneous batch size = {args.batch_size}")
    # logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    # logger.info(f"  Total optimization steps = {args.max_train_steps}")

    start = datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S')
    logger.info(f'Start time: {start}')
    
    
    torch.autograd.set_detect_anomaly(True)
    
    
    # train_loop = TrainLoop(args, model, train_dataloader, val_dataloader, train_ent_set=train_ent_sent, val_ent_sent=val_ent_sent, save_name=args.save_name)
    # train_loop.train()
    
    # logger.info("***** End Training *****")
    # logger.info("                          ")
    
    logger.info("***** Running test *****")
    
    test_loop = TestLoop(args, model, test_dataloader, test_ent_sent, save_name=args.save_name) #train_ent_sent, val_ent_sent, test_ent_sent)
    test_loop.test()
    
    end = datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S')
    logger.info(f'End time: {end}')