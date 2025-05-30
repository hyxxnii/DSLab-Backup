
import os
import random
import datetime

import numpy as np
from loguru import logger
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score
from termcolor import colored

from dataset.load_data import AscDataLoader
from config import parse_args
from utils import calculate_class_weights

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from transformers import get_scheduler


###############################################################################
## Environment Setting
###############################################################################

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

print(">>>>>>>>>>> CUDA available :", torch.cuda.is_available())


###############################################################################
## Evaluation Metric
###############################################################################

def evaluate_metrics(pred, true):
    pred_labels = torch.argmax(pred, dim=1)

    # 예측된 레이블과 실제 레이블을 numpy 배열로 변환
    pred_labels = pred_labels.cpu().numpy()
    true_labels = true.cpu().numpy()

    # accuracy 계산
    acc = accuracy_score(true_labels, pred_labels)

    # f1-score 계산
    weighted_f1 = f1_score(true_labels, pred_labels, average='weighted')
    macro_f1 = f1_score(true_labels, pred_labels, average='macro')

    return acc, weighted_f1, macro_f1


###############################################################################
## Definition Model for ASC 
###############################################################################

class ASCBert(nn.Module):
    def __init__(self, model, hidden_dim, num_labels, class_weights):
        super(ASCBert, self).__init__()
        self.bert = model        
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        # self.classifier = nn.Sequential(
        #                 nn.Linear(self.bert.config.hidden_size, hidden_dim),
        #                 nn.ReLU(),
        #                 nn.Dropout(0.2),
                        # nn.Linear(hidden_dim, num_labels))

        #self.loss_fn = nn.CrossEntropyLoss()
        self.weight_criterion = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None, freeze=False):
        outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        last_hidden_state, pooler_output = outputs[0], outputs[1]
        #last_hidden_state_cls = outputs[0][:, 0, :]
        
        if freeze == True:
            # probabilities of 3 sentiment label
            return 
        
        else:
            logits = self.classifier(pooler_output) # [batch_size, num_classes]
            #loss = self.loss_fn(logits, labels)
            loss = self.weight_criterion(logits, labels)
        
        return logits, loss
        
    def save_model(self, model_path):
        torch.save(self.state_dict(), f'{model_path}.bin')

    def load_model(self, model_path):
        self.load_state_dict(torch.load(f'{model_path}.bin'))


class ASCModel():
    def __init__(self, args, asc_model, train_dataloader, val_dataloader, save_name=None):
        self.args = args
        self.asc_model = asc_model
        self.save_name = save_name
        
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        self.best_metric_dir = os.path.join(args.output_dir, 'best', 'models')
        os.makedirs(self.best_metric_dir, exist_ok=True)
        
        num_training_steps = args.asc_num_epochs * len(train_dataloader)
        self.optimizer = optim.AdamW(self.asc_model.parameters(), lr=args.asc_learning_rate, weight_decay=args.weight_decay)
        
        self.lr_scheduler = get_scheduler(name="constant_with_warmup", optimizer=self.optimizer, 
                                    num_warmup_steps=200, num_training_steps=num_training_steps)
        
        # "constant_with_warmup: 학습률을 서서히 증가시킨 후 일정한 값으로 유지
        #  => 훈련 초기에 모델이 빠르게 수렴하지 않도록 하면서, 학습률이 너무 크거나 작아져 학습이 불안정해지는 것을 방지
        
    def train(self):
        logger.info('    ASC: Training and Valid Evaluation    ')
        
        for epoch in tqdm(range(self.args.asc_num_epochs)):
            train_loss = []
            self.asc_model.train()
            
            for step, batch in enumerate(self.train_dataloader):
                logits, loss = self.asc_model(**batch["context"])
                train_loss.append(loss.detach().cpu())
                
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
            
            train_loss = np.mean(train_loss)
            logger.info(f'epoch {epoch} : train loss {train_loss}') #sum([l for l in train_loss])/len(train_loss)
            del train_loss

            # evaluation
            logger.info("    ***** validation *****    ")
            self.val()
            
            intermed = datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S')
            logger.info(f'    ***** end of epoch {epoch} *****    ')
            logger.info(intermed)
            
        ## fine-tuning 후 save
        save_path = os.path.join(self.best_metric_dir, self.save_name)
        self.asc_model.save_model(model_path=save_path)
        
        print()
        print()
        logger.info("    ***** End of fine-tuning asc model *****    ")

    def val(self):
        valid_loss = []
        accuracy = []
        f1_weighted = []
        f1_macro = []
        correct = 0
        
        self.asc_model.eval()
        
        for step, batch in enumerate(self.val_dataloader):#(tqdm(self.val_dataloader)):
            with torch.no_grad():
                logits, loss = self.asc_model(**batch["context"])
                valid_loss.append(loss.detach().cpu())
                
                acc, weighted_f1, macro_f1 = evaluate_metrics(logits, batch["context"]["labels"])
                accuracy.append(acc)
                f1_weighted.append(weighted_f1)
                f1_macro.append(macro_f1)
                
                print('Val set: acc:%.3f%%' % (100. * acc))
        #         labels = batch["context"]["labels"].cpu().numpy()
        #         pred = torch.argmax(logits, dim=1)
        #         pred = pred.cpu().numpy()
        #         correct += pred.eq(labels).sum().item()
        
        # pre_acc = correct / len(self.val_dataloader.dataset)
        
        # print('Val set: acc:%d/%d(%.3f%%)' %(correct, len(self.val_dataloader.dataset), 100. * pre_acc))
        
        accuracy, f1_weighted, f1_macro = np.array(accuracy), np.array(f1_weighted), np.array(f1_macro)
        
        print("{}, {:s} {:>7.4f}, {:s} {:>7.4f}, {:s} {:>7.4f}, {:s} {:>7.4f}".format(
                datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S'),
                colored("valid loss", "blue"),
                np.mean(valid_loss),
                colored("acc mean", "blue"),
                np.mean(accuracy),
                colored("weighted f1 score", "blue"),
                np.mean(f1_weighted),
                colored("macro f1 score", "blue"),
                np.mean(f1_macro),
                ), flush=True)
                
        logger.info('End of validation')
        
               
        
class TEST_ASCModel():
    def __init__(self, args, asc_model, test_dataloader, save_name=None):
        self.asc_model = asc_model
        self.args = args
        self.save_name = save_name

        self.test_dataloader = test_dataloader 
        
        self.best_metric_dir = os.path.join(args.output_dir, 'best', 'models')
        os.makedirs(self.best_metric_dir, exist_ok=True)        
        
    def test(self):
        save_path = os.path.join(self.best_metric_dir, self.save_name)
        self.asc_model.load_model(model_path=save_path)       
        self.asc_model.eval()
        
        test_loss = []
        accuracy = []
        f1_weighted = []
        f1_macro = []
        
        logger.info("    ***** test *****    ")
        
        for step, batch in enumerate(tqdm(self.test_dataloader)):#(tqdm(self.val_dataloader)):
            with torch.no_grad():
                logits, loss = self.asc_model(**batch["context"])
                test_loss.append(loss.detach().cpu())
                
                acc, weighted_f1, macro_f1 = evaluate_metrics(logits, batch["context"]["labels"])
                accuracy.append(acc)
                f1_weighted.append(weighted_f1)
                f1_macro.append(macro_f1)
                
        accuracy, f1_weighted, f1_macro = np.array(accuracy), np.array(f1_weighted), np.array(f1_macro)
        
        print("{}, {:s} {:>7.4f}, {:s} {:>7.4f}, {:s} {:>7.4f}, {:s} {:>7.4f}".format(
                datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S'),
                colored("test loss", "blue"),
                np.mean(test_loss),
                colored("acc mean", "blue"),
                np.mean(accuracy),
                colored("weighted f1 score", "blue"),
                np.mean(f1_weighted),
                colored("macro f1 score", "blue"),
                np.mean(f1_macro),
                ), flush=True)

        intermed = datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S')
        logger.info(intermed)
        print()
        
        logger.info("    ***** End of testing fine-tuned asc model *****    ")
        



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
    
    model_path = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_path)
    tokenizer.add_special_tokens({'eos_token': '[EOS]'})
    model = BertModel.from_pretrained(model_path)
    model.resize_token_embeddings(len(tokenizer))

    asc_train_dataloader, asc_val_dataloader, asc_test_dataloader = AscDataLoader(args, tokenizer)
    
    class_weights = calculate_class_weights(asc_train_dataloader)
    class_weights = torch.tensor(class_weights).to(args.device)
    
    asc_model = ASCBert(model, args.asc_hidden_dim, args.asc_num_labels, class_weights)
    asc_model = asc_model.to(args.device)

    # training info
    logger.info("***** Running training for Aspect Sentiment Classification *****")
    logger.info(f"  Num Epochs = {args.asc_num_epochs}")
    logger.info(f"  Train batch size = {args.asc_train_batch_size}")

    start = datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S')
    logger.info(f'Start time: {start}')
    
    torch.autograd.set_detect_anomaly(True)
    
    model_loop = ASCModel(args, asc_model, asc_train_dataloader, asc_val_dataloader, save_name=args.asc_save_name)
    model_loop.train()
    
    logger.info("***** End Training *****")
    
    logger.info("***** Running test *****")
    
    test_model_loop = TEST_ASCModel(args, asc_model, asc_test_dataloader, save_name=args.asc_save_name)
    test_model_loop.test()
    
    end = datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S')
    logger.info(f'End time: {end}')
    
    
    ########################################
    # 1. weighted cross-entropy: 
    #### weights =  torch.tensor(self.class_weights).to(self.device)
    #### criterion = nn.CrossEntropyLoss(weight=weights)