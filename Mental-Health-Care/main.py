import os
from termcolor import colored

import torch
from torch.utils.data import Dataset, DataLoader

from parse import parse_args
from utils import set_seed
from dataloader import dailydialog_DataLoader, SupervisedDataset
from embedding import WORDEBD, CNNseq
from Model.mlp import MLPseq
from training import train, test


###############################################################################
## Environment Setting
###############################################################################
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

print(">>>>>>>>>>> CUDA available :", torch.cuda.is_available())

args = parse_args()
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


###############################################################################
## DataLoader
###############################################################################
set_seed(args.seed)
loader = dailydialog_DataLoader(args)
train_data, val_data, test_data, vocab = loader.load_dataset()

train_loader = DataLoader(SupervisedDataset(train_data, args), batch_size=args.batch_size, num_workers=2, shuffle=False)
val_loader = DataLoader(SupervisedDataset(val_data, args), batch_size=args.batch_size, num_workers=2, shuffle=False)
test_loader = DataLoader(SupervisedDataset(test_data, args), batch_size=args.batch_size, num_workers=2, shuffle=False)


###############################################################################
## Model Config
###############################################################################
model = {}

wordebd = WORDEBD(vocab, args.finetune_ebd)
ebd = CNNseq(wordebd, args).to(args.device)
model["ebd"] = ebd

clf = MLPseq(model["ebd"].ebd_dim, args).to(args.device)

model["clf"] = clf
train_loader = DataLoader(SupervisedDataset(train_data, args), batch_size=args.batch_size, num_workers=2, shuffle=False)
val_loader = DataLoader(SupervisedDataset(val_data, args), batch_size=args.batch_size, num_workers=2, shuffle=False)
test_loader = DataLoader(SupervisedDataset(test_data, args), batch_size=args.batch_size, num_workers=2, shuffle=False)

model["clf"] = clf


###############################################################################
## Train & Test
###############################################################################

train(train_data, val_data, model, args, loader=train_loader)

val_acc, val_std, _, _, _, _, = test(val_data, model, args, verbose=True, target='val', loader=val_loader)

print()
print( colored('test_data', 'green') )
print()