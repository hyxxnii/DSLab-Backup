import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--output_dir", type=str, default='/home/hyuns6100/[4]newCRS/save/', help="Where to store the final model.")
    parser.add_argument("--save_name", type=str, default='base_rec', help="Name of modeling type")
    parser.add_argument("--asc_save_name", type=str, default='asc_bert', help="Name of modeling type")
    
    parser.add_argument("--output_dataset_dir", type=str, default='/home/hyuns6100/[4]newCRS/data/redial/', help="Where to store the final model.")
    
    #parser.add_argument("--debug", action='store_true', help="Debug mode.")
    
    # data
    parser.add_argument("--dataset", type=str, default="/home/hyuns6100/[4]newCRS/data/redial/", help="A file containing all data.")
    parser.add_argument("--kg_dataset", type=str, default="/home/hyuns6100/[4]newCRS/data/dbpedia/")
    
    parser.add_argument("--context_max_length", type=int, default=128) #256, help="max input length in dataset.")
    parser.add_argument("--entity_max_length", type=int, default=50, help="max entity length in dataset.")
    parser.add_argument('--num_workers', type=int, default=0)
    
    parser.add_argument("--senti_tokenizer", type=str, default="roberta-base")
    parser.add_argument("--text_tokenizer", type=str, default="roberta-base")
    
    # model
    # parser.add_argument("--model", type=str, required=True,
    #                     help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--text_encoder", type=str, default="roberta-base")
    parser.add_argument("--num_bases", type=int, default=8, help="num_bases in RGCN.")
    
    parser.add_argument("--kg_emb_dim", type=int, default=128)
    #parser.add_argument("--prompt_encoder", type=str)
    
    # optim
    parser.add_argument("--num_epochs", type=int, default=30, help="Total number of training epochs to perform.")
    # parser.add_argument("--max_train_steps", type=int, default=None,
    #                     help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--batch_size", type=int, default=32, # 64
                        help="Batch size for the training dataloader.")
    
    parser.add_argument('--global_norm_clip', type=float, default=5, help='Threshold for gradient clipping')
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay to use.")
    parser.add_argument('--max_grad_norm', type=float)
    parser.add_argument('--num_warmup_steps', type=int)
    # parser.add_argument('--fp16', action='store_true')
    
    ## for ASC task
    parser.add_argument("--asc_max_seq_length", type=int, default=256, help="")
    parser.add_argument("--asc_train_batch_size", type=int, default=32, help="")
    parser.add_argument("--asc_eval_batch_size", type=int, default=32, help="")
    parser.add_argument("--asc_learning_rate", type=int, default=5e-5, help="") # lr_scheduler
    parser.add_argument("--asc_num_epochs", type=int, default=5, help="")
    parser.add_argument("--asc_num_labels", type=int, default=3, help="Number of polarity labels (like, dislike, unknown)")
    parser.add_argument("--asc_hidden_dim", type=int, default=128, help="")
    parser.add_argument("--asc_version", type=str, required=True, help=" asc 전처리 파일 버전 여러 개 중 하나 골라서 ~ ")
    
    
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #args.device = "cpu"
    return args