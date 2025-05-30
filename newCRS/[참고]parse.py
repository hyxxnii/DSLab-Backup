import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser(description='BERT4REC for CRS')
    
    ### Path ##
    parser.add_argument('--path', type=str, default="/home/hyuns6100/[3]CRS/redial/", 
                        help="Data path.")
    
    parser.add_argument('--export_root', type=str, default="/home/hyuns6100/[3]CRS/experiment/", 
                        help="Experiment path.")
    
    ## BERT4REC #
    parser.add_argument('--bert_max_len', type=int, default=50, #50,
                        help="Max length for BERT.")
    parser.add_argument('--hidden_units', type=int, default=32 ,#32,
                        help="Size of embedding.")
    parser.add_argument('--num_heads', type=int, default=2, #2, 
                        help="Number of multi-head layers.")
    parser.add_argument('--num_layers', type=int, default=2, 
                        help="Number of blocks (encoder layers).")
    parser.add_argument('--dropout_rate', type=float, default=0.2, 
                        help="Dropout rate.")
    parser.add_argument('--mask_prob', type=float, default=0.15, 
                        help="Masking probability for cloze task.")
    
    ## lr & optimizer ##
    parser.add_argument('--lr', type=float, default=1e-2, 
                        help="Learning rate.")
    parser.add_argument('--decay_step', type=int, default=15, help='Decay step for StepLR')
    parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for StepLR')
    parser.add_argument('--global_norm_clip', type=float, default=5, help='Threshold for gradient clipping')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay.')
    # parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for AdamW.')
    # parser.add_argument('--beta2', type=float, default=0.999, help='beta1 for AdamW.')
    # parser.add_argument('--epsilon', type=float, default=5.0, help='epsilon for AdamW.')
    
    ## dataloader & train ## 
    parser.add_argument('--batch_size', type=int, default=256,
                        help="Batch size.")
    parser.add_argument('--num_epochs', type=int, default=100, 
                        help="Number of epochs.")
    parser.add_argument('--num_workers', type=int, default=4, 
                        help="Number of workers.")
    parser.add_argument('--neg_sample_size', type=int, default=100, 
                        help="Negative sample size.")
    parser.add_argument('--duple_nums', type=int, default=10, 
                        help="Number of iteration for making different training data with different random masking")
    
    ## init ##
    parser.add_argument('--train_key', type=int, default=9347, 
                        help="Number of original train data")
    parser.add_argument('--num_items', type=int, default=7486, #7489 (순서 [1,2]), #7486 (Ours) #10895 (TSCR),
                        help="Number of items(including entities)")
    parser.add_argument('--num_labels', type=int, default=3, # dislike, like, unknown
                        help="Number of preference label of item")
    parser.add_argument('--seed', type=int, default=12345, 
                        help="Random seed.")
    parser.add_argument('--model_init_seed', type=int, default=12345)
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    
    # logger #
    # parser.add_argument('--log_period_as_iter', type=int, default=10000)
    parser.add_argument('--save_name', type=str, default="base_t1", # base model (TSCR) -> try 1 
                        help="Save path for the best model")
    
    # evaluation #
    parser.add_argument('--metric_ks', nargs='+', type=int, default=[1, 10, 50], help='ks for Metric@k')
    parser.add_argument('--best_metric', type=str, default='Recall@50', help='Metric for determining the best model')
    
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #args.device = torch.device("cpu")
    return args
