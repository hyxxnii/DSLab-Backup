import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Arguments for Mental Health Care model')

    # Add arguments
    parser.add_argument('--data_path', type=str, default="/home/hyuns6100/Mental-Heatlh-Care/data/dailydialog_conv35seq_splits.json", help='Path to the data')
    parser.add_argument('--result_path', type=str, default="/home/hyuns6100/Mental-Heatlh-Care/Result/best_results.pkl", help='Path to save results')
    parser.add_argument('--result_text_path', type=str, default="/home/hyuns6100/Mental-Heatlh-Care/Result/result_text/", help='Path to save result texts')
    parser.add_argument('--best_result_path', type=str, default="/home/hyuns6100/Mental-Heatlh-Care/Result/best_results.pkl", help='Path to save result texts')
    parser.add_argument('--wv_path', type=str, default="/home/hyuns6100/Mental-Heatlh-Care/data/", help='Path to word vectors')
    parser.add_argument('--word_vector', type=str, default="wiki-news-300d-1M.vec", help='Word vector file name')

    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--train_epochs', type=int, default=10000, help='Number of training epochs')
    parser.add_argument('--n_classes', type=int, default=7, help='Number of classes in dataset')
    parser.add_argument('--n_train_class', type=int, default=7, help='Number of training classes')
    parser.add_argument('--n_val_class', type=int, default=7, help='Number of validation classes')
    parser.add_argument('--n_test_class', type=int, default=7, help='Number of test classes')

    parser.add_argument('--cnn_num_filters', type=int, default=100, help='Number of CNN filters')
    parser.add_argument('--cnn_filter_sizes', type=list, default=[3,4,5], help='Size of CNN filters')
    parser.add_argument('--context_size', type=int, default=35, help='Context size')
    parser.add_argument('--maxtokens', type=int, default=30, help='Maximum number of tokens')
    parser.add_argument('--mlp_hidden', type=list, default=[300,300], help='MLP hidden layers size')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')

    parser.add_argument('--seed', type=int, default=330, help='Random seed')
    parser.add_argument('--patience_metric', type=str, default='f1', help='Metric for patience')
    parser.add_argument('--finetune_ebd', action='store_true', help='Whether to finetune embeddings')
    parser.add_argument('--patience', type=int, default=100, help='Patience for early stopping')
    parser.add_argument('--save', action='store_true', help='Whether to save the model')
    parser.add_argument('--authors', action='store_true', help='Flag for authors')
    parser.add_argument('--convmode', type=str, default='seq', help='')
    parser.add_argument('--embedding', type=str, default='cnn', help='')
    parser.add_argument('--classifier', type=str, default='mlp', help='')
    
    return parser