from torch.utils.data import DataLoader

from dataset.dataset_senti import CRSSentiDataset, CRSSentiDataCollator
from dataset.dataset_redial import CRSRecDataset, CRSRecDataCollator
from dataset.dataset_asc import CRSAscDataset, CRSAscDataCollator

def AscDataLoader(args, tokenizer):
    asc_train_dataset = CRSAscDataset(
            dpath=args.dataset, split='train', 
            tokenizer=tokenizer, context_max_length=args.asc_max_seq_length,
            version=args.asc_version
        )

    asc_valid_dataset = CRSAscDataset(
            dpath=args.dataset, split='valid', 
            tokenizer=tokenizer, context_max_length=args.asc_max_seq_length,
            version=args.asc_version
        )

    asc_test_dataset = CRSAscDataset(
            dpath=args.dataset, split='test', 
            tokenizer=tokenizer, context_max_length=args.asc_max_seq_length,
            version=args.asc_version
        )

    asc_data_collator = CRSAscDataCollator(
            tokenizer=tokenizer, context_max_length=args.asc_max_seq_length,
            device=args.device
        )

    asc_train_dataloader = DataLoader(
            asc_train_dataset,
            batch_size=args.asc_train_batch_size,
            collate_fn=asc_data_collator,
            shuffle=True
        )
    asc_val_dataloader = DataLoader(
            asc_valid_dataset,
            batch_size=args.asc_eval_batch_size,
            collate_fn=asc_data_collator,
        )
    asc_test_dataloader = DataLoader(
            asc_test_dataset,
            batch_size=args.asc_eval_batch_size,
            collate_fn=asc_data_collator,
        )
    
    return asc_train_dataloader, asc_val_dataloader, asc_test_dataloader


def SentDataLoader(args, tokenizer):
    sent_train_dataset = CRSSentiDataset(
            dpath=args.dataset, split='train', 
            tokenizer=tokenizer, context_max_length=args.context_max_length,
        )

    sent_valid_dataset = CRSSentiDataset(
            dpath=args.dataset, split='valid', 
            tokenizer=tokenizer, context_max_length=args.context_max_length,
        )

    sent_test_dataset = CRSSentiDataset(
            dpath=args.dataset, split='test', 
            tokenizer=tokenizer, context_max_length=args.context_max_length,
        )

    sent_data_collator = CRSSentiDataCollator(
            tokenizer=tokenizer, context_max_length=args.context_max_length,
            device=args.device
        )

    sent_train_dataloader = DataLoader(
            sent_train_dataset,
            batch_size=args.batch_size,
            collate_fn=sent_data_collator,
            shuffle=True
        )
    sent_val_dataloader = DataLoader(
            sent_valid_dataset,
            batch_size=args.batch_size,
            collate_fn=sent_data_collator,
        )
    sent_test_dataloader = DataLoader(
            sent_test_dataset,
            batch_size=args.batch_size,
            collate_fn=sent_data_collator,
        )
    
    return sent_train_dataloader, sent_val_dataloader, sent_test_dataloader


def RecDataLoader(args, tokenizer, pad_entity_id):
    train_dataset = CRSRecDataset(
                dpath=args.dataset, split='train', 
                tokenizer=tokenizer, context_max_length=args.context_max_length,
            )

    valid_dataset = CRSRecDataset(
            dpath=args.dataset, split='valid', 
            tokenizer=tokenizer, context_max_length=args.context_max_length,
        )

    test_dataset = CRSRecDataset(
            dpath=args.dataset, split='test', 
            tokenizer=tokenizer, context_max_length=args.context_max_length,
        )

    data_collator = CRSRecDataCollator(
            tokenizer=tokenizer, context_max_length=args.context_max_length,
            pad_entity_id=pad_entity_id, device=args.device
        )

    train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            collate_fn=data_collator,
            shuffle=True
        )
    val_dataloader = DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            collate_fn=data_collator,
        )
    test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            collate_fn=data_collator,
        )
    
    return train_dataloader, val_dataloader, test_dataloader
