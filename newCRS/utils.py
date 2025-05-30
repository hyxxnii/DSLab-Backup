import numpy as np
from typing import List, Union, Optional
import torch

# 주어진 배치에서 가장 max_lenght 에 맞춰 padding 되도록 
def padded_tensor( 
    items: List[Union[List[int], torch.LongTensor]],
    pad_idx: int = 0,
    pad_tail: bool = True,
    max_len: Optional[int] = None,
    debug: bool = False,
    device: torch.device = torch.device('cpu'),
    use_amp: bool = False
) -> torch.LongTensor:
    """Create a padded matrix from an uneven list of lists.

    Returns padded matrix.

    Matrix is right-padded (filled to the right) by default, but can be
    left padded if the flag is set to True.

    Matrix can also be placed on cuda automatically.

    :param list[iter[int]] items: List of items
    :param int pad_idx: the value to use for padding
    :param bool pad_tail:
    :param int max_len: if None, the max length is the maximum item length

    :returns: padded tensor.
    :rtype: Tensor[int64]

    """
    # number of items
    n = len(items)
    # length of each item
    lens: List[int] = [len(item) for item in items]
    # max in time dimension
    t = max(lens)
    # if input tensors are empty, we should expand to nulls
    t = max(t, 1)
    if debug and max_len is not None:
        t = max(t, max_len)

    if use_amp:
        t = t // 8 * 8

    output = torch.full((n, t), fill_value=pad_idx, dtype=torch.long, device=device)

    for i, (item, length) in enumerate(zip(items, lens)):
        if length == 0:
            continue
        if not isinstance(item, torch.Tensor):
            item = torch.tensor(item, dtype=torch.long, device=device)
        if pad_tail:
            output[i, :length] = item
        else:
            output[i, t - length:] = item

    return output


def calculate_class_weights(dataloader):
    train_polarity = {"like": 0, "dislike": 0, "unknown": 0}

    for batch in dataloader:
        labels = batch["context"]["labels"]
        train_polarity["dislike"] += torch.sum(labels == 0).item()
        train_polarity["like"] += torch.sum(labels == 1).item()  
        train_polarity["unknown"] += torch.sum(labels == 2).item()
        
    total_samples = sum(train_polarity.values())
    
    
    # 1. 역 빈도 가중치
    #weights = [total_samples / train_polarity["dislike"], total_samples / train_polarity["like"], total_samples / train_polarity["unknown"]]
    
    # 2. 루트 역 빈도 가중치 계산
    # weights = [torch.sqrt(torch.tensor(total_samples / train_polarity["dislike"])),
    #            torch.sqrt(torch.tensor(total_samples / train_polarity["like"])), 
    #            torch.sqrt(torch.tensor(total_samples / train_polarity["unknown"]))]
    
    # 3. 최대 빈도수 대비 가중치
    # 가장 많이 출현하는 클래스의 빈도수 찾기
    max_frequency = max(train_polarity.values())

    # 각 클래스에 대한 최대 빈도수 대비 가중치 계산
    weights = [max_frequency / train_polarity["dislike"], 
            max_frequency / train_polarity["like"], 
            max_frequency / train_polarity["unknown"]]
    
    return weights
