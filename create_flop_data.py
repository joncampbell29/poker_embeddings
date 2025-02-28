import numpy as np
import pandas as pd
from itertools import product, combinations, chain
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from utils import *


def flatten_tup(nested_tuple):
    return tuple(chain.from_iterable((x, *y) for x, y in nested_tuple))

class FlopDataset(Dataset):
    def __init__(self):
        super().__init__()
        RANKS = [str(i) for i in range(2,10)] + ['T','J','Q','K','A']
        SUITS = ['s','h','c','d']

        RANKS_DICT = {str(i): i for i in range(2, 10)}
        RANKS_DICT.update({'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14})
        SUITS_DICT = {suit: i for i, suit in enumerate(SUITS)}

        DECK = [rank+suit for rank, suit in product(RANKS, SUITS)]
        self.FLOPS = list(combinations(DECK, 3))
        
        self.SUITS_ONE_HOT = nn.functional.one_hot(torch.arange(4),4)

        DECK_ENCODED = list(
            product(
                [RANKS_DICT[i] for i in RANKS], 
                [tuple(i) for i in self.SUITS_ONE_HOT.tolist()]
                )
            )
        
        self.suitedness = torch.tensor([eval_suitedness(flop) for flop in self.FLOPS], dtype=torch.float)
        self.pairness = torch.tensor([eval_pairness(flop) for flop in self.FLOPS], dtype=torch.float)
        self.connectedness = torch.tensor([eval_connectedness(flop) for flop in self.FLOPS], dtype=torch.float)
        self.high_low_texture = torch.tensor([eval_high_low_texture(flop) for flop in self.FLOPS], dtype=torch.float)
        self.high_card = torch.tensor([eval_high_card(flop) for flop in self.FLOPS], dtype=torch.float)
        self.straightness = torch.tensor([eval_straightness(flop) for flop in self.FLOPS], dtype=torch.float)

        self.FLOPS_ENCODED = torch.tensor([flatten_tup(i) for i in combinations(DECK_ENCODED, 3)], dtype=torch.float)
        
    def __len__(self):
        return self.FLOPS_ENCODED.shape[0]

    def __getitem__(self, idx):
        return (
            self.FLOPS_ENCODED[idx], 
            self.FLOPS[idx], 
            self.suitedness[idx], 
            self.pairness[idx], 
            self.connectedness[idx], 
            self.high_low_texture[idx], 
            self.high_card[idx], 
            self.straightness[idx]
            )
        
def flop_collate_fn(batch):
    flops_encoded = torch.stack([item[0] for item in batch])
    raw_flops = [item[1] for item in batch]
    suitedness = torch.tensor([item[2] for item in batch])  
    pairness = torch.tensor([item[3] for item in batch])
    connectedness = torch.tensor([item[4] for item in batch])
    high_low_texture = torch.tensor([item[5] for item in batch])
    high_card = torch.tensor([item[6] for item in batch])
    straightness = torch.tensor([item[7] for item in batch])
    return (
        flops_encoded, 
        raw_flops, 
        suitedness, 
        pairness, 
        connectedness, 
        high_low_texture, 
        high_card, 
        straightness
        )
    
if __name__ == '__main__':
    data = FlopDataset()
    data_pandas = pd.DataFrame({
        'flop': data.FLOPS,
        'suitedness': data.suitedness,
        'pairness': data.pairness,
        'connectedness': data.connectedness,
        'high_low_texture': data.high_low_texture,
        'high_card': data.high_card,
        'straightness': data.straightness
        })
    data_pandas.to_csv('flopdata.csv')
    