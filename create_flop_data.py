import numpy as np
import pandas as pd
from itertools import product, combinations, chain
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from utils import *


class FlopDataset(Dataset):
    def __init__(self, data=None):
        super().__init__()
        if data is None:
            self.data = None
            RANKS = [str(i) for i in range(2,10)] + ['T','J','Q','K','A']
            SUITS = ['s','h','c','d']

            self.DECK = [rank+suit for rank, suit in product(RANKS, SUITS)]
            
            self.FLOPS = list(combinations(DECK, 3))
            self.FLOPS = [tuple(sorted(i)) for i in self.FLOPS]
            
            self.suitedness = torch.tensor([eval_suitedness(flop) for flop in self.FLOPS], dtype=torch.float)
            self.pairness = torch.tensor([eval_pairness(flop) for flop in self.FLOPS], dtype=torch.float)
            self.connectedness = torch.tensor([eval_connectedness(flop) for flop in self.FLOPS], dtype=torch.float)
            self.high_low_texture = torch.tensor([eval_high_low_texture(flop) for flop in self.FLOPS], dtype=torch.float)
            self.high_card = torch.tensor([eval_high_card(flop) for flop in self.FLOPS], dtype=torch.float)
            self.straightness = torch.tensor([eval_straightness(flop) for flop in self.FLOPS], dtype=torch.float)

        else:
            self.data = data
            

    def __len__(self):
        if self.data is None:
            return len(self.FLOPS)
        else:
            return self.data.shape[0]
        

    def __getitem__(self, idx):
        if self.data is None:
            flop_encoded = flop_to_vector(self.FLOPS[idx])
            return (
                flop_encoded,
                self.FLOPS[idx], 
                self.suitedness[idx], 
                self.pairness[idx], 
                self.connectedness[idx], 
                self.high_low_texture[idx], 
                self.high_card[idx], 
                self.straightness[idx]
                )
        else:
            samp = self.data.iloc[idx]
            return (
                torch.tensor(samp['flop_encoded']), 
                samp['flop'], 
                samp['suitedness'], 
                samp['pairness'], 
                samp['connectedness'], 
                samp['high_low_texture'], 
                samp['high_card'], 
                samp['straightness']
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
        'flop_encoded': torch.stack([flop_to_vector(i) for i in data.FLOPS]).tolist(),
        'suitedness': data.suitedness,
        'pairness': data.pairness,
        'connectedness': data.connectedness,
        'high_low_texture': data.high_low_texture,
        'high_card': data.high_card,
        'straightness': data.straightness
        })
    data_pandas['flop'] = data_pandas['flop'].apply(sorted)
    data_pandas[['card1', 'card2', 'card3']] = data_pandas['flop'].apply(pd.Series)
    data_pandas.to_parquet('data/flopdata.parquet')
    