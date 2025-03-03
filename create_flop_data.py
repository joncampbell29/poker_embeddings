import numpy as np
import pandas as pd
from itertools import product, combinations
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from utils import *


class FlopDataset(Dataset):
    def __init__(self, flop_to_vec_func=flop_to_vector, train=None):
        super().__init__()
        self.flop_to_vec_func = flop_to_vec_func
        
        RANKS = [str(i) for i in range(2,10)] + ['T','J','Q','K','A']
        SUITS = ['s','h','c','d']

        self.DECK = [rank+suit for rank, suit in product(RANKS, SUITS)]
            
        self.FLOPS = list(combinations(DECK, 3))
        self.FLOPS = [sorted(i) for i in self.FLOPS]
        
        self.flops_encoded = [self.flop_to_vec_func(flop) for flop in self.FLOPS]
        self.suitedness = [eval_suitedness(flop) for flop in self.FLOPS]
        self.pairness = [eval_pairness(flop) for flop in self.FLOPS]
        self.connectedness = [eval_connectedness(flop) for flop in self.FLOPS]
        self.high_low_texture = [eval_high_low_texture(flop) for flop in self.FLOPS]
        self.high_card = [eval_high_card(flop) for flop in self.FLOPS]
        self.straightness = [eval_straightness(flop) for flop in self.FLOPS]
        self.label_vec = [torch.tensor([a,b,c,d,e,f]) 
                          for a,b,c,d,e,f in 
                          zip(self.suitedness,self.pairness,
                              self.connectedness,self.high_low_texture,
                              self.high_card, self.straightness)]
        
        self.label_dict = {
            'suitedness': {'idx':0, **SUITEDNESS_DICT},
            'pairness': {'idx':1, **PAIRNESS_DICT}, 
            'connectedness': {'idx':2, **CONNECTEDNESS_DICT}, 
            'high_low_texture': {'idx':3, **HIGH_LOW_TEXTURE_DICT}, 
            'high_card': {'idx':4, **HIGH_CARD_DICT}, 
            'straightness': {'idx':5, **STRAIGHTNESS_DICT}
            }
        
        data = pd.DataFrame({
            'flop': self.FLOPS,
            'flop_encoded': self.flops_encoded,
            'label_vec': self.label_vec,
            'suitedness': self.suitedness,
            'pairness': self.pairness,
            'connectedness': self.connectedness,
            'high_low_texture': self.high_low_texture,
            'high_card': self.high_card,
            'straightness': self.straightness
        })
        data[['card1', 'card2', 'card3']] = data['flop'].apply(pd.Series)
        if train is not None:
            train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
            train_data.reset_index(drop=True, inplace=True)
            test_data.reset_index(drop=True, inplace=True)
            if train:
                self.df = train_data
            else:
                self.df = test_data
        else:
            self.df = data
                
    def __len__(self):
        return self.df.shape[0]
        
    def __getitem__(self, idx):
        samp = self.df.iloc[idx]
        return samp['flop_encoded'], "".join(samp['flop']), samp['label_vec']


if __name__ == '__main__':
    data = FlopDataset()
    data.df['flop_encoded'] = data.df['flop_encoded'].apply(np.array)
    data.df['label_vec'] = data.df['label_vec'].apply(np.array)
    data.df.to_parquet(f"data/flopdata.parquet")
    