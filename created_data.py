import torch
from torch.utils.data import Dataset
from utils import *
from itertools import combinations


class FlopDataset(Dataset):
    def __init__(self, train=True):
        super().__init__()
        self.flops = list(combinations(DECK, 3))

        self.flop_vectors = [hand_to_vector(flop) for flop in self.flops]

        self.labels = torch.tensor([
            [
                eval_suitedness(flop),
                eval_pairness(flop),
                eval_connectedness(flop),
                eval_high_low_texture(flop),
                eval_high_card(flop)
            ] for flop in self.flops
        ], dtype=torch.long) 
        
        self.label_dict = {
            'suitedness': {'idx':0, **SUITEDNESS_DICT},
            'pairness': {'idx':1, **PAIRNESS_DICT}, 
            'connectedness': {'idx':2, **CONNECTEDNESS_DICT}, 
            'high_low_texture': {'idx':3, **HIGH_LOW_TEXTURE_DICT}, 
            'high_card': {'idx':4, **HIGH_CARD_DICT}
            }

    def __len__(self):
        return len(self.flops)

    def __getitem__(self, idx):
        flop_vector = self.flop_vectors[idx]
        label_vector = self.labels[idx]
        return flop_vector, label_vector