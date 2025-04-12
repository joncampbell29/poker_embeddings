import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo 
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from poker_utils.constants import HANDS_DICT, DECK_DICT
from poker_utils.hands import normalize_hand
import random

class UCIrvineDataset(Dataset):
    def __init__(self, train=None):
        self.suit_id_mapping = {'c':0,'d':1,'h':2,'s':3}
        uc_irvine_suit_mapping = {1:'h', 2:'s', 3:'d', 4:'c'}
        uc_irvine_rank_mapping = {
            1:'A', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6',
            7:'7', 8:'8', 9:'9', 10:'T', 11:'J', 12:'Q', 13:'K'
            }
        uc_irvine_class_mapping = {
            0:'nothing', 1:'one_pair', 2:'two_pair', 3:'three_of_a_kind', 4:'straight', 
            5:'flush', 6:'full_house', 7:'four_of_a_kind', 8:'straight_flush', 9:'royal_flush'
            }
        poker_hand = fetch_ucirepo(id=158) 
        X = poker_hand.data.features.copy()
        y = poker_hand.data.targets.copy()
        suit_cols = [f'S{i}' for i in range(1, 6)]
        rank_cols = [f'C{i}' for i in range(1, 6)]
        X[suit_cols] = X[suit_cols].apply(lambda col: col.map(uc_irvine_suit_mapping))
        X[rank_cols] = X[rank_cols].apply(lambda col: col.map(uc_irvine_rank_mapping))
        hand_to_id = {hand: idx for idx, hand in HANDS_DICT.items()}
        card_to_id = {card:idx for idx,card in DECK_DICT.items()}
        for i in range(1, 6):
            X[f'card{i}'] = X[f'C{i}'] + X[f'S{i}']
            X[f'card{i}_id'] = X[f'card{i}'].map(card_to_id)
            
        X['hands_norm'] = list(zip(X['card1'], X['card2']))
        X['hands_norm'] = X['hands_norm'].apply(normalize_hand)
        X['hands_norm_id'] = X['hands_norm'].map(hand_to_id)
        
        self.deck = np.arange(52)
       
        X.drop(['C1','S1','C2','S2','C3','S3','C4','S4','C5','S5'], axis=1, inplace=True)
        if train is not None:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=29, stratify=y
            )
            if train:
                self.X = X_train.reset_index(drop=True)
                self.y = y_train.reset_index(drop=True)
                used_cards = self.X[['card1_id', 'card2_id', 'card3_id', 'card4_id', 'card5_id']].to_numpy()
                remaining_cards = []
                for row in used_cards:
                    mask = np.isin(self.deck, row, invert=True)
                    remaining = self.deck[mask]
                    remaining_cards.append(remaining)
                self.remaining_cards = np.stack(remaining_cards)
                
            else:
                self.X = X_val.reset_index(drop=True)
                self.y = y_val.reset_index(drop=True)
                used_cards = self.X[['card1_id', 'card2_id', 'card3_id', 'card4_id', 'card5_id']].to_numpy()
                remaining_cards = []
                for row in used_cards:
                    mask = np.isin(self.deck, row, invert=True)
                    remaining = self.deck[mask]
                    remaining_cards.append(remaining)
                self.remaining_cards = np.stack(remaining_cards)
                
        else:
            self.X = X
            self.y = y
            used_cards = self.X[['card1_id', 'card2_id', 'card3_id', 'card4_id', 'card5_id']].to_numpy()
            remaining_cards = []
            for row in used_cards:
                mask = np.isin(self.deck, row, invert=True)
                remaining = self.deck[mask]
                remaining_cards.append(remaining)
            self.remaining_cards = np.stack(remaining_cards)
            
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        y = self.y.iloc[index]['CLASS']
        row = self.X.iloc[index]
        suit1_id = self.suit_id_mapping[row['card1'][-1]]
        suit2_id = self.suit_id_mapping[row['card2'][-1]]
        hand_id = row['hands_norm_id']
        
        base_board = [row['card3_id'], row['card4_id'], row['card5_id']]
        remaining = self.remaining_cards[index]
        board_extension_size = random.randint(0, 2)
        sampled = random.sample(list(remaining), board_extension_size)
        full_board = base_board + sampled
        random.shuffle(full_board)
        while len(full_board) < 5:
            full_board.append(-1)
        
        return (
            torch.tensor(hand_id, dtype=torch.long),
            torch.tensor(suit1_id, dtype=torch.long),
            torch.tensor(suit2_id, dtype=torch.long),
            torch.tensor(full_board, dtype=torch.long),
            torch.tensor(y, dtype=torch.long)
        )