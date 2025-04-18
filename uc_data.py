import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo 
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from poker_utils.constants import HANDS_DICT, DECK_DICT
from poker_utils.hands import normalize_hand
import random
from treys import Card, Evaluator

class UCIrvineDataset(Dataset):
    def __init__(self, 
                 X_path="data/uc_irvine/X.csv", 
                 y_path="data/uc_irvine/y.csv", 
                 test_size=0.2, 
                 add_random_cards=True,
                 train=None):
        
        self.add_random_cards = add_random_cards
        X = pd.read_csv(X_path)
        y = pd.read_csv(y_path)
        self.evaluator = Evaluator()
        self.card_to_id = {card:idx for idx,card in DECK_DICT.items()}
        deck = np.array(list(DECK_DICT.values()))
        self.deck_treys = np.array([Card.new(i) for i in deck])
        if train is not None:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=test_size, random_state=29, stratify=y['CLASS']
            )
            if train:
                self.X = X_train.reset_index(drop=True)
                self.y = y_train.reset_index(drop=True)
                if self.add_random_cards:
                    self.refresh_boards()
            else:
                self.X = X_val.reset_index(drop=True)
                self.y = y_val.reset_index(drop=True)
                if self.add_random_cards:
                    self.refresh_boards()
        else:
            self.X = X
            self.y = y
            if self.add_random_cards:
                self.refresh_boards()
            
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        y = self.y.iloc[index]['CLASS']
        row = self.X.iloc[index]

        hand = row['card1_id'], row['card2_id']
        if self.add_random_cards:
            full_board = self.full_boards[index]
        else:
            full_board = row['card3_id'], row['card4_id'], row['card5_id']
        
        return (
            torch.tensor(hand, dtype=torch.long),
            torch.tensor(full_board, dtype=torch.long),
            torch.tensor(y, dtype=torch.long)
        )
    def refresh_boards(self):
        num_rows = len(self.X)
        full_boards = np.full((num_rows, 5), -1, dtype=np.int64)
        card_id_lookup = {Card.new(card): self.card_to_id[card] for card in self.card_to_id}
        deck_set = set(self.deck_treys.tolist())

        for row in self.X.itertuples(index=True):
            i = row.Index
            label_str = self.y.iloc[i]['CLASS_str']
            used_treys = row.card1_treys, row.card2_treys, row.card3_treys, row.card4_treys, row.card5_treys
            used_set = set(used_treys)
            remaining_treys = list(deck_set - used_set)

            base_board = [row.card3_id, row.card4_id, row.card5_id]
            board_extension_size = random.randint(0, 2)

            attempts = 0
            while True:
                attempts += 1
                sampled = random.sample(remaining_treys, board_extension_size)
                full_hand = list(used_treys) + sampled
                hand_rank = self.evaluator.get_rank_class(self.evaluator.evaluate([], full_hand))
                if self.evaluator.class_to_string(hand_rank) == label_str:
                    break
                if attempts > 10:
                    break

            sampled_ids = [card_id_lookup.get(c, -1) for c in sampled]
            full_board = base_board + sampled_ids
            
            while len(full_board) < 5:
                full_board.append(-1)
            random.shuffle(full_board)
            full_boards[i] = full_board

class UCIrvineDatasetDynamic(Dataset):
    def __init__(self, X, y, add_random_cards=True):
        self.X = X.reset_index(drop=True)
        self.y = y.reset_index(drop=True)
        self.add_random_cards = add_random_cards

        self.evaluator = Evaluator()
        self.card_to_id = {card: idx for idx, card in DECK_DICT.items()}
        self.deck_treys = np.array([Card.new(c) for c in DECK_DICT.values()])
        self.card_int_to_id = {Card.new(card): idx for idx, card in DECK_DICT.items()}

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        row = self.X.iloc[idx]
        y_label = self.y.iloc[idx]['CLASS']
        label_str = self.y.iloc[idx]['CLASS_str']

        hand = (row['card1_id'], row['card2_id'])
        used_treys = row['card1_treys'], row['card2_treys'], row['card3_treys'], row['card4_treys'], row['card5_treys']
        base_board = [row['card3_id'], row['card4_id'], row['card5_id']]

        if self.add_random_cards:
            board = self.sample_random_board(used_treys, base_board, label_str)
        else:
            board = base_board + [-1, -1]

        return (
            torch.tensor(hand, dtype=torch.long),
            torch.tensor(board, dtype=torch.long),
            torch.tensor(y_label, dtype=torch.long)
        )

    def sample_random_board(self, used_treys, base_board, label_str):
        used_set = set(used_treys)
        remaining = [card for card in self.deck_treys if card not in used_set]
        board_extension_size = random.randint(0, 2)
        attempts = 0
        evaluator = self.evaluator

        while True:
            attempts += 1
            sampled = random.sample(remaining, board_extension_size)
            full_hand = list(used_treys) + sampled
            hand_rank = evaluator.get_rank_class(evaluator.evaluate([], full_hand))
            if evaluator.class_to_string(hand_rank) == label_str:
                break
            if attempts > 10:
                sampled = []
                break
            

        sampled_ids = [self.card_int_to_id[card] for card in sampled]
        full_board = base_board + sampled_ids
        
        while len(full_board) < 5:
            full_board.append(-1)
        
        random.shuffle(full_board)
        return full_board
