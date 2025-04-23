import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from poker_utils.constants import HANDS_DICT, DECK_DICT
import random
from treys import Card, Evaluator
from torch_geometric.data import Data

class EquityDiffDataset:
    def __init__(self, path_to_handhand_equity):
        hand_to_id = {j:i for i,j in HANDS_DICT.items()}
        self.data = pd.read_csv(path_to_handhand_equity)
        self.data['hand1_id'] =self.data['hand1'].map(hand_to_id)
        self.data['hand2_id'] =self.data['hand2'].map(hand_to_id)
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        if random.random() < 0.5:
            hand1 = row['hand1_id']
            hand2 = row['hand2_id']
            equity_diff = row['hand1_equity'] - row['hand2_equity']
        else:
            hand1 = row['hand2_id']
            hand2 = row['hand1_id']
            equity_diff = row['hand2_equity'] - row['hand1_equity']

        return (
            torch.tensor(hand1, dtype=torch.long),
            torch.tensor(hand2, dtype=torch.long),
            torch.tensor(equity_diff, dtype=torch.float32)
        )


class UCIrvineDataset(Dataset):
    def __init__(self, X, y, add_random_cards=True, use_card_ids=True, graph=True, normalize_x=True):
        self.X = X
        self.y = y
        self.add_random_cards = add_random_cards
        self.use_card_ids = use_card_ids
        self.graph = graph
        self.normalize_x = normalize_x
        self.evaluator = Evaluator()
        self.card_to_id = {card: idx for idx, card in DECK_DICT.items()}
        self.deck_treys = np.array([Card.new(c) for c in DECK_DICT.values()])
        self.card_int_to_id = {Card.new(card): idx for idx, card in DECK_DICT.items()}
        
    def __getitem__(self, idx):
        row = self.X.iloc[idx]
        y = self.y.iloc[idx]
        cards_id = [i.item() for i in row.filter(regex='id')]
        if self.add_random_cards:
            card_treys = [i.item() for i in row.filter(regex='treys')]
            cards_id = self.sample_random_board(card_treys, cards_id, y['CLASS_str'])
        
        if not self.graph:
            while len(cards_id) < 7:
                cards_id.append(-1)
            return (
                torch.tensor(cards_id, dtype=torch.long),
                torch.tensor(y['CLASS'], dtype=torch.long)
            )
        cards_id = torch.tensor(cards_id)
        
        if self.use_card_ids:
            x = cards_id.unsqueeze(1)
        else:
            rank = cards_id // 4
            suit = cards_id % 4
            if self.normalize_x:
                rank = rank / 12.0
                suit = suit / 3.0
            x = torch.stack([rank, suit], dim=1)
            
        edges = []
        for i in range(len(cards_id)):
            for j in range(i + 1, len(cards_id)):
                id_i, id_j = cards_id[i].item(), cards_id[j].item()
                rank_i, rank_j = id_i // 4, id_j // 4
                suit_i, suit_j = id_i % 4, id_j % 4

                if suit_i == suit_j or abs(rank_i - rank_j) <= 1 or abs(rank_i - rank_j) == 12:
                    edges += [(i, j), (j, i)]

        if not edges:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            
        data = Data(x=x, edge_index=edge_index, y=y['CLASS'])
        return data
    
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
        
        random.shuffle(full_board)
        return full_board
    
    def __len__(self):
        return len(self.X)
    