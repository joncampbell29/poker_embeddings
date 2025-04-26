import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from .constants import HANDS_DICT, DECK_DICT
from .hands import card_distance, normalize_hand
import random
from treys import Card, Evaluator
from torch_geometric.data import Data

class CardDataset(Dataset):
    def __init__(self, normalize_dist_matrix=True):
        card_to_id = {j:i for i,j in DECK_DICT.items()}
        self.data = pd.DataFrame.from_dict(DECK_DICT, orient='index', columns=['card']).reset_index()
        self.data.rename({"index":'card_id'},axis=1, inplace=True)
        self.data['card_rank'] = self.data['card_id'] // 4
        self.data['card_suit'] = self.data['card_id'] % 4

        self.dist_matrix = np.zeros((52,52), dtype=int)
        for i in range(52):
            for j in range(52):
                card1 = self.data.iloc[i]['card']
                card2 = self.data.iloc[j]['card']
                self.dist_matrix[i,j] = card_distance(normalize_hand((card1,card2)))
        if normalize_dist_matrix:
            self.dist_matrix = self.dist_matrix / self.dist_matrix.max()
        self.dist_matrix = torch.tensor(self.dist_matrix, dtype=torch.float32)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        card_id = row['card_id']
        suit_id = row['card_suit']
        rank_id = row['card_rank']
        dist_vec = self.dist_matrix[idx]
        return {
            'card': torch.tensor(card_id, dtype=torch.long),
            'rank': torch.tensor(rank_id, dtype=torch.long),
            'suit': torch.tensor(suit_id, dtype=torch.long),
            'dist_vec': dist_vec
        }

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

        self.card_ids = self.X.filter(regex='id').to_numpy()
        self.card_treys = self.X.filter(regex='treys').to_numpy()
        self.y_CLASS = torch.tensor(self.y['CLASS'].to_numpy(), dtype=torch.long)

    def __getitem__(self, idx):
        y = self.y.iloc[idx]
        cards_id = self.card_ids[idx]
        if self.add_random_cards:
            card_treys = self.card_treys[idx]
            cards_id = self.sample_random_board(card_treys, cards_id, y['CLASS_str'])

        if not self.graph:
            cards_id = np.pad(cards_id, (0, 7 - len(cards_id)), constant_values=-1)
            return (
                torch.tensor(cards_id, dtype=torch.long),
                self.y_CLASS[idx]
            )
        cards_id = torch.tensor(cards_id, dtype=torch.long)

        if self.use_card_ids:
            x = cards_id.unsqueeze(1)
        else:
            rank = cards_id // 4
            suit = cards_id % 4
            if self.normalize_x:
                rank = rank / 12.0
                suit = suit / 3.0
            x = torch.stack([rank, suit], dim=1)
        data = self.create_graph(cards_id, x, self.y_CLASS[idx])

        return data

    def sample_random_board(self, used_treys, base_board, label_str):
        used_list = list(used_treys)
        remaining = np.setdiff1d(self.deck_treys, list(used_list), assume_unique=True).tolist()
        board_extension_size = random.randint(0, 2)

        evaluator = self.evaluator
        if board_extension_size > 0:
            for attempt in range(10):
                sampled = random.sample(remaining, board_extension_size)
                full_hand = used_list + sampled
                hand_rank = evaluator.get_rank_class(evaluator.evaluate([], full_hand))
                if evaluator.class_to_string(hand_rank) == label_str:
                    sampled_ids = np.array([self.card_int_to_id[card] for card in sampled])
                    full_board = np.concatenate((base_board, sampled_ids))
                    return full_board
            else:
                return base_board
        else:
            return base_board

    def create_graph(self, cards_id, x, y):
        ranks = cards_id // 4
        suits = cards_id % 4
        num_cards = cards_id.size(0)
        idx_i = cards_id.unsqueeze(1).expand(num_cards, num_cards)
        idx_j = cards_id.unsqueeze(0).expand(num_cards, num_cards)

        ranks_i = ranks.unsqueeze(1).expand(num_cards, num_cards)
        ranks_j = ranks.unsqueeze(0).expand(num_cards, num_cards)

        suits_i = suits.unsqueeze(1).expand(num_cards, num_cards)
        suits_j = suits.unsqueeze(0).expand(num_cards, num_cards)

        suit_match = suits_i == suits_j
        rank_close = (torch.abs(ranks_i - ranks_j) <= 1) | (torch.abs(ranks_i - ranks_j) == 12)

        not_self = idx_i != idx_j

        edge_mask = (suit_match | rank_close) & not_self

        edge_index = edge_mask.nonzero(as_tuple=False).t().contiguous()

        data = Data(x=x, edge_index=edge_index, y=y)
        return data

    def __len__(self):
        return len(self.X)
