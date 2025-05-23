import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from .constants import HANDS_DICT, DECK_DICT
from .hands import card_distance, normalize_hand, create_deck_graph, query_subgraph
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
    def __init__(self, X, y, add_random_cards=True, use_card_ids=True, graph=True, normalize_x=False):

        self.X = X
        self.y = y
        self.add_random_cards = add_random_cards
        self.use_card_ids = use_card_ids
        self.graph = graph
        self.normalize_x = normalize_x
        self.evaluator = Evaluator()

        self.deck_treys = np.array([Card.new(c) for c in DECK_DICT.values()])
        self.card_int_to_id = {Card.new(card): idx for idx, card in DECK_DICT.items()}

        self.card_ids = torch.tensor(self.X.filter(regex='id').to_numpy(), dtype=torch.long)
        self.card_treys = self.X.filter(regex='treys').to_numpy()
        self.y_CLASS = torch.tensor(self.y['CLASS'].to_numpy(), dtype=torch.long)

        self.deck_edge_index, self.deck_edge_attr = create_deck_graph()

    def __getitem__(self, idx):
        y = self.y.iloc[idx]
        cards_id = self.card_ids[idx]
        if self.add_random_cards:
            card_treys = self.card_treys[idx]
            cards_id, treys_score = self.sample_random_board(card_treys, cards_id, y['CLASS_str'])

        if not self.graph:
            cards_id = F.pad(cards_id, (0, 7 - len(cards_id)), value=-1)
            return (
                cards_id,
                self.y_CLASS[idx]
            )

        if self.use_card_ids:
            x = cards_id
        else:
            rank = cards_id // 4
            suit = cards_id % 4
            if self.normalize_x:
                rank = rank / 12.0
                suit = suit / 3.0
            x = torch.stack([rank, suit], dim=1)

        data = self.create_graph(cards_id, x, torch.tensor([[self.y_CLASS[idx], treys_score]], dtype=torch.long))
        return data # x = [rank, suit], edges = []

    def sample_random_board(self, used_treys, base_board, label_str):
        used_list = list(used_treys)
        remaining = np.setdiff1d(self.deck_treys, list(used_list), assume_unique=True).tolist()
        board_extension_size = random.randint(0, 2)

        evaluator = self.evaluator
        orig_t_rank = evaluator.evaluate([], used_list)

        if board_extension_size > 0:
            for attempt in range(10):
                sampled = random.sample(remaining, board_extension_size)
                full_hand = used_list + sampled
                t_rank = evaluator.evaluate([], full_hand)
                hand_rank = evaluator.get_rank_class(t_rank)
                if evaluator.class_to_string(hand_rank) == label_str:
                    sampled_ids = torch.tensor([self.card_int_to_id[card] for card in sampled])
                    full_board = torch.cat((base_board, sampled_ids), dim=0)
                    return full_board, t_rank
            else:
                return base_board, orig_t_rank
        else:
            return base_board, orig_t_rank

    def create_graph(self, cards_ids, x, y):
        subgraph_edge_index, subgraph_edge_attr = query_subgraph(cards_ids, self.deck_edge_index, self.deck_edge_attr)
        data = Data(x=x, edge_index=subgraph_edge_index, edge_attr=subgraph_edge_attr, y=y)
        return data

    def __len__(self):
        return len(self.X)


class PairwiseHandDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.length = len(base_dataset)

    def __len__(self):
        return self.length

    def __getitem__(self, idx1):
        x1 = self.base_dataset[idx1]
        ix2 = random.randint(0, self.length - 1)
        while ix2 == idx1:
            ix2 = random.randint(0, self.length - 1)
        x2 = self.base_dataset[ix2]
        score1 = x1.y[0,1].item()
        score2 = x2.y[0,1].item()
        label = 1 if score1 < score2 else -1 # Smaller is stronger in Treys
        return x1, x2, label