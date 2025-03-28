from constants import RANKS, SUITS, CardType
from dataclasses import dataclass, field
from typing import List
import torch

@dataclass
class Card:
    card: CardType
    rank: str = field(init=False)
    suit: str = field(init=False)
    rank_idx: int = field(init=False)
    suit_idx: int = field(init=False)
    idx: int = field(init=False)
    
    def __post_init__(self):
        self.rank = self.card[0]
        self.suit = self.card[1]
        self.rank_idx = RANKS.index(self.rank)
        self.suit_idx = SUITS.index(self.suit)
        self.idx = self.rank_idx * 4 + self.suit_idx
        
@dataclass
class Hand:
    hand: List[CardType] = field(default_factory=list)
    cards: List[Card] = field(init=False, default_factory=list)
    vector: torch.Tensor = field(init=False)
    rank_vector: torch.Tensor = field(init=False)
    suit_vector: torch.Tensor = field(init=False)
    
    def __post_init__(self):
        self.cards = [Card(c) for c in self.hand]
        self.vector = torch.tensor([c.idx for c in self.cards], dtype=torch.long)
        self.rank_vector = torch.tensor([c.rank_idx for c in self.cards], dtype=torch.long)
        self.suit_vector = torch.tensor([c.suit_idx for c in self.cards], dtype=torch.long)

def card_distance(hole_cards: List[Card]):
    rank_nums = [hole_cards[0].rank_idx, hole_cards[1].rank_idx]
    distance = abs(rank_nums[0] - rank_nums[1])
    alt_distance = None
    if 12 in rank_nums:
        other_rank = rank_nums[0] if rank_nums[1] == 12 else rank_nums[1]
        alt_distance = abs(-1 - other_rank)
    if alt_distance is not None:
        return min(distance, alt_distance)
    else:
        return distance

# def is_valid_flop(flop):
#     return len(flop) == 3 and len(set(flop)) == 3


# def eval_suitedness(flop):
#     if len(flop) != 3:
#         raise ValueError("A flop has only 3 cards")
#     suits = {get_suit_idx(card) for card in flop} 
#     unique_suits = len(suits)
#     if unique_suits == 3:
#         return 0 # 'rainbow'
#     elif unique_suits == 2:
#         return 1 # 'two_tone'
#     elif unique_suits == 1:
#         return 2 # 'monotone'
    
# def eval_pairness(flop):
#     ranks = [get_rank_idx(card) for card in flop]
#     rank_counts = {rank: ranks.count(rank) for rank in set(ranks)}

#     if max(rank_counts.values()) < 2:
#         return 0  # 'unpaired'
#     paired_rank = max(rank for rank, count in rank_counts.items() if count in {2,3})

#     if paired_rank >= 9:  # J, Q, K, A (9-12)
#         return 3  # 'high_pair'
#     elif paired_rank >= 5:  # 7, 8, 9, 10 (6-8)
#         return 2  # 'middle_pair'
#     else:  # 2, 3, 4, 5, 6 (0-5)
#         return 1  # 'low_pair'
    
# def eval_connectedness(flop):
#     ranks = sorted([get_rank_idx(card) for card in flop])
#     if ranks[2] - ranks[0] == 2 and ranks[1] - ranks[0] == 1:
#         return 2  # 'three_connected'
#     if ranks[1] - ranks[0] == 1 or ranks[2] - ranks[1] == 1:
#         return 1  # 'two_connected'
#     return 0  # 'disconnected'

# def eval_high_low_texture(flop):
#     ranks = [get_rank_idx(card) for card in flop]
#     low_count = sum(1 for r in ranks if r <= 4)  # 2-6 (rank index 0-4)
#     middle_count = sum(1 for r in ranks if 5 <= r <= 8)  # 7-T (rank index 5-8)
#     high_count = sum(1 for r in ranks if r >= 9)  # J-A (rank index 9-12)

#     if low_count == 1 and middle_count == 1 and high_count == 1:
#         return 1  # 'mixed'

#     if high_count >= 2:
#         return 3  # 'high'
#     elif middle_count >= 2:
#         return 2  # 'middle'
#     else:
#         return 0  # 'low'

# def eval_high_card(flop):
#     return max(get_rank_idx(card) for card in flop)



# def hand_to_vector(hand, ret_tensor=True):
#     if not isinstance(hand, (list, tuple)) or not all(isinstance(card, str) for card in hand):
#         raise ValueError("hand must be a list of card strings (e.g., ['As', 'Kd']).")
#     if not ret_tensor:
#         return [card_to_idx(card) for card in hand]
#     return torch.tensor([card_to_idx(card) for card in hand], dtype=torch.long)

# def vector_to_hand(vector, ret_tensor=False):
#     if isinstance(vector, torch.Tensor):
#         vector = vector.tolist()

#     if not isinstance(vector, (list, np.ndarray)):
#         raise ValueError("vector must be a list, numpy array, or torch tensor.")

#     if any(idx < 0 or idx > 51 for idx in vector):
#         raise ValueError("All indices must be between 0 and 51.")
#     if ret_tensor:
#         return torch.tensor([idx_to_card(idx) for idx in vector])
#     return [idx_to_card(idx) for idx in vector]

