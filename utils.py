from constants import CardType
from dataclasses import dataclass, field
from itertools import combinations
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
        self.rank_idx = '23456789TJQKA'.index(self.rank)
        self.suit_idx = 'cdhs'.index(self.suit)
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
