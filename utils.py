from constants import CardType, DECK, HOLE_CARD_DICT, HOLE_CARDS, normalize_hand
from dataclasses import dataclass, field
from itertools import combinations, product
from typing import List
import numpy as np
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

# def card_distance(hole_cards: List[Card]):
#     rank_nums = [hole_cards[0].rank_idx, hole_cards[1].rank_idx]
#     distance = abs(rank_nums[0] - rank_nums[1])
#     alt_distance = None
#     if 12 in rank_nums:
#         other_rank = rank_nums[0] if rank_nums[1] == 12 else rank_nums[1]
#         alt_distance = abs(-1 - other_rank)
#     if alt_distance is not None:
#         return min(distance, alt_distance)
#     else:
#         return distance
    
def get_possible_hands(hand):
    ''' 
    Takes a hand in the rank + suited/offsuit format (i.e JJo, 76s)
    and returns the hands in the deck statisfying it
    i.e. AAo -> [('Ac', 'Ad'),('Ac', 'Ah'), ('Ac', 'As'), ('Ad', 'Ah'), ('Ad', 'As'), ('Ah', 'As')]
    '''
    if hand[0]==hand[1]:
        return list(combinations([card for card in DECK if hand[0] in card], 2))
    else:
        if hand[-1] == 's':
            filter_func = lambda x:x[0][-1] == x[1][-1]
        else:
            filter_func = lambda x:x[0][-1] != x[1][-1]
        combos = product(
            [hand[0]+suit for suit in 'cdhs'], 
            [hand[1]+suit for suit in 'cdhs']
            )
        return list(filter(filter_func, combos))

def find_blocked_hands(hand):
    ''' 
    Returns dictionary of hands if blocks and the number of combos it blocks
    '''
    if hand[-1] == "s":
        card1 = hand[0]+"c"
        card2 = hand[1]+"c"
    else:
        card1 = hand[0]+"c"
        card2 = hand[1]+"d"
        
    blocked_hands = []
    
    for possible_hand in HOLE_CARDS:
        if possible_hand == hand:
            continue
            
        if card1 in possible_hand or card2 in possible_hand:
            blocked_hands.append(possible_hand)
    hands, counts = np.unique(np.array([normalize_hand(i) for i in blocked_hands]), return_counts=True)
    
    blocked_combos_dict = {i:j for i,j in zip(hands, counts)}
    return blocked_combos_dict

def find_dominated_hands(hand):
    if hand[-1] == "s":
        card1 = hand[0]+"c"
        card2 = hand[1]+"c"
    else:
        card1 = hand[0]+"c"
        card2 = hand[1]+"d"
        
    rank1, rank2 = card1[0], card2[0]
    
    card_ranks = '23456789TJQKA'
    
    dominated_hands = []
    for possible_hand in HOLE_CARDS:
        # Skip the hand itself
        if possible_hand == hand:
            continue
            
        # Extract ranks from the possible hand
        possible_rank1, possible_rank2 = possible_hand[0][0], possible_hand[1][0]
        if possible_rank1 == possible_rank2:
            continue
        
        if rank1 != possible_rank1 and rank1 != possible_rank2 and rank2 != possible_rank1 and rank2 != possible_rank2:
            continue
        # Check for domination scenarios
        
        # Case 1: Hand shares rank1 with possible hand
        if rank1 == possible_rank1:
            # Check if rank2 is higher than possible_rank2
            if card_ranks.index(rank2) > card_ranks.index(possible_rank2):
                dominated_hands.append(possible_hand)
        elif rank1 == possible_rank2:
            # Check if rank2 is higher than possible_rank1
            if card_ranks.index(rank2) > card_ranks.index(possible_rank1):
                dominated_hands.append(possible_hand)
                
        # Case 2: Hand shares rank2 with possible hand
        elif rank2 == possible_rank1:
            # Check if rank1 is higher than possible_rank2
            if card_ranks.index(rank1) > card_ranks.index(possible_rank2):
                dominated_hands.append(possible_hand)
        elif rank2 == possible_rank2:
            # Check if rank1 is higher than possible_rank1
            if card_ranks.index(rank1) > card_ranks.index(possible_rank1):
                dominated_hands.append(possible_hand)
    
    hands, counts = np.unique(np.array([normalize_hand(i) for i in dominated_hands]), return_counts=True)
    
    dominated_combos_dict = {i:j for i,j in zip(hands, counts)}
    return dominated_combos_dict
    