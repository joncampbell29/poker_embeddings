from itertools import product
import numpy as np
import torch
from constants import *

def card_to_idx(card) -> int:
    if isinstance(card, int):
        if 0 <= card <= 51:
            return card
        raise ValueError(f"Invalid card index: {card}. Must be between 0 and 51.")
    
    if isinstance(card, str) and len(card) == 2:
        try:
            rank_idx = RANKS.index(card[0])
            suit_idx = SUITS.index(card[1])
            return rank_idx * 4 + suit_idx
        except ValueError:
            raise ValueError(f"Invalid card string: {card}. Must be in format 'As', 'Td', etc.")
    raise TypeError("Invalid input type. Must be an integer (index) or a two-character string (e.g., 'As').")


def idx_to_card(card_idx):
    return DECK[card_idx]

def get_suit_idx(card):
    if isinstance(card, (int, np.integer)):
        return card % 4
    elif isinstance(card, torch.Tensor):
        return (card % 4).item()
    elif isinstance(card, str):
        return SUITS.index(card[1])
    else:
        raise ValueError("Invalid card format. Must be an index (0-51) or a string (e.g., 'As').")

def get_rank_idx(card):
    if isinstance(card, (int, np.integer)):
        return card // 4
    elif isinstance(card, torch.Tensor):
        return (card // 4).item()
    elif isinstance(card, str):
        return RANKS.index(card[0])
    else:
        raise ValueError("Invalid card format. Must be an index (0-51) or a string (e.g., 'As').")

def is_valid_flop(flop):
    return len(flop) == 3 and len(set(flop)) == 3


def eval_suitedness(flop):
    if len(flop) != 3:
        raise ValueError("A flop has only 3 cards")
    suits = {get_suit_idx(card) for card in flop} 
    unique_suits = len(suits)
    if unique_suits == 3:
        return 0 # 'rainbow'
    elif unique_suits == 2:
        return 1 # 'two_tone'
    elif unique_suits == 1:
        return 2 # 'monotone'
    
def eval_pairness(flop):
    ranks = [get_rank_idx(card) for card in flop]
    rank_counts = {rank: ranks.count(rank) for rank in set(ranks)}

    if max(rank_counts.values()) < 2:
        return 0  # 'unpaired'
    paired_rank = max(rank for rank, count in rank_counts.items() if count in {2,3})

    if paired_rank >= 9:  # J, Q, K, A (9-12)
        return 3  # 'high_pair'
    elif paired_rank >= 5:  # 7, 8, 9, 10 (6-8)
        return 2  # 'middle_pair'
    else:  # 2, 3, 4, 5, 6 (0-5)
        return 1  # 'low_pair'
    
def eval_connectedness(flop):
    ranks = sorted([get_rank_idx(card) for card in flop])
    if ranks[2] - ranks[0] == 2 and ranks[1] - ranks[0] == 1:
        return 2  # 'three_connected'
    if ranks[1] - ranks[0] == 1 or ranks[2] - ranks[1] == 1:
        return 1  # 'two_connected'
    return 0  # 'disconnected'

def eval_high_low_texture(flop):
    ranks = [get_rank_idx(card) for card in flop]
    low_count = sum(1 for r in ranks if r <= 4)  # 2-6 (rank index 0-4)
    middle_count = sum(1 for r in ranks if 5 <= r <= 8)  # 7-T (rank index 5-8)
    high_count = sum(1 for r in ranks if r >= 9)  # J-A (rank index 9-12)

    if low_count == 1 and middle_count == 1 and high_count == 1:
        return 1  # 'mixed'

    if high_count >= 2:
        return 3  # 'high'
    elif middle_count >= 2:
        return 2  # 'middle'
    else:
        return 0  # 'low'

def eval_high_card(flop):
    return max(get_rank_idx(card) for card in flop)



def hand_to_vector(hand, ret_tensor=True):
    if not isinstance(hand, (list, tuple)) or not all(isinstance(card, str) for card in hand):
        raise ValueError("hand must be a list of card strings (e.g., ['As', 'Kd']).")
    if not ret_tensor:
        return [card_to_idx(card) for card in hand]
    return torch.tensor([card_to_idx(card) for card in hand], dtype=torch.long)

def vector_to_hand(vector, ret_tensor=False):
    if isinstance(vector, torch.Tensor):
        vector = vector.tolist()

    if not isinstance(vector, (list, np.ndarray)):
        raise ValueError("vector must be a list, numpy array, or torch tensor.")

    if any(idx < 0 or idx > 51 for idx in vector):
        raise ValueError("All indices must be between 0 and 51.")
    if ret_tensor:
        return torch.tensor([idx_to_card(idx) for idx in vector])
    return [idx_to_card(idx) for idx in vector]

