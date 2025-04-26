from .constants import DECK_DICT, HANDS
from itertools import combinations, product
import numpy as np

def normalize_hand(hand: tuple):
    card1, card2 = hand
    rank1 = card1[0]
    rank2 = card2[0]
    suit1 = card1[1]
    suit2 = card2[1]
    ranks = [rank1, rank2]
    ranks.sort(key=lambda x: '23456789TJQKA'.index(x), reverse=True)

    suffix = 's' if suit1 == suit2 else 'o'
    return f"{ranks[0]}{ranks[1]}{suffix}"

def get_possible_hands(hand: str):
    '''
    Takes a hand in the rank + suited/offsuit format (i.e JJo, 76s)
    and returns the hands in the deck statisfying it
    i.e. AAo -> [('Ac', 'Ad'),('Ac', 'Ah'), ('Ac', 'As'), ('Ad', 'Ah'), ('Ad', 'As'), ('Ah', 'As')]
    '''
    if hand[0]==hand[1]:
        return list(combinations([card for card in DECK_DICT.values() if hand[0] in card], 2))
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

def card_distance(hand):
    rank_values = {
        '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
        'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14
    }
    rank1, rank2 = hand[0], hand[1]
    value1 = rank_values[rank1]
    value2 = rank_values[rank2]
    standard_sep = abs(value1 - value2)
    if rank1 == 'A' or rank2 == 'A':
        if rank1 == 'A':
            alt_sep = abs(1 - value2)
        else:
            alt_sep = abs(value1 - 1)

        return min(standard_sep, alt_sep)
    else:
        return standard_sep

def find_blocked_hands(hand: tuple):
    '''
    Returns list of hands if blocks
    '''
    card1, card2 = hand
    blocked_hands = []

    for possible_hand in HANDS:
        if (possible_hand == hand) or (possible_hand == (hand[1], hand[0])):
            continue

        if card1 in possible_hand or card2 in possible_hand:
            blocked_hands.append(possible_hand)
    return blocked_hands


def find_blocked_hands_simple(hand: str):
    '''
    Returns dictionary of hands if blocks and the number of combos it blocks
    '''
    if hand[-1] == "s":
        card1 = hand[0]+"c"
        card2 = hand[1]+"c"
    else:
        card1 = hand[0]+"c"
        card2 = hand[1]+"d"
    hand = (card1, card2)
    blocked_hands = []

    for possible_hand in HANDS:
        if (possible_hand == hand) or (possible_hand == (hand[1], hand[0])):
            continue

        if card1 in possible_hand or card2 in possible_hand:
            blocked_hands.append(possible_hand)
    hands, counts = np.unique(np.array([normalize_hand(i) for i in blocked_hands]), return_counts=True)

    blocked_combos_dict = {i:j for i,j in zip(hands, counts)}
    return blocked_combos_dict



def find_dominated_hands(hand: str):
    if hand[-1] == "s":
        card1 = hand[0]+"c"
        card2 = hand[1]+"c"
    else:
        card1 = hand[0]+"c"
        card2 = hand[1]+"d"

    rank1, rank2 = card1[0], card2[0]

    card_ranks = '23456789TJQKA'

    dominated_hands = []
    for possible_hand in HANDS:
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


