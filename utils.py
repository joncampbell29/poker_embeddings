from itertools import product, combinations
import numpy as np

with open('one_possible_straights.txt','r') as f:
    ONE_POSSIBLE_STRAIGHT = []
    for i in f.readlines():
        h = i.strip()
        ONE_POSSIBLE_STRAIGHT.append(tuple(sorted((h[0], h[2], h[4]))))
        
ONE_POSSIBLE_STRAIGHT = set(ONE_POSSIBLE_STRAIGHT)
        
with open('two_possible_straights.txt','r') as f:
    TWO_POSSIBLE_STRAIGHTS = []
    for i in f.readlines():
        h = i.strip()
        TWO_POSSIBLE_STRAIGHTS.append(tuple(sorted((h[0], h[2], h[4]))))

TWO_POSSIBLE_STRAIGHTS = set(TWO_POSSIBLE_STRAIGHTS)
 
        
with open('three_possible_straights.txt','r') as f:
    THREE_POSSIBLE_STRAIGHTS = []
    for i in f.readlines():
        h = i.strip()
        THREE_POSSIBLE_STRAIGHTS.append(tuple(sorted((h[0], h[2], h[4]))))
        
THREE_POSSIBLE_STRAIGHTS = set(THREE_POSSIBLE_STRAIGHTS)

CONNECTED_FLOPS = ONE_POSSIBLE_STRAIGHT\
    .union(TWO_POSSIBLE_STRAIGHTS)\
        .union(THREE_POSSIBLE_STRAIGHTS)
        

        
RANKS = [str(i) for i in range(2,10)] + ['T','J','Q','K','A']
SUITS = ['s','h','c','d']

RANKS_DICT = {str(i): i for i in range(2, 10)}
RANKS_DICT.update({'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14})
SUITS_DICT = {suit: i for i, suit in enumerate(SUITS)}

DECK = [rank+suit for rank, suit in product(RANKS, SUITS)]
FLOPS = list(combinations(DECK, 3))

def convert_to_numeric(flop, remove_dups=True):
    res = [RANKS_DICT[card[:-1]] for card in flop]
    if remove_dups:
        return sorted(set(res))
    else:
        return sorted(res)

SUITEDNESS_DICT = {
    'rainbow': 0, # No card of same suit
    'two_tone': 1, # 2 of 3 of same suit
    'monotone': 2 # All same suit
}

def eval_suitedness(flop):
    suits = [card[-1] for card in flop]
    unique_suits, counts = np.unique(suits, return_counts=True)

    if len(unique_suits) == 3:
        return 0 # 'rainbow'
    elif len(unique_suits) == 2:
        return 1 # 'two_tone'
    else:
        return 2 # 'monotone'


PAIRNESS_DICT = {
    'high_pair': 3, # AAx-JJx
    'middle_pair': 2, # TTx-77x
    'low_pair': 1, # 66x-22x
    'unpaired': 0 # No pair
}

def eval_pairness(flop):
    flop_numeric = convert_to_numeric(flop, remove_dups=True)
    if len(flop_numeric) == 3:
        return 0 # 'unpaired'
    flop_numeric = convert_to_numeric(flop, remove_dups=False)

    vals, counts = np.unique(flop_numeric, return_counts=True)
    if counts[0] == 3:
        pair_rank = vals[counts == 3] 
    else:
        pair_rank = vals[counts == 2][0]

    
    if pair_rank >= 11:
        return 3 # 'high_pair'
    elif 7 <= pair_rank <= 10:
        return 2 # 'middle_pair'
    else:  
        return 1  # 'low_pair'
    

CONNECTEDNESS = {
    'three_connected': 2,
    'two_connected': 1,
    'disconnected': 0
}

def eval_connectedness(flop):
    flop_numeric = convert_to_numeric(flop, remove_dups=False)
    if 14 in flop_numeric and 2 in flop_numeric:
        flop_numeric[flop_numeric.index(14)] = 1  # Convert A to 1 if A-2 scenario exists
        flop_numeric.sort()

    if flop_numeric[2] - flop_numeric[0] == 2 and flop_numeric[1] - flop_numeric[0] == 1:
        return 2  # 'three_connected'

    if flop_numeric[1] - flop_numeric[0] == 1 or flop_numeric[2] - flop_numeric[1] == 1:
        return 1  # 'two_connected'

    return 0  # 'disconnected'


HIGH_LOW_TEXTURE_DICT = {
    'low': 0, # 2 or 3 cards <= 8
    'middle': 1, # neither high or low
    'high': 2 # 2 or 3 broadway cards (T, J, Q, K, A)
}

def eval_high_low_texture(flop):
    broadway_ranks = {'T', 'J', 'Q', 'K', 'A'}

    ranks = [card[:-1] for card in flop]

    broadway_count = sum(1 for rank in ranks if rank in broadway_ranks)
    low_count = sum(1 for rank in ranks if rank.isdigit() and int(rank) <= 8)

    if broadway_count >= 2:
        return 2 # 'high'
    elif low_count >= 2:
        return 0 # 'low'
    else:
        return 1 # 'middle'

HIGH_CARD_DICT = {
    '2high': 0, # 2 high
    '3high': 1, # 3 high
    '4high': 2, # 4 high
    '5high': 3, # 5 high
    '6high': 4, # 6 high
    '7high': 5, # 7 high
    '8high': 6, # 8 high
    '9high': 7, # 9 high
    'Thigh': 8, # 10 high
    'Jhigh': 9, # J high
    'Qhigh': 10, # Q high
    'Khigh': 11, # K high
    'Ahigh': 12 # A high
    }

def eval_high_card(flop):
    flop_numeric = convert_to_numeric(flop)
    return flop_numeric[-1] - 2

STRAIGHTNESS_DICT = {
    'no_possible': 0, # no possible straights
    'one_possible': 1, # 1 possible straight
    'two_possible': 2, # 2 possible straights
    'three_possible': 3 # 3 possible straights
}

def eval_straightness(flop):
    ranks = tuple(sorted(i[0] for i in flop))
    if ranks in ONE_POSSIBLE_STRAIGHT:
        return 1
    elif ranks in TWO_POSSIBLE_STRAIGHTS:
        return 2
    elif ranks in THREE_POSSIBLE_STRAIGHTS:
        return 3
    else:
        return 0
        
