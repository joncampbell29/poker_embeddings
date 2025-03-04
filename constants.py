from itertools import product

RANKS = '23456789TJQKA'
SUITS = 'cdhs'
DECK = [rank+suit for rank, suit in product(RANKS, SUITS)]
RANKS_DICT = {i:rank for i,rank in enumerate(RANKS)}
SUITS_DICT = {i:suit for i,suit in enumerate(SUITS)}
DECK_DICT = {i:card for i,card in enumerate(DECK)}



SUITEDNESS_DICT = {
    0: 'rainbow', # No card of same suit
    1: 'two_tone', # 2 of 3 of same suit
    2: 'monotone' # All same suit
}

PAIRNESS_DICT = {
    3: 'high_pair', # AAx-JJx
    2: 'middle_pair', # TTx-77x
    1: 'low_pair', # 66x-22x
    0: 'unpaired' # No pair
}

CONNECTEDNESS_DICT = {
    2: 'three_connected',
    1: 'two_connected',
    0: 'disconnected'
}

HIGH_LOW_TEXTURE_DICT = {
    0: 'low', # 2 or 3 cards <= 6
    1: 'mixed', # all cards spread out
    2: 'middle', # 2 or 3 cards >= 7 & <= T
    3: 'high', # 2 or 3 broadway cards (J, Q, K, A)
}

HIGH_CARD_DICT = {
    0: '2 high', # 2 high
    1: '3 high', # 3 high
    2: '4 high', # 4 high
    3: '5 high', # 5 high
    4: '6 high', # 6 high
    5: '7 high', # 7 high
    6: '8 high', # 8 high
    7: '9 high', # 9 high
    8: 'T high', # 10 high
    9: 'J high', # J high
    10: 'Q high', # Q high
    11: 'K high', # K high
    12: 'A high' # A high
    }

# STRAIGHTNESS_DICT = {
#     0: 'no_possible', # no possible straights
#     1: 'one_possible', # 1 possible straight
#     2: 'two_possible', # 2 possible straights
#     3: 'three_possible' # 3 possible straights
# }

