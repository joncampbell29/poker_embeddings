from itertools import product
from typing import Literal

RANKS = '23456789TJQKA'
SUITS = 'cdhs'
DECK = [rank+suit for rank, suit in product(RANKS, SUITS)]
RANKS_DICT = {i:rank for i,rank in enumerate(RANKS)}
SUITS_DICT = {i:suit for i,suit in enumerate(SUITS)}
DECK_DICT = {i:card for i,card in enumerate(DECK)}

RankType = Literal['2','3','4','5','6','7','8','9','T','J','Q','K','A']
SuitType = Literal['c','d','h','s']
CardType = Literal[
    '2c', '2d', '2h', '2s', 
    '3c', '3d', '3h', '3s', 
    '4c', '4d', '4h', '4s', 
    '5c', '5d', '5h', '5s', 
    '6c', '6d', '6h', '6s', 
    '7c', '7d', '7h', '7s', 
    '8c', '8d', '8h', '8s', 
    '9c', '9d', '9h', '9s', 
    'Tc', 'Td', 'Th', 'Ts', 
    'Jc', 'Jd', 'Jh', 'Js', 
    'Qc', 'Qd', 'Qh', 'Qs', 
    'Kc', 'Kd', 'Kh', 'Ks', 
    'Ac', 'Ad', 'Ah', 'As'
]



# SUITEDNESS_DICT = {
#     0: 'rainbow', # No card of same suit
#     1: 'two_tone', # 2 of 3 of same suit
#     2: 'monotone' # All same suit
# }

# PAIRNESS_DICT = {
#     3: 'high_pair', # AAx-JJx
#     2: 'middle_pair', # TTx-77x
#     1: 'low_pair', # 66x-22x
#     0: 'unpaired' # No pair
# }

# CONNECTEDNESS_DICT = {
#     2: 'three_connected',
#     1: 'two_connected',
#     0: 'disconnected'
# }

# HIGH_LOW_TEXTURE_DICT = {
#     0: 'low', # 2 or 3 cards <= 6
#     1: 'mixed', # all cards spread out
#     2: 'middle', # 2 or 3 cards >= 7 & <= T
#     3: 'high', # 2 or 3 broadway cards (J, Q, K, A)
# }

# HIGH_CARD_DICT = {
#     0: '2 high', # 2 high
#     1: '3 high', # 3 high
#     2: '4 high', # 4 high
#     3: '5 high', # 5 high
#     4: '6 high', # 6 high
#     5: '7 high', # 7 high
#     6: '8 high', # 8 high
#     7: '9 high', # 9 high
#     8: 'T high', # 10 high
#     9: 'J high', # J high
#     10: 'Q high', # Q high
#     11: 'K high', # K high
#     12: 'A high' # A high
#     }

# STRAIGHTNESS_DICT = {
#     0: 'no_possible', # no possible straights
#     1: 'one_possible', # 1 possible straight
#     2: 'two_possible', # 2 possible straights
#     3: 'three_possible' # 3 possible straights
# }

