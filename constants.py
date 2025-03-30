from itertools import product, combinations
from typing import Literal

RANKS = '23456789TJQKA'
SUITS = 'cdhs'
DECK = [rank+suit for rank, suit in product(RANKS, SUITS)]
RANKS_DICT = {i:rank for i,rank in enumerate(RANKS)}
SUITS_DICT = {i:suit for i,suit in enumerate(SUITS)}
DECK_DICT = {i:card for i,card in enumerate(DECK)}
HOLE_CARDS = list(combinations(DECK, 2))

def normalize_hand(hole_cards: tuple):
    card1, card2 = hole_cards
    rank1 = card1[0]
    rank2 = card2[0]
    suit1 = card1[1]
    suit2 = card2[1]
    ranks = [rank1, rank2]
    ranks.sort(key=lambda x: '23456789TJQKA'.index(x), reverse=True)

    suffix = 's' if suit1 == suit2 else 'o'
    return f"{ranks[0]}{ranks[1]}{suffix}"

hole_card_list = sorted(list({normalize_hand(h_c) for h_c in HOLE_CARDS}))
HOLE_CARD_DICT = {i:j for i,j in enumerate(hole_card_list)}

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
