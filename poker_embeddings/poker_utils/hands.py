from .constants import DECK_DICT, HANDS, RANKS_DICT
from itertools import combinations, product
import numpy as np
import torch

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


_card_to_idx = {card: idx for idx, card in DECK_DICT.items()}
def card_distance(hands):
    if hands[0] == hands[1]: return 0
    def calc_dist(rank1,rank2):
        standard_sep = abs(rank1 - rank2)
        if 12 not in (rank1, rank2):
            return standard_sep
        if rank1 == 12:
            alt_sep = abs(rank2 + 1)
            return min(standard_sep, alt_sep)
        elif rank2 == 12:
            alt_sep = abs(rank1 + 1)
            return min(standard_sep, alt_sep)

    if isinstance(hands[0], str):
        rank1, rank2 = _card_to_idx[hands[0]] // 4, _card_to_idx[hands[1]] // 4
        return calc_dist(rank1,rank2)
    else:
        rank1, rank2 = hands[0] // 4, hands[1] // 4
        return calc_dist(rank1,rank2)


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


def fully_connected_edge_index(num_nodes):
    row = torch.arange(num_nodes).repeat_interleave(num_nodes)
    col = torch.arange(num_nodes).repeat(num_nodes)
    mask = row != col
    edge_index = torch.stack([row[mask], col[mask]], dim=0)
    return edge_index

def create_deck_graph(normalize=True):
    edge_index = fully_connected_edge_index(52)
    num_edges = edge_index.shape[1]
    edge_attr = torch.zeros((num_edges, 2), dtype=torch.float)

    for idx in range(num_edges):
        i, j = edge_index[:, idx]

        suit_i = i % 4
        suit_j = j % 4

        distance = card_distance((i, j))
        if normalize:
            dist_weight = (11 - distance) / 11  # scale: closer ranks → higher weight
        else:
            dist_weight = distance
        suited = int(suit_i == suit_j)

        edge_attr[idx, 0] = dist_weight
        edge_attr[idx, 1] = suited

    return edge_index, edge_attr

def query_subgraph(card_ids, full_edge_index, full_edge_attr):
    sources = full_edge_index[0]
    destinations = full_edge_index[1]
    edge_mask = torch.isin(sources, card_ids) & torch.isin(destinations, card_ids)
    sub_edge_index = full_edge_index[:, edge_mask]
    sub_edge_attr = full_edge_attr[edge_mask]
    card_id_to_new_idx = {card.item(): idx for idx, card in enumerate(card_ids)}
    new_src = torch.tensor([card_id_to_new_idx[s.item()] for s in sub_edge_index[0]])
    new_dst = torch.tensor([card_id_to_new_idx[d.item()] for d in sub_edge_index[1]])

    sub_edge_index = torch.stack([new_src, new_dst], dim=0)

    return sub_edge_index, sub_edge_attr