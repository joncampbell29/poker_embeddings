from treys import Card, Evaluator
from ucimlrepo import fetch_ucirepo
from poker_embeddings.poker_utils.constants import HANDS_DICT, DECK_DICT
import os
import pandas as pd
import numpy as np


suit_id_mapping = {'c':0,'d':1,'h':2,'s':3}
uc_irvine_suit_mapping = {1:'h', 2:'s', 3:'d', 4:'c'}
uc_irvine_rank_mapping = {
    1:'A', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6',
    7:'7', 8:'8', 9:'9', 10:'T', 11:'J', 12:'Q', 13:'K'
    }
uc_irvine_class_mapping = {
    0:'High Card', 1:'Pair', 2:'Two Pair', 3:'Three of a Kind', 4:'Straight',
    5:'Flush', 6:'Full House', 7:'Four of a Kind', 8:'Straight Flush', 9:'Royal Flush'
    }

def evaluate_hand(row):
    return evaluator.evaluate([], row.filter(regex='treys').tolist())

if __name__ == '__main__':

    data_dir = "./data"
    uc_dir = os.path.join(data_dir, "uc_irvine")

    for directory in [data_dir, uc_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    evaluator = Evaluator()
    poker_hand = fetch_ucirepo(id=158)

    X = poker_hand.data.features.copy()
    y = poker_hand.data.targets.copy()

    suit_cols = [f'S{i}' for i in range(1, 6)]
    rank_cols = [f'C{i}' for i in range(1, 6)]
    X[suit_cols] = X[suit_cols].apply(lambda col: col.map(uc_irvine_suit_mapping))
    X[rank_cols] = X[rank_cols].apply(lambda col: col.map(uc_irvine_rank_mapping))
    hand_to_id = {hand: idx for idx, hand in HANDS_DICT.items()}
    card_to_id = {card:idx for idx,card in DECK_DICT.items()}
    for i in range(1, 6):
        X[f'card{i}'] = X[f'C{i}'] + X[f'S{i}']
        X[f'card{i}_id'] = X[f'card{i}'].map(card_to_id)
        X[f'card{i}_treys'] = X[f'card{i}'].map(lambda x: Card.new(x))

    y['CLASS_str'] = y['CLASS'].map(uc_irvine_class_mapping)
    y['score_treys'] = X.apply(evaluate_hand, axis=1)
    X.drop(['C1','S1','C2','S2','C3','S3','C4','S4','C5','S5'], axis=1, inplace=True)

    X.to_csv(os.path.join(uc_dir,'X.csv'), index=False)
    y.to_csv(os.path.join(uc_dir,'y.csv'), index=False)
