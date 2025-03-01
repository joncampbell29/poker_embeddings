import pandas as pd
import numpy as np
import argparse
from utils import RANKS, SUITS
from utils import (
    eval_suitedness, SUITEDNESS_DICT,
    eval_pairness, PAIRNESS_DICT,
    eval_connectedness, CONNECTEDNESS_DICT,
    eval_high_low_texture, HIGH_LOW_TEXTURE_DICT,
    eval_high_card, HIGH_CARD_DICT,
    eval_straightness, STRAIGHTNESS_DICT
)


parser = argparse.ArgumentParser()
parser.add_argument("data_in", help="data to evaluate")
parser.add_argument("data_out", help="data name to save results as")
args = parser.parse_args()

data = pd.read_parquet(args.data_in)

prob_data = []

def contains_at_least_one(flop, rank_suit):
    return any(rank_suit in card for card in flop)

for i in RANKS + SUITS:
    p = data['flop'].apply(lambda x: contains_at_least_one(x, i)).value_counts(normalize=True)
    p_true = p[True] if True in p.index else 0
    prob_data.append({
        'condition': 'contains_at_least_one',
        'val': i,
        'probability': p_true
    })
    
    
def contains_at_least_two(flop, rank_suit):
    return sum(rank_suit in card for card in flop) >= 2


for i in RANKS + SUITS:
    p = data['flop'].apply(lambda x: contains_at_least_two(x, i)).value_counts(normalize=True)
    p_true = p[True] if True in p.index else 0
    
    prob_data.append({
        'condition': 'contains_at_least_two',
        'val': i,
        'probability': p_true
    })
    
def contains_three(flop, rank_suit):
    return sum(rank_suit in card for card in flop) == 3

for i in RANKS + SUITS:
    p = data['flop'].apply(lambda x: contains_three(x, i)).value_counts(normalize=True)
    p_true = p[True] if True in p.index else 0
    prob_data.append({
        'condition': 'contains_three',
        'val': i,
        'probability': p_true
    })
    
def contains_exactly_one(flop, rank_suit):
    return sum(rank_suit in card for card in flop) == 1

for i in RANKS + SUITS:
    p = data['flop'].apply(lambda x: contains_exactly_one(x, i)).value_counts(normalize=True)
    p_true = p[True] if True in p.index else 0
    prob_data.append({
        'condition': 'contains_exactly_one',
        'val': i,
        'probability': p_true
    })
    
def contains_exactly_two(flop, rank_suit):
    return sum(rank_suit in card for card in flop) == 2

for i in RANKS + SUITS:
    p = data['flop'].apply(lambda x: contains_exactly_two(x, i)).value_counts(normalize=True)
    p_true = p[True] if True in p.index else 0
    prob_data.append({
        'condition': 'contains_exactly_two',
        'val': i,
        'probability': p_true
    })
    
prob_df = pd.DataFrame(prob_data)
    
attribute_conditions = [
    (eval_suitedness, SUITEDNESS_DICT),
    (eval_pairness, PAIRNESS_DICT),
    (eval_connectedness, CONNECTEDNESS_DICT),
    (eval_high_low_texture, HIGH_LOW_TEXTURE_DICT),
    (eval_high_card, HIGH_CARD_DICT),
    (eval_straightness, STRAIGHTNESS_DICT)
    ]


for func, dict in attribute_conditions:
    prob_series = data['flop'].apply(func).value_counts(normalize=True)
    prob_series.index = prob_series.index.map(dict)
    prob_series = prob_series.reset_index(name='probability').rename({'flop':'condition'},axis=1)
    prob_df = pd.concat([prob_df, prob_series], axis=0)
    
prob_df.to_parquet(args.data_out)