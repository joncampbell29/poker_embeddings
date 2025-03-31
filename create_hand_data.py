from poker_utils.simulation import simulate_hand_randrange
from poker_utils.constants import HANDS_DICT
import pandas as pd
import argparse
from treys import Evaluator
from tqdm import tqdm

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simulate poker hands')
    parser.add_argument('--sims', type=int, default=1000, help='Number simulations (default: 1000)')
    parser.add_argument('--villains', type=int, default=1, help='Number Villians (default: 1)')
    args = parser.parse_args()
    
    num_sims = args.sims
    num_villains = args.villains
    res = []
    evaluator = Evaluator()
    for hand_idx, hand in tqdm(HANDS_DICT.items(), desc="Processing Hands"):
        hand_res = simulate_hand_randrange(
            hand, evaluator=evaluator, num_villans=num_villains, num_sims=num_sims)
        hand_res['hand_idx'] = hand_idx
        res.append(hand_res)
        
    df = pd.json_normalize(res)


    df['connectedness'] = df.apply(lambda x: card_distance(x.hand), axis=1)
    df['suited'] = df['hand'].str.endswith('s').astype(int)
    df['pair'] = (df['hand'].str[0] == df['hand'].str[1]).astype(int)

    df.to_csv("data/hand_data.csv", index=False)