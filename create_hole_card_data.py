from simulation import simulate_hand
from constants import HOLE_CARDS, HOLE_CARD_DICT
import pandas as pd

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
    
def normalize_hand_row(row):
    card1 = row['hole_card1']
    card2 = row['hole_card2']
    
    rank1 = card1[0]
    rank2 = card2[0]
    suit1 = card1[1]
    suit2 = card2[1]
    
    ranks = [rank1, rank2]
    ranks.sort(key=lambda x: '23456789TJQKA'.index(x), reverse=True)

    suffix = 's' if suit1 == suit2 else 'o'
    return f"{ranks[0]}{ranks[1]}{suffix}"

if __name__ == "__main__":
    num_sims = 1_000
    num_villans = 1
    res = []
    for hole_cards in HOLE_CARDS:
        hole_cards = sorted(hole_cards)
        river_win_rate = simulate_hand(hole_cards, num_villans=num_villans, num_sims=num_sims)
        res.append({
            'hole_card1': hole_cards[0],
            'hole_card2': hole_cards[1],
            'river_win_rate': river_win_rate
        })
    data = pd.DataFrame(res)


    data['hand'] = data.apply(normalize_hand_row, axis=1)

    df = data.groupby('hand')[['river_win_rate']].mean().reset_index()

    df['connectedness'] = df.apply(lambda x: card_distance(x.hand), axis=1)
    df['suited'] = df['hand'].str.endswith('s').astype(int)
    df['pair'] = (df['hand'].str[0] == df['hand'].str[1]).astype(int)
    
    rev_hole_card_dict = {j:i for i,j in HOLE_CARD_DICT.items()}
    df['hand_idx'] = df['hand'].apply(lambda x: rev_hole_card_dict[x])

    df.to_csv("data/hole_card_data.csv", index=False)