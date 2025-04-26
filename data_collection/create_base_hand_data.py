import pandas as pd
from poker_embeddings.poker_utils.constants import HANDS_DICT
from poker_embeddings.poker_utils.hands import card_distance
import os

data = []
for i, hand in HANDS_DICT.items():
    suited = 1 if hand[-1] == 's' else 0
    connectedness = card_distance(hand)
    pair = 1 if hand[0] == hand[1] else 0
    high_card = '23456789TJQKA'.index(hand[0])
    low_card = '23456789TJQKA'.index(hand[1])
    rank_diff = abs(high_card - low_card)
    if pair == 1:
        hand_type = 'pair'
    elif suited == 1:
        hand_type = 'suited'
    elif suited == 0:
        hand_type = 'offsuit'

    ace = 1 if 'A' in hand else 0
    broadway = (hand[0] in {'A', 'K', 'Q', 'J', 'T'}) + (hand[1] in {'A', 'K', 'Q', 'J', 'T'})
    hand_info = {
        'hand': hand,
        'hand_idx': i,
        'suited': suited,
        'connectedness': connectedness,
        'pair': pair,
        'high_card': high_card,
        'low_card': low_card,
        'rank_diff': rank_diff,
        'hand_type': hand_type,
        'ace': ace,
        'broadway': broadway
    }
    data.append(hand_info)

data = pd.DataFrame(data)

data['low_pair'] = ((data['pair'] == 1) & (data['high_card'] <= 4)).astype(int)
data['medium_pair'] = ((data['pair'] == 1) & (data['high_card'] > 4) & (data['high_card'] <= 8)).astype(int)
data['high_pair'] = ((data['pair'] == 1) & (data['high_card'] > 8)).astype(int)

data['broadway_score'] = (data['broadway'] - data['broadway'].min()) / (data['broadway'].max() - data['broadway'].min())
data['suited_broadway'] = (data['suited']*data['broadway']).astype(int)
data['suited_broadway_score'] = (data['suited_broadway'] - data['suited_broadway'].min()) / (data['suited_broadway'].max() - data['suited_broadway'].min())

data['connector'] = (data.connectedness == 1).astype(int)
data['one_gap'] = (data.connectedness == 2).astype(int)
data['two_gap'] = (data.connectedness == 3).astype(int)
data['suited_connector'] = (data['connector']*data['suited']).astype(int)
data['suited_one_gap'] = (data['one_gap']*data['suited']).astype(int)
data['suited_two_gap'] = (data['two_gap']*data['suited']).astype(int)


data['connectedness_score'] = (data['connectedness'].max() - data['connectedness']) / data['connectedness'].max()
data['rank_diff_score'] = (data['rank_diff'].max() - data['rank_diff']) / data['rank_diff'].max()
data['high_card_score'] = (data['high_card'] - data['high_card'].min()) / (data['high_card'].max() - data['high_card'].min())
data['low_card_score'] = (data['low_card'] - data['low_card'].min()) / (data['low_card'].max() - data['low_card'].min())
data['suited_connected_score'] = data['connectedness_score'] * data['suited']

if __name__ == '__main__':
    data_dir = "./data"
    raw_dir = os.path.join(data_dir, "raw")

    for directory in [data_dir, raw_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    data.to_csv(os.path.join(raw_dir, 'base_hand_data.csv'), index=False)