import pandas as pd
import numpy as np
from poker_utils.hands import find_blocked_hands, normalize_hand
from poker_utils.constants import HANDS_DICT
import os

data = []
for i, hand in HANDS_DICT.items():
    if hand[-1] == "s":
        card1 = hand[0]+"c"
        card2 = hand[1]+"c"
    else:
        card1 = hand[0]+"c"
        card2 = hand[1]+"d"
    blocked = find_blocked_hands((card1,card2))
    hands, counts = np.unique(np.array([normalize_hand(i) for i in blocked]), return_counts=True)
    blocked_combos_dict = {i.item():j.item() for i,j in zip(hands, counts)}
    
    premium_block = {
        'hand': hand,
        'block_aa': blocked_combos_dict.get('AAo', 0),
        'block_kk': blocked_combos_dict.get('KKo', 0),
        'block_qq': blocked_combos_dict.get('QQo', 0),
        'block_ak': blocked_combos_dict.get('AKo', 0) + blocked_combos_dict.get('AKs', 0),
        'block_aq': blocked_combos_dict.get('AQo', 0) + blocked_combos_dict.get('AQs', 0)
    }
    data.append(premium_block)
data = pd.DataFrame(data)

data['prem_blocker_score'] = (
    data['block_aa'] * 5.0 + 
    data['block_kk'] * 4.0 + 
    data['block_qq'] * 3.0 + 
    data['block_ak'] * 3.5 + 
    data['block_aq'] * 2.0
)

data['prem_blocker_score_norm'] = (data['prem_blocker_score'] - data['prem_blocker_score'].min()) / (data['prem_blocker_score'].max() - data['prem_blocker_score'].min())
data['block_aa_score'] = (data['block_aa'] - data['block_aa'].min()) / (data['block_aa'].max() - data['block_aa'].min())
data['block_kk_score'] = (data['block_kk'] - data['block_kk'].min()) / (data['block_kk'].max() - data['block_kk'].min())

if __name__ == '__main__':
    data_dir = "./data"
    raw_dir = os.path.join(data_dir, "raw")
    
    for directory in [data_dir, raw_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            
    data.to_csv(os.path.join(raw_dir, 'preflop_block_data.csv'), index=False)