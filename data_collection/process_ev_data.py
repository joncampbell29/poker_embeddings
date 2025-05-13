import pandas as pd
from poker_embeddings.poker_utils.constants import RANKS_DICT
import os

if __name__ == "__main__":
    data_dir = "./data"
    processed_dir = os.path.join(data_dir, "processed")
    raw_dir = os.path.join(data_dir, "raw")

    for directory in [data_dir, raw_dir, processed_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    ev_data = pd.read_excel(os.path.join(raw_dir, "hand_ev_full.xlsx"))

    card_to_rank_id = {card: i for i, card in RANKS_DICT.items()}
    positions = ['SB', 'BB', 'pos3', 'pos4', 'pos5', 'pos6', 'pos7', 'pos8', 'pos9','D']
    position_mapping = {pos: i for i, pos in enumerate(positions)}

    ev_data[['card1','card2','suited']] = ev_data['hand'].str.split('', expand=True).iloc[:, 1:-1]

    ev_data['card1_rank_id'] = ev_data['card1'].map(card_to_rank_id)
    ev_data['card2_rank_id'] = ev_data['card2'].map(card_to_rank_id)

    def get_hand_type(hand):
        if hand[0] == hand[1]:
            return 2 # pair
        elif hand[2] == 's':
            return 1 # suited
        else:
            return 0 # offsuit
    ev_data['EV_deviation'] = ev_data["EV"] - ev_data.groupby("hand")['EV'].transform("mean")
    ev_data['hand_type_id'] = ev_data['hand'].apply(get_hand_type)
    ev_data['position_id'] = ev_data['position'].map(position_mapping)
    ev_data['players_id'] = ev_data['players'] - 2
    ev_data['pos_play'] = ev_data['position'].astype(str) + ev_data['players'].astype(str)
    ev_data.to_csv(os.path.join(processed_dir, "hand_ev_processed.csv"), index=False)