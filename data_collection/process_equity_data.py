import pandas as pd
import os


if __name__ == "__main__":
    data_dir = "./data"
    raw_dir = os.path.join(data_dir, "raw")
    proc_dir = os.path.join(data_dir, "processed")
    for directory in [data_dir, raw_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)



    equity_data = pd.read_csv(os.path.join(raw_dir,'equity_data.csv'))
    equity_data.drop_duplicates(inplace=True)

    equity_data.drop(['not_found_hand1', 'not_found_hand2'], axis=1, inplace=True)
    equity_data.dropna(inplace=True)

    equity_data['hand1.hand'] = equity_data['hand1.hand'].apply(
        lambda x: x if x.endswith(('o','s')) else x + 'o')

    equity_data['hand2.hand'] = equity_data['hand2.hand'].apply(
        lambda x: x if x.endswith(('o','s')) else x + 'o')


    made_hands = [
        'high_card',
        'one_pair',
        'two_pair',
        'three_of_a_kind',
        'straight', 
        'flush',
        'full_house',
        'four_of_a_kind',
        'straight_flush', 
        ]


    hand_totals_dict = {}
    for i, row in equity_data.iterrows():
        for h in ['hand1','hand2']:
            hand = row[f'{h}.hand']
            if hand not in hand_totals_dict:
                hand_totals_dict[hand] = {made_hand+"_wins": 0 for made_hand in made_hands}
                hand_totals_dict[hand]['tot_wins'] = 0
                hand_totals_dict[hand]['tot_sims'] = 0
            for made_hand in made_hands:
                hand_totals_dict[hand][made_hand+"_wins"] += row[f"{h}.breakdown.{made_hand}"]
            hand_totals_dict[hand]['tot_wins'] += row[f'{h}.breakdown.total'] 
            hand_totals_dict[hand]['tot_sims'] += row['hand1.breakdown.total'] + row['hand2.breakdown.total'] 
        
        
    hand_win_totals = pd.DataFrame.from_dict(hand_totals_dict, orient='index')

    hand_win_totals['tot_win_perc'] = hand_win_totals['tot_wins'] / hand_win_totals['tot_sims']

    for made_hand in made_hands:
        col = f"{made_hand}_wins"
        hand_win_totals[f"{made_hand}_win_perc"] = hand_win_totals[col] / hand_win_totals["tot_wins"]

    hand_win_totals['straight_win_potential'] = hand_win_totals[
        ['straight_win_perc','straight_flush_win_perc']].sum(axis=1)

    hand_win_totals['flush_win_potential'] = hand_win_totals[
        ['flush_win_perc','straight_flush_win_perc']].sum(axis=1)

    hand_win_totals['value_win_potential'] = hand_win_totals[
        ['three_of_a_kind_win_perc','straight_win_perc','flush_win_perc','straight_flush_win_perc',
         'full_house_win_perc','four_of_a_kind_win_perc','straight_flush_win_perc']].sum(axis=1)


    hand_win_totals['highcard_win_potential'] = hand_win_totals['high_card_win_perc']
    hand_win_totals = hand_win_totals.reset_index().rename({"index":'hand'}, axis=1)


    hand_win_totals.to_csv(os.path.join(proc_dir, 'equity_totals.csv'), index=False)








