import requests
from bs4 import BeautifulSoup
from poker_utils.constants import HANDS_DICT
from itertools import product
import pandas as pd
from tqdm import tqdm

def extract_win_breakdown(soup_tr):
    hand_desc, val = soup_tr.find_all("td")
    hand_desc = hand_desc.text.strip().lower().replace(" ","_")
    val = val.text.strip().replace(",",'')
    return (hand_desc, val)

def extract_full_breakdown(table):
    hand_ranks = table.find_all('tr')[1:11] 
    return [extract_win_breakdown(i) for i in hand_ranks]

def extract_equity_data(text):
    try:
        return [i.strip()[6:] for i in text.strip("\n").split("%") if i.strip()]
    except Exception as e:
        return None
    
def extract_win_perc_data(text):
    try:
        return [float(i.strip()[3:]) for i in text.strip("\n").split("%") if i.strip()]
    except Exception as e:
        return None
    
def parse_hand_hand_html(soup_html):

    hand1_breakdown = extract_full_breakdown(soup_html.find_all("table")[3])
    hand2_breakdown = extract_full_breakdown(soup_html.find_all("table")[4])
    hand_text = soup_html.find_all('table')[1].find_all('tr')[2].text
    equity_text = soup_html.find_all('table')[1].find_all('tr')[5].text
    win_perc_text = soup_html.find_all('table')[1].find_all('tr')[4].text
    
    hand1, hand2 = hand_text.strip().split("\n\xa0\n")
    hand1_equity, hand2_equity = extract_equity_data(equity_text)
    hand1_win, tie, hand2_win = extract_win_perc_data(win_perc_text)
    hand1_breakdown_dict = {i: j for i, j in hand1_breakdown}
    hand2_breakdown_dict = {i: j for i, j in hand2_breakdown}
    return {
        "hand1": {
            "hand": hand1,
            "equity": hand1_equity,
            "win_perc": hand1_win,
            "breakdown": hand1_breakdown_dict
            },
        "hand2": {
            "hand": hand2,
            "equity": hand2_equity,
            "win_perc": hand2_win,
            "breakdown": hand2_breakdown_dict
            },
        "tie": tie
    }
    
if __name__ == "__main__":
    res = []
    for hand1, hand2 in tqdm(product(HANDS_DICT.values(), HANDS_DICT.values())):
        if hand1[0] == hand1[1]:
            hand1 = hand1[:2]
        
        if hand2[0] == hand2[1]:
            hand2 = hand2[:2]
        url = f"https://cardfight.com/{hand1}_{hand2}.html"
        resp = requests.get(url)
        if resp.status_code != 200:
            url = f"https://cardfight.com/{hand2}_{hand1}.html"
            resp = requests.get(url)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.content, 'html.parser')
            hand_hand_data = parse_hand_hand_html(soup)
        else:
            hand_hand_data = {'not_found_hand1': hand1,'not_found_hand2': hand2}
        res.append(hand_hand_data)
    equity_df = pd.json_normalize(res)
    equity_df.to_csv("data/equity_data.csv", index=False)
    