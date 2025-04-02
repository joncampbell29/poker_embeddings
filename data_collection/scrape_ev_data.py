import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

base_url = "https://flopturnriver.com/poker-strategy/texas-holdem-expected-value-hand-charts-"
url_dict = {
    3: {
        'url': base_url + '3-players-19155/',
        'colnames': ['hand', 'SB', 'BB', 'D'],
        "save_path": "data/raw/ev_data/hand_ev3.xlsx"
        },
    4: {
        'url': base_url + '4-players-19154/',
        'colnames': ['hand', 'SB', 'BB', 'pos3', 'D'],
        "save_path": "data/raw/ev_data/hand_ev4.xlsx"
        },
    5: {
        'url': base_url + '5-players-19153/',
        'colnames': ['hand', 'SB', 'BB', 'pos3', 'pos4','D'],
        "save_path": "data/raw/ev_data/hand_ev5.xlsx"
        },
    7: {
        'url': base_url + '7-players-19151/',
        'colnames': ['hand', 'SB', 'BB', 'pos3', 'pos4','pos5', 'pos6','D'],
        "save_path": "data/raw/ev_data/hand_ev7.xlsx"
        },
    8: {
        'url': base_url + '8-players-19150/',
        'colnames': ['hand', 'SB', 'BB', 'pos3', 'pos4','pos5', 'pos6','pos7','D'],
        "save_path": "data/raw/ev_data/hand_ev8.xlsx"
        },
    10: {
        'url': base_url + '10-players-19148/',
        'colnames': ['hand', 'SB', 'BB', 'pos3', 'pos4','pos5', 'pos6','pos7','pos8','pos9','D'],
        "save_path": "data/raw/ev_data/hand_ev10.xlsx"
        }
    }


for num_players, data in url_dict.items():
    url = data['url']
    colnames = data['colnames']
    save_path = data['save_path']
    resp = requests.get(url)
    if resp.status_code != 200:
        print("Failed: ", num_players)
    else:
        pulled_dat = pd.read_html(resp.content)[0].iloc[3:,:]
        pulled_dat.columns = colnames
        pulled_dat['hand'] = pulled_dat['hand'].str.replace(" ","").apply(
            lambda x: x if x.endswith(('o','s')) else x + 'o')
        pulled_dat.to_excel(save_path, index=False)
        
path_to_ev_dat = "data/raw/ev_data/"

dfs = []
for rel_path in os.listdir(path_to_ev_dat):
    num_players = int(rel_path.removeprefix("hand_ev").removesuffix(".xlsx"))
    full_path = path_to_ev_dat+rel_path
    dat = pd.read_excel(full_path)
    dat_melted = dat.melt(
        id_vars=['hand'],
        value_vars=dat.columns[1:],
        var_name='position',
        value_name='EV'
        )
    dat_melted['players'] = num_players
    dfs.append(dat_melted)

full_ev_data = pd.concat(dfs, ignore_index=True)

full_ev_data.to_excel("data/raw/hand_ev_full.xlsx", index=False)