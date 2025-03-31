import pandas as pd
import numpy as np
import networkx as nx
from poker_utils.hands import find_blocked_hands, find_dominated_hands
from sklearn.cluster import KMeans


hand_chart = pd.read_excel('data/raw/hand_chart.xlsx')

G = nx.DiGraph()
for row in range(13):
    for col in range(13):
        hand = hand_chart.iloc[row, col]
        G.add_node(hand)
                
for source in G.nodes():
    blocked_hands_dict = find_blocked_hands(source)
    dominated_hands_dict = find_dominated_hands(source)
    for dest in G.nodes():
        blocks = 1.0 if dest in blocked_hands_dict else 0.0
        if blocks == 1.0:
            combinations_blocked = blocked_hands_dict[dest]
        else:
            combinations_blocked = 0.0
            
        dominates = 1.0 if dest in dominated_hands_dict else 0.0
        if dominates == 1.0:
            combinations_dominated = dominated_hands_dict[dest]
        else:
            combinations_dominated = 0.0

        G.add_edge(source, dest, 
                   blocks=blocks, 
                   dominates=dominates, 
                   combinations_blocked=combinations_blocked,
                   combinations_dominated=combinations_dominated)
        
results = {}
for node in G.nodes():
    dominates_count = 0
    combos_dominated = 0
    combos_blocked = 0
    blocks_count = 0
    for source, dest, data in G.out_edges(node, data=True):
        dominates_count += data['dominates']
        combos_blocked += data['combinations_blocked']
        blocks_count += data['blocks']
        combos_dominated += data['combinations_dominated']

    dominated_by_count = 0
    blocked_by_count = 0
    combos_blocked_by_count = 0
    combos_dominated_by_count = 0
    for source, dest, data in G.in_edges(node, data=True):
        dominated_by_count += data['dominates']
        combos_blocked_by_count += data['combinations_blocked']
        blocked_by_count+= data['blocks']
        combos_dominated_by_count += data['combinations_dominated']
    
    results[node] = {
            'dominates_count': dominates_count,
            "combos_dominated": combos_dominated,
            
            'blocks_count': blocks_count,
            'combos_blocked': combos_blocked,
            
            'blocked_by_count': blocked_by_count,
            'combos_blocked_by_count': combos_blocked_by_count,
            
            'dominated_by_count': dominated_by_count,
            "combos_dominated_by_count": combos_dominated_by_count
    }
    
agg_dom_block_data = pd.DataFrame.from_dict(results, orient='index')

agg_dom_block_data["dom_ratio"] = agg_dom_block_data['combos_dominated'] / np.where(agg_dom_block_data['combos_dominated_by_count'] == 0, 1, agg_dom_block_data['combos_dominated_by_count'])
agg_dom_block_data["block_ratio"] = agg_dom_block_data['combos_blocked'] / np.where(agg_dom_block_data['combos_blocked_by_count'] == 0, 1, agg_dom_block_data['combos_blocked_by_count'])
agg_dom_block_data["dom_block_ratio"] = agg_dom_block_data['combos_dominated'] / np.where(agg_dom_block_data['combos_blocked_by_count'] == 0, 1, agg_dom_block_data['combos_blocked_by_count'])


agg_dom_block_data['block_ratio_cat'] = np.select(
    condlist=[
        agg_dom_block_data['block_ratio'] > 1.5,
        (agg_dom_block_data['block_ratio'] > 1) & (agg_dom_block_data['block_ratio'] < 1.5),
        agg_dom_block_data['block_ratio'] < 1
        ],
    choicelist=[
        2,
        1,
        0
    ]
)

X = agg_dom_block_data[['block_ratio_cat', 'dom_ratio','dom_block_ratio']].values
X_norm = (X - X.mean(axis=0)) / X.std(axis=0)

kmeans = KMeans(n_clusters=4, random_state=29)
agg_dom_block_data['cluster'] = kmeans.fit_predict(X_norm)

agg_dom_block_data.index.name = "hand"
agg_dom_block_data.reset_index(inplace=True)

agg_dom_block_data.to_csv("data/raw/dom_block_data.csv", index=False)

