import yaml

RANKS = '23456789TJQKA'
SUITS = 'cdhs'

with open("ranks.yaml", 'r') as f:
    RANKS_DICT = yaml.safe_load(f)
    
with open("suits.yaml", 'r') as f:
    SUITS_DICT = yaml.safe_load(f)
    
with open("deck.yaml", 'r') as f:
    DECK_DICT = yaml.safe_load(f)

with open("hands_full.yaml", 'r') as f:
    dat = yaml.safe_load(f)
    HANDS = [(hand[:2], hand[2:]) for hand in dat.values()]

with open("hands.yaml", 'r') as f:
    HANDS_DICT = yaml.safe_load(f)
