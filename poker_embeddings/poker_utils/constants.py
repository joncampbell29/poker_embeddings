import yaml
import os

UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(UTILS_DIR, ".."))
CONSTANTS_DIR = os.path.join(PROJ_ROOT, "constants")

RANKS = '23456789TJQKA'
SUITS = 'cdhs'

with open(os.path.join(CONSTANTS_DIR, "ranks.yaml"), 'r') as f:
    RANKS_DICT = yaml.safe_load(f)

with open(os.path.join(CONSTANTS_DIR, "suits.yaml"), 'r') as f:
    SUITS_DICT = yaml.safe_load(f)

with open(os.path.join(CONSTANTS_DIR, "deck.yaml"), 'r') as f:
    DECK_DICT = yaml.safe_load(f)

with open(os.path.join(CONSTANTS_DIR, "hands_full.yaml"), 'r') as f:
    dat = yaml.safe_load(f)
    HANDS = [(hand[:2], hand[2:]) for hand in dat.values()]

with open(os.path.join(CONSTANTS_DIR, "hands.yaml"), 'r') as f:
    HANDS_DICT = yaml.safe_load(f)

