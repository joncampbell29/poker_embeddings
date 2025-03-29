import numpy as np
from treys import Card, Evaluator
from itertools import product
import random


def simulate_hand(hole_cards: list, num_villans: int = 1, num_sims: int = 1_000):
    ''' 
    returns (river_win_rate)
    '''
    hero_hole_cards = [Card.new(c) for c in hole_cards]
    evaluator = Evaluator()
    # flop_wins = 0
    # turn_wins = 0
    river_wins = 0
    RANKS = '23456789TJQKA'
    SUITS = 'cdhs'
    DECK = [rank+suit for rank, suit in product(RANKS, SUITS)]
    for card in hole_cards:
        DECK.remove(card)
    DECK = [Card.new(c) for c in DECK]
    
    for _ in range(num_sims):
        deck = DECK[:]
        random.shuffle(deck)
        
        opponent_hands = [[deck.pop(), deck.pop()] for _ in range(num_villans)]
        
        board = [deck.pop() for _ in range(5)]
        # flop = board[:3]
        # turn = [board[4]]
        
        # hero_flop_score = evaluator.evaluate(board=flop, hand=hero_hole_cards)
        # if hero_flop_score < min(evaluator.evaluate(board=flop, hand=hand) for hand in opponent_hands):
        #     flop_wins += 1
        
        # hero_turn_score = evaluator.evaluate(board=flop+turn, hand=hero_hole_cards)
        # if hero_turn_score < min(evaluator.evaluate(board=flop+turn, hand=hand) for hand in opponent_hands):
        #     turn_wins += 1
            
        hero_river_score = evaluator.evaluate(board=board, hand=hero_hole_cards)
        if hero_river_score < min(evaluator.evaluate(board=board, hand=hand) for hand in opponent_hands):
            river_wins += 1
            
            #flop_wins / num_sims, turn_wins / num_sims, 
    return river_wins/ num_sims