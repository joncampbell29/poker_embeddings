import numpy as np
from treys import Card, Evaluator
from itertools import product
import random


def simulate_hand(hole_cards: list, num_villans: int = 1, num_sims: int= 1_000):
    ''' 
    returns (flop_win_rate, turn_win_rate, river_win_rate)
    '''
    hero_hole_cards = [Card.new(c) for c in hole_cards]
    evaluator = Evaluator()
    flop_wins = 0
    turn_wins = 0
    river_wins = 0
    for _ in range(num_sims):
        RANKS = '23456789TJQKA'
        SUITS = 'cdhs'
        DECK = [rank+suit for rank, suit in product(RANKS, SUITS)]
        random.shuffle(DECK)
        for card in hole_cards:
            DECK.remove(card)
        opponent_hands = []
        for _ in range(num_villans):
            villain_cards = [Card.new(DECK.pop()), Card.new(DECK.pop())]
            opponent_hands.append(villain_cards)
        
        board = [Card.new(DECK.pop()) for _ in range(5)]
        flop = board[:3]
        turn = [board[4]]
        
        hero_flop_score = evaluator.evaluate(board=flop, hand=hero_hole_cards)
        villian_flop_score = [evaluator.evaluate(board=flop, hand=hand) for hand in opponent_hands]
        if hero_flop_score < np.min(villian_flop_score):
            flop_wins += 1
        
        hero_turn_score = evaluator.evaluate(board=flop+turn, hand=hero_hole_cards)
        villian_turn_score = [evaluator.evaluate(board=flop+turn, hand=hand) for hand in opponent_hands]
        if hero_turn_score < np.min(villian_turn_score):
            turn_wins += 1
            
        hero_river_score = evaluator.evaluate(board=board, hand=hero_hole_cards)
        villian_river_score = [evaluator.evaluate(board=board, hand=hand) for hand in opponent_hands]
        if hero_river_score < np.min(villian_river_score):
            river_wins += 1
            
    return flop_wins / num_sims, turn_wins / num_sims, river_wins/ num_sims