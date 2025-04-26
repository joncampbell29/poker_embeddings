import numpy as np
from treys import Card, Evaluator
from itertools import product
import random
from .hands import get_possible_hands


def simulate_hand_randrange(hand: str, evaluator, num_villans: int = 1, num_sims: int = 1000):
    '''
    Simulates Hand vs random range (any 2 cards)
    Hand Format is ranks + suited/offsuit. i.e. AAo, KKo, 76s, 76o
    returns (river_win_rate)
    '''
    hero_possible_hands = get_possible_hands(hand)
    hero_possible_treys = [(Card.new(hand[0]), Card.new(hand[1])) for hand in hero_possible_hands]

    flop_wins = 0
    flop_losses = 0
    flop_ties = 0

    turn_wins = 0
    turn_losses = 0
    turn_ties = 0

    river_wins = 0
    river_losses = 0
    river_ties = 0


    tot_flop_score = 0
    tot_turn_score = 0
    tot_river_score = 0

    RANKS = '23456789TJQKA'
    SUITS = 'cdhs'
    DECK = [Card.new(rank+suit) for rank, suit in product(RANKS, SUITS)]

    for _ in range(num_sims):
        deck = DECK[:]
        random.shuffle(deck)
        hero_sampled_hand = list(random.choice(hero_possible_treys)) # in format [12332, 93874]
        for card in hero_sampled_hand:
            deck.remove(card)

        opponent_hands = [[deck.pop(), deck.pop()] for _ in range(num_villans)]

        board = [deck.pop() for _ in range(5)]
        flop = board[:3]
        turn = [board[4]]

        hero_flop_score = evaluator.evaluate(board=flop, hand=hero_sampled_hand)
        if hero_flop_score < min(evaluator.evaluate(board=flop, hand=hand) for hand in opponent_hands):
            flop_wins += 1
            tot_flop_score += hero_flop_score
        elif hero_flop_score == min(evaluator.evaluate(board=flop, hand=hand) for hand in opponent_hands):
            flop_ties += 1
        # elif hero_flop_score > min(evaluator.evaluate(board=flop, hand=hand) for hand in opponent_hands):
        #     flop_losses += 1


        hero_turn_score = evaluator.evaluate(board=flop+turn, hand=hero_sampled_hand)
        if hero_turn_score < min(evaluator.evaluate(board=flop+turn, hand=hand) for hand in opponent_hands):
            turn_wins += 1
            tot_turn_score += hero_turn_score
        elif hero_turn_score == min(evaluator.evaluate(board=flop+turn, hand=hand) for hand in opponent_hands):
            turn_ties += 1
        # elif hero_turn_score > min(evaluator.evaluate(board=flop+turn, hand=hand) for hand in opponent_hands):
        #     turn_losses += 1


        hero_river_score = evaluator.evaluate(board=board, hand=hero_sampled_hand)
        if hero_river_score < min(evaluator.evaluate(board=board, hand=hand) for hand in opponent_hands):
            river_wins += 1
            tot_river_score += hero_river_score
        elif hero_river_score == min(evaluator.evaluate(board=board, hand=hand) for hand in opponent_hands):
            river_ties +=1
        # elif hero_river_score > min(evaluator.evaluate(board=board, hand=hand) for hand in opponent_hands):
        #     river_losses += 1


        flop_equity = (flop_wins + (flop_ties / 2)) / num_sims
        turn_equity = (turn_wins + (turn_ties / 2)) / num_sims
        river_equity = (river_wins + (river_ties / 2)) / num_sims

    return {
        'hand': hand,
        "equity": {
            'flop': flop_equity,
            'turn': turn_equity,
            'river': river_equity
            },
        'avg_score': {
            'flop': tot_flop_score / num_sims,
            'turn': tot_turn_score / num_sims,
            'river': tot_river_score / num_sims
            }
        }

def simulate_hand_hand(hand1, hand2, evaluator, num_sims=1000):
    '''
    Hand Format is ranks + suited/offsuit. i.e. AAo, KKo, 76s, 76o
    '''

    hand1_possible_hands = get_possible_hands(hand1)
    hand1_possible_treys = [(Card.new(hand[0]), Card.new(hand[1])) for hand in hand1_possible_hands]

    hand2_possible_hands = get_possible_hands(hand2)
    hand2_possible_treys = [(Card.new(hand[0]), Card.new(hand[1])) for hand in hand2_possible_hands]

    RANKS = '23456789TJQKA'
    SUITS = 'cdhs'
    DECK = [Card.new(rank+suit) for rank, suit in product(RANKS, SUITS)]
    h1_wins = 0
    h2_wins = 0
    for _ in range(num_sims):
        deck = DECK[:]
        random.shuffle(deck)

        h1 = list(random.choice(hand1_possible_treys))
        hand2_possible_treys_filt = [hand for hand in hand2_possible_treys if not set(hand).intersection(h1)]
        h2 = list(random.choice(hand2_possible_treys_filt))

        for c1, c2 in zip(h1, h2):
            deck.remove(c1)
            deck.remove(c2)

        board = [deck.pop() for _ in range(5)]

        hand1_score = evaluator.evaluate(board=board, hand=h1)
        hand2_score = evaluator.evaluate(board=board, hand=h2)
        if hand1_score < hand2_score:
            h1_wins += 1
        elif hand2_score < hand1_score:
            h2_wins += 1
    return {
        'hand1': {'hand': hand1, 'win_perc': h1_wins / num_sims},
        'hand2': {'hand': hand2, 'win_perc': h2_wins / num_sims}
        }