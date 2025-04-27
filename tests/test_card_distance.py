from poker_embeddings.poker_utils.hands import card_distance, _rank_values
import pytest

test_cases = {
    '22': 0, '32': 1, '33': 0, '42': 2, '43': 1, '44': 0, '52': 3, '53': 2, '54': 1, '55': 0, '62': 4, '63': 3,
    '64': 2, '65': 1, '66': 0, '72': 5, '73': 4, '74': 3, '75': 2, '76': 1, '77': 0, '82': 6, '83': 5, '84': 4,
    '85': 3, '86': 2, '87': 1, '88': 0, '92': 7, '93': 6, '94': 5, '95': 4, '96': 3, '97': 2, '98': 1, '99': 0,
    'A2': 1, 'A3': 2, 'A4': 3, 'A5': 4, 'A6': 5, 'A7': 6, 'A8': 6, 'A9': 5, 'AA': 0, 'AJ': 3, 'AK': 1, 'AQ': 2,
    'AT': 4, 'J2': 9, 'J3': 8, 'J4': 7, 'J5': 6, 'J6': 5, 'J7': 4, 'J8': 3, 'J9': 2, 'JJ': 0, 'JT': 1, 'K2': 11,
    'K3': 10, 'K4': 9, 'K5': 8, 'K6': 7, 'K7': 6, 'K8': 5, 'K9': 4, 'KJ': 2, 'KK': 0, 'KQ': 1, 'KT': 3, 'Q2': 10,
    'Q3': 9, 'Q4': 8, 'Q5': 7, 'Q6': 6, 'Q7': 5, 'Q8': 4, 'Q9': 3, 'QJ': 1, 'QQ': 0, 'QT': 2, 'T2': 8, 'T3': 7,
    'T4': 6, 'T5': 5, 'T6': 4, 'T7': 3, 'T8': 2, 'T9': 1, 'TT': 0
    }
new_dict = {}
for key in test_cases.keys():
    val1 = _rank_values[key[0]]
    val2 = _rank_values[key[1]]
    new_dict[(val1, val2)] = test_cases[key]
test_cases.update(new_dict)

def test_card_distance_lookup():
    for hand, expected in test_cases.items():
        assert card_distance(hand) == expected, f"Failed for hand: {hand}"

