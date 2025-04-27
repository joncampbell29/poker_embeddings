from poker_embeddings.poker_utils.hands import card_distance, _card_to_idx
import pytest

test_cases = {
    ("2c", "2d"): 0, ("5h","2h"): 3, ('Ah',"2h"): 1, ("As",'5h'): 4,
    ("Jc","3d"): 8, ('Kh',"3h"): 10, ("Kc",'8s'): 5, ("Ts","Td"): 0, ("Tc","3c"): 7
    }
new_dict = {}
for key in test_cases.keys():
    card1_id = _card_to_idx[key[0]]
    card2_id = _card_to_idx[key[1]]
    new_dict[(card1_id, card2_id)] = test_cases[key]
test_cases.update(new_dict)

def test_card_distance_lookup():
    for hand, expected in test_cases.items():
        assert card_distance(hand) == expected, f"Failed for hand: {hand}"

