from poker_embeddings.poker_utils.hands import create_deck_graph, query_subgraph, card_distance


deck_edge_index, deck_edge_attr = create_deck_graph(normalize=False)

test_cards = [
    (0, 1),
    (6,10),
    (12, 13),
    (20, 21),
    (24, 25),
    (30, 31),
    (40, 41),
    (44, 45),
    (50, 51)
]

def test_query_subgraph():
    for cards in test_cards:
        distance = card_distance(cards)
        suited = int(cards[0] % 4 == cards[1] % 4)

        subgraph_edge_index, subgraph_edge_attr = query_subgraph(cards, deck_edge_index, deck_edge_attr)

        assert subgraph_edge_index.shape[0] == 2
        assert subgraph_edge_index.shape[1] == len(cards) * (len(cards) - 1)
        assert subgraph_edge_attr.shape[0] == len(cards) * (len(cards) - 1)
        assert subgraph_edge_attr.shape[1] == 2

        assert subgraph_edge_attr[0,0] == distance, f"Failed for cards {cards} with distance {distance}, got {subgraph_edge_attr[0,0].item()}"
        assert subgraph_edge_attr[0,1] == suited, f"Failed for cards {cards} with suited {suited}, got {subgraph_edge_attr[0,1].item()}"
