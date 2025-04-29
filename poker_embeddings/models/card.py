import torch.nn as nn
import torch_geometric as tg
from torch_geometric.nn import GINEConv
import torch.nn.functional as F
import torch


class CardEncoder(nn.Module):
    def __init__(self, rank_dim=2, suit_dim=2, dist_dim=12):
        super().__init__()
        self.rank_embedder = nn.Embedding(53, rank_dim, padding_idx=52)
        self.suit_embedder = nn.Embedding(53, suit_dim, padding_idx=52)
        self.dist_embedder = nn.Embedding(53, dist_dim, padding_idx=52)
        card_dim = rank_dim + suit_dim + dist_dim
        self.emb_proj = nn.Linear(card_dim, card_dim)
        self.rank_head = nn.Linear(card_dim, 13)
        self.suit_head = nn.Linear(card_dim, 4)
        self.dist_head = nn.Linear(card_dim, 52)

    def get_embeddings(self, card_id):
        rank_emb = self.rank_embedder(card_id)
        suit_emb = self.suit_embedder(card_id)
        dist_emb = self.dist_embedder(card_id)
        emb_cat = torch.cat([rank_emb, suit_emb, dist_emb], dim=1)
        return self.emb_proj(emb_cat)

    def forward(self, card_id):
        rank_emb = self.rank_embedder(card_id)
        suit_emb = self.suit_embedder(card_id)
        dist_emb = self.dist_embedder(card_id)

        emb_cat = torch.cat([rank_emb, suit_emb, dist_emb], dim=1)
        card_emb = self.emb_proj(emb_cat)
        rank_pred = self.rank_head(card_emb)
        suit_pred = self.suit_head(card_emb)
        dist_pred = self.dist_head(card_emb)
        return rank_pred, suit_pred, dist_pred

class HandGNN(nn.Module):
    def __init__(self, card_emb_dim=16, hidden_dim=16, out_dim=16, edge_attr_dim=2):
        super().__init__()
        self.card_embedder = nn.Embedding(53, card_emb_dim, padding_idx=52)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.card_emb_projector = nn.Linear(card_emb_dim, hidden_dim)
        self.gine1 = GINEConv(nn=self.node_mlp, edge_dim=edge_attr_dim)
        self.gine2 = GINEConv(nn=self.node_mlp, edge_dim=edge_attr_dim)
        self.final = nn.Linear(hidden_dim, out_dim)

    def forward(self, data):
        card_ids = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        x = self.card_embedder(card_ids)
        x = self.card_emb_projector(x)

        x = self.gine1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.gine2(x, edge_index, edge_attr)

        x = self.final(x)
        graphs_pooled = tg.utils.scatter(x, data.batch, dim=0, reduce='mean')
        return graphs_pooled

class HandClassifier(nn.Module):
    def __init__(self, card_emb_dim=16, hidden_dim=16, out_dim=16, edge_attr_dim=2):
        super().__init__()

        self.hand_encoder = HandGNN(
            card_emb_dim=card_emb_dim, hidden_dim=hidden_dim,
            out_dim=out_dim, edge_attr_dim=edge_attr_dim)
        self.output_layer = nn.Linear(out_dim, 10)

    def forward(self, data):
        hand_encoded = self.hand_encoder(data)
        return self.output_layer(hand_encoded)
