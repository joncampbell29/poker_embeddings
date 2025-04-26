import torch.nn as nn
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