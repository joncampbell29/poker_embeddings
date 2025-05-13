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
    def __init__(self,
                 rank_embedding_dim=8,
                 suit_embedding_dim=8,
                 hidden_dim=16,
                 edge_attr_dim=2,
                 node_mlp_layers=2,
                 gnn_layers=2,
                 reduction='mean'):
        super().__init__()
        self.reduction = reduction

        self.rank_embedder = nn.Embedding(13, rank_embedding_dim)
        self.suit_embedder = nn.Embedding(4, suit_embedding_dim)
        layers = []
        for i in range(node_mlp_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if i < node_mlp_layers-1:
                layers.append(nn.ReLU())
        self.node_mlp_layers = nn.Sequential(*layers)

        self.card_emb_projector = nn.Linear(rank_embedding_dim+suit_embedding_dim, hidden_dim)

        self.gnn_layers = nn.ModuleList()
        for i in range(gnn_layers):
            self.gnn_layers.append(GINEConv(nn=self.node_mlp_layers, edge_dim=edge_attr_dim))

    def forward(self, data):
        rank_emb = self.rank_embedder(data.x[:, 0])
        suit_emb = self.suit_embedder(data.x[:, 1])
        card_emb = torch.cat([rank_emb, suit_emb], dim=1)
        x = self.card_emb_projector(card_emb)
        for conv in self.gnn_layers:
            x = conv(x, data.edge_index, data.edge_attr)
            x = F.relu(x)
        x = tg.utils.scatter(x, data.batch, dim=0, reduce=self.reduction)
        return x

class HandClassifier(nn.Module):
    def __init__(self,
                 rank_embedding_dim=8,
                 suit_embedding_dim=8,
                 hidden_dim=16,
                 edge_attr_dim=2,
                 node_mlp_layers=2,
                 gnn_layers=2,
                 reduction='mean',
                 out_dim=16):
        super().__init__()
        self.hand_encoder = HandGNN(
            rank_embedding_dim=rank_embedding_dim, suit_embedding_dim=suit_embedding_dim, hidden_dim=hidden_dim,
            edge_attr_dim=edge_attr_dim, node_mlp_layers=node_mlp_layers,
            gnn_layers=gnn_layers, reduction=reduction)

        self.final = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        self.output_layer = nn.Linear(out_dim, 10)

    def forward(self, data):
        x = self.hand_encoder(data)
        x = self.final(x)
        return self.output_layer(x)

class HandScorer(nn.Module):
    def __init__(self,
                 rank_embedding_dim=8,
                 suit_embedding_dim=8,
                 hidden_dim=16,
                 edge_attr_dim=2,
                 node_mlp_layers=2,
                 gnn_layers=2,
                 reduction='mean',
                 out_dim=16):
        super().__init__()
        self.hand_encoder = HandGNN(
            rank_embedding_dim=rank_embedding_dim, suit_embedding_dim=suit_embedding_dim, hidden_dim=hidden_dim,
            edge_attr_dim=edge_attr_dim, node_mlp_layers=node_mlp_layers,
            gnn_layers=gnn_layers, reduction=reduction)

        self.final = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        self.output_layer = nn.Linear(out_dim, 1)

    def forward(self, data):
        x = self.hand_encoder(data)
        x = self.final(x)
        return self.output_layer(x).squeeze()