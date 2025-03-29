from .hand_embedder import HandEmbeddingModel
import torch.nn as nn
import torch

class HandWinPredictor(nn.Module):
    def __init__(self, card_embedding_dim=16, hidden_dims=[128, 64, 32]):
        super(HandWinPredictor, self).__init__()
        
        self.encoder = HandEmbeddingModel(card_embedding_dim)
        
        layers = []
        prev_dim = card_embedding_dim + 3
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.LeakyReLU())
            layers.append(nn.BatchNorm1d(dim))
            prev_dim = dim
        
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
    
    def forward(self, hole_card_idx, hole_card_attributes):
        hole_card_embedding = self.encoder(hole_card_idx)
        x = torch.concat([hole_card_embedding, hole_card_attributes], dim=1)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x