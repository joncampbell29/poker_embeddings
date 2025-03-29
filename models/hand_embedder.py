import torch.nn as nn

class HandEmbeddingModel(nn.Module):
    def __init__(self, embedding_dim=16):
        super().__init__()
        self.embedder = nn.Embedding(169, embedding_dim)
    def forward(self, x):
        x_embed = self.embedder(x)
        return x_embed