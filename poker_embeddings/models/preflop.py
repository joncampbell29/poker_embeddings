import torch.nn as nn
import torch
import torch.nn.functional as F

class EVcVAE(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, c_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LeakyReLU(),
        )
        self.mean_fc = nn.Linear(hidden_dim, embedding_dim)
        self.logvar_fc = nn.Linear(hidden_dim, embedding_dim)

        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim+c_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, ev, c):
        h = self.encoder(ev)
        mean = self.mean_fc(h)
        logvar = self.logvar_fc(h)
        z = self.reparameterize(mean, logvar)
        z_c = torch.cat((z, c), dim=1)
        reconstructed_x = self.decoder(z_c)
        return reconstructed_x, mean, logvar


class EquityDiffModel(nn.Module):
    def __init__(self, input_size=90, embedding_dim=8, hidden_dim=64):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, embedding_dim)
        )

        self.comparison = nn.Sequential(
            nn.Linear(embedding_dim*2, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim//2, 1)
        )

    def forward(self, hand1, hand2):
        emb1 = self.encoder(hand1)
        emb2 = self.encoder(hand2)
        combined = torch.cat([emb1, emb2], dim=1)
        equity_diff = self.comparison(combined)
        return emb1, emb2, equity_diff
