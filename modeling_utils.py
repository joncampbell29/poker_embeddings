import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from utils import *
from sklearn.manifold import TSNE

def visualize_learned_embeddings(model, reduction_tecq='pca', reduced_dims=3, **kwargs):
    with torch.no_grad():
        learned_embeddings = model.embedder.weight 

    if learned_embeddings.shape[1] > reduced_dims:
        embeddings_np = learned_embeddings.detach().cpu().numpy()
        
        if reduction_tecq == 'pca':
            reducer = PCA(n_components=reduced_dims)
        elif reduction_tecq == 'tsne':
            reducer = TSNE(n_components=reduced_dims, random_state=42)
        else:
            raise ValueError("Invalid reduction technique. Use 'pca' or 'tsne'.")
        
        x_reduced = reducer.fit_transform(embeddings_np)
    else:
        x_reduced = learned_embeddings.detach().cpu().numpy()

    card_indices = list(range(52))
    ranks = [get_rank_idx(idx) for idx in card_indices]
    suits = [get_suit_idx(idx) for idx in card_indices]

    rank_palette = sns.color_palette("hsv", 13)
    suit_palette = sns.color_palette("Set1", 4)

    rank_colors = [rank_palette[r] for r in ranks]
    suit_colors = [suit_palette[s] for s in suits]

    if reduced_dims == 3:
        fig = plt.figure(figsize=kwargs.get('figsize', (12,5)))
        fig.suptitle(f"{reduction_tecq.upper()}; Embedding Dim: {learned_embeddings.shape[1]}")
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(x_reduced[:, 0], x_reduced[:, 1], x_reduced[:, 2], c=rank_colors, s=50)
        ax1.set_title("Colored by Rank")

        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(x_reduced[:, 0], x_reduced[:, 1], x_reduced[:, 2], c=suit_colors, s=50)
        ax2.set_title("Colored by Suit")
        plt.show()

    elif reduced_dims == 2:
        fig, ax = plt.subplots(1, 2, figsize=kwargs.get('figsize', (12,5)))
        fig.suptitle(f"{reduction_tecq.upper()}; Embedding Dim: {learned_embeddings.shape[1]}")
        ax[0].scatter(x_reduced[:, 0], x_reduced[:, 1], c=rank_colors, s=50)
        ax[0].set_title("Colored by Rank")
        ax[1].scatter(x_reduced[:, 0], x_reduced[:, 1], c=suit_colors, s=50)
        ax[1].set_title("Colored by Suit")
        plt.show()
    else:
        print("Visualization is only supported for 2D and 3D embeddings.")
        
# def generate_flop_cpu(model, num_flops=1, device="cpu"):
#     model.eval()  

#     z_sample = torch.randn(num_flops, model.latent_dim).to(device)

#     with torch.no_grad():
#         flop_recon = model.decoder(z_sample)
#     return flop_recon

# def genflop_to_binary(flop_recon):
#     flop_binary = torch.zeros_like(flop_recon)

#     flop_binary[:, :13] = F.one_hot(flop_recon[:, :13].argmax(dim=1), num_classes=13)
#     flop_binary[:, 13:17] = F.one_hot(flop_recon[:, 13:17].argmax(dim=1), num_classes=4)
    
#     flop_binary[:, 17:30] = F.one_hot(flop_recon[:, 17:30].argmax(dim=1), num_classes=13)
#     flop_binary[:, 30:34] = F.one_hot(flop_recon[:, 30:34].argmax(dim=1), num_classes=4)
    
#     flop_binary[:, 34:47] = F.one_hot(flop_recon[:, 34:47].argmax(dim=1), num_classes=13)
#     flop_binary[:, 47:] = F.one_hot(flop_recon[:, 47:].argmax(dim=1), num_classes=4)

#     return flop_binary

# def decode_binary_flop(flop_batch):
#     RANKS = [str(i) for i in range(2, 10)] + ['T', 'J', 'Q', 'K', 'A']
#     SUITS = ['s', 'h', 'c', 'd']
    
#     card1_ranks = np.argmax(flop_batch[:, :13], axis=1)
#     card1_suits = np.argmax(flop_batch[:, 13:17], axis=1)
    
#     card2_ranks = np.argmax(flop_batch[:, 17:30], axis=1)
#     card2_suits = np.argmax(flop_batch[:, 30:34], axis=1)
    
#     card3_ranks = np.argmax(flop_batch[:, 34:47], axis=1)
#     card3_suits = np.argmax(flop_batch[:, 47:], axis=1)
#     flops = [
#         [f"{RANKS[card1_ranks[i]]}{SUITS[card1_suits[i]]}",
#          f"{RANKS[card2_ranks[i]]}{SUITS[card2_suits[i]]}",
#          f"{RANKS[card3_ranks[i]]}{SUITS[card3_suits[i]]}"]
#         for i in range(flop_batch.shape[0])
#     ]
#     return flops

# def generate_flop_human(model, num_flops = 1, device="cpu"):
#     flop_recon = generate_flop_cpu(model, num_flops, device)
#     flop_bin = genflop_to_binary(flop_recon)
#     decoded_flops = decode_binary_flop(flop_bin)
#     return decoded_flops

# def extract_z_samples(model, dataloader, label=None, device="cpu"):
#     label_dict = {
#         'suitedness': 0,
#         'pairness': 1, 
#         'connectedness': 2, 
#         'high_low_texture': 3, 
#         'high_card': 4, 
#         'straightness': 5
#     }
#     z_samples = []
#     labels = []
#     model.eval()
#     with torch.no_grad():
#         for flop_vec, flop, lab_vec in dataloader:
#             x = flop_vec.to(device).to(torch.float32)
#             if label:
#                 lab = lab_vec[:,label_dict[label]].to(device)
#                 labels.append(lab.cpu().numpy())
#             encoding = model.encoder(x)  
#             mu = model.fc_mu(encoding)
#             logvar = model.fc_log_var(encoding)
#             std = torch.exp(0.5 * logvar)
#             epsilon = torch.randn_like(std)
#             z = mu + std * epsilon
#             z_samples.append(z.cpu().numpy())
#     if label:
#         return np.vstack(z_samples), np.concatenate(labels)
#     else:
#         return np.vstack(z_samples)

# def plot_latent_space(model, dataloader, device='cpu'):
#     categories = ['suitedness','pairness','connectedness','high_low_texture','high_card', 'straightness']
#     fig, axes = plt.subplots(2, 3, figsize=(15, 10))
#     for idx, category in enumerate(categories):
#         z_samples, labs = extract_z_samples(model, dataloader, label=category, device=device)
#         pca = PCA(n_components=2)
#         z_pca = pca.fit_transform(z_samples)
#         row, col = divmod(idx, 3)
#         ax = axes[row, col]
    
#         scatter = ax.scatter(z_pca[:, 0], z_pca[:, 1], alpha=0.5, c=labs, cmap="coolwarm")
#         cbar = fig.colorbar(scatter, ax=ax)

#         ax.set_title(f"Latent Space: {category}", fontsize=12)
#         ax.set_xlabel("PC 1")
#         ax.set_ylabel("PC 2")

#     plt.tight_layout()
#     plt.show()