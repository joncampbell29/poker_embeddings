import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from poker_utils.constants import HANDS_DICT
import os
UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(UTILS_DIR, ".."))


def create_similarity_top_bottom(similarity_df, hands_of_interest):
    records = []

    for hand in hands_of_interest:
        subset = similarity_df.loc[hand].sort_values(ascending=False)
        top3 = subset.iloc[1:4]
        bottom3 = subset.iloc[-3:]
        records.append({
            'hand': hand,
            'top_1': f"{top3.index[0]} ({top3.iloc[0]:.4f})",
            'top_2': f"{top3.index[1]} ({top3.iloc[1]:.4f})",
            'top_3': f"{top3.index[2]} ({top3.iloc[2]:.4f})",
            'bottom_1': f"{bottom3.index[0]} ({bottom3.iloc[0]:.4f})",
            'bottom_2': f"{bottom3.index[1]} ({bottom3.iloc[1]:.4f})",
            'bottom_3': f"{bottom3.index[2]} ({bottom3.iloc[2]:.4f})"
        })

    return pd.DataFrame(records)


def analyze_embeddings(
    embeddings, 
    hands_of_interest=['AAo','KKo','QQo','AKs','AKo','JJo','TTo','A2s','72o','Q5s','76s','99o'],
    hand_feature_to_color='hand_type',
    **kwargs
    ):
    
    base_hand_data = pd.read_csv(os.path.join(PROJ_ROOT, "data/raw/base_hand_data.csv")).set_index("hand")
    
    hands = list(HANDS_DICT.values())
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    
    if embeddings.shape[0] != 169:
        raise ValueError(f"Embeddings of shape ({embeddings.shape[0]},{embeddings.shape[1]})")
    
    similarity_df = pd.DataFrame(cosine_similarity(embeddings), index=hands, columns=hands)
    top_bottom_df = create_similarity_top_bottom(similarity_df, hands_of_interest)
    
    embeddings_df = pd.DataFrame(embeddings, index=hands)
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    embeddings_2d_df = pd.DataFrame(embeddings_2d, index=hands, columns=['PC1', 'PC2'])

    pca_df = pd.concat([base_hand_data, embeddings_2d_df], axis=1).reset_index()
    pca_df.rename({'index':'hand'}, axis=1, inplace=True)
    
    plt.figure(figsize=kwargs.get("figsize", (12, 10)))
    sns.scatterplot(data=pca_df, 
                    x='PC1', y='PC2', 
                    hue=hand_feature_to_color, 
                    alpha=kwargs.get('alpha', 0.7))

    for hand in hands_of_interest:
        if hand in pca_df['hand'].values:
            row = pca_df[pca_df['hand'] == hand].iloc[0]
            plt.annotate(hand, (row['PC1'], row['PC2']), fontsize=12, ha='center')

    plt.title('PCA of Hand Embeddings', fontsize=14)
    plt.xlabel('PC1', fontsize=12)
    plt.ylabel('PC2', fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    print(top_bottom_df)
    return similarity_df

def get_feature_importance(model, input_tensor, feature_names, prediction_idx=None, target_idx=None):
    model.eval()
    num_hands = input_tensor.shape[0]
    input_tensor = input_tensor.clone().detach().requires_grad_(True)

    with torch.no_grad():
        output_sample = model(input_tensor[0].unsqueeze(0))
        if isinstance(output_sample, tuple):
            output_sample = output_sample[prediction_idx]
        num_targets = output_sample.shape[-1]

    all_importances = []

    for hand_idx in range(num_hands):
        hand_input = input_tensor[hand_idx].unsqueeze(0).clone().detach().requires_grad_(True)
        output = model(hand_input)
        if isinstance(output, tuple):
            output = output[prediction_idx]

        output = output.squeeze(0)

        hand_importances = []
        for i in range(num_targets):
            model.zero_grad()
            hand_input.grad = None
            output[i].backward(retain_graph=True)
            grad = hand_input.grad.abs().squeeze().detach().clone()
            hand_importances.append(grad)

        hand_importances = torch.stack(hand_importances)  # [num_targets, num_features]
        all_importances.append(hand_importances)

    all_importances = torch.stack(all_importances)  # [num_hands, num_targets, num_features]
    return all_importances
    # mean_importance_per_hand = pd.DataFrame(
    #     all_importances.mean(dim=1).numpy(), 
    #     index=list(HANDS_DICT.values()), 
    #     columns=feature_names)
    # plt.figure(figsize=(20, 12))

    # ax = sns.heatmap(mean_importance_per_hand, 
    #                 cmap='YlOrRd',
    #                 robust=True,
    #                 cbar_kws={'label': 'importance'})

    # plt.title('Importance Heatmap by Hand and Feature')
    # plt.tight_layout()
    # plt.show()



def save_model_and_embeddings(embeddings, embedding_filename, model=None, state_dict_filename=None):
    weight_dir = os.path.join(PROJ_ROOT, "model_weights")
    if isinstance(embeddings, np.ndarray):
        embeddings = torch.tensor(embeddings)
        torch.save(embeddings, os.path.join(weight_dir, embedding_filename+".pt"))
    if model is not None:
        torch.save(model.state_dict(), os.path.join(weight_dir, state_dict_filename+".pth"))
        
    