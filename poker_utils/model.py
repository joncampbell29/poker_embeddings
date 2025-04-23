import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier, MLPRegressor
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
import warnings
from sklearn.exceptions import ConvergenceWarning
# from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
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

def plot_train_loss(train_losses, val_losses=None, figsize=(5,5)):
    plt.figure(figsize=figsize)
    plt.plot(train_losses, label='Training Loss')
    if val_losses is not None:
        plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    
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
    tsne = TSNE(n_components=2, perplexity=30, random_state=29)
    embeddings_2d = tsne.fit_transform(embeddings)
    # pca = PCA(n_components=2)
    # embeddings_2d = pca.fit_transform(embeddings)
    embeddings_2d_df = pd.DataFrame(embeddings_2d, index=hands, columns=['C1', 'C2'])

    tsne_df = pd.concat([base_hand_data, embeddings_2d_df], axis=1).reset_index()
    tsne_df.rename({'index':'hand'}, axis=1, inplace=True)
    
    plt.figure(figsize=kwargs.get("figsize", (12, 10)))
    sns.scatterplot(data=tsne_df, 
                    x='C1', y='C2', 
                    hue=hand_feature_to_color, 
                    alpha=kwargs.get('alpha', 0.7))

    for hand in hands_of_interest:
        if hand in tsne_df['hand'].values:
            row = tsne_df[tsne_df['hand'] == hand].iloc[0]
            plt.annotate(hand, (row['C1'], row['C2']), fontsize=12, ha='center')

    plt.title('TSNE of Hand Embeddings', fontsize=14)
    plt.xlabel('C1', fontsize=12)
    plt.ylabel('C2', fontsize=12)
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

def prob_embeddings(embedding, prob_data):
    if isinstance(embedding, torch.Tensor):
        embedding = embedding.detach().cpu().numpy()
    attributes_cls = [
        'suited', 'connectedness', 'pair', 'high_card', 'low_card',
        'rank_diff', 'hand_type', 'ace', 'broadway', 'low_pair', 'medium_pair',
        'high_pair','suited_broadway','connector', 'one_gap', 'two_gap',
        'suited_connector', 'suited_one_gap', 'suited_two_gap'
    ]
    attributes_reg = [
        'tot_win_perc', 'high_card_win_perc',
        'one_pair_win_perc', 'two_pair_win_perc', 'three_of_a_kind_win_perc',
        'straight_win_perc', 'flush_win_perc', 'full_house_win_perc',
        'four_of_a_kind_win_perc', 'straight_flush_win_perc',
        'BB_play10', 'BB_play2', 'BB_play3', 'BB_play4', 'BB_play5', 'BB_play6',
        'BB_play7', 'BB_play8', 'BB_play9', 'D_play10', 'D_play3', 'D_play4',
        'D_play5', 'D_play6', 'D_play7', 'D_play8', 'D_play9', 'SB_play10',
        'SB_play2', 'SB_play3', 'SB_play4', 'SB_play5', 'SB_play6', 'SB_play7',
        'SB_play8', 'SB_play9', 'pos3_play10', 'pos3_play4', 'pos3_play5',
        'pos3_play6', 'pos3_play7', 'pos3_play8', 'pos3_play9', 'pos4_play10',
        'pos4_play5', 'pos4_play6', 'pos4_play7', 'pos4_play8', 'pos4_play9',
        'pos5_play10', 'pos5_play6', 'pos5_play7', 'pos5_play8', 'pos5_play9',
        'pos6_play10', 'pos6_play7', 'pos6_play8', 'pos6_play9', 'pos7_play10',
        'pos7_play8', 'pos7_play9', 'pos8_play10', 'pos8_play9', 'pos9_play10'
        ]
    
    reg_results = []
    cls_results = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        for attr in attributes_cls:
            y = prob_data[attr].to_numpy()
        
            classifier = MLPClassifier(
            hidden_layer_sizes=(32,),
            activation='relu',
            solver='adam',
            max_iter=1000,
            random_state=29
            ).fit(embedding, y)
            pred = classifier.predict(embedding)
            acc = np.mean(pred == y)
            report = classification_report(y, pred, output_dict=True, zero_division=0)

            for label, metrics in report.items():
                if label not in ['accuracy', 'macro avg', 'weighted avg']:
                    cls_results.append({
                        'attribute': attr,
                        'class': label,
                        'accuracy': acc,
                        'precision': metrics['precision'],
                        'recall': metrics['recall'],
                        'f1_score': metrics['f1-score'],
                        'support_frac': metrics['support'] / len(y)
                    })
        for attr in attributes_reg:
            y = prob_data[attr].to_numpy()
            regressor = MLPRegressor(
                hidden_layer_sizes=(32,),
                activation='relu',
                solver='adam',
                max_iter=1000,
                random_state=29
                ).fit(embedding, y)
            pred = regressor.predict(embedding)
            rmse = np.sqrt(np.mean((pred - y)**2))
            reg_results.append({
                'attribute': attr,
                'rmse': rmse
            })
    return pd.concat([pd.DataFrame(cls_results), pd.DataFrame(reg_results)])

def evaluate_hand_hand_equity(embeddings, equity_matrix):
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    emb_sim = embeddings @ embeddings.T
    equity_sim = 1 - squareform(pdist(equity_matrix, metric='cosine'))
    embedding_sims = emb_sim.flatten()
    equity_sims = equity_sim.flatten()

    corr, pval = spearmanr(embedding_sims, equity_sims)
    return {'spear_corr': corr, "pval": pval}

def save_model_and_embeddings(embeddings, embedding_filename, model=None, state_dict_filename=None):
    weight_dir = os.path.join(PROJ_ROOT, "model_weights")
    emb_dir = os.path.join(PROJ_ROOT, "embeddings")
    if isinstance(embeddings, np.ndarray):
        embeddings = torch.tensor(embeddings)
    torch.save(embeddings, os.path.join(emb_dir, embedding_filename+".pt"))
    if model is not None:
        torch.save(model.state_dict(), os.path.join(weight_dir, state_dict_filename+".pth"))
        
    