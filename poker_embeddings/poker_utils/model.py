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
from .constants import HANDS_DICT, PROJ_ROOT
import os
import timeit
from torch.utils.data import DataLoader
from torch_geometric import loader



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

def plot_train_loss(train_losses: list, val_losses: list=None, figsize: tuple=(5,5))->None:
    '''
    Plots trian losses, and val losses if provided
    '''
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


def benchmark_dataloader(
        dataset: torch.utils.data.Dataset,
        batch_sizes: list=[64, 128, 256, 512, 1024],
        num_workers_list: list=[0, 1, 2, 4, 8],
        num_runs: int=10,
        graph: bool=False)->None:
    '''
    Times a pytorch dataloader for the fastest combination of batch size, num workers
    If dataset is a graph dataset set graph to True
    '''
    print(f"Dataset size: {len(dataset)} samples")

    best_time = float('inf')
    for batch_size in batch_sizes:
        for num_workers in num_workers_list:
            if graph:
                dloader = loader.DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=True
                )
            else:
                dloader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=True
                )

            def load_one_epoch():
                for batch in dloader:
                    pass

            total_time = timeit.timeit(load_one_epoch, number=num_runs)
            avg_time = total_time / num_runs
            print(f"Batch size: {batch_size}, num_workers: {num_workers}, time: {avg_time:.3f} seconds")
            if avg_time < best_time:
                best_time = avg_time
                best_params = (batch_size, num_workers)

    print(f"Best params: batch_size={best_params[0]}, num_workers={best_params[1]}, time {best_time:.3f}")
