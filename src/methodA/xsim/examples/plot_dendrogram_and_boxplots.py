import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

# Languages in evaluation order
langs = [
    'ar', 'az', 'bg', 'cs', 'da', 'de', 'el', 'en', 'es', 'et',
    'fi', 'fr', 'hi', 'hu', 'kk', 'lt', 'lv', 'nl', 'no', 'pl',
    'ru', 'sv', 'sw', 'th', 'tr', 'ur', 'uz', 'vi', 'zh'
]

def load_pkl_data(pkl_path):
    """Loads the pickle file and returns the nested dict of CKA scores."""
    with open(pkl_path, 'rb') as f:
        content = pickle.load(f)
    model_key = list(content.keys())[0]
    return model_key, content[model_key]

def build_layer_dataframe_and_matrices(data, langs):
    """
    Parses pair scores from the pkl data:
    1. Builds a DataFrame with columns ['Layer', 'CKA', 'lang_pair'] for all layers.
    2. Constructs a dict of symmetric N x N matrices per layer index.

    Note: layer_idx 0 corresponds to mean_0, the embedding layer (no transformer
    blocks applied), NOT a numbered paper layer. layer_idx >= 1 corresponds directly
    to the paper's layer numbers (mean_1 = paper Layer 1, ..., mean_12 = paper Layer 12).
    """
    n_langs = len(langs)
    sample_pair = next(iter(data.keys()))
    num_layers = len(data[sample_pair])
    
    rows = []
    matrices = {l: np.ones((n_langs, n_langs)) for l in range(num_layers)}
    
    for i, l1 in enumerate(langs):
        for j, l2 in enumerate(langs):
            if i == j:
                continue
            pair_key1 = f"{l1}-{l2}"
            pair_key2 = f"{l2}-{l1}"
            
            if pair_key1 in data:
                scores = data[pair_key1]
            elif pair_key2 in data:
                scores = data[pair_key2]
            else:
                scores = [np.nan] * num_layers
                
            for layer_idx, score in enumerate(scores):
                matrices[layer_idx][i, j] = score
                if i < j and layer_idx >= 1:      # skip embedding layer (idx 0)
                    rows.append({
                        'lang_pair': f"{l1}-{l2}",
                        'CKA': score,
                        'Layer': str(layer_idx)    # no +1; layer_idx already matches paper's layer number
                    })
                    
    big_df = pd.DataFrame(rows)
    return big_df, matrices

def author_plot_dendrogram(model, labels, ax, **kwargs):
    """
    Author's exact dendrogram plotting function from multilingual-case_study.ipynb:
    Creates linkage matrix from AgglomerativeClustering model and calls scipy dendrogram.
    """
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)

    dendrogram(
        linkage_matrix,
        labels=labels,
        orientation='top',
        ax=ax,
        **kwargs
    )

def plot_boxplots(big_df_mbert, big_df_xlmr, save_path):
    """Plots layer-wise CKA boxplots for both models side-by-side."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    layer_order = [str(i) for i in range(1, 13)]
    
    # mBERT Box Plot
    sns.boxplot(data=big_df_mbert, x='Layer', y='CKA', order=layer_order, ax=axes[0])
    axes[0].set_title("(a) mBERT Layer-wise CKA")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("CKA Similarity")
    axes[0].grid(True, linestyle='--', alpha=0.5)
    
    # XLM-R Box Plot
    sns.boxplot(data=big_df_xlmr, x='Layer', y='CKA', order=layer_order, ax=axes[1])
    axes[1].set_title("(b) XLM-R Layer-wise CKA")
    axes[1].set_xlabel("Layer")
    axes[1].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved layer-wise box plots to {save_path}")

def plot_dendrogram_figure_pair(matrix_mbert, matrix_xlmr, langs, save_path):
    """Plots side-by-side dendrograms for (a) mBERT (layer 8) and (b) XLM-R (layer 7)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # (a) mBERT (Layer 8, index 8)
    model_mbert = AgglomerativeClustering(distance_threshold=0, n_clusters=None).fit(matrix_mbert)
    author_plot_dendrogram(model_mbert, labels=langs, ax=axes[0])
    axes[0].tick_params(axis='x', which='major', labelsize=10, rotation=45)
    axes[0].set_title("(a) mBERT")

    # (b) XLM-R (Layer 7, index 7)
    model_xlmr = AgglomerativeClustering(distance_threshold=0, n_clusters=None).fit(matrix_xlmr)
    author_plot_dendrogram(model_xlmr, labels=langs, ax=axes[1])
    axes[1].tick_params(axis='x', which='major', labelsize=10, rotation=45)
    axes[1].set_title("(b) XLM-R")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved dendrograms side-by-side to {save_path}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(script_dir, "..", "experiments", "encoded_datasets", "xnli"))
    
    mbert_pkl = os.path.join(base_dir, "mbert_repro-full_pairwise_cka.pkl")
    xlmr_pkl = os.path.join(base_dir, "xlmr_repro-full_pairwise_cka.pkl")
    
    key_mbert, data_mbert = load_pkl_data(mbert_pkl)
    key_xlmr, data_xlmr = load_pkl_data(xlmr_pkl)
    
    sample_pair_mbert = next(iter(data_mbert.keys()))
    sample_pair_xlmr = next(iter(data_xlmr.keys()))
    print(f"mBERT: pair={sample_pair_mbert}, num layer-scores={len(data_mbert[sample_pair_mbert])}")
    print(f"XLM-R: pair={sample_pair_xlmr}, num layer-scores={len(data_xlmr[sample_pair_xlmr])}")

    df_mbert, matrices_mbert = build_layer_dataframe_and_matrices(data_mbert, langs)
    df_xlmr, matrices_xlmr = build_layer_dataframe_and_matrices(data_xlmr, langs)
    
    print(f"mBERT matrices keys: {sorted(matrices_mbert.keys())}")
    print(f"XLM-R matrices keys: {sorted(matrices_xlmr.keys())}")

    # 1. Plot layer-wise box plots
    boxplot_path = os.path.join(script_dir, "cka_layer_boxplots.png")
    plot_boxplots(df_mbert, df_xlmr, boxplot_path)
    
    # 2. Dendrograms for Layer 8 of mBERT (index 8) and Layer 7 of XLM-R (index 7)
    fig8_path = os.path.join(script_dir, "fig8_dendrograms.png")
    plot_dendrogram_figure_pair(
        matrices_mbert[8],
        matrices_xlmr[7],
        langs,
        fig8_path
    )
    
    print("Execution complete.")

if __name__ == "__main__":
    main()
