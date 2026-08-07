import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Languages in alphabetical order (matching Fig 7)
langs = [
    'ar', 'az', 'bg', 'cs', 'da', 'de', 'el', 'en', 'es', 'et',
    'fi', 'fr', 'hi', 'hu', 'kk', 'lt', 'lv', 'nl', 'no', 'pl',
    'ru', 'sv', 'sw', 'th', 'tr', 'ur', 'uz', 'vi', 'zh'
]

def load_and_build_matrix(pkl_path, model_key, layer_idx):
    """
    Loads CKA scores from the pickle file and constructs a symmetric DataFrame (29 x 29).
    layer_idx: 8 for 8th layer, 7 for 7th layer.
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)[model_key]
    
    n = len(langs)
    matrix = np.ones((n, n))
    
    for i, l1 in enumerate(langs):
        for j, l2 in enumerate(langs):
            if i == j:
                continue
            pair_key1 = f"{l1}-{l2}"
            pair_key2 = f"{l2}-{l1}"
            
            if pair_key1 in data:
                val = data[pair_key1][layer_idx]
            elif pair_key2 in data:
                val = data[pair_key2][layer_idx]
            else:
                val = np.nan
                
            matrix[i, j] = val
            
    df = pd.DataFrame(matrix, index=langs, columns=langs)
    return df

def plot_matrices():
    mbert_pkl = "../experiments/encoded_datasets/xnli/mbert_repro-full_pairwise_cka.pkl"
    xlmr_pkl = "../experiments/encoded_datasets/xnli/xlmr_repro-full_pairwise_cka.pkl"
    
    if not os.path.exists(mbert_pkl) or not os.path.exists(xlmr_pkl):
        # Fallback path if running from root directory
        mbert_pkl = "experiments/encoded_datasets/xnli/mbert_repro-full_pairwise_cka.pkl"
        xlmr_pkl = "experiments/encoded_datasets/xnli/xlmr_repro-full_pairwise_cka.pkl"

    # Layer 8 for mBERT (layer_idx 8) and Layer 7 for XLM-R (layer_idx 7)
    df_mbert = load_and_build_matrix(mbert_pkl, "bert-base-multilingual-cased", layer_idx=8)
    df_xlmr  = load_and_build_matrix(xlmr_pkl, "xlm-roberta-base", layer_idx=7)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Reverse y-axis to match paper convention (zh at top, ar at bottom)
    df_mbert_plot = df_mbert.iloc[::-1]
    df_xlmr_plot  = df_xlmr.iloc[::-1]

    # Plot (a) mBERT
    sns.heatmap(df_mbert_plot, ax=axes[0], cmap="Blues_r", vmin=0.4, vmax=1.0, 
                cbar_kws={'label': 'CKA', 'location': 'top', 'shrink': 0.5})
    axes[0].set_title("(a) mBERT", y=-0.15, fontsize=14)
    axes[0].set_xlabel("src")
    axes[0].set_ylabel("tgt")
    axes[0].tick_params(axis='x', rotation=45)

    # Plot (b) XLM-R
    sns.heatmap(df_xlmr_plot, ax=axes[1], cmap="Blues_r", vmin=0.5, vmax=1.0, 
                cbar_kws={'label': 'CKA', 'location': 'top', 'shrink': 0.5})
    axes[1].set_title("(b) XLM-R", y=-0.15, fontsize=14)
    axes[1].set_xlabel("src")
    axes[1].set_ylabel("tgt")
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    output_path = "pairwise_language_similarities.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {output_path}")

if __name__ == "__main__":
    plot_matrices()
