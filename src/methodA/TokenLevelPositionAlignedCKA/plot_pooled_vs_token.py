"""
plot_pooled_vs_token.py

Step 15: plot pooled vs. token-level CKA curves on the same axes, per pair,
per model. This comparison -- not any single number -- is the Method A
deliverable for this diagnostic.

Reads the single shared results pickle that both the pooled path and the
--token-level path in cka.py write to (../experiments/results/flores_cka_scores.pkl),
matching pair_key "{a}-{b}-{split}" against "{a}-{b}-{split}-token".

Usage:
    python plot_pooled_vs_token.py [--out-dir ../experiments/results/plots]
"""

import argparse
import os
import pickle

import matplotlib.pyplot as plt

RESULTS_PATH = "../experiments/results/flores_cka_scores.pkl"
PAIRS = [("urd_Arab", "hin_Deva"), ("knc_Arab", "knc_Latn")]
SPLITS = ["dev", "devtest"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=RESULTS_PATH)
    ap.add_argument("--out-dir", default="../experiments/results/plots")
    args = ap.parse_args()

    with open(args.results, "rb") as f:
        scores_dict = pickle.load(f)

    os.makedirs(args.out_dir, exist_ok=True)

    for hf_id, pair_scores in scores_dict.items():
        for lang_a, lang_b in PAIRS:
            for split in SPLITS:
                pooled_key = f"{lang_a}-{lang_b}-{split}"
                token_key = f"{pooled_key}-token"

                pooled = pair_scores.get(pooled_key)
                token = pair_scores.get(token_key)

                if pooled is None and token is None:
                    continue  # neither variant computed yet for this pair/split

                fig, ax = plt.subplots(figsize=(7, 4.5))
                if pooled is not None:
                    ax.plot(range(len(pooled)), pooled, marker="o", label="pooled (mean)")
                if token is not None:
                    ax.plot(range(len(token)), token, marker="s", label="token-level (aligned words)")

                ax.set_xlabel("Layer")
                ax.set_ylabel("CKA")
                ax.set_title(f"{hf_id}\n{lang_a}-{lang_b} [{split}]")
                ax.legend()
                ax.grid(alpha=0.3)
                fig.tight_layout()

                safe_model = hf_id.replace("/", "_")
                out_path = os.path.join(
                    args.out_dir, f"{safe_model}_{lang_a}-{lang_b}_{split}_pooled_vs_token.png"
                )
                fig.savefig(out_path, dpi=150)
                plt.close(fig)
                print(f"wrote {out_path}")

                if pooled is not None and token is not None:
                    # quick numeric divergence summary alongside the plot
                    diffs = [
                        abs(p - t)
                        for p, t in zip(pooled, token)
                        if p == p and t == t  # filter out NaNs (Step 14 zero-alignment cases)
                    ]
                    if diffs:
                        print(
                            f"  {hf_id} {lang_a}-{lang_b} [{split}]: "
                            f"mean |pooled - token| = {sum(diffs)/len(diffs):.4f}, "
                            f"max = {max(diffs):.4f}"
                        )


if __name__ == "__main__":
    main()
