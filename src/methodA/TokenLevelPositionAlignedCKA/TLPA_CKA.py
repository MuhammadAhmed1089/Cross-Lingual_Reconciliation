import sys
import os
import json
import pickle
import argparse
from collections import defaultdict
from util import get_hf_model_ids, cka_score, load_from_disk

# ---------------------------------------------------------------------------
# --token-level mode (Steps 7-14): everything in this block is additive.
# The original pooled-CKA path below is untouched when this flag is off.
# ---------------------------------------------------------------------------

ALIGN_DIR = os.environ.get("ALIGN_DIR", "../alignments")
# Alignment files are keyed by the pair *name* used in build_alignments.py,
# not the raw lang codes -- map lang-code pairs to that name here.
ALIGN_FILE_KEY = {
    frozenset(("urd_Arab", "hin_Deva")): "urd_hin",
    frozenset(("knc_Arab", "knc_Latn")): "knc",
}


def load_alignments(lang_a, lang_b, split):
    """Step 6/10: load the (word_idx_a, word_idx_b) pairs saved by build_alignments.py."""
    key = ALIGN_FILE_KEY[frozenset((lang_a, lang_b))]
    path = os.path.join(ALIGN_DIR, f"{key}_{split}.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)  # {sent_idx_str: [[word_idx_a, word_idx_b], ...]}


def read_flores_split(lang, split, flores_root=None):
    """Raw sentences for a split -- needed here because pooled encodings alone
    don't expose per-word boundaries (Step 7)."""
    flores_root = flores_root or os.environ.get("FLORES_ROOT", "../../data/splits/original")
    path = os.path.join(flores_root, f"{lang}.{split}.txt")
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def extract_first_subtoken_vectors(sentences, tokenizer, model, device):
    """
    Steps 7-8: for every sentence, every layer, every whitespace-split word,
    keep only that word's first subtoken hidden state (no averaging).

    Words are defined by str.split(), matching the same split used by
    build_alignments.py's aligner input and the Kanuri positional check --
    so word indices here line up with the indices stored in the alignment
    files without any extra remapping.

    Returns: list (one entry per sentence) of dict {layer_idx: ndarray [n_words, hidden_dim]}
    """
    import torch
    import numpy as np

    model.eval()
    per_sentence = []
    with torch.no_grad():
        for sent in sentences:
            words = sent.split()
            if not words:
                per_sentence.append({})
                continue
            enc = tokenizer(
                words,
                is_split_into_words=True,
                return_tensors="pt",
                truncation=True,
            ).to(device)
            word_ids = enc.word_ids(batch_index=0)  # subtoken idx -> word idx (or None)

            # first subtoken index per word
            first_subtok_idx = {}
            for subtok_idx, w_id in enumerate(word_ids):
                if w_id is not None and w_id not in first_subtok_idx:
                    first_subtok_idx[w_id] = subtok_idx

            out = model(**enc, output_hidden_states=True)
            hidden_states = out.hidden_states  # tuple: (num_layers+1) x [1, seq_len, dim]

            layer_vecs = {}
            for layer_idx, layer_h in enumerate(hidden_states):
                layer_h = layer_h[0]  # [seq_len, dim]
                vecs = np.stack([
                    layer_h[first_subtok_idx[w]].cpu().numpy()
                    for w in range(len(words))
                    if w in first_subtok_idx
                ]) if first_subtok_idx else np.zeros((0, layer_h.shape[-1]))
                layer_vecs[layer_idx] = vecs
            per_sentence.append(layer_vecs)
    return per_sentence


def get_or_build_word_vectors(model_name, lang, split, tokenizer, model, device, cache_root=None):
    """Step 9: cache these separately from the pooled encode_batch output --
    they were never part of it."""
    import pickle as _pickle

    cache_root = cache_root or os.environ.get(
        "WORD_VEC_CACHE_DIR", "../../data/encoded_datasets/flores_token_level"
    )
    cache_dir = os.path.join(cache_root, model_name, lang)
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{split}.pkl")

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return _pickle.load(f)

    sentences = read_flores_split(lang, split)
    vectors = extract_first_subtoken_vectors(sentences, tokenizer, model, device)
    with open(cache_path, "wb") as f:
        _pickle.dump(vectors, f)
    return vectors


def build_aligned_matrices(alignments, vecs_a, vecs_b, layer):
    """Steps 10-11: pick out matched word-vector pairs per the alignment,
    stack into two row-aligned matrices for one layer."""
    import numpy as np

    rows_a, rows_b = [], []
    for sent_idx_str, idx_pairs in alignments.items():
        i = int(sent_idx_str)
        if i >= len(vecs_a) or i >= len(vecs_b):
            continue
        va = vecs_a[i].get(layer)
        vb = vecs_b[i].get(layer)
        if va is None or vb is None or len(va) == 0 or len(vb) == 0:
            continue
        for word_idx_a, word_idx_b in idx_pairs:
            if word_idx_a < len(va) and word_idx_b < len(vb):
                rows_a.append(va[word_idx_a])
                rows_b.append(vb[word_idx_b])

    if not rows_a:
        return None, None
    return np.stack(rows_a), np.stack(rows_b)


def run_token_level(hf_model_ids, pairs, splits, scores_dict, out_path):
    """Steps 7-14 end to end, using the SAME cka_score / skip-resume / output
    pickle as the pooled path, distinguished only by the '-token' pair_key
    suffix (Step 13) so pooled and token-level results never collide."""
    import torch
    from transformers import AutoTokenizer, AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for model_name, hf_id in list(reversed(list(hf_model_ids.items()))):
        print(f"\n\n[token-level] {hf_id}")
        tokenizer = AutoTokenizer.from_pretrained(hf_id)
        model = AutoModel.from_pretrained(hf_id).to(device)

        for split in splits:
            for lang_a, lang_b in pairs:
                pair_key = f"{lang_a}-{lang_b}-{split}-token"

                alignments = load_alignments(lang_a, lang_b, split)
                vecs_a = get_or_build_word_vectors(model_name, lang_a, split, tokenizer, model, device)
                vecs_b = get_or_build_word_vectors(model_name, lang_b, split, tokenizer, model, device)

                num_layers = model.config.num_hidden_layers + 1  # + embedding layer

                if len(scores_dict[hf_id].get(pair_key, [])) == num_layers:
                    print(f"\n pair: {pair_key} (already done, skipping)", flush=True)
                    continue

                print(f"\n pair: {pair_key}", flush=True)

                pair_scores = []
                for l in range(num_layers):
                    mat_a, mat_b = build_aligned_matrices(alignments, vecs_a, vecs_b, l)

                    # Step 14: sanity check -- near-zero count means the aligner
                    # or the Kanuri positional-matching assumption silently failed.
                    n_pairs = 0 if mat_a is None else mat_a.shape[0]
                    print(f"l{l}: n_aligned_pairs={n_pairs}", end=" ", flush=True)
                    if n_pairs == 0:
                        print("[WARNING: zero aligned pairs -- check alignment file / cache]", end=" ")
                        pair_scores.append(float("nan"))
                        continue

                    s = cka_score(mat_a, mat_b)
                    pair_scores.append(s)
                print()

                scores_dict[hf_id][pair_key] = pair_scores
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                with open(out_path, "wb") as f:
                    pickle.dump(dict(scores_dict), f)


# ---------------------------------------------------------------------------
# Original pooled-CKA path (unchanged)
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("model_class")
parser.add_argument("--token-level", action="store_true",
                     help="Run Steps 7-14 token-level CKA instead of pooled CKA")
args = parser.parse_args()
model_class = args.model_class

hf_model_ids = get_hf_model_ids(model_class)
langs = ["hin_Deva", "urd_Arab", "knc_Arab", "knc_Latn"]
splits = ["dev", "devtest"]

# Only these two pairs are in scope for Method A -- no cross-pair combinations.
pairs = [
    ("urd_Arab", "hin_Deva"),
    ("knc_Arab", "knc_Latn"),
]

scores_dict = defaultdict(dict)

out_path = "../experiments/results/flores_cka_scores.pkl"
if os.path.exists(out_path):
    with open(out_path, "rb") as f:
        scores_dict.update(pickle.load(f))

if args.token_level:
    run_token_level(hf_model_ids, pairs, splits, scores_dict, out_path)
    print("\nFinished (token-level)")
    sys.exit(0)

for model_name, hf_id in list(reversed(list(hf_model_ids.items()))):
    print(f"\n\n{hf_id}")

    for split in splits:
        dataset = {}
        for lang in langs:
            dataset[lang] = load_from_disk(
                f"../../data/encoded_datasets/flores/{model_name}/{lang}/{split}"
            )

        src_check = dataset[langs[0]]
        num_layers = sum([n.startswith("mean") for n in src_check.column_names])
        print(f"{num_layers} layers [{split}]", flush=True)

        for lang_a, lang_b in pairs:
            pair_key = f"{lang_a}-{lang_b}-{split}"

            # skip pairs/splits already finished on a previous run
            if len(scores_dict[hf_id].get(pair_key, [])) == num_layers:
                print(f"\n pair: {pair_key} (already done, skipping)", flush=True)
                continue

            print(f"\n pair: {pair_key}", flush=True)

            a_data = dataset[lang_a]
            b_data = dataset[lang_b]

            pair_scores = []
            for l in range(num_layers):
                s = cka_score(a_data[f"mean_{l}"], b_data[f"mean_{l}"])
                pair_scores.append(s)
                print(f"l{l}: {s:.4f}", end=" ", flush=True)
            print()

            scores_dict[hf_id][pair_key] = pair_scores

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "wb") as f:
                pickle.dump(dict(scores_dict), f)

print("\nFinished")