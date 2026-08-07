import sys
import os
import pickle
from itertools import combinations
from collections import defaultdict

from datasets import load_from_disk

from util import cka_score, get_hf_model_ids, get_langs_list

print("Entered script")

model_class = sys.argv[1]  # e.g. "mbert_repro"

hf_model_ids = get_hf_model_ids(model_class)
langs = get_langs_list(model_class)

print(hf_model_ids)
print(langs)

savename = f"../experiments/encoded_datasets/xnli/{model_class}-full_pairwise_cka.pkl"

# --- resume support: load whatever was already saved ---
if os.path.exists(savename):
    with open(savename, 'rb') as f:
        loaded = pickle.load(f)
    scores_dict = defaultdict(lambda: defaultdict(list))
    for k, v in loaded.items():
        scores_dict[k] = defaultdict(list, v)
    print(f"Resuming from existing file, found {sum(len(v) for v in scores_dict.values())} pairs already done")
else:
    scores_dict = defaultdict(lambda: defaultdict(list))


def save_progress():
    """Write to a temp file then atomically replace, so a crash mid-write
    can't corrupt the pickle."""
    scores_dfs = {k: dict(v) for k, v in scores_dict.items()}
    tmpname = savename + ".tmp"
    with open(tmpname, 'wb') as f:
        pickle.dump(scores_dfs, f)
    os.replace(tmpname, savename)


for model_name, hf_model_id in list(reversed(list(hf_model_ids.items()))):
    print(f"\n\n{hf_model_id}")

    dataset = {}
    for lang in langs:
        savedir = model_name
        dataset[lang] = load_from_disk(f"../experiments/encoded_datasets/xnli/{savedir}/{lang}")

    src_check = dataset[langs[0]]
    num_layers = sum([n.startswith("mean") for n in src_check.column_names])
    print(f"{num_layers} layers", flush=True)

    for lang_a, lang_b in combinations(langs, 2):
        pair_key = f"{lang_a}-{lang_b}"

        # --- skip pairs we already finished on a previous run ---
        if len(scores_dict[hf_model_id].get(pair_key, [])) == num_layers:
            print(f"\n pair: {pair_key} (already done, skipping)", flush=True)
            continue

        print(f"\n pair: {pair_key}", flush=True)

        a_data = dataset[lang_a]
        b_data = dataset[lang_b]

        pair_scores = []
        for l in range(num_layers):
            s = cka_score(a_data[f'mean_{l}'], b_data[f'mean_{l}'])
            pair_scores.append(s)
            print(f"l{l}: {s:.4f}", end=' ', flush=True)

        scores_dict[hf_model_id][pair_key] = pair_scores

        # --- save after every pair ---
        save_progress()

print('\n\nFinished \n')
print(f"saved scores at {savename}")