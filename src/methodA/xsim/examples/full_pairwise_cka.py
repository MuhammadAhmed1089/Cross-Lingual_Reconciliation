import sys
import pickle
from itertools import combinations
from collections import defaultdict

from datasets import load_from_disk

from util import cka_score_gpu, get_hf_model_ids, get_langs_list

print("Entered script")

model_class = sys.argv[1]  # e.g. "mbert_repro"

hf_model_ids = get_hf_model_ids(model_class)
langs = get_langs_list(model_class)

print(hf_model_ids)
print(langs)

scores_dict = defaultdict(lambda: defaultdict(list))

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
        print(f"\n pair: {lang_a}-{lang_b}", flush=True)

        a_data = dataset[lang_a]
        b_data = dataset[lang_b]

        for l in range(num_layers):
            s = cka_score_gpu(a_data[f'mean_{l}'], b_data[f'mean_{l}'])
            scores_dict[hf_model_id][f"{lang_a}-{lang_b}"].append(s)
            print(f"l{l}: {s:.4f}", end=' ', flush=True)

print('\n\nFinished \n')

scores_dfs = dict(scores_dict)
scores_dfs = {k: dict(v) for k, v in scores_dfs.items()}

savename = f"../experiments/encoded_datasets/xnli/{model_class}-full_pairwise_cka.pkl"
pickle.dump(scores_dfs, open(savename, 'wb'))
print(f"saved scores at {savename}")