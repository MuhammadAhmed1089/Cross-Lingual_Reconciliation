import sys
import os
import pickle
from collections import defaultdict
from util import get_hf_model_ids, cka_score, load_from_disk

model_class = sys.argv[1]

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