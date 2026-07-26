import pickle
import pandas as pd

savename = "experiments/encoded_datasets/xnli/mbert_repro-full_pairwise_cka.pkl"

with open(savename, "rb") as f:
    scores = pickle.load(f)

model_id = list(scores.keys())[0]
pair_scores = scores[model_id]

print(pair_scores)  # sanity check the raw shape first

df = pd.DataFrame(pair_scores)
df.index.name = "layer"
df.columns.name = "pair"

# safety net: if pairs ended up as rows instead of columns, flip it
if set(df.columns) <= {"en-ur", "en-hi", "en-sw", "en-th"}:
    pass  # already correct: pairs as columns
else:
    df = df.T
    df.index.name = "layer"

print(df.round(4).to_string())
df.to_csv("cka_mbert_repro.csv")