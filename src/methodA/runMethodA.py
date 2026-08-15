import sys

from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

from util import encode_batch, get_hf_model_ids
import torch
import os


model_class = sys.argv[1]
device = sys.argv[2]
# model_class = "mT5"
# model_class = "xlmr"


batch_size = 100


hf_model_ids = get_hf_model_ids(model_class)
langs = ["urd_Arab", "hin_Deva", "knc_Arab", "knc_Latn"]

splits = ["dev", "devtest"]

# lang -> pair-directory mapping, avoids nested-quote f-string issue
pair_dir = {
    "urd_Arab": "hindi_urdu",
    "hin_Deva": "hindi_urdu",
    "knc_Arab": "kanuri",
    "knc_Latn": "kanuri",
}

# load once per (lang, split); each load_dataset('text', ...) call returns a
# DatasetDict with a single 'train' key by default, so unwrap it here to
# store the actual Dataset directly.
dataset = {}
for l in langs:
    dataset[l] = {}
    for s in splits:
        data_path = f"../../../data/splits/{pair_dir[l]}/{l}.{s}.txt"
        dataset[l][s] = load_dataset("text", data_files=data_path)["train"]

for model_name, hf_model_id in list(reversed(hf_model_ids.items())):
    print("\n loading ", hf_model_id, "\n")

    # load model
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)

    if model_class == "mT5":
        model = AutoModel.from_pretrained(hf_model_id).encoder
    else:
        model = AutoModel.from_pretrained(hf_model_id)

    _ = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    for lang in langs:
        for split in splits:
            print(f"\n encoding {lang} [{split}]")

            savedir = model_name if not model_class.startswith("norm") else f"{model_class}_{model_name}"
            outpath = f"../../data/encoded_datasets/flores/{savedir}/{lang}/{split}"
            if os.path.exists(outpath):
                print(f"\n skipping {lang} [{split}], already encoded at {outpath}")
                continue

            dataset_enc = dataset[lang][split].map(
                function=encode_batch,
                fn_kwargs={
                    "field": "text",
                    "tokenizer": tokenizer,
                    "model": model,
                    "detok": False,
                    "lang_code": lang,
                    "encode_token1": False,
                    "encode_cls": False,
                },
                batched=True,
                batch_size=batch_size,
            )

            if model_class.startswith("norm"):
                savedir = f"{model_class}_{model_name}"
            else:
                savedir = model_name

            dataset_enc.save_to_disk(f"../../data/encoded_datasets/flores/{savedir}/{lang}/{split}")

print("Finished")