# Day 4 Execution Log: Method A Reproduction — CKA-Outlier Check (Del & Fishel, 2021)

**Companion to:** Isolating Tokenization Fertility from the Mean-Pooling Confound (Urdu–Hindi / Central Kanuri Arabic-Latin)
**Covers:** Day 4 — Method A reproduction, pass 1 (per Day-by-Day Execution Plan)
**Reference paper:** Del & Fishel (2021), _Similarity of Sentence Representations in Multilingual LMs: Resolving Conflicting Literature and a Case Study of Baltic Languages_, arXiv:2109.01207 — not the 2022 AACL paper (which covers ANC and is a separate, later reproduction target).

---

## 1. Objective

Reproduce the published Urdu/Hindi/Swahili/Thai CKA-outlier finding using the forked `xsim` pipeline, as a sanity check before trusting the fork for real target-language data. Log any deviation immediately rather than at week's end.

---

## 2. Pipeline gap found and fixed

`get_langs_list()` in the fork had no branch returning `ur/hi/sw/th` — added a new `mbert_repro` model class:

```python
elif model_class == "mbert_repro":
    langs = ['en', 'ur', 'hi', 'sw', 'th', 'fr', 'de', 'es']
```

`fr`, `de`, `es` added as "main branch" reference languages, needed to test whether Urdu/Hindi/Swahili/Thai are actually isolated from a normal-similarity cluster, not just from each other.

`get_hf_model_ids()` given a matching dict-valued branch (`{'mbert': 'bert-base-multilingual-cased'}`) — the encode script's Day 1–3 fix assumed `hf_model_ids` is always a dict, but only the `norm_1M` branch actually was one; every other branch (including the new one, initially) returned a list, which broke on `.items()`.

`run_analysis.py` had the same list-vs-dict inconsistency the encode script was patched for on Day 1–3, but only in the encode script — `run_analysis.py` itself was never touched. It used `hf_model_id.split('/')[-1]` (the model _id string_) to build `savedir`, while the patched encode script now saves using `model_name` (the dict _key_) instead — two different folder names for the same data, causing a `FileNotFoundError` the first time this was run. Fixed by mirroring the same `model_name, hf_model_id = ...items()` unpacking in `run_analysis.py` that was already applied to the encode script.

**New script written:** `run_analysis.py` only ever computes `en-X` similarity, so it can't test a clustering claim, which depends on the full pairwise matrix. Wrote `full_pairwise_cka.py` (computes CKA for all C(8,2)=28 language pairs, all 13 layers) and `cluster_cka.py` (agglomerative clustering + dendrogram via `scipy.cluster.hierarchy`).

---

## 3. Result

Layer-wise CKA obtained for all 8 languages × all pairs (mBERT, `bert-base-multilingual-cased`). At layer 8:

- `en-fr` / `en-de` / `en-es` / `fr-de` / `fr-es` / `de-es`: 0.61–0.78 (tight main-branch cluster, as expected)
- `ur-hi`: 0.3080 — the single highest pairwise value among the outlier set
- `hi-th`: 0.3065 — close second
- `sw-th`: 0.2963 — next tightest
- `ur-sw`: 0.2193, `ur-th`: 0.2305 — clearly the weakest pairings involving Urdu

**Qualitative match to the paper's Figure 8 claim:** confirmed. Urdu pairs closest with Hindi (not Swahili/Thai), Swahili pairs closest with Thai, and both pairings sit clearly apart from the tight European-language main branch — consistent with the paper's description of Urdu+Hindi as one outlier branch and Swahili+Thai as a separately isolated branch.

**Not yet confirmed:** exact dendrogram topology. With only 8 languages (vs. the paper's 29), the average `en`-to-{Urdu,Hindi} distance (~0.565) is smaller than the {Urdu,Hindi}-to-{Swahili,Thai} distance (~0.739) — meaning `en` could plausibly merge onto the Urdu-Hindi cluster before Swahili-Thai does, giving a differently-shaped tree at this small scale even though the core pairwise signal matches. `scipy.cluster.hierarchy.linkage`/`dendrogram` has not actually been run yet (requires matplotlib/scipy on a machine that can execute it — pending lab PC / Colab access).

**Open ambiguity, not yet resolved:** the paper's "layer 8" labeling is inconsistent across its own figures (some axes appear 0-indexed, some 1-indexed), so it isn't confirmed whether the paper's "layer 8" corresponds to this pipeline's `mean_8` or `mean_7`. `cluster_cka.py` is written to generate dendrograms for layers 7, 8, and 9 so the actual tree shape can be compared against Figure 8 directly, rather than assuming the label maps literally.

---

## 4. Checkpoint 1 verdict

**Proceed with documented partial match**, not a clean pass: the qualitative outlier relationship (Urdu↔Hindi tightest, Swahili↔Thai tightest, both isolated from main branch) reproduces correctly. Full dendrogram topology and the exact layer-index mapping remain unverified pending execution of `cluster_cka.py` on a machine with matplotlib/scipy available.

---

## 5. Naming inconsistency flagged

This log's own filename lineage (see Day 1–3 log's header: "Serbian Cyrillic–Latin") does not match the Research Proposal's committed secondary pair, which is **Central Kanuri Arabic/Latin** (`knc_Arab`/`knc_Latn`), not Serbian. Tamasheq was screened and rejected earlier; Serbian does not appear in the proposal as a candidate at all. This appears to be a leftover from an earlier proposal draft — worth confirming explicitly before Day 5, since Day 5 shifts from reproduction-language data (XNLI: en/ur/hi/sw/th/fr/de/es) to the actual project's target pair, which needs FLORES-200 Kanuri data, not Serbian.

---

## 6. Still outstanding (carried forward from Day 1–3, unresolved)

- [ ] Pull and verify FLORES-200 splits for the actual target pair (Urdu/Hindi + Central Kanuri Arabic/Latin) — separate from the XNLI reproduction data used for Days 1–4.
- [ ] Rewrite the Section 6 sentence flagged as lifted verbatim from the COLING 2025 source.
- [ ] Draft the OSF pre-registration skeleton (hypotheses RQ1–RQ4, primary analysis plan, TOST SESOI).

---

## 7. What's next (Day 5)

Per the Day-by-Day Execution Plan, Day 5 is Method A reproduction, pass 2:

- Extend to cosine similarity and nearest-neighbor accuracy, identically across layers/models.
- Compute the secondary-pair equivalents to the Urdu×Hindi numbers — **Central Kanuri Arabic × Central Kanuri Latin**, using FLORES-200 (not XNLI, which doesn't cover Kanuri).

This requires access to a machine that can run matplotlib/scipy and handle FLORES-200 pulls/encoding at reasonable speed — flagged as needing lab PC or Colab going forward.
