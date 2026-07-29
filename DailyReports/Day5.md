# Day 5 Execution Log: Method A Reproduction — Pass 2 (Cosine, NN-Accuracy, Reconciliation Analysis)

**Companion to:** Isolating Tokenization Fertility from the Mean-Pooling Confound (Urdu–Hindi / Central Kanuri Arabic-Latin)
**Covers:** Day 5 — Method A reproduction, pass 2 (per Day-by-Day Execution Plan v2)
**Reference papers:** Pires, Schlinger & Garrette (2019), *How Multilingual is Multilingual BERT?*, ACL 2019; Del & Fishel (2021), arXiv:2109.01207 — continuation of Day 4's reproduction check.

---

## 1. Objective

Extend Day 4's CKA-only reproduction to the two remaining metrics specified in the proposal (Section 7.1): anisotropy-corrected cosine similarity and nearest-neighbor accuracy. Compute the Kanuri-Arabic × Kanuri-Latin equivalents. Separately, begin the Checkpoint 1 reconciliation write-up (Day 7 deliverable) by pinning down exactly what Pires (2019) and Del & Fishel (2021) each found for Urdu–Hindi, since the two papers use fundamentally different metric types and the "disagreement" between them needed to be characterized precisely rather than assumed.

---

## 2. Pipeline gaps found and fixed

### 2.1 `cosine_score` in `util.py` does not implement the metric the proposal specifies
- `cosine_score` computes **neuron-wise** cosine (cosine across the sentence-sample dimension, treating each neuron/dimension as a unit of comparison, averaged over all neurons) — the same family as `svcca_score`/`pwcca_score`/`cca_score`, which are network-alignment measures, not sentence-similarity measures.
- The proposal (Section 7.1) specifies **per-sentence** anisotropy-corrected cosine similarity — comparing sentence pairs, matching CKA's unit of comparison.
- No centering/anisotropy-correction option exists in `cosine_score` at all.
- **Fix:** added `anisotropic_cosine_score()` to `util.py` alongside (not replacing) `cosine_score`, computing per-sentence cosine after corpus-mean centering (`axis=0`), matching Ethayarajh (2019)'s anisotropy-correction method as cited in the proposal.
- **Provenance note for the methods write-up (Day 29):** flag explicitly that cosine similarity in this study is computed per-sentence over mean-pooled, corpus-mean-centered representations, distinct from the neuron-wise formulation already present in the base xsim toolkit — otherwise a reviewer comparing against other xsim-based work may expect the neuron-wise number.

### 2.2 `run_analysis.py`'s cosine branch uses the wrong pooling type
- Calls `cosine_score(src['cls_{l}'], tgt['cls_{l}'])` — CLS-token pooling. The mean-pooled line is present but commented out.
- Proposal spec is mean-pooled throughout for Method A. Not used going forward — `full_pairwise_cosine.py` (below) uses `mean_{l}` correctly.

### 2.3 `compute_cosine_gpu`'s centering axis was wrong
- `center=True` branch did `a -= a.mean(1, keepdims=True)` — centers each sentence's own vector across its own dimensions (per-sentence normalization), not the shared corpus-level direction.
- True anisotropy correction requires subtracting the **corpus mean vector** (the same direction shared across all sentences in that language/layer) — `axis=0`, not `axis=1`.
- **Fix applied:** `a = a - a.mean(0, keepdims=True)`. This affects `matching_accuracy(..., center=True)`, i.e., the `acc-cent` task in `run_analysis.py`.
- **Decision:** per proposal Section 7.1 wording ("...anisotropy-corrected cosine similarity... plus nearest-neighbor accuracy" — no correction qualifier on NN-accuracy), the required NN-accuracy run is **uncorrected** (`acc`, `center=False`). The `acc-cent` fix is retained as available but not part of the required Day 5 deliverable.

### 2.4 `run_analysis.py` only ever computes `en-X` pairs
- Confirmed again (same limitation Day 4 noted for CKA): no `lang_a`–`lang_b` combinations, only English against everything else.
- Not sufficient for the reconciliation rank-comparison (below), which specifically needs `ur-hi` and `sw-th` as pairs, not just their relationship to English.

---

## 3. New/updated files

| File | Status | Purpose |
|---|---|---|
| `util.py` | updated | Added `anisotropic_cosine_score()`; fixed centering axis in `compute_cosine_gpu()` |
| `full_pairwise_cosine.py` | written, running | Mirrors `full_pairwise_cka.py` structure exactly; all-pairs, all-layers, mean-pooled, anisotropy-corrected cosine |
| `full_pairwise_accuracy.py` | written, not yet run | Mirrors same structure; all-pairs, all-layers, mean-pooled, **uncorrected** NN-accuracy (`center=False`) |

**Scope note on `full_pairwise_cosine.py`:** the proposal only requires cosine for two pairs (Urdu-Hindi, Kanuri-Arabic/Latin), not the full 8-language/28-pair matrix. The full-matrix run currently executing overnight is a bonus/sanity check, not a required deliverable — its output doesn't need to appear in the paper unless it's later decided to serve some other purpose.

**Scope note on `full_pairwise_accuracy.py`:** unlike cosine, the full 8-language matrix **is** actually needed here — specifically to rank-compare NN-accuracy against Day 4's CKA ranks for the reconciliation analysis (Section 5 below). This is the one full-matrix run with a concrete justification beyond "for completeness."

**Open implementation decision, not yet resolved:** `matching_accuracy` is directional/asymmetric — it checks whether `a`'s nearest neighbor in `b` is correct, which isn't necessarily true in reverse. Pires's Table 4 reports both directions separately (fine-tune-language → eval-language). Not yet decided whether `full_pairwise_accuracy.py` should compute both directions per pair or just one — needs a decision before running tomorrow.

---

## 4. FLORES-200 Kanuri pull

Not yet done — still an open, blocking item carried forward from Days 1–4. Needed before the Kanuri-Arabic × Kanuri-Latin cosine/NN-accuracy numbers can be computed (currently only the Urdu-Hindi / XNLI-based reproduction data has been run against the new metrics).

---

## 5. Reconciliation Analysis (Checkpoint 1 / Day 7 prep) — detailed

### 5.1 What actually needed reconciling

Initially framed loosely as "rerun the metrics and compare numbers against Pires." That framing doesn't work: **Pires and Del & Fishel don't measure the same kind of thing.** Pinning down exactly what each paper reports for Urdu-Hindi was necessary before any reconciliation claim could be written honestly.

### 5.2 What Pires (2019) actually reports for Urdu–Hindi

Two separate results, both favorable:

1. **POS zero-shot transfer accuracy** (Table 4): fine-tune on Hindi → eval on Urdu: **85.9%**. Fine-tune on Urdu → eval on Hindi: **91.1%**. Presented as one of the paper's headline positive findings — strong transfer despite zero lexical overlap (Devanagari vs. Arabic script).
2. **Nearest-neighbor translation accuracy** (Section 5.1–5.2, Figure 3): using WMT16 sentence pairs, mean-pooled (excluding [CLS]/[SEP]) representations, an averaged translation-offset vector is computed per language pair and *added* to the source sentence's vector before nearest-neighbor search. Urdu-Hindi's curve is reported as similar to EN-DE and EN-RU — over 50% accuracy for most layers.

**Both findings are framed as evidence of strong, well-behaved cross-lingual representation for this pair.**

### 5.3 What Del & Fishel / this reproduction found for Urdu–Hindi

From Day 4's reproduction: at layer 8, `ur-hi` CKA = **0.3080** — the single highest value *within* the isolated/outlier cluster (which also includes `hi-th`: 0.3065, `sw-th`: 0.2963), but far below the tight European main-branch cluster (`en-fr`/`en-de`/`en-es`/`fr-de`/`fr-es`/`de-es`: 0.61–0.78). I.e., by CKA's standard, Urdu-Hindi is still comparatively far apart in representational space — just the *least* far apart among the non-European pairs.

### 5.4 The actual tension, stated precisely

- **Pires:** Urdu-Hindi is a success case — strong transfer, strong NN-accuracy in the shared subspace.
- **Del & Fishel:** Urdu-Hindi's raw representational similarity (CKA) is still low relative to the main branch — closer to "isolated" than "well-aligned," even if the best of a weak group.

These are not directly contradictory *numbers* — no shared metric exists between them to literally disagree on — but they support different qualitative pictures of the same pair, and the proposal (RQ3) explicitly asks whether the choice of metric changes the conclusion drawn. That's the actual research question here, not a foregone conclusion either way.

### 5.5 The concrete, checkable hypothesis (this week's brainstorm)

**Question:** does Urdu-Hindi's *NN-accuracy* — the metric closest in spirit to what underlies Pires's transfer-success finding — rank comparatively high (closer to the main-branch pairs) even while its CKA ranks low/isolated?

- **If NN-accuracy ranks Urdu-Hindi high, CKA ranks it low** → clean reconciliation: the two metrics are capturing genuinely different structural properties (a usable shared subspace for nearest-neighbor matching can exist even when overall representational geometry, as CKA measures it, remains comparatively distant). This would directly explain why Pires and Del & Fishel drew different-sounding conclusions without either being wrong.
- **If NN-accuracy also ranks Urdu-Hindi low/isolated, matching CKA** → the tension is real and not resolved by this reproduction alone; worth reporting as-is rather than forcing a tidy story. Still a legitimate answer to RQ3 ("does choice of metric affect the conclusion" → in this case, no, both metrics agree).

Either outcome is a valid, reportable finding — the point of running it is to find out which, not to confirm a preferred narrative.

### 5.6 A methodological wrinkle to flag before comparing

Pires's NN-accuracy is **not** identical to `matching_accuracy` in this pipeline's `util.py`. Pires first computes an averaged translation-offset vector between the two languages and translates each sentence by that offset *before* nearest-neighbor search. `matching_accuracy` does raw nearest-neighbor matching on the pooled representations directly — no offset step. This is a real methodological difference, not just an implementation detail, and should be named explicitly in the reconciliation write-up if Pires's NN-accuracy figures are cited alongside this pipeline's own NN-accuracy numbers. Not yet decided whether to additionally implement the offset-based version for a stricter apples-to-apples comparison — worth deciding once the raw numbers are in hand and it's clear whether the distinction actually changes the conclusion.

### 5.7 What's still needed to actually answer the question

1. Run `full_pairwise_accuracy.py` (tomorrow) across all 8 languages, uncorrected.
2. Rank Urdu-Hindi's NN-accuracy among all 28 pairs; compare against its CKA rank (already known from Day 4).
3. Decide whether the offset-based NN-accuracy variant (matching Pires's exact method) is needed, or whether the raw version is sufficient to make the reconciliation point.
4. Draft the reconciliation paragraph (Day 7) using the actual rank comparison, not a hypothesized outcome.

---

## 6. Still outstanding (carried forward)

- [ ] FLORES-200 Kanuri splits — still not pulled (blocking Kanuri cosine/NN-accuracy numbers).
- [ ] Rewrite the Section 6 sentence flagged as lifted verbatim from the COLING 2025 source (carried from Day 1–4, still unresolved).
- [ ] Decide symmetric vs. single-direction NN-accuracy for `full_pairwise_accuracy.py`.
- [ ] Draft the OSF pre-registration skeleton — still outstanding per Day 1–4's carryover, now further behind schedule relative to the Day 13 pre-registration-finalize deadline.

---

## 7. What's next (tomorrow)

- Run `full_pairwise_accuracy.py mbert_repro` across all 8 XNLI languages.
- Check `full_pairwise_cosine.py`'s overnight output for the two required pairs (Urdu-Hindi, and once FLORES-200 Kanuri data is pulled, Kanuri-Arabic/Latin).
- Rank-compare NN-accuracy vs. CKA for Urdu-Hindi specifically; begin drafting the reconciliation paragraph with real numbers.
- Pull FLORES-200 Kanuri splits — increasingly urgent given the pre-registration deadline (Day 13) and the Day 6 token-level CKA task depends on having both pairs' data ready.
