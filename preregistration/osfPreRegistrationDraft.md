# OSF Pre-Registration

**Study title:** Isolating Tokenization Fertility from the Mean-Pooling
Confound: A Matched Subword-Regularization Test on Urdu–Hindi and Central
Kanuri Arabic–Latin Cross-Lingual Representations

**Status:** Draft — to be timestamped on OSF before Method A or Method B is
run on either language pair.

Full hypotheses (H1–H4, RQ1–RQ4) are maintained separately in
`hypotheses.md` and are incorporated here by reference; this document covers
design, analysis plan, and pre-specified decision rules only.

## 1. Design summary

Observational, representational-analysis design on frozen pretrained
encoders (mBERT, XLM-R). No fine-tuning, no new text generation, no human
subjects. Two phases:

- **Method A** — reconciliation baseline (prerequisite gate; reproduces
  Pires et al. 2019 / Del & Fishel 2021 via the xsim fork).
- **Method B** — matched subword-regularization test (core contribution).

## 2. Language pairs

- Primary: Urdu–Hindi (Indo-Aryan, South Asia).
- Secondary: Central Kanuri Arabic/Latin (Nilo-Saharan, Lake Chad Basin).
- Whether the secondary pair receives full statistical depth or a lighter
  preliminary check is decided at the Section-12 interim checkpoint based on
  remaining runway — **not** decided post-hoc from results.

## 3. Primary analysis plan

1. **Method A gate:** pass / proceed-with-discrepancy / stop. An unresolved
   reproduction is not carried into Method B.
2. **Method B, per language pair, per model (mBERT, XLM-R):**
   - Real-vs-count-matched comparison on CKA, anisotropy-corrected cosine,
     and ANC.
   - Per-layer regression, Benjamini–Hochberg FDR at α = 0.05, one family per
     (model × outcome variable × language pair).
   - Benjamini–Yekutieli correction reported as an appendix-level robustness
     check (valid under arbitrary dependence).
3. **Secondary / non-headline:** nested-model likelihood-ratio test or
   partial R² — fertility vs. subtoken-count-only fit. First item cut if
   time-constrained.

## 4. Equivalence test (TOST) — smallest effect size of interest

- **SESOI: [NOT YET SPECIFIED — decide before Method A/B is run on either
  pair].**
- This must be a concrete number, fixed before any results are seen. Options
  to choose between:
  - Fixed absolute threshold on the similarity metric's own scale (e.g., a
    CKA or cosine delta of ~0.02–0.05).
  - Standardized effect-size threshold (e.g., Cohen's d ~0.2, "small effect,"
    scaled to the observed metric's variance).
- Where the real-vs-matched difference is non-significant, TOST against this
  threshold is reported instead of treating a non-significant p-value as
  evidence of no effect.

## 5. Discard-rate and sampling-bias checks (pre-committed)

- Per-language discard rate logged for every pair run; asymmetry between
  languages reported as a limitation, not absorbed silently.
- Discard rate reported as a function of (a) corpus word-frequency quartile
  and (b) the real segmentation's subtoken count, to test for systematic
  skew toward shorter/higher-frequency retained words.
- Sampling-bias correction: adopt Cognetta et al. (2024) uniform-sampling
  (FST lattice) as the matched-control sampler, subject to a time-boxed
  feasibility check; if not implemented in time, the asymmetric
  BPE-Dropout/MaxMatch-Dropout bias is stated explicitly as a limitation.

## 6. Tokenizer viability gate (already run, reported for transparency)

- Urdu baseline: mBERT fertility 1.76 / UNK 0.13%; XLM-R fertility 1.36 / UNK
  0.00%.
- Tamasheq screened and **rejected** (mBERT UNK 69.6%, XLM-R UNK 47.1%, Llama
  3.1 8B fertility 12.71 — silent byte-fallback shredding).
- Central Kanuri screened and **passed**: XLM-R clean on both scripts
  (Arabic fertility 4.29 / UNK 0.02%; Latin fertility 2.50 / UNK 0.00%);
  mBERT usable but elevated UNK (Arabic 3.98%, Latin 7.97%) — carried forward
  as a documented, non-disqualifying confound.

## 7. Deviations policy

Any deviation from this plan after timestamping (e.g., promoting/demoting
the secondary pair's statistical depth, changing the SESOI, adopting a
fallback sampler) will be logged as an explicit amendment with rationale,
not folded into the main text silently.

## 8. Registration checklist before submission

- [ ] SESOI numerically fixed (Section 4)
- [ ] Hypotheses file (`hypotheses.md`) finalized and attached
- [ ] Method A smoke test passed
- [ ] Interim-checkpoint decision rule for secondary-pair depth documented
- [ ] Uniform-sampling feasibility check scheduled (time-boxed)
