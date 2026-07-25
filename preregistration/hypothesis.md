# Hypotheses

Isolating Tokenization Fertility from the Mean-Pooling Confound: A Matched
Subword-Regularization Test on Urdu–Hindi and Central Kanuri Arabic–Latin
Cross-Lingual Representations

## RQ1 / H1 — Core isolation test

**RQ1:** Once the mean-pooling artifact is held constant via a count-matched
stochastic-segmentation control, does tokenization fertility independently
predict degradation in cross-lingual sentence representations produced by
frozen mBERT and XLM-R?

- **H1 (alternative):** Real-tokenization representations show significantly
  greater cross-lingual degradation (lower CKA / anisotropy-corrected cosine /
  NN-accuracy) than count-matched alternative-segmentation representations,
  after controlling for subtoken count.
- **H1 (null):** No significant difference between real and count-matched
  representations once subtoken count is held constant — degradation
  previously attributed to fertility is fully explained by the pooling
  artifact.
- **Equivalence framing:** Where the real-vs-matched difference is
  non-significant, this is evaluated via TOST equivalence testing against a
  pre-specified SESOI (see `osf_preregistration.md`), not treated as
  automatic evidence of no effect.

## RQ2 / H2 — Generalization across script-divergence type

**RQ2:** Does the RQ1 relationship hold consistently across two structurally
different forms of script divergence (Urdu–Hindi vs. Central Kanuri
Arabic/Latin), or does it diverge between them?

- **H2 (alternative):** The fertility effect (if present under H1) is
  consistent in direction and comparable in magnitude across both language
  pairs.
- **H2 (null):** The effect differs meaningfully between pairs — treated as a
  finding in its own right, not noise.
- **Reporting constraint:** Because mBERT shows an elevated UNK rate on
  Central Kanuri relative to its Urdu baseline (3.98% Arabic / 7.97% Latin vs.
  0.13% Urdu), RQ2 is answered **separately per model**, not pooled. The
  XLM-R comparison (UNK-clean on both pairs) is the cleaner test; mBERT-on-
  Kanuri results are reported as lower-confidence.

## RQ3 / H3 — Reconciliation baseline (Method A, prerequisite)

**RQ3:** Do previously reported cross-lingual similarity findings for
Urdu–Hindi (Pires et al., 2019; Del & Fishel, 2021) reproduce under
independent re-implementation, and does similarity metric choice (CKA,
cosine, ANC) affect the conclusion?

- **H3a:** The Pires et al. and Del & Fishel results are reconcilable once
  metric and pooling choices are matched.
- **H3b:** The conclusion is sensitive to metric choice (i.e., CKA, cosine,
  and ANC do not agree).
- **Gate condition:** An unresolved reproduction under H3 is **not** carried
  forward into Method B — this is a pass/proceed-with-discrepancy/stop
  checkpoint, not a soft prior.

## RQ4 / H4 — Marginal contribution of fertility (secondary, non-headline)

**RQ4:** To what extent does fertility improve statistical fit over subtoken
count alone, once both are modeled jointly against representational
similarity?

- **H4 (alternative):** A model including fertility as a predictor fits
  significantly better (likelihood-ratio test / partial R²) than subtoken
  count alone.
- **H4 (null):** Fertility adds no significant explanatory power beyond raw
  subtoken count.
- **Status:** Explicitly labeled secondary; first item dropped if project
  time runs short.

## Notes on interpretation ordering

1. H3 (Method A) must clear before H1/H2 (Method B) are interpreted.
2. H1 is the headline result; H2 qualifies its generalizability; H4 is
   supplementary and expendable under time pressure.
3. All hypotheses above are fixed prior to running Method A or Method B on
   either language pair, per the OSF pre-registration timestamp.
