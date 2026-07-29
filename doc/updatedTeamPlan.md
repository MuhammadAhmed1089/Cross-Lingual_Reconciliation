# 🎯 Nastaliq Cross-Lingual Reconciliation — Execution Plan v3

_Isolating Tokenization Fertility from the Mean-Pooling Confound (Urdu–Hindi / Central Kanuri Arabic–Latin)_
_Reconciled with Research Proposal v6 · Restructured into weekly deliverables_

![Window](https://img.shields.io/badge/Window-Jul%2019%20→%20Aug%2025%2C%202026-1e293b?style=flat-square) ![Buffer](https://img.shields.io/badge/Buffer-Aug%2026–30-64748b?style=flat-square) ![Target](https://img.shields.io/badge/Targets-arXiv%20%2B%20BlackboxNLP%20ARR-16a34a?style=flat-square)

---

### 🗝️ Legend

| Badge                                                                          | Owner          | Meaning                        |
| ------------------------------------------------------------------------------ | -------------- | ------------------------------ |
| ![YOU](https://img.shields.io/badge/-YOU-2563eb?style=for-the-badge)           | Muhammad       | Deliverables assigned to you   |
| ![AHMAD](https://img.shields.io/badge/-AHMAD-16a34a?style=for-the-badge)       | Ahmad Ali Khan | Deliverables assigned to Ahmad |
| ![COMBINED](https://img.shields.io/badge/-COMBINED-9333ea?style=for-the-badge) | Both           | Joint checkpoint / gate        |

> Each week = **5 deliverables** → 2 🔵 You, 2 🟢 Ahmad, 1 🟣 Combined checkpoint.
> Swap names/ownership any time — the structure doesn't care who's assigned where.

---

## ⚠️ Immediate Action — Before Week 1 Begins

> 🔴 **Blocks nothing, but sits as a standing liability until closed.**

- [ ] Rewrite the Section 6 sentence flagged as lifted from COLING 2025
- [ ] Re-read the surrounding paragraph for any other close paraphrase and fix it too

---

## 📅 Week 1 — Setup, Pre-Registration, Method A Launch

### ![YOU](https://img.shields.io/badge/-YOU-2563eb?style=for-the-badge) Deliverable 1 — Lit Integrity & Pre-Registration Draft

**🎯 Goal:** Close the paraphrase liability; get a pre-reg skeleton ready to finalize later.

| #   | Step                                                                                                                                              |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Finish the COLING-2025 rewrite (carry-over from Immediate Action if not done)                                                                     |
| 2   | Resolve xsim's environment/dependency issues — pin mBERT/XLM-R weights + tokenizer versions                                                       |
| 3   | Draft OSF pre-registration skeleton: **(a)** RQ1–RQ3 written in full **(b)** primary analysis plan **(c)** a concrete TOST SESOI number — not TBD |
| 4   | Confirm only **three** RQs remain — v6 cut the old RQ4 (nested fertility-vs-subtoken-count fit)                                                   |

✅ **Done when:** environment builds clean + OSF-ready skeleton exists with RQ1–RQ3, analysis plan, and a real TOST number.

---

### ![YOU](https://img.shields.io/badge/-YOU-2563eb?style=for-the-badge) Deliverable 2 — Method A Reproduction

**🎯 Goal:** Confirm Del & Fishel's numbers reproduce before trusting anything built on top.

| #   | Step                                                                                         |
| --- | -------------------------------------------------------------------------------------------- |
| 1   | Reproduce Del & Fishel's Urdu/Hindi/Swahili/Thai CKA-outlier clustering exactly as published |
| 2   | Log any deviation **immediately** — don't wait for end of week                               |
| 3   | Extend to anisotropy-corrected cosine + NN-accuracy, identically across layers & both models |
| 4   | Compute Kanuri-Arabic × Kanuri-Latin equivalents of every Urdu × Hindi number                |

✅ **Done when:** side-by-side table of CKA / cosine / NN-accuracy exists for both pairs, deviations explicitly noted.

---

### ![AHMAD](https://img.shields.io/badge/-AHMAD-16a34a?style=for-the-badge) Deliverable 3 — Environment, Data & Pipeline Verification

**🎯 Goal:** Build the scaffolding and _prove_ it runs before Method A depends on it.

| #   | Step                                                                                                                                   |
| --- | -------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Create repo skeleton: `code/` `data/` `results/` `preregistration/`                                                                    |
| 2   | Pull FLORES-200/FLORES+; verify Urdu/Hindi + Kanuri (`knc_Arab`/`knc_Latn`) split sizes **by direct inspection**, not the dataset card |
| 3   | Fork Del & Fishel's xsim codebase; confirm clean import                                                                                |
| 4   | Run xsim end-to-end on a trivial toy input                                                                                             |
| 5   | Fix any plumbing issues the smoke test surfaces                                                                                        |

✅ **Done when:** repo exists, FLORES splits verified directly, toy-input smoke test runs error-free.

---

### ![AHMAD](https://img.shields.io/badge/-AHMAD-16a34a?style=for-the-badge) Deliverable 4 — Token-Level CKA

**🎯 Goal:** Test whether pooled CKA hides a length/fertility sensitivity.

| #   | Step                                                                 |
| --- | -------------------------------------------------------------------- |
| 1   | Implement token-level, position-aligned CKA (no mean-pooling)        |
| 2   | Run alongside standard pooled CKA for direct comparison              |
| 3   | 🚫 Do **not** implement ANC — future work only (proposal §6.3/7.2/8) |

✅ **Done when:** token-level CKA computed for both pairs, ready to compare against pooled CKA (Deliverable 2).

---

### ![COMBINED](https://img.shields.io/badge/-COMBINED-9333ea?style=for-the-badge) Deliverable 5 — 🚦 Checkpoint 1: Freeze & Reconciliation

| #   | Step                                                                                                                                    |
| --- | --------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Freeze final Method A outputs: pooled CKA (primary), token-level CKA, anisotropy-corrected cosine (secondary), NN-accuracy — both pairs |
| 2   | Write plain-language answer: does repro resolve Pires vs. Del & Fishel? Does token-level CKA show the Mitra & Kumar (2026) sensitivity? |
| 3   | Apply the gate — see below                                                                                                              |

> **🚦 Gate rule**
> 🟢 **Clean pass** → straight into Week 2
> 🟡 **Explainable discrepancy** (version drift, metric definition) → proceed, document why it's explainable
> 🔴 **Unexplained failure** → **STOP.** Do not start Method B with an unresolved reproduction.

✅ **Done when:** outputs frozen in a results file, write-up exists, gate outcome explicitly agreed by both.

---

## 📅 Week 2 — Method A Close-Out, Method B Prep

### ![YOU](https://img.shields.io/badge/-YOU-2563eb?style=for-the-badge) Deliverable 1 — Method A Write-Up & Kanuri Depth Confirmation

| #   | Step                                                                                                               |
| --- | ------------------------------------------------------------------------------------------------------------------ |
| 1   | Write Method A section: pooled CKA, token-level CKA, cosine, NN-accuracy side by side, both pairs                  |
| 2   | Confirm pre-reg draft already reflects Kanuri's pre-committed lighter depth — **confirmation, not a new decision** |
| 3   | Add the mBERT-on-Kanuri UNK-rate caveat (§7.1.1) into the write-up explicitly                                      |

✅ **Done when:** Method A section written coherently; UNK-rate caveat + Kanuri depth both documented.

---

### ![YOU](https://img.shields.io/badge/-YOU-2563eb?style=for-the-badge) Deliverable 2 — Pre-Registration Finalization

| #   | Step                                                                              |
| --- | --------------------------------------------------------------------------------- |
| 1   | Finalize RQ1–RQ3 wording + primary analysis plan from Week 1 skeleton             |
| 2   | Lock in writing: CKA primary, cosine secondary, BH FDR correction, TOST threshold |
| 3   | Timestamp on OSF — **must** happen before any Method B result exists              |

✅ **Done when:** pre-registration is timestamped and public on OSF, no post-hoc analysis choices.

---

### ![AHMAD](https://img.shields.io/badge/-AHMAD-16a34a?style=for-the-badge) Deliverable 3 — Tokenizer Sampling Infrastructure

| #   | Step                                                                                          |
| --- | --------------------------------------------------------------------------------------------- |
| 1   | Enable SentencePiece sampling mode (BPE-dropout, alpha) on XLM-R                              |
| 2   | Manually inspect sampled segmentations on real Urdu/Hindi/Kanuri (Arabic/Ajami + Latin) words |
| 3   | Get Hiraoka's MaxMatch-Dropout running against mBERT's WordPiece vocab                        |
| 4   | Treat step 3 as a small research codebase — budget real adaptation time                       |

✅ **Done when:** both tokenizers produce sampled alternatives on demand, real-word examples visually confirmed correct.

---

### ![AHMAD](https://img.shields.io/badge/-AHMAD-16a34a?style=for-the-badge) Deliverable 4 — Sampling-Bias Feasibility Check

| #   | Step                                                                                                                 |
| --- | -------------------------------------------------------------------------------------------------------------------- |
| 1   | Time-box: can Cognetta et al.'s uniform-sampling (FST lattice) be implemented for both tokenizers in time remaining? |
| 2   | **If yes** → begin implementation                                                                                    |
| 3   | **If no** → draft explicit limitation language for the asymmetric sampling-bias fallback                             |

✅ **Done when:** either a working implementation exists, or a clear limitation paragraph is written — no open question left hanging.

---

### ![COMBINED](https://img.shields.io/badge/-COMBINED-9333ea?style=for-the-badge) Deliverable 5 — 🚦 Checkpoint 2

| #   | Step                                                                                |
| --- | ----------------------------------------------------------------------------------- |
| 1   | Confirm both tokenizer sampling mechanisms (or replacements) are verified and ready |
| 2   | Confirm pre-registration is live and timestamped on OSF                             |
| 3   | Only once **both** are true → begin the Week 3 count-matching pilot                 |

✅ **Done when:** both conditions checked off together.

---

## 📅 Week 3 — Method B Pilot and Scale-Up

### ![AHMAD](https://img.shields.io/badge/-AHMAD-16a34a?style=for-the-badge) Deliverable 1 — Count-Matching Pilot: Setup & Run

| #   | Step                                                                                                                |
| --- | ------------------------------------------------------------------------------------------------------------------- |
| 1   | Implement count-matching: record default split → sample alternative → resample-or-discard on mismatched piece count |
| 2   | Pilot on a few hundred sentence pairs, per language pair                                                            |
| 3   | Log per-language discard rate **immediately**, as the pilot runs                                                    |

✅ **Done when:** pilot-scale matched-control data + logged discard rate exist for both pairs.

---

### ![AHMAD](https://img.shields.io/badge/-AHMAD-16a34a?style=for-the-badge) Deliverable 2 — Pilot Review

| #   | Step                                                                                                   |
| --- | ------------------------------------------------------------------------------------------------------ |
| 1   | Check discard-rate asymmetry between languages before scaling (§7.3.4 failure mode)                    |
| 2   | Break discard rate down by word-frequency quartile **and** real subtoken count, not just the aggregate |
| 3   | Fix any systematic resampling issue found                                                              |

✅ **Done when:** both aggregate and breakdown checked; any systematic issue fixed, not just flagged.

---

### ![YOU](https://img.shields.io/badge/-YOU-2563eb?style=for-the-badge) Deliverable 3 — Scale-Up: Urdu–Hindi

| #   | Step                                                                              |
| --- | --------------------------------------------------------------------------------- |
| 1   | Run matched-control procedure at full scale for Urdu–Hindi                        |
| 2   | Compute real vs. count-matched-alternative mean-pooled representations            |
| 3   | Compute CKA (primary) and cosine (secondary) between real and matched-alternative |

✅ **Done when:** complete full-scale real-vs-matched dataset exists for Urdu–Hindi with both metrics.

---

### ![YOU](https://img.shields.io/badge/-YOU-2563eb?style=for-the-badge) Deliverable 4 — Scale-Up: Kanuri & Discard Audit

| #   | Step                                                                                            |
| --- | ----------------------------------------------------------------------------------------------- |
| 1   | Run same procedure for Central Kanuri Arabic/Latin — pre-committed **preliminary depth only**   |
| 2   | Finalize per-language discard-rate logs, both pairs, including the frequency/subtoken breakdown |
| 3   | Write up any asymmetry/skew as an explicit limitation — not raw numbers with no interpretation  |

✅ **Done when:** Kanuri data exists at agreed depth; discard-rate write-up drafted as limitation text.

---

### ![COMBINED](https://img.shields.io/badge/-COMBINED-9333ea?style=for-the-badge) Deliverable 5 — 🚦 Checkpoint 3

| #   | Step                                                                                           |
| --- | ---------------------------------------------------------------------------------------------- |
| 1   | Confirm both pairs have complete real-vs-matched data, ready for statistical analysis          |
| 2   | If Urdu–Hindi is behind: trim **Kanuri depth further first** — never Urdu–Hindi (primary pair) |

✅ **Done when:** both datasets confirmed complete; fallback order agreed if either is behind.

---

## 📅 Week 4 — Statistical Analysis

### ![YOU](https://img.shields.io/badge/-YOU-2563eb?style=for-the-badge) Deliverable 1 — Primary Regression, BH-FDR & MRL Stretch

| #   | Step                                                                                                                                |
| --- | ----------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Run per-layer regressions of CKA (primary) on real vs. matched-alternative tokenization                                             |
| 2   | Apply Benjamini–Hochberg FDR at α = 0.05 — **sole** correction procedure, one family per (model × outcome × pair)                   |
| 3   | Urdu–Hindi → full formal inference. Kanuri → descriptive only                                                                       |
| 4   | Check MRL Workshop @ EMNLP deadline (this week, **stretch only**) — submit short paper only if Method A alone is publication-shaped |

✅ **Done when:** regression tables exist with correct BH-FDR application; MRL decision explicitly made.

---

### ![YOU](https://img.shields.io/badge/-YOU-2563eb?style=for-the-badge) Deliverable 2 — TOST Equivalence & Cross-Metric Consistency

| #   | Step                                                                                                 |
| --- | ---------------------------------------------------------------------------------------------------- |
| 1   | Run pre-specified TOST equivalence on every non-significant real-vs-matched comparison, primary pair |
| 2   | 🚫 Do **not** run Benjamini–Yekutieli robustness check — dropped in v6                               |
| 3   | Compare CKA (primary) vs. cosine (secondary) direction/significance across both pairs                |
| 4   | Write down any metric disagreement explicitly — this **is** the answer to RQ3                        |

✅ **Done when:** TOST results complete; explicit written answer on CKA/cosine agreement for both pairs.

---

### ![AHMAD](https://img.shields.io/badge/-AHMAD-16a34a?style=for-the-badge) Deliverable 3 — Regression Review / Buffer

| #   | Step                                                                                                      |
| --- | --------------------------------------------------------------------------------------------------------- |
| 1   | Review Deliverable 1's regression output for completeness — every (model × outcome × pair) family covered |
| 2   | 🚫 Nested-model test stays cut entirely in v6 — do not run it even if it seems like an obvious next step  |
| 3   | If Week 3 ran long, use this purely as catch-up buffer                                                    |

✅ **Done when:** regression output reviewed and gaps flagged, or buffer used to close Week 3 backlog.

---

### ![AHMAD](https://img.shields.io/badge/-AHMAD-16a34a?style=for-the-badge) Deliverable 4 — Results Consolidation

| #   | Step                                                                  |
| --- | --------------------------------------------------------------------- |
| 1   | Produce final per-layer degradation curves (figures)                  |
| 2   | Produce final discard-rate tables, incl. frequency/subtoken breakdown |
| 3   | Write RQ-by-RQ answer summary — one direct statement per RQ           |

✅ **Done when:** all figures/tables final; each RQ has a one-paragraph direct answer.

---

### ![COMBINED](https://img.shields.io/badge/-COMBINED-9333ea?style=for-the-badge) Deliverable 5 — 🚦 Checkpoint 4

| #   | Step                                                                                    |
| --- | --------------------------------------------------------------------------------------- |
| 1   | Confirm every RQ (1–3) has a data-backed answer                                         |
| 2   | Identify which of the three pre-drafted write-ups (§9) matches actual outcome           |
| 3   | Lock it in — **no further analysis choices** after seeing which write-up "looks better" |

✅ **Done when:** all RQs answered, matching write-up identified, further analysis choices explicitly frozen.

---

## 📅 Week 5 — Writing

### ![YOU](https://img.shields.io/badge/-YOU-2563eb?style=for-the-badge) Deliverable 1 — Draft: Methods & Results

| #   | Step                                                                                 |
| --- | ------------------------------------------------------------------------------------ |
| 1   | Write Methods A & B directly from pre-registration + analysis logs — not from memory |
| 2   | Write Results directly against Week 4's consolidated tables/figures                  |

✅ **Done when:** both sections drafted, every claim traceable to a logged result.

---

### ![YOU](https://img.shields.io/badge/-YOU-2563eb?style=for-the-badge) Deliverable 2 — Draft: Related Work & Discussion

| #   | Step                                                                                                                                        |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Related Work: fold in DEPART + Mitra & Kumar (corroborating), Del & Fishel open question, ANC/nested-model/BY as **documented future work** |
| 2   | Discussion: tie results to RQ1–RQ3 + the two-pair generalization question                                                                   |

✅ **Done when:** Related Work positions all four threads correctly; Discussion addresses generalization.

---

### ![AHMAD](https://img.shields.io/badge/-AHMAD-16a34a?style=for-the-badge) Deliverable 3 — Draft: Limitations, Ethics & Consistency Pass

| #   | Step                                                                                           |
| --- | ---------------------------------------------------------------------------------------------- |
| 1   | Write Limitations (§10) incl. mBERT-on-Kanuri UNK-rate caveat + discard-sample-skew limitation |
| 2   | Write Ethical Considerations, adapted near-verbatim from proposal                              |
| 3   | Full read-through: flag any claim that overstates or contradicts the actual data               |

✅ **Done when:** both sections exist; consistency pass done with mismatches flagged back to the relevant author.

---

### ![AHMAD](https://img.shields.io/badge/-AHMAD-16a34a?style=for-the-badge) Deliverable 4 — Internal Review & Artifact Finalization

| #   | Step                                                                                     |
| --- | ---------------------------------------------------------------------------------------- |
| 1   | Send full draft to Dr. Zeeshan Ali Rana (+ other readers) for feedback                   |
| 2   | Run a plagiarism/similarity check on the **entire** draft — not just the Week 1 sentence |
| 3   | Incorporate feedback once received                                                       |
| 4   | Finalize code repo, OSF artifacts, result CSV/JSON releases (§13)                        |

✅ **Done when:** feedback incorporated, similarity check clean, artifacts finalized and release-ready.

---

### ![COMBINED](https://img.shields.io/badge/-COMBINED-9333ea?style=for-the-badge) Deliverable 5 — 🚦 Checkpoint 5

| #   | Step                                                                                  |
| --- | ------------------------------------------------------------------------------------- |
| 1   | Confirm full draft is feature-complete and internally reviewed                        |
| 2   | Confirm all deliverables (code repo, OSF pre-reg, result artifacts) are release-ready |

✅ **Done when:** both true — remaining work is polish/submission logistics only, no new analysis or writing.

---

## 🏁 Final Stretch — Submission (Days 36–38 + Buffer Aug 26–30)

### ![YOU](https://img.shields.io/badge/-YOU-2563eb?style=for-the-badge) Deliverable 1 — Polish

| #   | Step                                                                       |
| --- | -------------------------------------------------------------------------- |
| 1   | Copyedit the full draft                                                    |
| 2   | Check every citation against §14 reference list                            |
| 3   | Verify figure/table numbering is sequential and matches in-text references |

✅ **Done when:** draft copyedited, citations + numbering verified.

---

### ![YOU](https://img.shields.io/badge/-YOU-2563eb?style=for-the-badge) Deliverable 2 — arXiv & ARR Submission

| #   | Step                                                                                    |
| --- | --------------------------------------------------------------------------------------- |
| 1   | Format to arXiv template and submit (closes 38-day window)                              |
| 2   | Aug 26–27: adapt to BlackboxNLP/ARR formatting — reformatting only, **no new analysis** |
| 3   | Aug 28: submit to BlackboxNLP via ARR                                                   |

✅ **Done when:** both submissions confirmed received.

---

### ![AHMAD](https://img.shields.io/badge/-AHMAD-16a34a?style=for-the-badge) Deliverable 3 — Formatting & Public Release

| #   | Step                                                               |
| --- | ------------------------------------------------------------------ |
| 1   | Format paper to arXiv/venue template                               |
| 2   | Publish code repo + OSF artifacts publicly                         |
| 3   | Manually click through every published link to confirm it resolves |

✅ **Done when:** repo/OSF public, every link manually verified.

---

### ![AHMAD](https://img.shields.io/badge/-AHMAD-16a34a?style=for-the-badge) Deliverable 4 — Buffer Slack

| #   | Step                                                                                        |
| --- | ------------------------------------------------------------------------------------------- |
| 1   | Aug 29–30: reserved **only** for genuinely unresolved technical issues that slipped through |

✅ **Done when:** either unused, or a specific slipped issue is resolved.

---

### ![COMBINED](https://img.shields.io/badge/-COMBINED-9333ea?style=for-the-badge) Deliverable 5 — 🚦 Final Sanity Check

| #   | Step                                                                       |
| --- | -------------------------------------------------------------------------- |
| 1   | Confirm both arXiv and ARR submissions went through                        |
| 2   | Re-click every public link one final time                                  |
| 3   | Compare draft claims against actually-released artifacts for discrepancies |

✅ **Done when:** both submissions live, all links resolve, no discrepancy found.

---

## 🛟 Standing Contingencies _(apply at any checkpoint)_

> **Collaborator-fallback cut order**, if a checkpoint is missed and no capacity exists to catch up:
> 1️⃣ TOST reporting depth first (go descriptive)
> 2️⃣ Kanuri statistical depth second (trim further toward descriptive)
> 3️⃣ Dashboard deliverable last (already optional)
>
> 🚫 ANC, Benjamini–Yekutieli, and the nested-model test are **already out of scope** in v6 — not part of this cut order.
>
> **If Urdu–Hindi reproduction fails at Checkpoint 1:** promote Kanuri to primary-pair statistical depth; report Urdu–Hindi as the secondary, caveated pair.
