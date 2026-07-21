# Day 1 Report — FLORES-200 Data Pipeline Setup

**Project:** Cross-Lingual Reconciliation (Paper A / Paper B)
**Date:** July 22, 2026

## Objective

Get `fetchFLORES200.py` running end-to-end to pull the FLORES-200 corpus (`facebook/flores` on Hugging Face) for the reconciliation study's language set.

## Timeline of Issues, Diagnoses, and Fixes

### 1. Missing `datasets` package

- **Problem:** `ModuleNotFoundError: No module named 'datasets'`
- **Fix:** Installed the Hugging Face `datasets` library.

### 2. Gated dataset access

- **Problem:** `facebook/flores` is a gated dataset. Script sent unauthenticated requests and failed with "You must be authenticated to access it."
- **Cause:** Two things both needed to be true: (a) the dataset's terms had to be accepted on the HF account, and (b) the script needed to actually pass a token, not just have one saved locally.
- **Fix:**
  - Accepted the gating terms on the `facebook/flores` dataset page (screenshot reviewed: evaluation-only use, no ML training, redistribution-gating requirements).
  - Ran `huggingface-cli login` with a token.
  - **Root cause of a later repeat failure:** the `load_dataset(...)` call in the script never passed a `token` argument, so the CLI login wasn't actually being used by the script. Fixed by adding `token=True` to the call.

### 3. `trust_remote_code` no longer supported

- **Problem:** `datasets` v4.0.0+ completely removed dataset loading-script support — no flag can re-enable it. `facebook/flores` is script-based (not plain Parquet), so it's incompatible with `datasets>=4.0.0` regardless of authentication.
- **Fix (Option A, chosen):** Downgraded to `datasets<4.0.0`, kept `trust_remote_code=True` in the call.
- **Alternative considered (Option B, not taken):** Switch to `openlanguagedata/flores_plus`, the Parquet-based FLORES successor with no loading script. Left as a fallback if further script-based issues resurface.

### 4. Pickling crash after downgrade

- **Problem:** `Pickler._batch_setitems() takes 2 positional arguments but 3 were given`
- **Cause:** Version mismatch between `dill` (used internally by `datasets` for caching) and a newer Python interpreter (3.13/3.14) whose internal `pickle` API changed. This is a known, reported issue — not specific to this project.
- **Fix:** Recommended running in a Python 3.10–3.12 environment (e.g. a dedicated conda env) rather than the newer interpreter, since `dill` hasn't caught up to the newest Python releases.
- **Status:** Ran successfully — three of four language configs downloaded (dev + devtest).

### 5. Invalid language config: `srp_Latn`

- **Problem:** `BuilderConfig 'srp_Latn' not found`
- **Cause:** FLORES-200 only includes Serbian in Cyrillic script (`srp_Cyrl`). There is no Latin-script Serbian config in this dataset, despite Serbian being officially digraphic in reality.
- **Status:** **Unresolved — open design question, not just an engineering bug.** The current `LANG_CODES` dict has:
  ```python
  LANG_CODES = {
      "urdu": "urd_Arab",
      "hindi": "hin_Deva",
      "serbian_cyrillic": "srp_Cyrl",
      "serbian_latin": "srp_Latn",   # does not exist in FLORES-200
  }
  ```
  A substitute (e.g. Tamasheq `taq_Latn` / `taq_Tfng`, the one same-language pair in FLORES-200 with two scripts available) was floated as a possible drop-in, but **this is a methodology decision, not something to resolve unilaterally** — it depends on what role the Serbian pair was playing in the 2×2 script-typology design (a second same-language/different-script replicate vs. some other cell).

## Carried Over to Tomorrow

- **Decide the fix for the missing `srp_Latn` config.** Needs to be worked out with reference to the full 2×2 script-typology matrix and what Serbian was meant to test — not just swapped for the first available substitute.
- Once resolved, rerun `fetchFLORES200.py` for the corrected/final language set and confirm `checksums.txt` and `manifest.json` are fully populated (currently `checksums.txt` had 0 entries).
- Worth flagging to Dr. Zeeshan given this touches the experimental design of Paper B, not just data engineering.

## Net Outcome

By end of day: authentication, gating, and environment issues are all resolved and the pipeline runs cleanly for valid configs. The only remaining blocker is a genuine data-availability gap for one of the four planned language variants, to be resolved as a design decision before rerunning.
