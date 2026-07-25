# Day 1–3 Execution Log: xsim Environment Setup & Smoke Test

**Companion to:** Isolating Tokenization Fertility from the Mean-Pooling Confound (Urdu–Hindi / Serbian Cyrillic–Latin)
**Covers:** Day 1 (repo skeleton, fork xsim), Day 2 (environment/tokenizer setup), Day 3 (smoke test)
**Actual dates worked:** Jul 25–26, 2026 (compressed from the plan's Jul 19–21 window — see "Schedule Position" below)

---

## 1. Objective

Per the Day-by-Day Execution Plan, Days 1–3 required:
- Forking Del & Fishel's `xsim` codebase and getting it importable
- Resolving environment/dependency issues (mBERT/XLM-R weights, tokenizer versions pinned)
- Running the forked `xsim` pipeline end-to-end on a trivial input to confirm plumbing works, before trusting any real number

This log documents what it actually took to clear that gate — both because it's a useful record for the eventual Methods/reproducibility section, and because most of the failures below will recur on any machine (or any teammate's machine) running this stack.

---

## 2. Environment Setup — What Broke and Why

### 2.1 Conda / pip connectivity (Windows)
- `conda create -n norm python=3.8` failed with `CondaHTTPError: CONNECTION FAILED` against `repo.anaconda.com`.
- Resolved by creating the env via `conda-forge` instead of the default channel, avoiding Anaconda's CDN.
- Subsequent `pip install scipy` timed out mid-download (slow connection, ~150 kB/s). Fixed with `--default-timeout=1000 --retries 10`.

**Takeaway:** network flakiness to Anaconda/PyPI servers was environmental, not a code issue — worth checking `--default-timeout`/`--retries` first on any future slow-connection install.

### 2.2 Windows SSL certificate store crash (`ASN1: NOT_ENOUGH_DATA`)
This was the single largest time sink. Root cause: Python's `ssl.create_default_context()` crashes when Windows' certificate store contains a malformed certificate. Two expired legacy certs (**"Copyright (c) 1997 Microsoft Corp."** and **"NO LIABILITY ACCEPTED, (c)97 VeriSign..."**) were the likely culprits, though the fix that ultimately worked was a code-level bypass rather than cert-store surgery.

**Failed/complicating attempts, in order:**
1. `python-certifi-win32` — itself calls `ssl.create_default_context()` internally, so it crashed identically, and eventually caused a **fatal interpreter-startup crash** once its `.pth` autoload hook was in place.
2. Attempting `pip uninstall` while the broken package's startup hook was active — `pip` itself couldn't start (`python -S -m pip` fixed startup but hid `pip` from `sys.path`, since `-S` disables site-packages entirely).
3. Manual removal of `certifi_win32`'s package directory (since pip couldn't run) — successful.
4. Leftover `.pth` file (`python-certifi-win32-init.pth`) continued throwing a *non-fatal* import warning on every startup until manually deleted.

**Actual working fix:** a `sitecustomize.py` dropped into `Lib/site-packages/` of the `norm` conda env, monkey-patching `ssl.create_default_context` to build its context from `certifi`'s bundled CA list instead of asking Windows for the store:
```python
import ssl, certifi

def _patched_default_context(purpose=ssl.Purpose.SERVER_AUTH, *, cafile=None, capath=None, cadata=None):
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT if purpose == ssl.Purpose.SERVER_AUTH else ssl.PROTOCOL_TLS_SERVER)
    ctx.load_verify_locations(cafile=certifi.where())
    ctx.verify_mode = ssl.CERT_REQUIRED
    ctx.check_hostname = True
    return ctx

ssl.create_default_context = _patched_default_context
```
One real gotcha along the way: the file was initially saved as `sitecustomize.py.py` (double extension from the text editor) and silently never loaded — worth checking filename exactness (`dir sitecustomize*`) before assuming a `sitecustomize.py` patch isn't working.

**Verification command going forward:**
```cmd
python -c "import ssl; ctx = ssl.create_default_context(); print('OK:', ctx)"
```

### 2.3 Shell syntax mismatches (cmd.exe vs bash)
The plan and repo READMEs assume Unix shell (`mv`, `rm -rf`, `wget`, `unzip`). Anaconda Prompt runs `cmd.exe`, which has none of these. Windows 10/11's built-in `curl`/`tar` substitute cleanly:
```cmd
curl -L -o XNLI-15way.zip https://dl.fbaipublicfiles.com/XNLI/XNLI-15way.zip
tar -xf XNLI-15way.zip
move XNLI-15way xnli_15way\data
```
`mv`'s absence caused one silent data-layout bug (see 3.2 below) that wasn't caught until several steps later.

### 2.4 Stale Python 2–era dependencies
- `ordereddict` package tried `from UserDict import DictMixin` — a Python 2 module that doesn't exist in Python 3. Fixed by replacing the shim's contents with `from collections import OrderedDict`.
- Wrong `procrustes` package installed (a different PyPI package than the one `xsim` expects). Correct package: `qc-procrustes` (installs under the name `qc-procrustes`, imports as `procrustes`).
- `qc-procrustes`'s `generic.py` imported `pinv2` from `scipy.linalg`, removed in scipy ≥1.9. Fixed by importing/using `pinv` instead (functionally identical).
- `ecco` (used for CKA/CCA scoring) pins `scikit-learn~=0.23`, `transformers~=4.2`, `PyYAML~=5.4` — none buildable on modern Python (3.12, Colab). Installed with `pip install --no-deps ecco`, relying on already-installed modern `scikit-learn`/`transformers` to satisfy its actual runtime calls. Pip's resulting dependency-conflict warnings are cosmetic, not functional.

---

## 3. Data Fetch Issues

### 3.1 Empty `xnli_extension` folder
Root cause: `mv xnli_ext_repo/data xnli_extension` was run in `cmd.exe`, where `mv` doesn't exist — it failed silently at the time and wasn't noticed until the encoding script threw `FileNotFoundError` several steps later looking for `multinli.train.en.tsv`.

### 3.2 Directory-nesting mismatch (Colab)
Even after moving data with real `mv` in Colab (bash), `.tsv` files landed one level shallower than expected (`xnli_extension/*.tsv` instead of `xnli_extension/data/*.tsv`) — an `mv` semantics quirk when the destination folder already exists. Fixed with:
```bash
mkdir xnli_extension/data
mv xnli_extension/multinli.train.*.tsv xnli_extension/data/
```

**Takeaway for future data-fetch steps in this project:** always `ls`/`dir` the destination immediately after a move step, rather than assuming it landed as expected — this class of bug is easy to carry silently for several steps.

---

## 4. Code Fixes to `xsim` (Applied to Fork)

The original repo hardcodes the author's personal HPC cluster path (`/gpfs/space/home/maksym95/third-paper/saved_models/...`) instead of the public Hugging Face Hub checkpoints listed in the README. Two files needed patching:

### `examples/util.py` — `get_hf_model_ids()`
Replaced the `norm_1M` (and related) branches' local gpfs paths with a **dict** mapping model name → Hub id:
```python
def get_hf_model_ids(model_class):
    if model_class == "norm_1M":
        hf_model_ids = {
            'scale_post': 'delmaksym/aacl22.scale_post',
            'scale_pre': 'delmaksym/aacl22.scale_pre',
            'scale_normformer': 'delmaksym/aacl22.scale_normformer'
        }
    ...
```
Using a dict (rather than a list) avoids `add_model_names`'s fragile `split('/')[-2]` path-parsing, which worked for local checkpoint paths but breaks on Hub-style ids (`namespace/repo`).

Also made every `.cuda()` call device-agnostic via a module-level:
```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
applied to `load_models_tokenizers`, `compute_cosine_gpu`, `torch_corr_all`, `torch_cosines_all` — needed for CPU-only local runs; harmless on GPU (Colab) since it still resolves to `cuda` there.

### `examples/encode_dataset_with_models.py`
- Original code did `for hf_model_id in list(reversed(hf_model_ids)):` — iterating a dict yields **keys**, not values, so this passed the bare model name (`scale_normformer`) to `AutoTokenizer.from_pretrained()` instead of the full Hub id, producing a `401 Unauthorized` / `RepositoryNotFoundError`. Fixed to unpack both key and value:
  ```python
  for model_name, hf_model_id in list(reversed(list(hf_model_ids.items()))):
  ```
- `savedir` construction assumed local-path-style splitting (`hf_model_id.split('/')[-2]`); replaced with the already-available `model_name`.
- Device handling: original code built `torch.device(f"cuda:{device}")` unconditionally, which breaks for a `cpu` argument. Fixed:
  ```python
  device_obj = torch.device("cpu") if device == "cpu" else torch.device(f"cuda:{device}")
  ```

---

## 5. Result: Smoke Test Passed (Colab)

Full run of all three `norm_1M` checkpoints (`scale_post`, `scale_pre`, `scale_normformer`) against all four NormFormer-paper languages (`en`, `fr`, `et`, `bg`) completed cleanly on Colab GPU — models downloaded, tokenized, encoded, and saved to disk without error. Ends in `Finished`.

Non-fatal warnings observed (expected, not bugs):
- `UNEXPECTED`/`MISSING` key warnings when loading NormFormer-architecture checkpoints into standard `XLMRobertaModel` — expected, since the extra LayerNorm variants aren't part of the base architecture class.
- HF Hub unauthenticated rate-limit notice — cosmetic, downloads succeeded regardless.
- Dataset fingerprinting/hashing warning — affects caching dedup only, not correctness.

Local (CPU-only) replication was set up in parallel with the same code fixes, plus the CPU-device patch above — expected to be markedly slower per language/model than the GPU run but functionally equivalent.

---

## 6. Schedule Position

The plan's original Day 1–3 window was **Jul 19–21**. This work was actually done **Jul 25–26**, roughly 5–6 days behind the nominal schedule (today, Jul 25, was nominally Day 7 / Checkpoint 1).

Given the full runway to Aug 25 (arXiv) and Aug 28 (BlackboxNLP ARR) remains, this is recoverable without restructuring the plan — but Week 1's remaining items need to be compressed rather than run day-by-day as originally laid out.

**Still outstanding from Days 1–2 (not yet done):**
- [ ] Pull and verify FLORES-200 splits (Urdu/Hindi + Serbian Cyrillic/Latin) directly against the dataset — separate from the XNLI data fetched for `xsim`'s own reproduction scripts.
- [ ] Rewrite the Section 6 sentence flagged as lifted verbatim from the COLING 2025 source (blocks nothing else, but is a standing liability the longer it sits).
- [ ] Draft the OSF pre-registration skeleton (hypotheses RQ1–RQ4, primary analysis plan, TOST SESOI).

---

## 7. What's Coming Next (Day 4 onward)

### Day 4 — Method A reproduction, pass 1
- Reproduce Del & Fishel's published **Urdu/Hindi/Swahili/Thai CKA-outlier clustering**. Note: the smoke test just completed used the NormFormer paper's own language set (en/fr/et/bg) — Day 4 requires either extending `get_langs_list`/data fetch to the actual target languages, or confirming which script (`run_analysis.py` vs `run_analysis_torch_corr.py`) is the right entry point for this specific reproduction target before assuming today's run directly feeds it.
- Log any deviation from published numbers **immediately**, not at week's end.

### Day 5 — Method A reproduction, pass 2
- Extend to cosine similarity and nearest-neighbor accuracy, identically across layers/models.
- Compute Serbian-Cyrillic × Serbian-Latin equivalents to the Urdu×Hindi numbers.

### Day 6 — Token-level CKA + ANC
- Implement token-level, position-aligned CKA (no pooling step) alongside standard pooled CKA.
- Implement ANC (Del & Fishel, 2022) as a third similarity index.

### Day 7 — Checkpoint 1 (gate)
- Freeze Method A's raw outputs (pooled CKA, token-level CKA, cosine, ANC, NN-accuracy) for both pairs.
- Apply the pass / proceed-with-discrepancy / stop rule before continuing into Method B prep.

### Known carry-forward risk for Day 4+
`util.py`'s `compute_cosine_gpu`, `torch_corr_all`, and `torch_cosines_all` are now device-agnostic (see Section 4), which should mean `run_analysis.py` / `run_analysis_torch_corr.py` inherit the same CPU/GPU flexibility automatically — but this hasn't been confirmed by an actual run of those scripts yet. Worth a quick smoke test of *those* specific scripts before assuming Day 4's numbers will run cleanly on either machine.

---

## 8. Repository State

All fixes above were committed to the personal fork (not the original `TartuNLP/xsim` — forking creates an independent remote, and `git push` only affects `origin`, which points to the fork unless explicitly redirected). Commit message used:
> "Fix hardcoded gpfs paths to use public HF Hub checkpoints; make device handling CPU/GPU agnostic"

Recommended for the eventual Methods/reproducibility write-up: cite this fork + commit hash explicitly, noting the reproduction used the publicly released Hub checkpoints via a patched encoding script (same models, same metrics) rather than the original author's exact runtime environment — checkpoint fidelity, not script fidelity, is what matters for the reproduction claim.
