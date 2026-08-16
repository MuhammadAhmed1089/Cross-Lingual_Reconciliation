"""
build_alignments.py

Part 1 of the token-level CKA protocol: produces word-index alignment pairs
and saves them to disk. This is a STANDALONE preprocessing pass — it does not
touch your encoding pipeline or CKA orchestrator. Its only output is JSON
alignment files consumed later by token_level_cka.py (Step 6).

Covers:
  Step 1: separate aligner environment/script (this file)
  Step 2: transliteration-assisted input for Urdu-Hindi
  Step 3: run SimAlign on transliterated pairs
  Step 4: pilot-before-scaling (mandatory ~20-50 sentence check)
  Step 5: Kanuri-Arab/Latin positional matching (no aligner)
  Step 6: save alignments to disk, keyed by sentence index

Install (separate env recommended):
    pip install simalign indic-transliteration
    # or, if you prefer awesome-align, see the ALIGNER_BACKEND note below.

Usage:
    # 1. Pilot first (mandatory) -- inspect output by eye before scaling
    python build_alignments.py pilot --pair urd_hin --split dev --n 30

    # 2. Full run after pilot looks sane
    python build_alignments.py run --pair urd_hin --split dev
    python build_alignments.py run --pair urd_hin --split devtest

    # 3. Kanuri: positional pilot + check, then full run
    python build_alignments.py pilot --pair knc --split dev --n 30
    python build_alignments.py run --pair knc --split dev
"""

import argparse
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Config -- adjust paths to match your FLORES-200 layout
# ---------------------------------------------------------------------------

FLORES_ROOT = os.environ.get("FLORES_ROOT", "../../data/splits/original")
ALIGN_OUT_DIR = Path("alignments")
ALIGN_OUT_DIR.mkdir(exist_ok=True)

LANG_CODE_MAP = {
    "urd_hin": {"a": "urd_Arab", "b": "hin_Deva"},
    "knc": {"a": "knc_Arab", "b": "knc_Latn"},
}

# Which backbone the aligner uses for its own internal similarity scoring.
# This is independent of the mBERT/XLM-R backbone you use later for CKA --
# it's just what SimAlign uses to produce alignments.
ALIGNER_BACKBONE = os.environ.get("ALIGNER_BACKBONE", "bert")  # simalign model key


def read_flores_split(lang_code: str, split: str):
    """
    Reads a FLORES-200 split file for a given language code.
    split is one of: 'dev', 'devtest'.
    Expects one sentence per line, matching FLORES-200's standard layout:
        {FLORES_ROOT}/{split}/{lang_code}.{split}
    Adjust this if your local copy uses a different naming convention.
    """
    path = Path(FLORES_ROOT) / f"{lang_code}.{split}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Set FLORES_ROOT env var or edit read_flores_split()."
        )
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


# ---------------------------------------------------------------------------
# Step 2: transliteration-assisted input (Urdu-Hindi only)
# ---------------------------------------------------------------------------

def transliterate_urdu_to_devanagari(sentences):
    """
    Transliterates Urdu (Arabic script) sentences into Devanagari so the
    aligner sees two scripts it can meaningfully compare.

    Two viable approaches -- pick ONE and pilot it (per Step 2):
      (a) indic-transliteration: Urdu -> Devanagari via ITRANS/Sanscript
      (b) uroman: both Urdu and Hindi -> shared Roman scheme

    This implementation uses approach (a). Swap in uroman if your pilot
    (Step 4) shows better alignment quality with the Roman-scheme route.
    """
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate
    except ImportError:
        raise ImportError(
            "pip install indic-transliteration  (or switch to uroman -- see docstring)"
        )

    transliterated = []
    for sent in sentences:
        # Urdu is Perso-Arabic script; there isn't a single universally-agreed
        # scheme name in sanscript for Urdu specifically, so most pipelines
        # roman-ize Urdu first (via a light Arabic->Roman mapping or uroman)
        # and then go Roman -> Devanagari, OR go straight uroman for both.
        # If this raises / produces garbage on your data, switch to the
        # uroman path below -- flagged explicitly so you don't debug this
        # silently mid-pipeline.
        try:
            translit = transliterate(sent, sanscript.URDU, sanscript.DEVANAGARI)
        except Exception as e:
            raise RuntimeError(
                f"Urdu->Devanagari transliteration failed on: {sent!r}\n"
                f"Consider switching to the uroman shared-Roman-scheme approach.\n"
                f"Original error: {e}"
            )
        transliterated.append(translit)
    return transliterated


def transliterate_both_to_roman(urdu_sentences, hindi_sentences):
    """
    Alternative to transliterate_urdu_to_devanagari(): uroman both languages
    into a shared Roman scheme. Requires the `uroman` package or CLI.
    """
    try:
        import uroman as ur
    except ImportError:
        raise ImportError("pip install uroman")

    uroman = ur.Uroman()
    urdu_roman = [uroman.romanize_string(s) for s in urdu_sentences]
    hindi_roman = [uroman.romanize_string(s) for s in hindi_sentences]
    return urdu_roman, hindi_roman


# ---------------------------------------------------------------------------
# Step 3: run the aligner (SimAlign) on transliterated pairs
# ---------------------------------------------------------------------------

def get_aligner():
    """
    Lazily builds a SimAlign aligner. Swap model="bert" for "xlmr" etc. to
    match whichever backbone your pilot (Step 4) selects.

    If you prefer awesome-align instead of SimAlign, it's CLI/script-based
    rather than a Python object -- wrap its subprocess call here and keep
    the same get_word_alignments() call signature so the rest of this file
    doesn't need to change.
    """
    try:
        from simalign import SentenceAligner
    except ImportError:
        raise ImportError("pip install simalign")

    return SentenceAligner(model=ALIGNER_BACKBONE, token_type="bpe", matching_methods="mai")


def get_word_alignments(aligner, src_sent: str, tgt_sent: str):
    """
    Runs the aligner on one sentence pair.
    Returns a list of [src_word_idx, tgt_word_idx] pairs (word positions,
    not text), per Step 3.
    """
    alignments = aligner.get_word_aligns(src_sent.split(), tgt_sent.split())
    # SimAlign returns a dict keyed by matching method (mai/inter/itermax);
    # "mai" (match-align-intersect) is a reasonable default -- change if
    # your pilot inspection (Step 4) shows another method aligns better.
    pairs = alignments.get("mai", [])
    return [[int(a), int(b)] for a, b in pairs]


# ---------------------------------------------------------------------------
# Step 5: Kanuri-Arab/Latin positional matching (no aligner needed)
# ---------------------------------------------------------------------------

def check_positional_word_counts(arab_sentences, latin_sentences, n=None):
    """
    Step 5 pilot check: compares per-sentence whitespace-split word counts
    between the two Kanuri scripts. Returns a list of dicts with per-sentence
    diagnostics so you can eyeball where (if anywhere) counts diverge.
    """
    n = n or len(arab_sentences)
    results = []
    mismatches = 0
    for i, (a, l) in enumerate(zip(arab_sentences[:n], latin_sentences[:n])):
        a_words = a.split()
        l_words = l.split()
        match = len(a_words) == len(l_words)
        if not match:
            mismatches += 1
        results.append({
            "sent_idx": i,
            "arab_word_count": len(a_words),
            "latin_word_count": len(l_words),
            "match": match,
        })
    mismatch_rate = mismatches / n if n else 0.0
    return results, mismatch_rate


def positional_alignment(arab_sentence: str, latin_sentence: str):
    """
    Direct positional alignment for the Kanuri script pair: word i (Arabic)
    <-> word i (Latin). Only valid if check_positional_word_counts() showed
    a low/zero mismatch rate on the pilot.
    """
    n = min(len(arab_sentence.split()), latin_sentence.split().__len__())
    return [[i, i] for i in range(n)]


# ---------------------------------------------------------------------------
# Step 4: pilot-before-scaling
# ---------------------------------------------------------------------------

def run_pilot(pair: str, split: str, n: int):
    codes = LANG_CODE_MAP[pair]
    sents_a = read_flores_split(codes["a"], split)[:n]
    sents_b = read_flores_split(codes["b"], split)[:n]

    print(f"[pilot] {pair} / {split} -- inspecting {len(sents_a)} sentence pairs\n")

    if pair == "knc":
        results, mismatch_rate = check_positional_word_counts(sents_a, sents_b, n=n)
        print(f"[pilot] word-count mismatch rate: {mismatch_rate:.1%} "
              f"({sum(not r['match'] for r in results)}/{len(results)} sentences)")
        if mismatch_rate > 0.05:
            print("[pilot] WARNING: mismatch rate is non-trivial. Do NOT use direct "
                  "positional indices -- fall back to the aligner treatment (Step 5, "
                  "'if they diverge' branch) instead.")
        else:
            print("[pilot] Mismatch rate low -- positional indexing looks safe. "
                  "Spot-check a few sentences below by eye anyway.")
        for r in results[:10]:
            print(f"  sent {r['sent_idx']:3d}: arab={r['arab_word_count']:2d} words, "
                  f"latin={r['latin_word_count']:2d} words, match={r['match']}")
    else:
        translit_a = transliterate_urdu_to_devanagari(sents_a)
        aligner = get_aligner()
        for i, (orig_a, tr_a, b) in enumerate(zip(sents_a, translit_a, sents_b)):
            pairs = get_word_alignments(aligner, tr_a, b)
            print(f"  sent {i}:")
            print(f"    urd (orig):  {orig_a}")
            print(f"    urd (translit): {tr_a}")
            print(f"    hin:         {b}")
            print(f"    aligned idx pairs: {pairs}")
            print(f"    -> spot-check by eye: do these index pairs point at "
                  f"translation-equivalent words?\n")

    print("[pilot] Manually inspect the above before running the full split. "
          "Do not proceed to `run` until this looks sane (your anti-spiral rule).")


# ---------------------------------------------------------------------------
# Step 6: full run + save to disk
# ---------------------------------------------------------------------------

def run_full(pair: str, split: str):
    codes = LANG_CODE_MAP[pair]
    sents_a = read_flores_split(codes["a"], split)
    sents_b = read_flores_split(codes["b"], split)
    assert len(sents_a) == len(sents_b), (
        f"Sentence count mismatch: {len(sents_a)} vs {len(sents_b)} -- "
        f"FLORES splits should be parallel and equal length."
    )

    out = {}
    if pair == "knc":
        results, mismatch_rate = check_positional_word_counts(sents_a, sents_b)
        if mismatch_rate > 0.05:
            print(f"[run] WARNING: {mismatch_rate:.1%} mismatch rate on full split -- "
                  f"positional alignment may be unreliable for some sentences. "
                  f"Proceeding, but treat downstream token-level Kanuri numbers "
                  f"with caution (consistent with your mBERT-Kanuri UNK-rate caveat).")
        for i, (a, b) in enumerate(zip(sents_a, sents_b)):
            out[str(i)] = positional_alignment(a, b)
    else:
        translit_a = transliterate_urdu_to_devanagari(sents_a)
        aligner = get_aligner()
        for i, (tr_a, b) in enumerate(zip(translit_a, sents_b)):
            out[str(i)] = get_word_alignments(aligner, tr_a, b)
            if i % 100 == 0:
                print(f"[run] {i}/{len(sents_a)} sentences aligned")

    out_path = ALIGN_OUT_DIR / f"{pair}_{split}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[run] wrote {len(out)} sentence alignments -> {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pilot_p = sub.add_parser("pilot", help="Run on a small sample and print for manual inspection (Step 4)")
    pilot_p.add_argument("--pair", choices=list(LANG_CODE_MAP.keys()), required=True)
    pilot_p.add_argument("--split", choices=["dev", "devtest"], required=True)
    pilot_p.add_argument("--n", type=int, default=30)

    run_p = sub.add_parser("run", help="Full run, saves alignments to disk (Step 6)")
    run_p.add_argument("--pair", choices=list(LANG_CODE_MAP.keys()), required=True)
    run_p.add_argument("--split", choices=["dev", "devtest"], required=True)

    args = p.parse_args()
    if args.cmd == "pilot":
        run_pilot(args.pair, args.split, args.n)
    elif args.cmd == "run":
        run_full(args.pair, args.split)


if __name__ == "__main__":
    main()
