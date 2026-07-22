#!/usr/bin/env python3
"""
fetch_flores.py — Day 1 task: pull FLORES-200 (frozen original release),
verify coverage, extract the Urdu/Hindi and Central Kanuri Arabic/Latin
subsets, checksum everything.

Single source, used throughout the whole project:

  facebook/flores (HF) == frozen facebookresearch/flores GitHub repo.
  No longer updated. Used for BOTH Method A (reproduction of Del & Fishel's
  published numbers, which were computed on this exact snapshot) AND
  Method B onward (count-matching), so Method A and Method B run on
  byte-identical source text throughout — no cross-corpus confound to
  explain or defend in the proposal.

  Central Kanuri Arabic/Latin (knc_Arab, knc_Latn) is confirmed present
  in this frozen original release (FLORES-200 full language list), so
  there's no availability reason to reach for a second, actively-revised
  source (e.g. openlanguagedata/flores_plus) — that route was dropped.

Requires: pip install datasets huggingface_hub

Usage:
    python fetch_flores.py
"""

import hashlib
import json
import sys
from pathlib import Path

# Language codes (FLORES-200 naming: <iso639-3>_<script>)
LANG_CODES = {
    "urdu": "urd_Arab",
    "hindi": "hin_Deva",
    "kanuri_arabic": "knc_Arab",
    "kanuri_latin": "knc_Latn",
}

# Expected sentence counts, per the FLORES-200 docs.
# Mismatches here are a signal to stop and investigate, not to proceed.
EXPECTED_COUNTS = {
    "dev": 997,
    "devtest": 1012,
}

DATA_ROOT = Path(__file__).parent
RAW_DIR = DATA_ROOT / "raw"
SPLITS_DIR = DATA_ROOT / "splits" / "original"
CHECKSUM_FILE = DATA_ROOT / "checksums.txt"

HF_REPO = "facebook/flores"


def load_source():
    """Load FLORES-200 (original, frozen) via the HF `datasets` library.
    Returns a dict of split_name -> {lang_code -> list[str]}."""
    from datasets import load_dataset

    print(f"loading {HF_REPO} ...")

    out = {}
    # facebook/flores requires a per-language config name, e.g. "urd_Arab".
    # This config is public/ungated, so no auth token is needed here.
    for split in ("dev", "devtest"):
        out[split] = {}
        for lang_name, code in LANG_CODES.items():
            ds = load_dataset(HF_REPO, code, split=split, trust_remote_code=True)
            out[split][code] = ds["sentence"]

    return out


def verify_coverage(data: dict) -> bool:
    """Check split sizes against the documented counts. Don't trust the
    dataset card — this checks the pulled data directly."""
    ok = True
    for split, expected_n in EXPECTED_COUNTS.items():
        for lang_name, code in LANG_CODES.items():
            n = len(data.get(split, {}).get(code, []))
            status = "OK" if n == expected_n else "MISMATCH"
            if n != expected_n:
                ok = False
            print(f"  {split:8s} {code:10s} n={n:5d} expected={expected_n:5d}  {status}")
    return ok


def write_splits(data: dict):
    """Write per-language, per-split plaintext files under data/splits/original/."""
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for split, langs in data.items():
        for code, sentences in langs.items():
            fpath = SPLITS_DIR / f"{code}.{split}.txt"
            fpath.write_text("\n".join(sentences) + "\n", encoding="utf-8")
            print(f"  wrote {fpath} ({len(sentences)} lines)")


def compute_checksums() -> dict:
    """SHA-256 every file under data/splits/ (and raw/ if present)."""
    checksums = {}
    for base in (RAW_DIR, SPLITS_DIR):
        if not base.exists():
            continue
        for fpath in sorted(base.rglob("*")):
            if fpath.is_file():
                h = hashlib.sha256(fpath.read_bytes()).hexdigest()
                checksums[str(fpath.relative_to(DATA_ROOT))] = h
    return checksums


def write_checksum_file(checksums: dict):
    lines = [f"{h}  {relpath}" for relpath, h in sorted(checksums.items())]
    CHECKSUM_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {CHECKSUM_FILE} ({len(checksums)} entries)")


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any coverage check fails (use in CI / smoke test).",
    )
    args = parser.parse_args()

    print("\n=== facebook/flores (frozen original) ===")
    try:
        data = load_source()
    except Exception as e:
        print(f"  FAILED to load {HF_REPO}: {e}", file=sys.stderr)
        sys.exit(1)

    ok = verify_coverage(data)
    write_splits(data)

    manifest = {
        "source": {
            "hf_repo": HF_REPO,
            "coverage_ok": ok,
        }
    }
    checksums = compute_checksums()
    write_checksum_file(checksums)

    manifest_path = DATA_ROOT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {manifest_path}")

    if args.strict and not ok:
        print("\nCoverage check failed for at least one language/split — "
              "per Day 1: verify before trusting anything downstream.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()