#!/usr/bin/env python3
"""
fetch_flores.py — Day 1 task: pull FLORES-200, verify coverage, extract
the Urdu/Hindi and Central Kanuri Arabic/Latin subsets, checksum everything.

Two sources, used for different purposes in the plan:

  original     facebook/flores (HF) == frozen facebookresearch/flores GitHub repo.
               No longer updated. Use this for the Method A reproduction
               (Day 4-7) so you're computing on the exact data Del & Fishel's
               published numbers came from.

  flores_plus  openlanguagedata/flores_plus (HF). Actively maintained by OLDI,
               content has been revised/corrected since the original release.
               Use this for Method B onward (count-matching, everything after
               Checkpoint 1).

NOTE: Method A and Method B therefore run on two different corpus snapshots
(frozen original vs. revised flores_plus). This is deliberate — reproduction
fidelity for Method A, corrected data for Method B onward — but it means the
two methods are NOT computed on byte-identical source text. State this
explicitly in the proposal's Section 7.1 rather than letting a reader assume
a single static FLORES-200 source throughout.

Requires: pip install datasets huggingface_hub
flores_plus is gated — run `huggingface-cli login` and accept the dataset's
terms on the HF website first, or this will fail with a 403.

Usage:
    python fetch_flores.py --source original
    python fetch_flores.py --source flores_plus
    python fetch_flores.py --source both   # default
"""

import argparse
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

# Expected sentence counts, per the FLORES-200 / FLORES+ docs.
# Mismatches here are a signal to stop and investigate, not to proceed.
EXPECTED_COUNTS = {
    "dev": 997,
    "devtest": 1012,
}

DATA_ROOT = Path(__file__).parent
RAW_DIR = DATA_ROOT / "raw"
SPLITS_DIR = DATA_ROOT / "splits"
CHECKSUM_FILE = DATA_ROOT / "checksums.txt"

SOURCE_CONFIG = {
    "original": {
        "hf_repo": "facebook/flores",
        "subdir": "original",
    },
    "flores_plus": {
        "hf_repo": "openlanguagedata/flores_plus",
        "subdir": "flores_plus",
    },
}


def load_source(source_key: str):
    """Load a FLORES source via the HF `datasets` library. Returns a dict of
    split_name -> {lang_code -> list[str]}."""
    from datasets import load_dataset

    cfg = SOURCE_CONFIG[source_key]
    repo = cfg["hf_repo"]
    print(f"[{source_key}] loading {repo} ...")

    out = {}
    if source_key == "original":
        # facebook/flores requires a per-language config name, e.g. "urd_Arab".
        # This config is public/ungated, so no auth token is needed here.
        for split in ("dev", "devtest"):
            out[split] = {}
            for lang_name, code in LANG_CODES.items():
                ds = load_dataset(repo, code, split=split, trust_remote_code=True)
                out[split][code] = ds["sentence"]
    else:
        # flores_plus ships a single "all" config with an `iso_639_3` + `iso_15924`
        # (or an `id`) column rather than one config per language — filter it.
        ds_all = load_dataset(repo, split="dev+devtest")

        # Determine which column actually identifies language+script before
        # filtering at scale, rather than silently falling back mid-filter.
        candidate_cols = ["iso_639_3_and_script", "id"]
        id_col = next((c for c in candidate_cols if c in ds_all.column_names), None)
        if id_col is None:
            raise RuntimeError(
                f"flores_plus schema has none of {candidate_cols} — "
                f"actual columns: {ds_all.column_names}. Inspect schema and "
                f"update id_col logic before trusting any downstream filter."
            )
        print(f"  [flores_plus] using '{id_col}' as the language+script identifier column")

        for split in ("dev", "devtest"):
            out[split] = {}
        for lang_name, code in LANG_CODES.items():
            subset = ds_all.filter(lambda row, c=code, col=id_col: row.get(col, "") == c)
            if len(subset) == 0:
                raise RuntimeError(
                    f"[flores_plus] filtering '{id_col}' == '{code}' ({lang_name}) "
                    f"returned 0 rows. This means the code doesn't match this "
                    f"schema's values — do NOT treat this as a size mismatch; "
                    f"inspect ds_all['{id_col}'] unique values and fix the code "
                    f"mapping before proceeding."
                )
            # flores_plus doesn't always expose split as a column post-concat;
            # if your pulled schema differs, split manually before filtering instead.
            for split in ("dev", "devtest"):
                split_subset = subset.filter(lambda row, s=split: row.get("split", s) == s)
                out[split][code] = split_subset["text"] if "text" in split_subset.column_names else split_subset["sentence"]

    return out


def verify_coverage(source_key: str, data: dict) -> bool:
    """Check split sizes against the documented counts. Don't trust the
    dataset card — this checks the pulled data directly."""
    ok = True
    for split, expected_n in EXPECTED_COUNTS.items():
        for lang_name, code in LANG_CODES.items():
            n = len(data.get(split, {}).get(code, []))
            status = "OK" if n == expected_n else "MISMATCH"
            if n != expected_n:
                ok = False
            print(f"  [{source_key}] {split:8s} {code:10s} n={n:5d} expected={expected_n:5d}  {status}")
    return ok


def write_splits(source_key: str, data: dict):
    """Write per-language, per-split plaintext files under data/splits/<source>/."""
    out_dir = SPLITS_DIR / SOURCE_CONFIG[source_key]["subdir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, langs in data.items():
        for code, sentences in langs.items():
            fpath = out_dir / f"{code}.{split}.txt"
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source",
        choices=["original", "flores_plus", "both"],
        default="both",
        help="Which FLORES source to pull. 'original' for Method A reproduction, "
             "'flores_plus' for Method B onward, 'both' to do everything now.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any coverage check fails (use in CI / smoke test).",
    )
    args = parser.parse_args()

    sources = ["original", "flores_plus"] if args.source == "both" else [args.source]

    all_ok = True
    manifest = {}
    for source_key in sources:
        print(f"\n=== {source_key} ===")
        try:
            data = load_source(source_key)
        except Exception as e:
            print(f"  FAILED to load {source_key}: {e}", file=sys.stderr)
            if source_key == "flores_plus":
                print("  (flores_plus is gated — did you `huggingface-cli login` "
                      "and accept the dataset terms on huggingface.co?)", file=sys.stderr)
            all_ok = False
            continue

        ok = verify_coverage(source_key, data)
        all_ok = all_ok and ok
        write_splits(source_key, data)
        manifest[source_key] = {
            "hf_repo": SOURCE_CONFIG[source_key]["hf_repo"],
            "coverage_ok": ok,
        }

    checksums = compute_checksums()
    write_checksum_file(checksums)

    manifest_path = DATA_ROOT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {manifest_path}")

    if args.strict and not all_ok:
        print("\nCoverage check failed for at least one source/language/split — "
              "per Day 1: verify before trusting anything downstream.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()