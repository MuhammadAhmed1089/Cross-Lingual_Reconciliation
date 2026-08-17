"""
run_methodA.py
==============
Unified orchestration script for Method A of the Cross-Lingual Reconciliation
pipeline.

Pipeline stages (in order):
  1. encode   – encodeFLORES.py         (sentence-level pooled encodings)
  2. cka      – cka.py                  (pooled-mean CKA)
  3. cosine   – cosineNN.py             (anisotropy-corrected cosine similarity)
  4. nn       – nearestNeighbourAccuracy.py  (nearest-neighbour accuracy)
  5. tlpa     – TokenLevelPositionAlignedCKA/ subfolder:
                  build_alignments.py pilot  (per pair x split, manual review)
                  build_alignments.py run    (per pair x split)
                  TLPA_CKA.py --token-level
                  plot_pooled_vs_token.py

Usage
-----
  # Full pipeline for both model families (default), auto-detected device
  python run_methodA.py

  # Specific model classes
  python run_methodA.py --model-class mbert xlmr

  # Run only encoding and CKA
  python run_methodA.py --model-class mbert --steps encode cka

  # Skip the alignment pilot confirmation gate
  python run_methodA.py --skip-pilot

  # Preview commands without executing them
  python run_methodA.py --dry-run

  # Force CPU even if CUDA is available
  python run_methodA.py --device cpu
"""

import argparse
import datetime
import os
import subprocess
import sys
import textwrap

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Default model classes to run (in the order they will be processed).
DEFAULT_MODEL_CLASSES = ["mbert", "xlmr"]

# Language-pair names used by build_alignments.py
TLPA_PAIRS = ["urd_hin", "knc"]
SPLITS = ["dev", "devtest"]

# Script locations -- all relative to THIS file's directory.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TLPA_DIR = os.path.join(SCRIPT_DIR, "TokenLevelPositionAlignedCKA")

# Maps stage name -> (script_path, cwd) for the four standard stages.
# TLPA is handled explicitly in the main loop.
STAGE_SCRIPTS = {
    "encode": (os.path.join(SCRIPT_DIR, "encodeFLORES.py"),           SCRIPT_DIR),
    "cka":    (os.path.join(SCRIPT_DIR, "cka.py"),                    SCRIPT_DIR),
    "cosine": (os.path.join(SCRIPT_DIR, "cosineNN.py"),               SCRIPT_DIR),
    "nn":     (os.path.join(SCRIPT_DIR, "nearestNeighbourAccuracy.py"), SCRIPT_DIR),
}

ALL_STAGES = ["encode", "cka", "cosine", "nn", "tlpa"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now():
    return datetime.datetime.now().strftime("%H:%M:%S")


def _banner(msg):
    width = 72
    line = "=" * width
    print("\n" + line, flush=True)
    print(f"  [{_now()}]  {msg}", flush=True)
    print(line, flush=True)


def _sub_banner(msg):
    print(f"\n  >> [{_now()}]  {msg}", flush=True)


def _run(cmd, cwd, dry_run):
    """Run a command as a subprocess, raising SystemExit on non-zero return."""
    cmd_str = " ".join(cmd)
    print(f"\n  CWD : {cwd}", flush=True)
    print(f"  CMD : {cmd_str}", flush=True)

    if dry_run:
        print("  [DRY-RUN] -- skipping execution\n", flush=True)
        return

    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(
            f"\n[ERROR] Command failed (exit {result.returncode}):\n"
            f"  {cmd_str}\n"
            f"Aborting pipeline.",
            file=sys.stderr,
        )
        sys.exit(result.returncode)


def _detect_device():
    """Return 'cuda' if a CUDA-capable GPU is visible, else 'cpu'."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def run_standard_stage(stage, model_class, device, dry_run):
    """Run one of the four standard stages (encode / cka / cosine / nn).

    Equivalent shell commands (run from src/methodA/):
      encode:  python encodeFLORES.py <model_class> <device>
      cka:     python cka.py <model_class>
      cosine:  python cosineNN.py <model_class>
      nn:      python nearestNeighbourAccuracy.py <model_class>
    """
    script_path, cwd = STAGE_SCRIPTS[stage]
    cmd = [sys.executable, script_path, model_class]
    if stage == "encode":
        # encodeFLORES.py reads device from sys.argv[2]
        cmd.append(device)
    _run(cmd, cwd, dry_run)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        prog="run_methodA.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-class",
        nargs="+",
        default=DEFAULT_MODEL_CLASSES,
        metavar="CLASS",
        help=(
            "One or more model-class strings accepted by get_hf_model_ids(). "
            f"Default: {DEFAULT_MODEL_CLASSES}"
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        metavar="DEVICE",
        help=(
            "PyTorch device string passed to encodeFLORES.py "
            "(e.g. 'cuda', 'cuda:1', 'cpu'). "
            "Default: auto-detect (cuda if available, else cpu)."
        ),
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        default=ALL_STAGES,
        choices=ALL_STAGES,
        metavar="STAGE",
        help=(
            "Subset of pipeline stages to run. "
            f"Choices: {ALL_STAGES}. Default: all stages."
        ),
    )
    parser.add_argument(
        "--skip-pilot",
        action="store_true",
        help=(
            "Skip the interactive alignment pilot confirmation gate. "
            "Useful for automated runs after pilots have already been verified."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print each command that would be executed without actually "
            "running it. Implies --skip-pilot."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ----------------------------------------------------------------
    # Full set of commands this orchestrator dispatches
    # (shown here for reference; <model> = mbert | xlmr, <device> = cuda | cpu)
    #
    # Stage 1 — ENCODE  (src/methodA/)
    #   python encodeFLORES.py <model> <device>
    #
    # Stage 2 — CKA  (src/methodA/)
    #   python cka.py <model>
    #
    # Stage 3 — COSINE  (src/methodA/)
    #   python cosineNN.py <model>
    #
    # Stage 4 — NN  (src/methodA/)
    #   python nearestNeighbourAccuracy.py <model>
    #
    # Stage 5 — TLPA  (src/methodA/TokenLevelPositionAlignedCKA/)
    #
    #   Step A  alignment pilots (model-agnostic, 4 runs: 2 pairs x 2 splits)
    #     python build_alignments.py pilot --pair urd_hin --split dev     --n 30
    #     python build_alignments.py pilot --pair urd_hin --split devtest --n 30
    #     python build_alignments.py pilot --pair knc     --split dev     --n 30
    #     python build_alignments.py pilot --pair knc     --split devtest --n 30
    #
    #   Step B  [interactive gate -- user inspects pilot output, presses Enter]
    #
    #   Step C  full alignment runs (model-agnostic, 4 runs: 2 pairs x 2 splits)
    #     python build_alignments.py run --pair urd_hin --split dev
    #     python build_alignments.py run --pair urd_hin --split devtest
    #     python build_alignments.py run --pair knc     --split dev
    #     python build_alignments.py run --pair knc     --split devtest
    #
    #   Step D  token-level CKA (per model class)
    #     python TLPA_CKA.py <model> --token-level
    #
    #   Step E  pooled vs. token-level comparison plots (once)
    #     python plot_pooled_vs_token.py
    # ----------------------------------------------------------------

    args = parse_args()

    # Resolve device once so it is consistent across all model classes.
    device = args.device if args.device else _detect_device()

    # --dry-run implies --skip-pilot (nothing executes anyway).
    skip_pilot = args.skip_pilot or args.dry_run

    # Preserve pipeline order regardless of how --steps was supplied.
    requested_stages = [s for s in ALL_STAGES if s in args.steps]

    # ----------------------------------------------------------------
    # Pipeline header
    # ----------------------------------------------------------------
    _banner("Method A -- pipeline start")
    print(f"  Model classes : {args.model_class}", flush=True)
    print(f"  Device        : {device}", flush=True)
    print(f"  Stages        : {requested_stages}", flush=True)
    print(f"  Skip pilot    : {skip_pilot}", flush=True)
    print(f"  Dry run       : {args.dry_run}", flush=True)

    # ----------------------------------------------------------------
    # Stages 1-4: encode / cka / cosine / nn  (per model class)
    # ----------------------------------------------------------------
    for model_class in args.model_class:
        for stage in requested_stages:
            if stage == "tlpa":
                continue  # handled separately below

            _banner(f"Stage: {stage.upper()}  |  model_class={model_class}")
            run_standard_stage(stage, model_class, device, args.dry_run)
            _banner(f"Stage: {stage.upper()} DONE  |  model_class={model_class}")

    # ----------------------------------------------------------------
    # Stage 5: TLPA
    #
    # Alignment build is model-agnostic -- run once.
    # Token-level CKA (TLPA_CKA.py) loads models internally, so it runs
    # once per model class.
    # Plots are generated once at the end (they read the shared results pkl).
    # ----------------------------------------------------------------
    if "tlpa" in requested_stages:

        build_script    = os.path.join(TLPA_DIR, "build_alignments.py")
        tlpa_cka_script = os.path.join(TLPA_DIR, "TLPA_CKA.py")
        plot_script     = os.path.join(TLPA_DIR, "plot_pooled_vs_token.py")

        # ---- Step A: pilot runs (model-agnostic) ----
        # Commands (run from TokenLevelPositionAlignedCKA/):
        #   python build_alignments.py pilot --pair urd_hin --split dev     --n 30
        #   python build_alignments.py pilot --pair urd_hin --split devtest --n 30
        #   python build_alignments.py pilot --pair knc     --split dev     --n 30
        #   python build_alignments.py pilot --pair knc     --split devtest --n 30
        _banner("Stage: TLPA -- Step A: alignment pilots")
        for pair in TLPA_PAIRS:
            for split in SPLITS:
                _sub_banner(f"Pilot  pair={pair}  split={split}")
                _run(
                    [sys.executable, build_script,
                     "pilot", "--pair", pair, "--split", split, "--n", "30"],
                    cwd=TLPA_DIR,
                    dry_run=args.dry_run,
                )

        # ---- Step B: confirmation gate ----
        if not skip_pilot:
            print(
                textwrap.dedent("""
                +---------------------------------------------------------------+
                |  PILOT INSPECTION REQUIRED                                    |
                |                                                               |
                |  Review the pilot output above:                               |
                |  * Urdu-Hindi: do aligned index pairs point at               |
                |    translation-equivalent words?                              |
                |  * Kanuri: is the word-count mismatch rate < 5 %?            |
                |                                                               |
                |  Press  [Enter]  to continue to the full alignment run.       |
                |  Press  Ctrl-C   to abort.                                    |
                +---------------------------------------------------------------+
                """),
                flush=True,
            )
            try:
                input()
            except KeyboardInterrupt:
                print("\n[Aborted by user during pilot gate]", flush=True)
                sys.exit(0)
        else:
            print(
                "  [--skip-pilot] Skipping pilot confirmation gate.",
                flush=True,
            )

        # ---- Step C: full alignment runs (model-agnostic) ----
        # Commands (run from TokenLevelPositionAlignedCKA/):
        #   python build_alignments.py run --pair urd_hin --split dev
        #   python build_alignments.py run --pair urd_hin --split devtest
        #   python build_alignments.py run --pair knc     --split dev
        #   python build_alignments.py run --pair knc     --split devtest
        _banner("Stage: TLPA -- Step C: full alignment runs")
        for pair in TLPA_PAIRS:
            for split in SPLITS:
                _sub_banner(f"Full run  pair={pair}  split={split}")
                _run(
                    [sys.executable, build_script,
                     "run", "--pair", pair, "--split", split],
                    cwd=TLPA_DIR,
                    dry_run=args.dry_run,
                )

        # ---- Step D: token-level CKA (per model class) ----
        # Command (run from TokenLevelPositionAlignedCKA/):
        #   python TLPA_CKA.py <model> --token-level
        for model_class in args.model_class:
            _banner(
                f"Stage: TLPA -- Step D: token-level CKA  |  model_class={model_class}"
            )
            _run(
                [sys.executable, tlpa_cka_script, model_class, "--token-level"],
                cwd=TLPA_DIR,
                dry_run=args.dry_run,
            )

        # ---- Step E: pooled vs. token-level comparison plots ----
        # Command (run from TokenLevelPositionAlignedCKA/):
        #   python plot_pooled_vs_token.py
        _banner("Stage: TLPA -- Step E: pooled vs. token-level plots")
        _run(
            [sys.executable, plot_script],
            cwd=TLPA_DIR,
            dry_run=args.dry_run,
        )

        _banner("Stage: TLPA DONE")

    # ----------------------------------------------------------------
    # Done
    # ----------------------------------------------------------------
    _banner("Method A -- pipeline complete")
    print(
        f"  Finished at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
