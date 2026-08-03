#!/usr/bin/env python3
"""oww_wrapper.py — Run the vendored ``oww_train_xpu`` with compat patches.

This wrapper applies monkey-patches for torchaudio 2.10+, speechbrain, and
piper-sample-generator BEFORE openwakeword is imported, then delegates to
``oww_train_xpu`` — this project's device-portable (xpu > cuda > cpu) copy of
``openwakeword.train``.  All ``sys.argv`` arguments are forwarded transparently.

Usage (instead of ``python -m openwakeword.train``):
    python oww_wrapper.py --training_config ... --generate_clips
"""

import os
import sys

# Ensure the project directory is on sys.path so ``import compat`` works.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# ── Apply patches BEFORE any openwakeword / speechbrain imports ──
import compat  # noqa: E402

compat.apply_all()

# ── Verify patches are functional ──
results = compat.verify_all()
failed = [k for k, v in results.items() if not v]
if failed:
    print(f"WARNING: Some compat patches failed verification: {failed}", file=sys.stderr)
    print("Training may still work — proceeding.", file=sys.stderr)

# ── Delegate to the vendored, device-portable copy of openwakeword.train ──
import runpy  # noqa: E402

sys.argv[0] = "oww_train_xpu"
runpy.run_module("oww_train_xpu", run_name="__main__", alter_sys=True)
