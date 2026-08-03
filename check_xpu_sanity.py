"""check_xpu_sanity.py — verify torch XPU kernel correctness before trusting a box.

Born 2026-07-11: torch 2.13.0+xpu shipped a broken `nonzero` kernel (wrong row
counts, out-of-bounds indices) on Lunar Lake iGPU, which silently poisons ALL
boolean-mask indexing (masked_select, t[mask], index_put). torch 2.9.1+xpu is
correct on the same driver stack. Run this on every new machine/driver/torch
combination (Arc Pro B70 included) BEFORE any training run.

Exit code 0 = sane, 1 = broken (details printed).
"""

from __future__ import annotations

import sys

import torch


def main() -> int:
    print(f"torch {torch.__version__}")
    if not torch.xpu.is_available():
        print("XPU not available — nothing to check (CPU path is trusted).")
        return 0
    print(f"device: {torch.xpu.get_device_name(0)}")

    failures: list[str] = []

    # Deterministic nonzero
    t = torch.tensor([1, 0, 1, 0, 1], device="xpu")
    got = t.nonzero().flatten().tolist()
    if got != [0, 2, 4]:
        failures.append(f"nonzero([1,0,1,0,1]) -> {got}, expected [0, 2, 4]")

    # Random masks across the shapes that broke on torch 2.13.0+xpu / LNL
    for shape in [(23,), (1, 23), (1, 1, 23), (1, 2, 23), (1, 1, 24),
                  (1, 1, 100), (1, 1, 1000), (1, 1, 1, 23)]:
        m = (torch.rand(shape) > 0.5).to("xpu")
        s, nz = int(m.sum()), int(m.nonzero().shape[0])
        if s != nz:
            failures.append(f"shape {shape}: sum()={s} but nonzero() rows={nz}")

    # Gapped-view masked_select (the exact pattern from piper's spline transform)
    base = torch.randn(1, 2, 23, device="xpu")
    view = base[:, 1:, :]
    mask = view > 0
    s, sel = int(mask.sum()), int(view[mask].numel())
    if s != sel:
        failures.append(f"gapped view: mask sum={s} but masked_select numel={sel}")

    if failures:
        print("XPU SANITY: FAILED")
        for f in failures:
            print(f"  - {f}")
        print("Do NOT train on this stack — masked indexing returns wrong results.")
        return 1

    print("XPU SANITY: PASSED (nonzero / masked indexing correct)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
