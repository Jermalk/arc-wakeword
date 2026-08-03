"""device.py — single point of device resolution for the training pipeline.

The whole point of this project is that no code path may hardcode "cuda":
resolve the device ONCE here and thread it through. Preference order is
xpu > cuda > cpu so the same code runs on Intel Arc, on an NVIDIA box (the
upstream pipeline's original target), and on CPU-only machines, unmodified.
"""

from __future__ import annotations

import torch


def resolve_device() -> str:
    """Return the best available torch device string: "xpu", "cuda", or "cpu"."""
    if torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def empty_cache(device: str) -> None:
    """Free the accelerator's cached memory pool, whichever backend is active."""
    if device == "xpu":
        torch.xpu.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


def synchronize(device: str) -> None:
    """Block until pending kernels finish — needed before wall-clock timing."""
    if device == "xpu":
        torch.xpu.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def describe(device: str) -> str:
    """Human-readable device description for logs and run-parameter records."""
    if device == "xpu":
        props = torch.xpu.get_device_properties(0)
        return f"xpu: {torch.xpu.get_device_name(0)} ({props.total_memory / (1 << 30):.1f} GB)"
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        return f"cuda: {torch.cuda.get_device_name(0)} ({props.total_memory / (1 << 30):.1f} GB)"
    return "cpu"
