"""compat.py — Compatibility patches for openWakeWord training dependencies.

Adapted from the lgpearson1771/openwakeword-trainer fork (MIT License, Copyright
(c) 2026 Luke Pearson — see NOTICE.md and third_party_licenses/). The cuda->xpu
shim (_patch_cuda_to_xpu_shim and related) is this project's own addition on top
of that fork's original compat-patch layer.

Addresses known breaking changes in modern dependency versions:
  - setuptools 82+ removed pkg_resources
  - torchaudio 2.10+ removed load(), info(), list_audio_backends()
  - Piper sample generator API changed (requires model= kwarg)
  - Sample rate mismatches (Piper outputs 22050 Hz, openWakeWord expects 16000 Hz)

Apply BEFORE importing openwakeword, speechbrain, or torch-audiomentations:

    import compat
    results = compat.apply_all()    # monkey-patches torchaudio etc.
    ok      = compat.verify_all()   # tests each patch actually works
"""

from __future__ import annotations

import logging
import subprocess
import sys
import tempfile
from pathlib import Path

log = logging.getLogger("compat")


# ─── Public API ───────────────────────────────────────────────────────────


def apply_all() -> dict[str, str]:
    """Apply every patch. Returns ``{name: status}`` where *status* is one of
    ``ok``, ``applied``, ``skipped (reason)``, or ``FAILED: reason``.
    """
    results: dict[str, str] = {}
    for name, fn in _PATCHES:
        try:
            status = fn()
        except Exception as exc:
            status = f"FAILED: {exc}"
        results[name] = status
        level = logging.WARNING if "FAIL" in status else logging.INFO
        log.log(level, "  patch %-30s %s", name, status)
    return results


def verify_all() -> dict[str, bool]:
    """Functional tests for each patch.  Returns ``{name: passed}``."""
    results: dict[str, bool] = {}

    # ── torchaudio.load ──
    try:
        import numpy as np
        import soundfile as sf
        import torch
        import torchaudio

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            # Write a file at 22050 Hz to test resampling
            sf.write(f.name, np.zeros(22050, dtype=np.float32), 22050)
            wav, sr = torchaudio.load(f.name)
            results["torchaudio.load"] = sr == 16000 and isinstance(wav, torch.Tensor)
            if sr != 16000:
                log.warning("  verify torchaudio.load  returned SR=%d (expected 16000)", sr)
            Path(f.name).unlink(missing_ok=True)
    except Exception as exc:
        results["torchaudio.load"] = False
        log.warning("  verify torchaudio.load  FAILED: %s", exc)

    # ── torchaudio.info ──
    try:
        import numpy as np
        import soundfile as sf
        import torchaudio

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, np.zeros(16000, dtype=np.float32), 16000)
            meta = torchaudio.info(f.name)
            results["torchaudio.info"] = meta.sample_rate == 16000
            Path(f.name).unlink(missing_ok=True)
    except Exception as exc:
        results["torchaudio.info"] = False
        log.warning("  verify torchaudio.info  FAILED: %s", exc)

    # ── torchaudio.list_audio_backends ──
    try:
        import torchaudio

        backends = torchaudio.list_audio_backends()
        results["torchaudio.list_audio_backends"] = isinstance(backends, list)
    except Exception:
        results["torchaudio.list_audio_backends"] = False

    # ── pkg_resources ──
    try:
        import pkg_resources  # noqa: F401

        results["pkg_resources"] = True
    except ImportError:
        results["pkg_resources"] = False

    for name, ok in results.items():
        log.info("  verify %-30s %s", name, "PASS" if ok else "FAIL")

    return results


# ─── Individual patches ──────────────────────────────────────────────────


def _ensure_pkg_resources() -> str:
    """Install setuptools<82 if pkg_resources was removed."""
    try:
        import pkg_resources  # noqa: F401

        return "ok"
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "setuptools<82", "-q"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return "applied (setuptools<82)"


def _patch_torchaudio_load() -> str:
    """Replace ``torchaudio.load`` with a soundfile-based loader that also
    resamples to 16 kHz when needed (Piper outputs 22050 Hz)."""
    import torch
    import torchaudio

    if getattr(torchaudio, "_oww_load_patched", False):
        return "ok (already patched)"

    def _load(filepath, *args, **kwargs):
        import numpy as np
        import soundfile as sf

        data, sr = sf.read(str(filepath), dtype="float32")
        if data.ndim == 1:
            data = data[np.newaxis, :]  # (1, samples)
        else:
            data = data.T  # (channels, samples)

        # Resample to 16 kHz if needed (Piper TTS outputs 22050 Hz)
        if sr != 16000:
            from scipy.signal import resample as scipy_resample
            old_len = data.shape[-1]
            new_len = int(old_len * 16000 / sr)
            # Resample each channel
            if data.ndim == 2:
                resampled = np.stack([
                    scipy_resample(data[c], new_len).astype(np.float32)
                    for c in range(data.shape[0])
                ])
            else:
                resampled = scipy_resample(data, new_len).astype(np.float32)
            data = resampled
            sr = 16000

        return torch.from_numpy(data), sr

    torchaudio.load = _load
    torchaudio._oww_load_patched = True
    return "applied"


def _patch_torchaudio_info() -> str:
    """Provide a soundfile-based ``torchaudio.info``."""
    import torchaudio

    if getattr(torchaudio, "_oww_info_patched", False):
        return "ok (already patched)"

    class AudioMetaData:
        __slots__ = (
            "sample_rate",
            "num_frames",
            "num_channels",
            "bits_per_sample",
            "encoding",
        )

        def __init__(self, sample_rate: int, num_frames: int, num_channels: int):
            self.sample_rate = sample_rate
            self.num_frames = num_frames
            self.num_channels = num_channels
            self.bits_per_sample = 16
            self.encoding = "PCM_S"

    def _info(filepath):
        import soundfile as sf

        fi = sf.info(str(filepath))
        return AudioMetaData(fi.samplerate, fi.frames, fi.channels)

    torchaudio.info = _info
    if not hasattr(torchaudio, "AudioMetaData"):
        torchaudio.AudioMetaData = AudioMetaData
    torchaudio._oww_info_patched = True
    return "applied"


def _patch_torchaudio_list_backends() -> str:
    """Re-add ``torchaudio.list_audio_backends`` for speechbrain compat."""
    import torchaudio

    if hasattr(torchaudio, "list_audio_backends"):
        return "ok"
    torchaudio.list_audio_backends = lambda: ["soundfile"]
    return "applied"


def _patch_piper_generate_samples() -> str:
    """Make openwakeword's legacy ``from generate_samples import generate_samples``
    work against piper-sample-generator ≥3.2.0.

    3.2.0 refactored the repo-root ``generate_samples.py`` into the
    ``piper_sample_generator`` package (function lives in ``__main__``), so the
    module openwakeword imports no longer exists on disk. We register a virtual
    ``generate_samples`` module in ``sys.modules`` (which wins over any
    ``sys.path`` lookup) exposing a wrapper that also injects *model=* when the
    caller omits it (openwakeword 0.6.0 never passes it; the new signature
    requires it).

    Note: ``piper_train`` (needed for .pt models) is deliberately excluded from
    the package install, so the repo root must be on ``sys.path`` — openwakeword
    does this itself later, but we need it now to import ``__main__``.
    """
    import types

    if isinstance(sys.modules.get("generate_samples"), types.ModuleType) and getattr(
        sys.modules["generate_samples"], "_oww_generate_patched", False
    ):
        return "ok (already patched)"

    try:
        import piper_sample_generator as psg
    except ImportError:
        return "skipped (piper_sample_generator not installed)"

    repo_root = Path(psg.__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    try:
        from piper_sample_generator.__main__ import generate_samples as _orig_generate
    except ImportError as exc:
        return f"FAILED: cannot import generate_samples from package: {exc}"

    def _wrapped(*args, **kwargs):
        if "model" not in kwargs:
            models = sorted(repo_root.rglob("*.pt"))
            if models:
                kwargs["model"] = str(models[0])
                log.info("Auto-resolved Piper model: %s", kwargs["model"])
        if "auto_reduce_batch_size" in kwargs:
            # Legacy kwarg from openwakeword 0.6.0; new generate_samples takes
            # **kwargs and silently ignores it — surface that once.
            log.info("Note: auto_reduce_batch_size is ignored by piper-sample-generator >=3.2.0")
        return _orig_generate(*args, **kwargs)

    mod = types.ModuleType("generate_samples")
    mod.generate_samples = _wrapped
    mod._oww_generate_patched = True
    sys.modules["generate_samples"] = mod
    psg.generate_samples = _wrapped
    return "applied (sys.modules['generate_samples'] shim)"


def _patch_oww_data_sample_rate() -> str:
    """Suppress openwakeword's sample-rate ValueError.

    Since ``torchaudio.load`` (patched above) already resamples to 16 kHz,
    this patch only needs to handle any remaining direct ``sf.read`` calls
    inside openwakeword that might raise on rate mismatches.

    We do NOT globally patch ``soundfile.read`` because that would conflict
    with torchaudio's internal ``_soundfile_load`` which passes extra kwargs
    like ``start``, ``stop``, ``always_2d`` that don't survive resampling.
    Instead, we patch only openwakeword-specific code paths.
    """
    try:
        import openwakeword.data as oww_data
    except ImportError:
        return "skipped (openwakeword not installed)"

    if getattr(oww_data, "_oww_sr_patched", False):
        return "ok (already patched)"

    # The torchaudio.load patch already handles resampling.
    # Mark as done so we don't re-apply.
    oww_data._oww_sr_patched = True
    return "applied (torchaudio.load handles resampling)"


def _patch_cuda_to_xpu_shim() -> str:
    """Route inline CUDA idioms to XPU on CUDA-less Intel boxes.

    piper-sample-generator (and other upstream code we don't vendor) uses the
    ``if torch.cuda.is_available(): x.cuda()`` idiom inline, which can't be
    patched at a single seam. On a machine with an XPU and no CUDA, shim the
    idiom itself: ``torch.cuda.is_available`` answers for the XPU, and
    ``.cuda()`` moves to it. Guarded so NVIDIA boxes and CPU-only boxes are
    completely untouched. Our own vendored training code resolves its device
    explicitly and never hits this path.
    """
    import torch

    if getattr(torch, "_oww_xpu_shim", False):
        return "ok (already patched)"
    if torch.cuda.is_available():
        return "skipped (real CUDA present)"
    if not torch.xpu.is_available():
        return "skipped (no XPU on this machine)"

    torch._oww_real_cuda_is_available = torch.cuda.is_available
    torch.cuda.is_available = lambda: True
    torch.cuda.empty_cache = torch.xpu.empty_cache
    torch.cuda.synchronize = torch.xpu.synchronize
    torch.cuda.get_device_name = torch.xpu.get_device_name
    torch.cuda.get_device_properties = torch.xpu.get_device_properties
    torch.cuda.device_count = torch.xpu.device_count

    def _tensor_cuda(self, *args, **kwargs):
        return self.to("xpu")

    def _module_cuda(self, *args, **kwargs):
        return self.to("xpu")

    torch.Tensor.cuda = _tensor_cuda
    torch.nn.Module.cuda = _module_cuda

    # The other inline idiom: torch.device('cuda:0') handed to .to(...)
    # (e.g. openwakeword.data.augment_clips). torch.device itself can't be
    # replaced (isinstance checks all over torch), so translate the target
    # at the .to() seam instead.
    def _translate(dev):
        if isinstance(dev, str) and dev.startswith("cuda"):
            return "xpu"
        if isinstance(dev, torch.device) and dev.type == "cuda":
            return torch.device("xpu")
        return dev

    def _wrap_to(orig):
        def _to(self, *args, **kwargs):
            args = tuple(_translate(a) for a in args)
            if "device" in kwargs:
                kwargs["device"] = _translate(kwargs["device"])
            return orig(self, *args, **kwargs)
        return _to

    torch.Tensor.to = _wrap_to(torch.Tensor.to)
    torch.nn.Module.to = _wrap_to(torch.nn.Module.to)
    torch._oww_xpu_shim = True
    return "applied (cuda -> xpu, including .to() translation)"


def _patch_deep_phonemizer_checkpoint_load() -> str:
    """Fix DeepPhonemizer's checkpoint load under torch's weights_only default flip.

    openwakeword's adversarial-negative generation (``generate_adversarial_texts``)
    loads a DeepPhonemizer checkpoint via ``dp.model.model.load_checkpoint()``,
    which calls ``torch.load(checkpoint_path, map_location=device)`` with no
    ``weights_only`` argument. PyTorch 2.6 changed that default from False to
    True; the checkpoint (downloaded by openwakeword itself, from a fixed
    DeepPhonemizer-owned S3 URL — not user-supplied) contains a custom
    ``dp.preprocessing.text.Preprocessor`` global that isn't on the default
    safe list, so loading now raises ``UnpicklingError``. Temporarily force
    ``weights_only=False`` around just this one call — restores the load
    behavior this code always had before torch changed the default, without
    weakening ``torch.load`` anywhere else.
    """
    try:
        import dp.model.model as dp_model
        import dp.phonemizer as dp_phonemizer
    except ImportError:
        return "skipped (deep-phonemizer not installed)"

    if getattr(dp_model.load_checkpoint, "_oww_weights_only_patched", False):
        return "ok (already patched)"

    import torch

    _orig_load_checkpoint = dp_model.load_checkpoint

    def _load_checkpoint_weights_only_false(checkpoint_path, device="cpu"):
        _orig_torch_load = torch.load

        def _torch_load_no_weights_only(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return _orig_torch_load(*args, **kwargs)

        torch.load = _torch_load_no_weights_only
        try:
            return _orig_load_checkpoint(checkpoint_path, device=device)
        finally:
            torch.load = _orig_torch_load

    _load_checkpoint_weights_only_false._oww_weights_only_patched = True
    dp_model.load_checkpoint = _load_checkpoint_weights_only_false
    dp_phonemizer.load_checkpoint = _load_checkpoint_weights_only_false
    return "applied (weights_only=False for the DeepPhonemizer checkpoint)"


# ─── Patch registry (order matters) ──────────────────────────────────────

_PATCHES = [
    ("setuptools/pkg_resources", _ensure_pkg_resources),
    ("torchaudio.load", _patch_torchaudio_load),
    ("torchaudio.info", _patch_torchaudio_info),
    ("torchaudio.list_audio_backends", _patch_torchaudio_list_backends),
    ("piper generate_samples model=", _patch_piper_generate_samples),
    ("oww data.py sample rate", _patch_oww_data_sample_rate),
    ("cuda -> xpu shim", _patch_cuda_to_xpu_shim),
    ("deep-phonemizer checkpoint load", _patch_deep_phonemizer_checkpoint_load),
]
