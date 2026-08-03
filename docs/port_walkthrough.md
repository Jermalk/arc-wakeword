# Article material — porting openWakeWord training from NVIDIA CUDA to Intel Arc

Raw material for the article, written 2026-07-11 while it happened. Companion to
`20260711_beats.md` (story beats/reactions) — this is the technical narrative:
every step of the port, in order, with the actual code.

---

## 1. The starting line

openWakeWord is arguably the best open wake-word framework, but its training pipeline
is frozen in 2022: pinned to PyTorch 1.13.1, hard CUDA requirement, README instructions
that assume WSL2 with NVIDIA passthrough. The `lgpearson1771/openwakeword-trainer` fork
already modernized the *Python* side (current torchaudio/Piper/speechbrain via a runtime
monkeypatch layer, `compat.py`) — but "GPU" still meant "CUDA" everywhere.

Meanwhile PyTorch has shipped a native Intel GPU backend (`torch.xpu`, upstreamed since
2.5, with Intel's IPEX bridge sunset in its favor). Nobody had connected the two.

Target: train on an Intel Arc Pro B70 (32 GB, Battlemage). Development/validation: a
Lunar Lake laptop iGPU (Arc 140V class, Xe2 — same architecture family, tiny). That
choice paid off: the *entire port* was provable on a laptop before the workstation ever
entered the picture — same `xpu` device string, same driver stack, different silicon.

## 2. Step 0 — map the enemy before writing a line

Before changing anything, we grepped the full dependency chain for the CUDA surface.
The result reshaped the plan — the trainer repo itself was nearly clean; the CUDA
assumptions live in the *pip-installed* packages it drives:

| Where | The CUDA idiom | Framework |
|---|---|---|
| `openwakeword/train.py` `Model.__init__` | `torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')` | PyTorch |
| `openwakeword/train.py` main | `device="gpu" if torch.cuda.is_available() else "cpu"` ×4 | **ONNX Runtime** (!) |
| `openwakeword/train.py` | `torch.cuda.empty_cache()` ×4 | PyTorch |
| `openwakeword/data.py` `augment_clips` | `torch.device('cuda:0' ...)` + `.to(device)` | PyTorch |
| `piper-sample-generator` | inline `if torch.cuda.is_available(): x.cuda()` (with an MPS elif!) | PyTorch |
| trainer's `check-env` | `torch.cuda.get_device_name(0)` (cosmetic) | PyTorch |

Two findings that mattered:

**The sleeper: `device="gpu"` doesn't always mean PyTorch.** openwakeword's
`AudioFeatures(device="gpu")` maps to ONNX Runtime's **CUDAExecutionProvider** — a
second, independent CUDA dependency in a different framework. A naive
"s/cuda/xpu/" port would have produced a nonsense ORT provider string. The port keeps
feature extraction on ORT's CPU provider (correct everywhere); switching it to
`onnxruntime-openvino`'s GPU provider is a separate optimization, deliberately deferred.

**The precedent: piper already has an MPS branch.** The generator's device dispatch is
`if cuda: ... elif mps: ...` — Apple Silicon got its branch; Intel is just the branch
nobody wrote. That shaped the strategy: don't fork piper, make the *existing* branch
take the right exit.

## 3. Strategy — vendor one file, shim the rest

Two porting styles were on the table:

- **Pure shim**: monkeypatch `torch.cuda.is_available` and `.cuda()` globally. Zero
  vendored files, great demo, but it can mask real incompatibilities.
- **Vendor everything**: copy each CUDA-touching file and edit. Explicit, but now you
  maintain three upstream files forever.

We went hybrid, and the split rule is worth stating precisely: **vendor what you can
audit as a unit, shim what you'd otherwise have to fork.**

- `openwakeword/train.py` (the training entrypoint — 900 lines, inline device logic
  everywhere) → vendored once as `oww_train_xpu.py`, device logic rewritten.
- piper-sample-generator and `openwakeword/data.py` (device idioms buried inside
  functions we don't otherwise touch) → covered by extending the trainer's existing
  `compat.py` monkeypatch architecture. The fork already patches upstream at runtime
  *before openwakeword imports* — a cuda→xpu shim is just one more patch in that list.

## 4. The port, in code

### 4a. Resolve the device once

The core discipline (it's a hard rule in this repo's CLAUDE.md): no code path may
hardcode `"cuda"` — or `"xpu"`, for that matter. One function decides, everything else
threads it through:

```python
# device.py
def resolve_device() -> str:
    if torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def empty_cache(device: str) -> None:
    if device == "xpu":
        torch.xpu.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()
```

Note the preference order means the same code still runs on an NVIDIA box — the port
*adds* a backend, it doesn't swap one prison for another.

### 4b. The vendored training file

`oww_train_xpu.py` is openwakeword 0.6.0's `train.py` with exactly four classes of
change (plus a header documenting provenance):

```python
TRAIN_DEVICE = resolve_device()            # once, at module level

# 1. Model device
self.device = torch.device(TRAIN_DEVICE)   # was: 'cuda:0' if cuda.is_available() else 'cpu'

# 2. Cache management
empty_cache(TRAIN_DEVICE)                  # was: torch.cuda.empty_cache()

# 3. AudioFeatures — ONNX Runtime, NOT torch; "gpu" == CUDA EP only
AudioFeatures(..., device="gpu" if TRAIN_DEVICE == "cuda" else "cpu",
              ncpu=n_cpus if TRAIN_DEVICE != "cuda" else 1)

# 4. TFLite conversion dropped (crashes on abandoned onnx_tf; our target is
#    ONNX -> OpenVINO IR anyway)
logging.info("Skipping TFLite conversion (this pipeline targets ONNX -> OpenVINO IR)")
```

The dispatch (`oww_wrapper.py`) changes one line: `runpy.run_module("oww_train_xpu")`
instead of `"openwakeword.train"`.

### 4c. The shim — making `.cuda()` land on Arc

The interesting engineering is in `compat.py`. Upstream code we don't vendor uses two
idioms, and they need different treatment:

```python
def _patch_cuda_to_xpu_shim() -> str:
    # Guarded: NVIDIA boxes and CPU-only boxes are completely untouched
    if torch.cuda.is_available():
        return "skipped (real CUDA present)"
    if not torch.xpu.is_available():
        return "skipped (no XPU on this machine)"

    # Idiom 1: `if torch.cuda.is_available(): x.cuda()`
    torch._oww_real_cuda_is_available = torch.cuda.is_available  # keep the truth around
    torch.cuda.is_available = lambda: True
    torch.cuda.empty_cache = torch.xpu.empty_cache
    torch.Tensor.cuda = lambda self, *a, **k: self.to("xpu")
    torch.nn.Module.cuda = lambda self, *a, **k: self.to("xpu")

    # Idiom 2: `torch.device('cuda:0')` handed to `.to(...)`.
    # torch.device itself can't be replaced (isinstance checks all over torch),
    # so translate at the .to() seam instead:
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
```

Idiom 2 was found the hard way: the first shim draft only patched `.cuda()`, which
made `openwakeword.data.augment_clips` *worse* — with `is_available()` lying True, its
`torch.device('cuda:0')` + `.to(device)` path would run and explode, where before it
fell back to CPU. **A lying `is_available()` obliges you to catch every consequence of
the lie.** Which foreshadows…

### 4d. Where the lie leaks: torch.onnx.export

On torch 2.9, `torch.onnx.export`'s dynamo path probes `torch.cuda.is_available()` and
then initializes CUDA proper → `AssertionError: Torch not compiled with CUDA enabled`,
*after training succeeded*. The fix is to scope the lie: the shim saves the real
function, and the vendored export sites temporarily restore it:

```python
@contextlib.contextmanager
def _unshimmed_cuda():
    real = getattr(torch, "_oww_real_cuda_is_available", None)
    ...
    torch.cuda.is_available = real      # tell the truth during export
    try:
        yield
    finally:
        torch.cuda.is_available = shimmed

with _unshimmed_cuda():
    torch.onnx.export(model.to("cpu"), ..., opset_version=13, dynamo=False)
```

(`dynamo=False` also pins the legacy tracer the upstream opset-13 export was written
for.) Lesson worth a pull-quote: **global monkeypatches are a loan, and torch internals
collect the interest.**

## 5. What actually broke — in chronological order

### 5a. Six blockers before CUDA was even involved

Establishing the *unmodified CPU baseline* (so port failures couldn't be blamed on the
fork) required six fixes, all pure 2026-ecosystem drift, zero CUDA: uv venvs shipping
without pip; scipy 1.17 removing `sph_harm` (used by the `acoustics` dep); piper 3.2.0
refactoring away the top-level module openwakeword imports (`sys.modules` shim);
datasets 4.x killing the MIT-RIR script-dataset (rewritten to `snapshot_download` of
the plain WAVs); openwakeword's base feature models not being bundled in the pip
package; and a stale size check (expects ≥600 MB for a 204 MB model). Full details in
DECISIONS.md. Article framing: **the CUDA pin is one axis of bitrot; this stack decays
on every axis simultaneously. Reproducibility work IS the port work.**

### 5b. The centerpiece: the wheel whose `nonzero` lies

First XPU run of TTS generation → SIGABRT with device-side asserts
(`Indexing.h: index out of bounds`) deep in piper's VITS spline transform.

The diagnosis method is the reusable part:

1. **Stage bisect with sync points.** Device asserts are asynchronous — the Python
   traceback points at the wrong line. Re-run the model forward stage by stage with
   `torch.xpu.synchronize()` after each; the crash pins to the stochastic duration
   predictor → `outputs[mask] = inputs[mask]` in the spline.
2. **CPU control.** Same tensors, same code on CPU: clean. So the data is fine.
3. **Shrink to pure torch.** Boolean-mask ops in isolation… all pass?! Only the
   *combined* run crashed. Suspicion shifts from "our port" to "op-level state".
4. **Interrogate the primitive.** `mask.sum()` says 9. `mask.nonzero()` returns 0 rows.
   And the deterministic kill shot:

   ```python
   torch.tensor([1, 0, 1, 0, 1], device="xpu").nonzero()
   # -> [0, 0, 4]        expected: [0, 2, 4]        torch 2.13.0+xpu, LNL iGPU
   ```

   `nonzero` is broken for a wide band of ordinary shapes — `(23,)`, `(1,1,100)`,
   `(1,1,1000)` returns 263 of 503 rows, one gapped-view case returned **535 rows from
   a 23-element tensor** (the source of the OOB asserts). Every masked-indexing op in
   torch funnels through `nonzero`. Nothing in a loss curve would ever tell you.
5. **Version bisect on the same driver.** 2.13.0+xpu broken, 2.12.1+xpu correct,
   2.9.1+xpu correct → wheel regression, not driver. Pinned 2.9.1 (the only version
   below 2.13 with a matching torchaudio on the xpu index — torchaudio skipped 2.10,
   and 2.11.0 hard-requires torch 2.13.0).
6. **Institutionalize the check.** `check_xpu_sanity.py` now gates every new
   machine/driver/torch combo — it must pass on the B70 before anything trains there.

Pull-quote: **"should work the same" is exactly the assumption this project exists to
test — and the failure mode isn't a crash, it's silently wrong arithmetic.**

### 5c. Shared-memory reality: the iGPU OOM and the assistant that got shot

Two memory incidents unique to integrated GPUs (the B70's dedicated VRAM has neither):

- Mid-run `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` during validation-heavy training
  sequences — the caching allocator holds blocks a 24 GB discrete card would never
  miss. Fix: explicit `empty_cache(TRAIN_DEVICE)` at sequence starts and after each
  validation pass, in the vendored trainer.
- systemd-oomd killed a co-located OpenVINO inference service under memory pressure
  from training. Beat: on shared silicon, training and serving compete for the same
  memory pool; dedicated VRAM isn't a luxury, it's what lets them coexist.

## 6. The numbers (smoke config: 500 steps, 800 TTS clips)

| Step | CPU (laptop, 8 cores) | LNL iGPU (Arc 140V) | Notes |
|---|---|---|---|
| TTS generation (800 clips) | ~2.5 min | ~10 min | iGPU loses: batch 10, host-bound |
| Augment + features | ~1.5 min | ~1.3 min | ORT CPU either way (by design) |
| Train step (all 3 sequences) | 8m24s | 11m07s | see below |
| — pure training loop | ~1 it/s | **38–45 it/s** | **~40× on XPU** |
| Extrapolated full run (50k steps) | ~14 h | ~1–1.5 h | loop dominates at scale |
| Upstream README reference | 12–24 h CPU | — | "1–2 h" on NVIDIA GPU |

The honest headline: at *smoke* scale the iGPU loses end-to-end (validation passes are
tiny-batch, transfer-bound). At *real* scale the training loop dominates and the ~40×
loop speedup wins. And this is the wrong Arc — the B70 numbers get their own column.

## 7. CUDA → XPU cheat sheet (as actually used)

| CUDA | XPU | Caveat learned here |
|---|---|---|
| `torch.cuda.is_available()` | `torch.xpu.is_available()` | never patch globally without scoping (see §4d) |
| `.cuda()` / `.to('cuda')` | `.to('xpu')` | no `.xpu()` shorthand exists |
| `torch.cuda.empty_cache()` | `torch.xpu.empty_cache()` | you'll need it MORE on shared-memory iGPUs |
| `torch.cuda.synchronize()` | `torch.xpu.synchronize()` | essential for bisecting async device asserts |
| ORT `CUDAExecutionProvider` | no analog; CPU or `OpenVINOExecutionProvider` | "gpu" strings aren't always torch |
| install: `pip install torch` | `--index-url https://download.pytorch.org/whl/xpu` | version pairs are sparse; check both torch AND torchaudio exist |

## 8. Takeaways

1. **Map the device surface before editing** — ours spanned two frameworks (PyTorch
   *and* ONNX Runtime), and the grep took an hour that saved days.
2. **Vendor what you can audit, shim what you'd have to fork** — and keep the shim
   guarded so NVIDIA/CPU environments are untouched.
3. **A lying `is_available()` must be scoped** — torch internals also ask.
4. **Validate kernels, not just pipelines** — a sanity script for primitive ops
  (`nonzero`, masked indexing) is now a permanent gate before any new stack trains.
5. **Baseline on CPU first, unmodified** — every later failure was attributable in
   minutes because the un-ported pipeline was known-good.
6. **The portability work and the reproducibility work are the same work.** Six of the
   eight fixes had nothing to do with CUDA. The pin was just the loudest symptom of an
   unmaintained stack.
