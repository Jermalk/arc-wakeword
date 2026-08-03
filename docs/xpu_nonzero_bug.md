# Bug hunt: Piper TTS generation crashes on XPU with a device-side index assertion

**Date:** 2026-07-11 · **Context:** validating the CUDA→XPU port (`oww_train_xpu.py` +
the `cuda->xpu` shim in `compat.py`, which patches `torch.cuda.is_available`, `.cuda()`,
and translates `.to('cuda*')` → xpu) on a Lunar Lake iGPU. The unmodified pipeline
already passed on CPU, isolating this as an XPU-specific issue; `generate` is the first
pipeline step that actually touches the device.

## Symptom

`python train_wakeword.py --config configs/smoke.yaml --step generate` →
subprocess dies with SIGABRT (exit -6). Log shows the shim applied, piper loaded
`en_US-libritts_r-medium.pt`, took the GPU branch ("CUDA available, using GPU"), then:

```
/__w/pytorch/pytorch/third_party/torch-xpu-ops/src/ATen/native/xpu/sycl/Indexing.h:622:
operator(): global id: [129,0,0] ... Assertion `index >= -sizes_[i] && index < sizes_[i]
&& "index out of bounds"` failed.   (repeated for many work-items)
AssertHandler::printMessage
```

Environment: torch 2.13.0+xpu, LNL iGPU (Arc 130V/140V-class), piper-sample-generator
3.2.0 (master), piper-tts 1.3.0. Same code path works on CPU (baseline) — difference is
model+inputs moved to xpu via the shim.

## Hypotheses (unranked, to test)

- H1: genuine OOB index in piper's generate path that CUDA never traps (device asserts
  compiled out on release CUDA builds) but XPU traps — e.g. phoneme id ≥ embedding size.
- H2: torch-xpu-ops bug in an indexing op (embedding / index_select / gather) for this
  shape/dtype combo.
- H3: shim artifact — some tensor stayed on CPU / wrong device mix produces garbage
  indices only on the xpu path (e.g. `x.cuda()` on a LongTensor path we translated but
  a sibling tensor untouched).

## Findings

- (2026-07-11 18:31) Crash reproduced on first run; 0 clips generated. Not intermittent.
- (18:35) Stage-bisect (scratchpad/repro_xpu_indexing.py): CPU control passes all stages;
  XPU dies inside `model.dp` (stochastic duration predictor, reverse) →
  `piper_train/vits/transforms.py:75` `outputs[outside_interval_mask] = inputs[outside_interval_mask]`
  → `RuntimeError: numel: integer multiplication overflow` (sync) / device assert (async).
- (18:40) H1 eliminated: phoneme ids max 120 < vocab 256; same data passes on CPU.
- (18:45) Instrumented spy at the spline: inputs (1,1,23) gapped view (stride 46,23,1),
  widths/heights/derivs non-contiguous. Saved tensors to scratchpad/spline_args.pt.
- (18:50) **ROOT CAUSE — H2 confirmed, and it's worse than an indexing edge case:
  `torch.nonzero()` itself returns wrong results on this stack.** Evidence:
  - `torch.tensor([1,0,1,0,1], device='xpu').nonzero()` → `[0,0,4]` (expect `[0,2,4]`). Deterministic.
  - bool masks: `sum()`=9 but `nonzero()`=0 rows for shapes (23,), (1,23), (1,1,23), (1,1,100);
    (1,1,1000) → 263 of 503; one gapped-view case returned **535 rows from a 23-element
    tensor** — the source of the `Indexing.h index out of bounds` device asserts.
  - Shapes (1,2,23), (2,1,23) are correct. Failure correlates with small/flat shapes, not
    provenance (fresh tensors fail equally; `.clone()` doesn't help).
  - All masked indexing (`t[mask]`, `masked_select`, index_put via mask) uses nonzero
    internally → piper's spline transform is collateral damage.
- Environment: torch 2.13.0+xpu, intel-opencl-icd 26.05.37020.3, kernel 7.0.0-27,
  LNL iGPU (Arc 130V/140V). NOTE: may be specific to this driver/GPU combo — B70 box
  must re-run scratchpad/repro grid before trusting any XPU result.

## Resolution

- (19:05) Version bisect across torch xpu wheels ON THE SAME driver stack:
  - torch 2.13.0+xpu: nonzero BROKEN (all evidence above)
  - torch 2.12.1+xpu: nonzero CORRECT (throwaway venv test)
  - torch 2.9.1+xpu: nonzero CORRECT, full grid + gapped-view masked_select correct
  → regression in the torch xpu wheel line, not in the driver. Nearest public issues:
  pytorch#170166 (count_nonzero XPU crash regression, ~2.10 nightlies), pytorch#146883
  (aten.nonzero layout deviation), intel/torch-xpu-ops#1506 (BMG/LNL accuracy fails).
  None is an exact match — candidate for an upstream bug report.
- **Fix applied: project pinned to torch==2.9.1+xpu + torchaudio==2.9.1+xpu** (the only
  coherent pair below 2.13 on the xpu index — torchaudio skipped 2.10, and 2.11.0
  requires torch 2.13.0 exactly, so torch 2.12.x has no torchaudio at all).
- Permanent guard: `check_xpu_sanity.py` — run on every new box/driver/torch combo
  before trusting any XPU result. PASSES on this stack.

## Follow-up finding (same session): shim collateral in torch.onnx.export

On torch 2.9.1, the XPU train step completed BOTH training sequences but died at ONNX
export: the dynamo exporter probes `torch.cuda.is_available()` (shimmed → True) and then
initializes CUDA proper → `AssertionError: Torch not compiled with CUDA enabled`.
Fix: shim now saves the real `is_available` (`torch._oww_real_cuda_is_available`);
vendored export sites wrap in `_unshimmed_cuda()` and pass `dynamo=False` (legacy
tracer — upstream's opset-13 export was written for it anyway).
Lesson: a lying `is_available()` is a loaded gun for any torch-internal feature gate;
scope the lie tightly.

## Status: RESOLVED (2026-07-11) — pipeline unblocked, XPU smoke rerun pending.
