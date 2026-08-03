# arc-wakeword

Retargeting [openWakeWord](https://github.com/dscripka/openWakeWord)'s training
pipeline from NVIDIA CUDA to Intel Arc, using PyTorch's native `xpu` backend —
**training** a wake-word model on Intel hardware, not just running inference on one.

## Why this exists

openWakeWord is one of the best open-source wake-word frameworks, but its *training*
pipeline is frozen in 2022: pinned to PyTorch 1.13.1, with a hard CUDA requirement and
setup instructions that assume WSL2 with NVIDIA passthrough. Inference — running an
already-trained model — works fine anywhere ONNX Runtime does. Training doesn't, unless
you own an NVIDIA card.

PyTorch has shipped a native Intel GPU backend (`torch.xpu`, upstreamed since 2.5) for a
while now. Nobody had connected the two. This repo is that port: every CUDA-only
assumption in the training pipeline, found and retargeted, plus the diagnostic tooling
that caught a real correctness bug along the way (see below) — proof that the training
side of this ecosystem can move off CUDA too, not just the inference side.

## What's actually demonstrated here

- **A single point of device resolution** (`device.py`) and a **monkeypatch shim
  layer** (`compat.py`) that lets un-vendored upstream code (Piper TTS, openWakeWord's
  own augmentation) run on Arc without being forked — vs. one vendored, explicitly
  diffed copy of the actual training entrypoint (`oww_train_xpu.py`) where the device
  logic was inline and unpatchable. The split is deliberate: *vendor what you can
  audit as a unit, shim what you'd otherwise have to fork.*
- **A standalone kernel-correctness check** (`check_xpu_sanity.py`, run by hand
  before trusting a new machine) that exists because of a real bug this project
  found: PyTorch 2.13.0+xpu shipped a broken `nonzero` kernel that silently returns
  wrong results for a wide range of ordinary tensor shapes — not a crash, just
  wrong arithmetic, which is worse. Full investigation, including the bisection
  method used to find it: `docs/xpu_nonzero_bug.md`.
- **A pinned, reproducible environment** — exact `torch`/`torchaudio` xpu-wheel
  versions, plus a full `requirements.lock.txt` frozen environment, not just "should
  work with recent versions."

**Hardware validated:** this exact code (not a variant) has run end-to-end on a CPU
baseline, a Lunar Lake iGPU (Arc 130V/140V-class) for the full pipeline, and an Intel Arc
Pro B70 (32 GB, Battlemage) for a full-scale production training run in a downstream
project. See `docs/port_walkthrough.md` for the CPU-vs-iGPU timing breakdown.

## Prerequisites

- **Linux** (the pipeline hard-checks this; `piper-phonemize` also has no non-Linux
  wheels). Not WSL2-specific — any Linux box, including bare-metal.
- **Python 3.11** (3.14 is too new for this stack's `piper-phonemize` wheels).
- **~15 GB free disk** and a working internet connection. The `download` pipeline
  step fetches this regardless of which config you run — see the warning below,
  it applies to the smoke config too.
- **Optional, for Intel Arc:** the Intel compute runtime / GPU driver stack that
  makes `torch.xpu.is_available()` return `True` (this repo doesn't install or
  configure that — see your distro's Intel GPU driver docs). Without it, the
  pipeline still runs correctly on CPU — see `check_xpu_sanity.py` below.

## Quickstart

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # if uv is missing
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python \
    torch==2.9.1+xpu torchaudio==2.9.1+xpu --index-url https://download.pytorch.org/whl/xpu
uv pip install --python .venv/bin/python -r requirements.txt
```

Both commands matter, in order — `requirements.txt` alone has no index URL and
can't resolve the pinned `+xpu` wheels by itself. (`requirements.lock.txt` is a
full frozen snapshot of an environment that passed end-to-end, kept for reference/
debugging dependency issues — it's not meant to be installed directly instead of
the two commands above.)

**Before training on any new machine, run:**

```bash
.venv/bin/python check_xpu_sanity.py
```

This is a manual step, not something the pipeline runs for you automatically. If
it reports a failure, don't train — fix the driver/torch combination first (see
`docs/xpu_nonzero_bug.md` for exactly what this catches and why). If your machine
has no Intel/NVIDIA GPU at all, it prints that XPU isn't available and exits
cleanly — that's expected, and training will proceed on CPU (slow, but correct).

**⚠️ Before running anything below: the `download` pipeline step fetches roughly
15 GB** (a ~7.5 GB precomputed negative-feature file, the Piper TTS model, and
assorted background-noise/room-impulse datasets) **regardless of config** — this
includes the "tiny" smoke config below, since sample/step counts don't change what
`download` fetches. Budget ~30 min and the disk space before starting either
command below; it's a one-time cost, cached in `data/` for subsequent runs.

**Run the smoke config** (tiny, ~500 steps — proves the pipeline runs end-to-end, not a
usable model):

```bash
.venv/bin/python train_wakeword.py --config configs/smoke.yaml
```

**Train your own wake word:** copy `configs/hey_echo.yaml`, edit `model_name`,
`target_phrase`, and `custom_negative_phrases`, then run
`train_wakeword.py --config configs/your_word.yaml`. Steps in order:
`check-env → apply-patches → download → verify-data → resolve-config → generate →
resample-clips → verify-clips → augment → verify-features → train → verify-model →
export`. Running the command above with no `--from`/`--step` flag runs all 13 steps
back to back with no confirmation prompt — use `--step download` first if you'd
rather watch that step alone before committing to the rest; resume from any step
with `--from <step>`, or run one step alone with `--step <step>`.

**Where your model ends up:** `export/<model_name>.onnx` plus
`export/<model_name>.onnx.data` (external weights) — both files are required
together, copy both wherever you deploy.

## Reproducibility notes

The pinned environment, lock file, and the sanity check above are meant to make
this actually reproducible, not just "should work." One honest limitation, stated
plainly rather than glossed over: **the training loop itself has no seed control.**
In the downstream project this pipeline was used for, an identical config reran on
identical hardware produced a 14x swing in one key metric — a real, measured
finding, not a hypothetical caveat. Treat any single training run's numbers as one
sample, not a reproducible ground truth, until you've confirmed otherwise on your
own runs.

## Architecture crib

- `train_wakeword.py` — the pipeline orchestrator (13 steps, each with its own
  verification).
- `compat.py` — runtime monkeypatches applied *before* openWakeWord imports:
  torchaudio 2.10+ shims, a Piper API-shape shim, and the guarded **cuda→xpu shim**
  (`torch.cuda.is_available` → true on Intel hardware, `.cuda()`/`.to('cuda*')` → xpu)
  that lets unmodified upstream code run on Arc.
- `oww_wrapper.py` — applies the compat patches, then runs `oww_train_xpu`.
- `oww_train_xpu.py` — vendored `openwakeword/train.py` (0.6.0) with device logic
  resolved once via `device.py` instead of hardcoded CUDA; header documents exactly
  what changed vs. upstream.
- `device.py` — the single point of device resolution (`xpu > cuda > cpu`). No code
  path anywhere else may hardcode `"cuda"`.
- `check_xpu_sanity.py` — standalone kernel-correctness check (`nonzero` / masked
  indexing), run manually, not wired into the pipeline's own steps.

## Further reading

- `docs/port_walkthrough.md` — the technical narrative of the port: every CUDA
  touchpoint found, the vendor-vs-shim strategy, the actual code, the timing numbers.
- `docs/port_walkthrough_for_students.md` — the same story, written for a reader who
  knows how to program but has never trained a model.
- `docs/xpu_nonzero_bug.md` — the full bisection log for the broken-kernel bug found
  during the port, kept in the form it was written during debugging.
- `docs/training_notes.md` — carried over from the upstream trainer fork: torchaudio
  2.10+ breakage, sample-rate handling, ONNX export, WSL2 filesystem notes, and a
  short section on architecture levers (layer size, RNN vs. DNN).
- `UPSTREAM-README.md` — the upstream fork's original README, kept verbatim for
  attribution (see the banner at its top — it predates this port and assumes
  NVIDIA/CUDA, don't follow it directly).

## What this is not

Not a pretrained-model repository — no `.onnx`/`.xml`/`.bin` artifacts are committed
(regenerable, and large; see `.gitignore`). Not a specific product's wake word — bring
your own target phrase via `configs/`.

## License

Apache License 2.0 for this project's own contributions (the CUDA→XPU port itself).
This repo also incorporates modified code from two upstream projects under their own
license terms — see `NOTICE.md` for full attribution.

## Acknowledgments

- [openWakeWord](https://github.com/dscripka/openWakeWord) by David Scripka
- [openwakeword-trainer](https://github.com/lgpearson1771/openwakeword-trainer) by
  Luke Pearson — modernized the training pipeline for current
  `torchaudio`/Piper/`speechbrain` before this project retargeted its device handling
- [Piper](https://github.com/rhasspy/piper) by Rhasspy, for synthetic TTS training data
