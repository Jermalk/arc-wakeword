# Third-party provenance

This repository is licensed under the Apache License, Version 2.0 (see `LICENSE`)
for its own original contributions — the Intel Arc/`xpu` port itself (`device.py`,
the XPU-specific portions of `compat.py`, `check_xpu_sanity.py`, and this project's
documentation). It also incorporates modified code from two upstream projects,
credited here per each project's own license terms.

## openWakeWord (Apache License 2.0)

**Copyright © David Scripka.** https://github.com/dscripka/openWakeWord

`oww_train_xpu.py` is a modified, vendored copy of openWakeWord 0.6.0's
`openwakeword/train.py`, retargeted from a hardcoded CUDA device to a
resolved-once `xpu > cuda > cpu` device (see the file's own header comment for
what changed). Same license as this repository (Apache-2.0), so no separate
license text is needed here beyond this attribution.

## openwakeword-trainer fork (MIT License)

**Copyright (c) 2026 Luke Pearson.** https://github.com/lgpearson1771/openwakeword-trainer

`train_wakeword.py`, `oww_wrapper.py`, the original (non-XPU) portions of
`compat.py`, and the `configs/hey_echo.yaml`/`configs/smoke.yaml` structure are
adapted from this fork, which modernized openWakeWord's training pipeline for
current `torchaudio`/Piper TTS/`speechbrain` before this project retargeted its
device handling to Intel Arc. Full MIT license text preserved at
`third_party_licenses/LICENSE-openwakeword-trainer-MIT.txt`, per that license's
own terms.

## Piper / piper-sample-generator

**Rhasspy.** https://github.com/rhasspy/piper-sample-generator — used, not
modified or vendored; installed by this pipeline's own `download` step at
runtime (see `requirements.lock.txt`'s note).
