# Third-party provenance

This repository is licensed under the Apache License, Version 2.0 (see `LICENSE`)
for its own original contributions — the Intel Arc/`xpu` port itself (`device.py`,
the XPU-specific portions of `compat.py`, `check_xpu_sanity.py`, and the README/
`docs/port_walkthrough*.md`/`docs/xpu_nonzero_bug.md` documentation of that port).
It also incorporates modified or unmodified code and text from upstream projects,
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
`compat.py`, the `configs/hey_echo.yaml`/`configs/smoke.yaml` structure,
`docs/training_notes.md`, and `UPSTREAM-README.md` (kept verbatim, see the banner
at its top) are adapted or copied from this fork, which modernized openWakeWord's
training pipeline for current `torchaudio`/Piper TTS/`speechbrain` before this
project retargeted its device handling to Intel Arc. Full MIT license text
preserved at `third_party_licenses/LICENSE-openwakeword-trainer-MIT.txt`, per that
license's own terms.

## Piper / piper-sample-generator

**Rhasspy.** https://github.com/rhasspy/piper-sample-generator — used, not
modified or vendored; installed by this pipeline's own `download` step at
runtime (see `requirements.lock.txt`'s note).

## Training data and models fetched at runtime (not distributed here)

This repository contains no datasets, voice checkpoints, or trained models. The
`download` step fetches the following third-party resources onto the user's machine;
each remains under its own terms. Licence statements below are quoted or paraphrased
from the upstream pages as retrieved on 2026-09-29 — check the linked sources before
relying on them.

| Resource | Used for | Licence stated upstream |
|---|---|---|
| [Piper `en_US-libritts_r-medium`](https://huggingface.co/datasets/rhasspy/piper-checkpoints/tree/main/en/en_US/libritts_r/medium) (Rhasspy) | Synthetic speech generation | Repository licence: MIT. Its model card names the training data as [LibriTTS-R](http://www.openslr.org/141/), **CC BY 4.0** (attribution required; cite Koizumi et al., 2023, "LibriTTS-R: A Restored Multi-Speaker Text-to-Speech Corpus") |
| [openWakeWord pre-computed features](https://huggingface.co/datasets/davidscripka/openwakeword_features) (ACAV100M negatives; validation set built from DiPCo, Santa Barbara Corpus and MUSDB per its card) | Negative training data; false-positive validation | **CC BY-NC-SA 4.0** |
| [MIT Environmental Impulse Response Survey](https://mcdermottlab.mit.edu/Reverb/IR_Survey.html), 16 kHz copy at [`davidscripka/MIT_environmental_impulse_responses`](https://huggingface.co/datasets/davidscripka/MIT_environmental_impulse_responses) | Reverb augmentation | Dataset card: "unknown"; the MIT page states no terms. Cite MIT's Computational Audition Lab |
| [AudioSet](https://huggingface.co/datasets/agkphysics/AudioSet) (`agkphysics/AudioSet`) | Background-noise augmentation | Card: CC BY 4.0. Clips are YouTube audio; the card does not address the rights of individual videos |
| [FMA `fma_small`](https://github.com/mdeff/fma) | Background-noise augmentation | Metadata CC BY 4.0. Audio: "We do not hold the copyright on the audio and distribute it under the license chosen by the artist" — licences vary per track |
| openWakeWord embedding model | Frozen embedding stage of the cascade | Apache-2.0 (Google TFHub speech embedding, re-implemented by openWakeWord) |
| openWakeWord melspectrogram model | Mel stage of the cascade | Part of the Apache-2.0 openWakeWord repository; no separate statement found |

### Models trained with this pipeline

The non-commercial, share-alike features above are used to train the negative class,
and several augmentation sources have unknown or mixed clip-level licences. openWakeWord
itself licenses all of its pre-trained models CC BY-NC-SA 4.0 "due to the inclusion of
datasets with unknown or restrictive licensing as part of the training data". Anyone
publishing a model trained with this pipeline as-is should assume the same and
distribute it under **CC BY-NC-SA 4.0** with the attributions above. Whether trained
weights legally constitute a derivative of their training data is unsettled; this is a
conservative default, not legal advice. Obtaining a permissively licensed model requires
replacing the non-permissive data sources and retraining.

Any recordings of the user's own voice or of other speakers added to training are the
responsibility of whoever supplies them; none are included in this repository.
