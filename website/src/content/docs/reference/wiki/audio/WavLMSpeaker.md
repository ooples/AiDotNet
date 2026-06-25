---
title: "WavLMSpeaker<T>"
description: "WavLM-based speaker verification and embedding extraction model (Chen et al., 2022)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Speaker`

WavLM-based speaker verification and embedding extraction model (Chen et al., 2022).

## For Beginners

WavLM first learned to understand speech generally by "listening"
to thousands of hours of audio. Then it was specialized to recognize individual voices.
Because of this broad training, it handles noisy phone calls, different microphones, and
room echoes much better than models trained only on clean speech.

**Usage:**

## How It Works

WavLM is a self-supervised speech model that, when fine-tuned for speaker verification,
achieves 0.59% EER on VoxCeleb1—among the best results for any single model. Its
Transformer encoder is pre-trained with masked speech prediction and denoising, making
it robust to noise and reverberation.

