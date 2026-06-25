---
title: "BSRoFormer<T>"
description: "BS-RoFormer (Band-Split Rotary Transformer) for music source separation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.SourceSeparation`

BS-RoFormer (Band-Split Rotary Transformer) for music source separation.

## For Beginners

BS-RoFormer separates a mixed song into stems (vocals, drums, bass, other)
by dividing frequencies into bands and using AI to figure out which parts belong to which instrument.

**Usage:**

## How It Works

BS-RoFormer (Lu et al., 2023) splits the spectrogram into frequency bands, processes each
with Transformer layers using rotary position embeddings, then fuses results for mask estimation.
Achieves 12.8 dB SDR on vocals separation (MUSDB18-HQ).

