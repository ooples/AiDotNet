---
title: "Squeezeformer<T>"
description: "Squeezeformer speech recognition model with temporal U-Net structure."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.ConformerFamily`

Squeezeformer speech recognition model with temporal U-Net structure.

## For Beginners

Squeezeformer restructures Conformer with: (1) Temporal U-Net: downsamples time resolution in middle layers, upsamples at the end, (2) Micro-macro design: pre-norm (more stable training) replaces post-norm, (3) Simplified block: MHA → Conv → FF (r...

## How It Works

**References:**

- Paper: "Squeezeformer: An Efficient Transformer for Automatic Speech Recognition" (Kim et al., 2022)

Squeezeformer restructures Conformer with:
(1) Temporal U-Net: downsamples time resolution in middle layers, upsamples at the end,
(2) Micro-macro design: pre-norm (more stable training) replaces post-norm,
(3) Simplified block: MHA → Conv → FF (removes redundant second FF from Conformer).
These changes reduce compute by ~30% while matching or exceeding Conformer accuracy.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using Squeezeformer's temporal U-Net encoder. |

