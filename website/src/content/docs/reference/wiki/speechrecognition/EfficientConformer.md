---
title: "EfficientConformer<T>"
description: "Efficient Conformer with progressive frequency/time downsampling."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.ConformerFamily`

Efficient Conformer with progressive frequency/time downsampling.

## For Beginners

Progressively downsamples both frequency and time dimensions through the encoder layers, reducing computation while preserving accuracy. Groups of encoder layers operate at different resolutions, with strided convolution transitions between groups...

## How It Works

**References:**

- Paper: "Efficient Conformer: Progressive Downsampling and Grouped Attention" (Burchi & Vielzeuf, 2021)

Progressively downsamples both frequency and time dimensions through the encoder layers,
reducing computation while preserving accuracy. Groups of encoder layers operate at
different resolutions, with strided convolution transitions between groups.
Achieves similar accuracy to standard Conformer with ~30% fewer FLOPs.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using progressive downsampling Conformer encoder. |

