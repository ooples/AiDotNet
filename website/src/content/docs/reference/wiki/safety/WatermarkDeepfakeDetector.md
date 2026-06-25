---
title: "WatermarkDeepfakeDetector<T>"
description: "Detects AI-generated audio by looking for the presence or absence of known watermark patterns (e.g., AudioSeal-style localized watermarks)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Audio`

Detects AI-generated audio by looking for the presence or absence of known watermark
patterns (e.g., AudioSeal-style localized watermarks).

## For Beginners

Some AI companies embed invisible "watermarks" in the audio they
generate — like a secret signature proving it's AI-made. This module looks for those
signatures. If found, we know it's AI-generated. If we also detect other AI artifacts
but no watermark, it might be an attempt to hide AI origin.

## How It Works

Many responsible AI audio generators embed watermarks into their output. This detector
checks for characteristic patterns of known watermarking schemes: spectral energy in
specific sub-bands, phase-based encoding, and spread-spectrum signatures. The absence
of any camera/microphone artifacts combined with no watermark is also a signal.

**References:**

- AudioSeal: Localized watermarking for speech (Meta AI, 2024, arxiv:2401.17264)
- SoK: Systematization of watermarking across modalities (2024, arxiv:2411.18479)
- Only 38% of AI generators implement adequate watermarking (2025, arxiv:2503.18156)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `WatermarkDeepfakeDetector(Double,Int32)` | Initializes a new watermark-based deepfake detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ModuleName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `EvaluateAudio(Vector<>,Int32)` |  |

