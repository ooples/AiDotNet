---
title: "AudioWatermarker<T>"
description: "Embeds and detects invisible watermarks in audio content using spread-spectrum techniques."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Watermarking`

Embeds and detects invisible watermarks in audio content using spread-spectrum techniques.

## For Beginners

Audio watermarking hides an invisible signal inside sound. The signal
is so quiet compared to the actual audio that humans can't hear it. But a computer can
detect it even after the audio has been compressed or had noise added.

## How It Works

Uses a spread-spectrum approach inspired by AudioSeal (Meta AI, 2024). The watermark signal
is spread across the audio spectrum at imperceptible energy levels. Detection uses correlation
analysis in the frequency domain to identify hidden patterns. The watermark survives common
audio transformations (compression, noise, filtering, resampling).

**Detection algorithm:**

1. Segment audio into overlapping frames
2. Apply FFT to each frame to get frequency-domain representation
3. Analyze mid-frequency magnitude patterns for watermark signatures
4. Compute spectral regularity — watermarks create unnaturally uniform patterns
5. Detect energy clustering in specific frequency bands
6. Aggregate per-frame scores into final detection confidence

**References:**

- AudioSeal: Proactive localized watermarking for speech (Meta AI, ICML 2024)
- WavMark: High-capacity audio watermarking (2024)
- Timbre watermarking: Robust audio watermarking via timbre modulation (2024)
- Audio watermark resilience under codec transformations (IEEE, 2024)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AudioWatermarker(Double,Double)` | Initializes a new audio watermarker. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsReady` |  |
| `ModuleName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeNormalizedCorrelation(Vector<>,Vector<>,Int32)` | Computes normalized cross-correlation between two magnitude vectors. |
| `DetectWatermarkSpreadSpectrum(Vector<>,Int32)` | Detects watermarks using spread-spectrum analysis in the frequency domain. |
| `Evaluate(Vector<>)` |  |
| `EvaluateAudio(Vector<>,Int32)` | Detects whether the given audio contains a watermark. |

