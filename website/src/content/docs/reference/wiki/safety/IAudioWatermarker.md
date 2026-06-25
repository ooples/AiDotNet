---
title: "IAudioWatermarker<T>"
description: "Interface for audio watermarking modules that embed and detect watermarks in audio."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Safety.Watermarking`

Interface for audio watermarking modules that embed and detect watermarks in audio.

## For Beginners

An audio watermarker adds an invisible signature to audio content.
Even after the audio is compressed or slightly modified, the watermark can still be
detected to prove the audio was AI-generated.

## How It Works

Audio watermarkers embed imperceptible watermarks in audio using spread-spectrum,
frequency domain, or AudioSeal-style localized techniques. The watermark survives
common transformations like compression, resampling, and noise addition.

**References:**

- AudioSeal: Localized watermarking (Meta AI, 2024, arxiv:2401.17264)
- Only 38% of AI generators implement adequate watermarking (2025, arxiv:2503.18156)

## Methods

| Method | Summary |
|:-----|:--------|
| `DetectWatermark(Vector<>,Int32)` | Detects the watermark confidence score in the given audio (0.0 = no watermark, 1.0 = certain). |

