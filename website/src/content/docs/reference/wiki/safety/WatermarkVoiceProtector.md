---
title: "WatermarkVoiceProtector<T>"
description: "Protects voice recordings by embedding imperceptible watermarks that survive voice cloning and can be detected in cloned output."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Audio`

Protects voice recordings by embedding imperceptible watermarks that survive voice cloning
and can be detected in cloned output.

## For Beginners

This embeds a secret "tag" in the audio that follows the voice
even if someone tries to clone it. If cloned audio shows up later, the tag proves where
the original voice came from.

## How It Works

Embeds a spread-spectrum watermark into the audio's mid-frequency band that is robust to
typical voice processing (resampling, compression, noise addition) and voice cloning.
The watermark can later be detected using `WatermarkDeepfakeDetector`.
Uses frequency-domain embedding with psychoacoustic masking to ensure inaudibility.

**References:**

- AudioSeal: Localized watermarking for speech (Meta AI, 2024, arxiv:2401.17264)
- SoK: Systematization of watermarking across modalities (2024, arxiv:2411.18479)
- Watermarking survey: unified framework (2025, arxiv:2504.03765)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `WatermarkVoiceProtector(Double,Int32,Int32)` | Initializes a new watermark-based voice protector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ModuleName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `EmbedWatermark(Vector<>,Int32)` | Embeds a watermark into the audio and returns the watermarked audio. |
| `EvaluateAudio(Vector<>,Int32)` |  |

