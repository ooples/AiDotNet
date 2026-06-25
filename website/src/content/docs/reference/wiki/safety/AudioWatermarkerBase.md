---
title: "AudioWatermarkerBase<T>"
description: "Abstract base class for audio watermarking modules."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Safety.Watermarking`

Abstract base class for audio watermarking modules.

## For Beginners

This base class provides common code for all audio watermarkers.
Each watermarker type extends this and adds its own way of embedding invisible
signatures in audio content.

## How It Works

Provides shared infrastructure for audio watermarkers including strength
configuration and spectral utilities. Concrete implementations provide
the actual watermarking technique (spread-spectrum, AudioSeal, spectral).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AudioWatermarkerBase(Double)` | Initializes the audio watermarker base. |

## Methods

| Method | Summary |
|:-----|:--------|
| `DetectWatermark(Vector<>,Int32)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `WatermarkStrength` | The watermark strength factor (0.0 to 1.0). |

