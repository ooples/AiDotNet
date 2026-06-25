---
title: "SpectralAudioWatermarker<T>"
description: "Audio watermarker that embeds watermarks in the frequency domain using spectral modification."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Watermarking`

Audio watermarker that embeds watermarks in the frequency domain using spectral modification.

## For Beginners

This watermarker hides a signature in the audio's frequency
spectrum — the mathematical representation of pitch and tone. The watermark is placed
in frequencies that the human ear is least sensitive to, making it inaudible.

## How It Works

Embeds watermark bits by modifying specific frequency band magnitudes in the audio spectrum.
Uses sub-band coding to spread the watermark across psychoacoustically masked frequencies,
making it inaudible while robust to compression and resampling.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SpectralAudioWatermarker(Double)` | Initializes a new spectral audio watermarker. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ModuleName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DetectWatermark(Vector<>,Int32)` |  |
| `EvaluateAudio(Vector<>,Int32)` |  |

