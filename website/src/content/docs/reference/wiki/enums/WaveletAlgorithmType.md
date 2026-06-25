---
title: "WaveletAlgorithmType"
description: "Represents different types of wavelet transform algorithms for signal processing."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums.AlgorithmTypes`

Represents different types of wavelet transform algorithms for signal processing.

## For Beginners

Wavelet transforms are mathematical techniques that break down signals (like audio, 
images, or any data that changes over time) into different frequency components, similar to how 
a prism breaks light into different colors.

Unlike traditional Fourier transforms that only give frequency information, wavelets show both:

- What frequencies are present (like bass or treble in music)
- When these frequencies occur in time (like knowing exactly when a drum hit happens)

This makes wavelets extremely useful for:

- Analyzing signals that change over time
- Compressing images and audio (like JPEG2000)
- Removing noise from signals
- Detecting patterns or features in data
- Many scientific and engineering applications

Think of wavelets as special measuring tools that can zoom in on both short-lived and long-lasting 
patterns in your data, giving you a more complete picture than traditional methods.

## Fields

| Field | Summary |
|:-----|:--------|
| `DWT` | Discrete Wavelet Transform - the standard wavelet transform algorithm. |
| `MODWT` | Maximal Overlap Discrete Wavelet Transform - a redundant wavelet transform that preserves time invariance. |
| `SWT` | Stationary Wavelet Transform - another non-decimated wavelet transform similar to MODWT. |

