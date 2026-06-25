---
title: "ImageWatermarker<T>"
description: "Embeds and detects invisible watermarks in images using frequency-domain techniques."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Watermarking`

Embeds and detects invisible watermarks in images using frequency-domain techniques.

## For Beginners

Image watermarking hides an invisible "stamp" inside a picture.
The stamp is embedded in the image's frequency patterns (mathematical patterns that make
up the image), not in the visible pixels. This means you can't see the watermark, but a
computer can detect it even after the image is compressed or resized.

## How It Works

Uses a frequency-domain approach inspired by SynthID-Image (Google DeepMind, 2025) and
StegaStamp. The watermark is detected by analyzing mid-frequency band statistics in the
image's spectral representation. Watermarked images exhibit characteristic patterns in
their frequency coefficients that differ from natural images.

**Detection algorithm:**

1. Extract rows of pixel data from the image
2. Apply FFT to each row to get frequency-domain representation
3. Analyze mid-frequency band statistics (where watermarks are typically embedded)
4. Compute deviation from expected natural image statistics
5. Aggregate per-row scores into a final detection confidence

**References:**

- SynthID-Image: Internet-scale AI image watermarking (Google DeepMind, 2025)
- StegaStamp: Robust image steganography (Berkeley, 2019, still state-of-art robustness)
- Tree-Ring Watermarks: Invisible but detectable in diffusion images (2023)
- Gaussian Shading: Provable watermarking for diffusion models (CVPR 2024)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ImageWatermarker(Double,Double)` | Initializes a new image watermarker. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsReady` |  |
| `ModuleName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeBimodalityScore(Vector<>,Int32)` | Detects bimodality in magnitude distribution using Sarle's bimodality coefficient. |
| `ComputeSpectralFlatness(Vector<>,Int32)` | Computes spectral flatness of a magnitude vector: geometric mean / arithmetic mean. |
| `DetectWatermarkFrequencyDomain(Tensor<>)` | Detects watermarks by analyzing frequency-domain characteristics of the image. |
| `Evaluate(Vector<>)` |  |
| `EvaluateImage(Tensor<>)` | Detects whether the given image contains a watermark by analyzing frequency-domain statistics. |

