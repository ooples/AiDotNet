---
title: "ProvenanceDeepfakeDetector<T>"
description: "Detects deepfake/AI-generated images by analyzing provenance signals: compression artifacts, statistical fingerprints, and embedded watermark traces."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Image`

Detects deepfake/AI-generated images by analyzing provenance signals: compression artifacts,
statistical fingerprints, and embedded watermark traces.

## For Beginners

Real photos carry invisible "fingerprints" from the camera that took
them — specific noise patterns, compression artifacts, and color processing signatures. AI
images don't have these, or they have different ones. This module checks for the presence
or absence of these fingerprints to determine if an image is AI-generated.

## How It Works

AI-generated images often lack natural camera processing artifacts (JPEG quantization patterns,
sensor noise, optical aberrations) or contain telltale signs of specific generators (GAN
checkerboard artifacts, diffusion model smoothing). This detector analyzes these provenance
signals without requiring frequency domain analysis.

**Detection signals:**

1. JPEG artifact analysis — real photos have characteristic quantization patterns
2. Noise floor consistency — cameras produce consistent sensor noise
3. Color channel correlation — natural images have specific cross-channel statistics
4. Local Binary Pattern (LBP) texture analysis — AI textures differ from natural

**References:**

- C2P-CLIP: Content provenance detection (2024, arxiv:2404.09677)
- SynthID-Image: Internet-scale watermarking for provenance (DeepMind, 2025, arxiv:2510.09263)
- Only 38% of AI generators have adequate watermarking (2025, arxiv:2503.18156)
- AI-generated media detection survey (2025, arxiv:2502.05240)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ProvenanceDeepfakeDetector(Double)` | Initializes a new provenance-based deepfake detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ModuleName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AnalyzeColorStatistics(ReadOnlySpan<>,ProvenanceDeepfakeDetector<>.ImageLayout)` | Analyzes color statistics. |
| `AnalyzeJPEGArtifacts(ReadOnlySpan<>,ProvenanceDeepfakeDetector<>.ImageLayout)` | Analyzes JPEG-like quantization artifacts. |
| `AnalyzeLBPTexture(ReadOnlySpan<>,ProvenanceDeepfakeDetector<>.ImageLayout)` | Analyzes Local Binary Pattern (LBP) texture distribution. |
| `AnalyzeNoiseFloor(ReadOnlySpan<>,ProvenanceDeepfakeDetector<>.ImageLayout)` | Analyzes noise floor characteristics. |
| `EvaluateImage(Tensor<>)` |  |

