---
title: "FrequencyDeepfakeDetector<T>"
description: "Detects AI-generated/deepfake images by analyzing frequency domain artifacts."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Image`

Detects AI-generated/deepfake images by analyzing frequency domain artifacts.

## For Beginners

Every image can be decomposed into waves of different frequencies
(like separating sound into bass, mid, and treble). AI-generated images have unusual
patterns in these frequency waves — like a fingerprint left by the AI. This module
detects those fingerprints.

## How It Works

AI-generated images (GANs, diffusion models) leave characteristic artifacts in the
frequency domain that are invisible to the human eye. This detector applies FFT to
image rows and columns, then analyzes the resulting spectrum for anomalies such as
periodic peaks (GAN fingerprints), unusual spectral roll-off patterns, and frequency
band energy distribution abnormalities.

**References:**

- Frequency analysis for deepfake detection via spectral artifacts (2020, arxiv:2003.08685)
- Generalizable deepfake detection across benchmarks (CVPR 2025, arxiv:2508.06248)
- NACO: Self-supervised natural consistency for face forgery detection (ECCV 2024, arxiv:2407.10550)
- AI-generated media detection survey: non-MLLM to MLLM (2025, arxiv:2502.05240)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FrequencyDeepfakeDetector(Double)` | Initializes a new frequency-domain deepfake detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ModuleName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `EvaluateImage(Tensor<>)` |  |

