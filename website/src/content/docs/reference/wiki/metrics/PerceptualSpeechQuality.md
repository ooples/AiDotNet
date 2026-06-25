---
title: "PerceptualSpeechQuality<T>"
description: "Perceptual Evaluation of Speech Quality (PESQ) approximation metric."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Metrics`

Perceptual Evaluation of Speech Quality (PESQ) approximation metric.

## How It Works

This is a simplified approximation of the ITU-T P.862 PESQ algorithm.
For production use, consider using an official PESQ implementation.

PESQ scores range from -0.5 to 4.5, where higher values indicate better quality.

- >4.0: Excellent quality
- 3.5-4.0: Good quality
- 3.0-3.5: Fair quality
- <3.0: Poor quality

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PerceptualSpeechQuality(Int32)` | Initializes a new instance of PESQ approximation calculator. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Compute(Tensor<>,Tensor<>)` | Computes an approximation of PESQ score between degraded and reference speech. |
| `MapToPesqScale(Double,Double)` | Maps STOI and SNR to an approximate PESQ score. |

