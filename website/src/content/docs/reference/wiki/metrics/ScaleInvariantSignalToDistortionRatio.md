---
title: "ScaleInvariantSignalToDistortionRatio<T>"
description: "Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) metric for source separation evaluation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Metrics`

Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) metric for source separation evaluation.

## How It Works

SI-SDR is the standard metric for evaluating source separation quality. It measures
how well the estimated signal matches the target signal, ignoring scale differences.

Formula: SI-SDR = 10 * log10(||s_target||^2 / ||e_noise||^2)
where s_target is the projection of the estimate onto the target, and e_noise is the residual.

Higher values indicate better separation quality. Typical values:

- >15 dB: Excellent separation
- 10-15 dB: Good separation
- 5-10 dB: Fair separation
- <5 dB: Poor separation

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ScaleInvariantSignalToDistortionRatio` | Initializes a new instance of SI-SDR calculator. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Compute(Tensor<>,Tensor<>)` | Computes SI-SDR between an estimated signal and the target signal. |
| `ComputeImprovement(Tensor<>,Tensor<>,Tensor<>)` | Computes SI-SDR improvement relative to a baseline (typically the mixture). |

