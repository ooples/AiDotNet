---
title: "SignalToNoiseRatio<T>"
description: "Signal-to-Noise Ratio (SNR) metric for audio quality assessment."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Metrics`

Signal-to-Noise Ratio (SNR) metric for audio quality assessment.

## How It Works

SNR measures the ratio of signal power to noise power. Higher values indicate cleaner audio.

Formula: SNR = 10 * log10(P_signal / P_noise)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SignalToNoiseRatio` | Initializes a new instance of SNR calculator. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Compute(Tensor<>,Tensor<>)` | Computes SNR between a clean signal and a noisy signal. |
| `ComputeSegmental(Tensor<>,Tensor<>,Int32)` | Computes segmental SNR (average SNR over short segments). |

