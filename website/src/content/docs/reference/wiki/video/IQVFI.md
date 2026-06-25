---
title: "IQVFI<T>"
description: "IQ-VFI: image quality-aware video frame interpolation with degradation adaptation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.FrameInterpolation`

IQ-VFI: image quality-aware video frame interpolation with degradation adaptation.

## For Beginners

Most frame interpolation methods assume input frames are clean and
high-quality. IQ-VFI first checks how good each part of the input frames is, then adjusts
its interpolation strategy accordingly. This means it works better on real-world videos
that may have noise, blur, or compression artifacts.

**Usage:**

## How It Works

IQ-VFI (2024) adapts interpolation based on input image quality:

- Quality assessment module: estimates per-pixel quality scores for input frames using a

learned no-reference image quality assessment (NR-IQA) branch, identifying regions with
noise, blur, compression artifacts, or other degradations

- Degradation-adaptive flow: the optical flow estimation network receives quality maps as

additional conditioning, so it can be more conservative in degraded regions (where flow
estimation is unreliable) and more aggressive in clean regions

- Quality-guided fusion: the blending weights between warped frames incorporate quality

scores, favoring the higher-quality frame contribution in each spatial region

- Quality-aware training: training uses a quality-stratified sampling strategy that ensures

the model sees diverse degradation levels and learns robust interpolation for each

**Reference:** "IQ-VFI: Image Quality-Aware Video Frame Interpolation" (2024)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `IQVFI(NeuralNetworkArchitecture<>,IQVFIOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates an IQ-VFI model in native training mode. |
| `IQVFI(NeuralNetworkArchitecture<>,String,IQVFIOptions)` | Creates an IQ-VFI model in ONNX inference mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Interpolate(Tensor<>,Tensor<>,Double)` |  |

