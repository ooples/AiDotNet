---
title: "IQVFIOptions"
description: "Configuration options for IQ-VFI image quality-aware video frame interpolation."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Video.Options`

Configuration options for IQ-VFI image quality-aware video frame interpolation.

## For Beginners

Most frame interpolation methods assume input frames are clean and
high-quality. IQ-VFI first checks how good each part of the input frames is, then adjusts
its interpolation strategy accordingly. This means it works better on real-world videos
that may have noise, blur, or compression artifacts.

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

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `IQVFIOptions` | Initializes a new instance with default values. |
| `IQVFIOptions(IQVFIOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumFeatures` | Gets or sets the number of feature channels. |
| `NumFlowRefinementIters` | Gets or sets the number of flow refinement iterations. |
| `NumPyramidLevels` | Gets or sets the number of pyramid levels for multi-scale processing. |
| `NumQualityBlocks` | Gets or sets the number of quality assessment blocks. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `QualityThreshold` | Gets or sets the quality threshold below which conservative interpolation is used. |
| `Variant` | Gets or sets the model variant. |

